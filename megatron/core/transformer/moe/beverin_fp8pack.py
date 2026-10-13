# beverin-rocm (lab T4): fused Triton pack/unpack for T1's fp8 forward dispatch (BEVERIN_FP8_DISPATCH).
# pack:   bf16 tokens [n, h] + fp32 probs [n] -> uint8 rows [q (h bytes e4m3fn) | per-128-group fp32 scales | fp32 prob]
# unpack: those rows -> bf16 tokens (q * scale, rounded to bf16) and fp32 probs.
# Bit-identical to the eager torch path (_beverin_fp8_quant + cat, and the slice/float/mul/to(bf16) unpack):
# scale = max(amax(|x_g|) / 448, 1e-30) in fp32, q = e4m3fn(x / scale) with round-to-nearest-even, encoded with
# integer arithmetic (no reliance on Triton's fp8 conversion on gfx942). One kernel each instead of ~12 eager ops.
import torch
import triton
import triton.language as tl


@triton.jit
def _e4m3fn_encode(v):
    # v: fp32 with |v| <= 448 (+ rounding); returns int32 code 0..255 (sign bit 0x80), RNE like torch
    bits = v.to(tl.int32, bitcast=True)
    sign = (bits >> 24) & 0x80
    a = tl.abs(v)
    abits = a.to(tl.int32, bitcast=True)
    # normal range (a >= 2^-6): round mantissa 23 -> 3 bits, RNE
    r = (abits + 0x7FFFF + ((abits >> 20) & 1)) >> 20
    code_n = r - ((127 - 7) << 3)
    # subnormal range (a < 2^-6): code = rne(a * 2^9)
    t = a * 512.0
    fl = tl.floor(t)
    frac = t - fl
    fli = fl.to(tl.int32)
    up = (frac > 0.5) | ((frac == 0.5) & ((fli & 1) == 1))
    code_s = fli + up.to(tl.int32)
    code = tl.where(a < 0.015625, code_s, code_n)
    code = tl.minimum(code, 0x7E)  # 448; inputs never exceed it after scaling (torch would give NaN beyond)
    return code | sign


@triton.jit
def _pack_kernel(x_ptr, p_ptr, out_ptr, H: tl.constexpr, G: tl.constexpr, GP: tl.constexpr, W: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    gi = tl.arange(0, GP)[:, None]
    ci = tl.arange(0, 128)[None, :]
    msk = gi < G
    x = tl.load(x_ptr + row * H + gi * 128 + ci, mask=msk, other=0.0).to(tl.float32)  # [GP, 128]
    amax = tl.max(tl.abs(x), axis=1)
    sf = tl.maximum(amax * (1.0 / 448.0), 1e-30)  # torch computes tensor / scalar as a multiply by the fp32 reciprocal
    q = _e4m3fn_encode(tl.math.div_rn(x, tl.broadcast_to(sf[:, None], x.shape)))
    base = out_ptr + row * W
    tl.store(base + gi * 128 + ci, q.to(tl.uint8), mask=msk)
    sfb = sf.to(tl.int32, bitcast=True)  # little-endian bytes of the fp32 scales
    gj = tl.arange(0, GP)
    for k in tl.static_range(4):
        tl.store(base + H + gj * 4 + k, ((sfb >> (8 * k)) & 0xFF).to(tl.uint8), mask=gj < G)
    pb = tl.load(p_ptr + row).to(tl.int32, bitcast=True)
    for k in tl.static_range(4):
        tl.store(base + H + 4 * G + k, ((pb >> (8 * k)) & 0xFF).to(tl.uint8))


@triton.jit
def _unpack_kernel(in_ptr, y_ptr, p_ptr, H: tl.constexpr, G: tl.constexpr, GP: tl.constexpr, W: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    base = in_ptr + row * W
    gi = tl.arange(0, GP)[:, None]
    ci = tl.arange(0, 128)[None, :]
    msk = gi < G
    c = tl.load(base + gi * 128 + ci, mask=msk, other=0).to(tl.int32)
    e = (c >> 3) & 15
    mi = c & 7
    # exact decode: normal values built from fp32 bits, subnormals m * 2^-9 (exact)
    nbits = ((e + (127 - 7)) << 23) | (mi << 20)
    mag = tl.where(e == 0, mi.to(tl.float32) * 0.001953125, nbits.to(tl.float32, bitcast=True))
    val = (mag.to(tl.int32, bitcast=True) | ((c & 0x80) << 24)).to(tl.float32, bitcast=True)  # keeps -0.0
    gj = tl.arange(0, GP)
    sfb = tl.zeros([GP], dtype=tl.int32)
    for k in tl.static_range(4):
        sfb = sfb | (tl.load(base + H + gj * 4 + k, mask=gj < G, other=0).to(tl.int32) << (8 * k))
    sf = sfb.to(tl.float32, bitcast=True)
    tl.store(y_ptr + row * H + gi * 128 + ci, (val * sf[:, None]).to(tl.bfloat16), mask=msk)
    pb = tl.zeros([1], dtype=tl.int32)
    for k in tl.static_range(4):
        pb = pb | (tl.load(base + H + 4 * G + k + tl.arange(0, 1)).to(tl.int32) << (8 * k))
    tl.store(p_ptr + row + tl.arange(0, 1), pb.to(tl.float32, bitcast=True))


def pack(x, probs):
    n, h = x.shape
    g = h // 128
    assert h % 128 == 0 and x.dtype == torch.bfloat16 and probs.dtype == torch.float32
    w = h + 4 * g + 4
    out = torch.empty((n, w), dtype=torch.uint8, device=x.device)
    if n:
        _pack_kernel[(n,)](x.contiguous(), probs.contiguous(), out, H=h, G=g, GP=triton.next_power_of_2(g), W=w, num_warps=4)
    return out


def unpack(packed, h):
    m, w = packed.shape
    g = h // 128
    assert w == h + 4 * g + 4
    y = torch.empty((m, h), dtype=torch.bfloat16, device=packed.device)
    p = torch.empty((m,), dtype=torch.float32, device=packed.device)
    if m:
        _unpack_kernel[(m,)](packed.contiguous(), y, p, H=h, G=g, GP=triton.next_power_of_2(g), W=w, num_warps=4)
    return y, p
