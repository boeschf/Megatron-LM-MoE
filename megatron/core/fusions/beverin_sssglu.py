# beverin-rocm patch (lab T4): fused Triton SSSGLU (weighted and plain) for AMD.
# With TORCH_COMPILE_DISABLE=1 (needed on Beverin, per-shape recompiles hang at scale) Megatron's
# @jit_fuser SSSGLU runs eagerly: ~15 fp32/bf16 elementwise kernels over [rows, 2 x 2048] per expert
# call. These kernels do the same math in one pass each way, all in fp32 (as inductor would on
# GH200), with a deterministic per-row reduction for the probs gradient (no atomics).
# Enabled with BEVERIN_TRITON_SSSGLU=1 in megatron/core/fusions/fused_bias_sssglu.py.
import torch
import triton
import triton.language as tl


@triton.jit
def _fwd(x_ptr, p_ptr, y_ptr, M, D, HAS_P: tl.constexpr, BM: tl.constexpr, BD: tl.constexpr):
    pm = tl.program_id(0)
    pd = tl.program_id(1)
    rm = (pm * BM + tl.arange(0, BM)).to(tl.int64)
    rd = pd * BD + tl.arange(0, BD)
    mask = (rm[:, None] < M) & (rd[None, :] < D)
    xo = rm[:, None] * (2 * D) + rd[None, :]
    a = tl.load(x_ptr + xo, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(x_ptr + xo + D, mask=mask, other=0.0).to(tl.float32)
    u = a - 1.0
    y = (0.5 + u / (1.0 + tl.abs(u))) * b
    if HAS_P:
        p = tl.load(p_ptr + rm, mask=rm < M, other=0.0)
        y = y * p[:, None]
    tl.store(y_ptr + rm[:, None] * D + rd[None, :], y.to(y_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _bwd(x_ptr, p_ptr, g_ptr, gx_ptr, gp_ptr, M, D, HAS_P: tl.constexpr, BM: tl.constexpr,
         BD: tl.constexpr):
    pm = tl.program_id(0)
    rm = (pm * BM + tl.arange(0, BM)).to(tl.int64)
    mrow = rm < M
    if HAS_P:
        p = tl.load(p_ptr + rm, mask=mrow, other=0.0)
    acc = tl.zeros([BM], dtype=tl.float32)
    for d0 in range(0, D, BD):
        rd = d0 + tl.arange(0, BD)
        mask = mrow[:, None] & (rd[None, :] < D)
        xo = rm[:, None] * (2 * D) + rd[None, :]
        go = rm[:, None] * D + rd[None, :]
        a = tl.load(x_ptr + xo, mask=mask, other=0.0).to(tl.float32)
        b = tl.load(x_ptr + xo + D, mask=mask, other=0.0).to(tl.float32)
        g = tl.load(g_ptr + go, mask=mask, other=0.0).to(tl.float32)
        u = a - 1.0
        den = 1.0 + tl.abs(u)
        gate = 0.5 + u / den
        if HAS_P:
            acc += tl.sum(tl.where(mask, gate * b * g, 0.0), axis=1)
            g = g * p[:, None]
        ga = g * b / (den * den)
        gb = g * gate
        tl.store(gx_ptr + xo, ga.to(gx_ptr.dtype.element_ty), mask=mask)
        tl.store(gx_ptr + xo + D, gb.to(gx_ptr.dtype.element_ty), mask=mask)
    if HAS_P:
        tl.store(gp_ptr + rm, acc, mask=mrow)


_FBM, _FBD, _BBM, _BBD = 8, 512, 4, 512


def fwd(x2d, p=None):
    M, D2 = x2d.shape
    D = D2 // 2
    y = torch.empty((M, D), device=x2d.device, dtype=x2d.dtype)
    if M == 0:
        return y
    grid = (triton.cdiv(M, _FBM), triton.cdiv(D, _FBD))
    _fwd[grid](x2d, p if p is not None else x2d, y, M, D, HAS_P=p is not None, BM=_FBM, BD=_FBD,
               num_warps=8)
    return y


def bwd(g2d, x2d, p=None):
    M, D2 = x2d.shape
    D = D2 // 2
    gx = torch.empty_like(x2d)
    gp = torch.empty((M,), device=x2d.device, dtype=torch.float32) if p is not None else None
    if M == 0:
        return gx, gp
    grid = (triton.cdiv(M, _BBM),)
    _bwd[grid](x2d, p if p is not None else x2d, g2d, gx, gp if gp is not None else x2d, M, D,
               HAS_P=p is not None, BM=_BBM, BD=_BBD, num_warps=8)
    return gx, gp


class TritonWeightedSSSGLU(torch.autograd.Function):
    """y = gate(x1) * x2 * w, w per row ([M, 1], usually fp32 router probs)."""

    @staticmethod
    def forward(ctx, x, w):
        x = x.contiguous()
        p = w.reshape(-1).float().contiguous()
        ctx.save_for_backward(x, p)
        ctx.w_dtype, ctx.w_shape = w.dtype, w.shape
        return fwd(x, p)

    @staticmethod
    def backward(ctx, g):
        x, p = ctx.saved_tensors
        gx, gp = bwd(g.contiguous(), x, p)
        return gx, gp.to(ctx.w_dtype).view(ctx.w_shape)


class TritonSSSGLU(torch.autograd.Function):
    """y = gate(x1) * x2."""

    @staticmethod
    def forward(ctx, x):
        x = x.contiguous()
        ctx.save_for_backward(x)
        return fwd(x)

    @staticmethod
    def backward(ctx, g):
        (x,) = ctx.saved_tensors
        gx, _ = bwd(g.contiguous(), x)
        return gx
