# Beverin (bench2 K2): KDA short convolution fused with the in_proj split.
#
# One autograd Function from the fused in_proj output `projected` [S, B, W] (W = 3*CS + tail,
# e.g. 7956 = 3 x 2560 + 128 + 128 + 20 at mid-4b) to
#   q, k, v: SiLU(causal depthwise conv1d(projected[..., :3*CS])), each a contiguous [B, S, CS]
#   tail:    projected[..., 3*CS:] as a contiguous [S, B, W - 3*CS] (decay / gate bottlenecks, beta)
# The backward writes d(projected) [S, B, W] in place: the conv backward fills the first 3*CS
# columns (reading dq, dk, dv separately), the tail gradient is copied into the rest.
#
# Replaces, per KDA layer: the 8-alignment copy of the qkv slice, the causal_conv1d channel-last
# kernels, the three .contiguous() copies of q, k, v, the cat of dq/dk/dv and the cat of the
# split backward. Each program walks BT rows of a BC-channel block with a sliding window in
# registers (every x row is read once, plus a 3-row halo), so loads are whole-row segments.
#
# Packed sequences: seq_idx [B, S] int32 (document id per token, as causal_conv1d's CUDA backend
# takes); taps from another document are zero.
import os

import torch
import triton
import triton.language as tl


@triton.jit
def _ldrow(base, r, sxt: tl.constexpr, c, lo, hi):
    return tl.load(base + r.to(tl.int64) * sxt + c, mask=(r >= lo) & (r < hi) & (c >= 0), other=0.0)


@triton.jit
def _ldseq(sb, r, T):
    return tl.load(sb + tl.minimum(tl.maximum(r, 0), T - 1))


# Rows are prefetched 8 ahead (p0..p7 rotate in registers): the stores of y / dx may alias the
# loads as far as the compiler knows, so without the explicit prefetch every row load waits for
# the previous store (one load in flight per wave, about 1.2 TB/s).
@triton.jit
def _k2_conv_fwd(x_ptr, w_ptr, seq_ptr, y_ptr, T, B,
                 sxt: tl.constexpr, sxb: tl.constexpr, CS: tl.constexpr,
                 BT: tl.constexpr, BC: tl.constexpr, HAS_SEQ: tl.constexpr, CFIRST: tl.constexpr):
    if CFIRST:
        pid_c = tl.program_id(0)
        pid_t = tl.program_id(1)
    else:
        pid_t = tl.program_id(0)
        pid_c = tl.program_id(1)
    pid_b = tl.program_id(2)
    c = pid_c * BC + tl.arange(0, BC)
    seg = (pid_c * BC) // CS
    cl = c - seg * CS
    yb = y_ptr + (seg * B + pid_b).to(tl.int64) * T * CS
    xb = x_ptr + pid_b.to(tl.int64) * sxb
    w0 = tl.load(w_ptr + c * 4 + 0).to(tl.float32)
    w1 = tl.load(w_ptr + c * 4 + 1).to(tl.float32)
    w2 = tl.load(w_ptr + c * 4 + 2).to(tl.float32)
    w3 = tl.load(w_ptr + c * 4 + 3).to(tl.float32)
    t0 = pid_t * BT
    hi = tl.minimum(t0 + BT, T)
    sb = seq_ptr + pid_b.to(tl.int64) * T
    x0 = _ldrow(xb, t0 - 3, sxt, c, 0, hi).to(tl.float32)
    x1 = _ldrow(xb, t0 - 2, sxt, c, 0, hi).to(tl.float32)
    x2 = _ldrow(xb, t0 - 1, sxt, c, 0, hi).to(tl.float32)
    p0 = _ldrow(xb, t0 + 0, sxt, c, 0, hi)
    p1 = _ldrow(xb, t0 + 1, sxt, c, 0, hi)
    p2 = _ldrow(xb, t0 + 2, sxt, c, 0, hi)
    p3 = _ldrow(xb, t0 + 3, sxt, c, 0, hi)
    p4 = _ldrow(xb, t0 + 4, sxt, c, 0, hi)
    p5 = _ldrow(xb, t0 + 5, sxt, c, 0, hi)
    p6 = _ldrow(xb, t0 + 6, sxt, c, 0, hi)
    p7 = _ldrow(xb, t0 + 7, sxt, c, 0, hi)
    if HAS_SEQ:
        s0 = _ldseq(sb, t0 - 3, T)
        s1 = _ldseq(sb, t0 - 2, T)
        s2 = _ldseq(sb, t0 - 1, T)
    for i in tl.static_range(BT):
        r = t0 + i
        x3 = p0.to(tl.float32)
        p0 = p1
        p1 = p2
        p2 = p3
        p3 = p4
        p4 = p5
        p5 = p6
        p6 = p7
        p7 = _ldrow(xb, r + 8, sxt, c, 0, hi)
        if HAS_SEQ:
            s3 = _ldseq(sb, r, T)
            z = (tl.where(s0 == s3, x0, 0.0) * w0 + tl.where(s1 == s3, x1, 0.0) * w1
                 + tl.where(s2 == s3, x2, 0.0) * w2 + x3 * w3)
            s0 = s1
            s1 = s2
            s2 = s3
        else:
            z = x0 * w0 + x1 * w1 + x2 * w2 + x3 * w3
        y = z * tl.sigmoid(z)
        tl.store(yb + r.to(tl.int64) * CS + cl, y.to(yb.dtype.element_ty), mask=(r < T) & (c >= 0))
        x0 = x1
        x1 = x2
        x2 = x3


@triton.jit
def _k2_conv_bwd(x_ptr, w_ptr, seq_ptr, dy_ptr, dx_ptr, dw_ptr, T, c_off,
                 sxt: tl.constexpr, sxb: tl.constexpr, CS: tl.constexpr,
                 BT: tl.constexpr, BC: tl.constexpr, HAS_SEQ: tl.constexpr, CFIRST: tl.constexpr):
    # Rows r = t0 .. t0+BT+2 walked in order. z[r], dz[r] (needs x[r-3..r], dy[r]);
    # dx[r-3] = dz[r-3] w3 + dz[r-2] w2 + dz[r-1] w1 + dz[r] w0 once r-3 >= t0;
    # dw[j] += dz[r] x[r-3+j] for r in [t0, t0+BT).
    if CFIRST:
        pid_c = tl.program_id(0)
        pid_t = tl.program_id(1)
    else:
        pid_t = tl.program_id(0)
        pid_c = tl.program_id(1)
    pid_b = tl.program_id(2)
    NT = tl.num_programs(1) if CFIRST else tl.num_programs(0)
    cl = pid_c * BC + tl.arange(0, BC)
    c = c_off + cl
    db = dy_ptr + pid_b.to(tl.int64) * T * CS
    xb = x_ptr + pid_b.to(tl.int64) * sxb
    dxb = dx_ptr + pid_b.to(tl.int64) * sxb
    w0 = tl.load(w_ptr + c * 4 + 0).to(tl.float32)
    w1 = tl.load(w_ptr + c * 4 + 1).to(tl.float32)
    w2 = tl.load(w_ptr + c * 4 + 2).to(tl.float32)
    w3 = tl.load(w_ptr + c * 4 + 3).to(tl.float32)
    t0 = pid_t * BT
    hi = tl.minimum(t0 + BT + 3, T)
    sb = seq_ptr + pid_b.to(tl.int64) * T
    x0 = _ldrow(xb, t0 - 3, sxt, c, 0, hi).to(tl.float32)
    x1 = _ldrow(xb, t0 - 2, sxt, c, 0, hi).to(tl.float32)
    x2 = _ldrow(xb, t0 - 1, sxt, c, 0, hi).to(tl.float32)
    p0 = _ldrow(xb, t0 + 0, sxt, c, 0, hi)
    p1 = _ldrow(xb, t0 + 1, sxt, c, 0, hi)
    p2 = _ldrow(xb, t0 + 2, sxt, c, 0, hi)
    p3 = _ldrow(xb, t0 + 3, sxt, c, 0, hi)
    p4 = _ldrow(xb, t0 + 4, sxt, c, 0, hi)
    p5 = _ldrow(xb, t0 + 5, sxt, c, 0, hi)
    p6 = _ldrow(xb, t0 + 6, sxt, c, 0, hi)
    p7 = _ldrow(xb, t0 + 7, sxt, c, 0, hi)
    d0 = _ldrow(db, t0 + 0, CS, cl, 0, hi)
    d1 = _ldrow(db, t0 + 1, CS, cl, 0, hi)
    d2 = _ldrow(db, t0 + 2, CS, cl, 0, hi)
    d3 = _ldrow(db, t0 + 3, CS, cl, 0, hi)
    d4 = _ldrow(db, t0 + 4, CS, cl, 0, hi)
    d5 = _ldrow(db, t0 + 5, CS, cl, 0, hi)
    d6 = _ldrow(db, t0 + 6, CS, cl, 0, hi)
    d7 = _ldrow(db, t0 + 7, CS, cl, 0, hi)
    if HAS_SEQ:
        s0 = _ldseq(sb, t0 - 3, T)
        s1 = _ldseq(sb, t0 - 2, T)
        s2 = _ldseq(sb, t0 - 1, T)
    z0 = tl.zeros([BC], dtype=tl.float32)  # dz[r-3], dz[r-2], dz[r-1]
    z1 = tl.zeros([BC], dtype=tl.float32)
    z2 = tl.zeros([BC], dtype=tl.float32)
    a0 = tl.zeros([BC], dtype=tl.float32)
    a1 = tl.zeros([BC], dtype=tl.float32)
    a2 = tl.zeros([BC], dtype=tl.float32)
    a3 = tl.zeros([BC], dtype=tl.float32)
    for i in tl.static_range(BT + 3):
        r = t0 + i
        x3 = p0.to(tl.float32)
        dy = d0.to(tl.float32)
        p0 = p1
        p1 = p2
        p2 = p3
        p3 = p4
        p4 = p5
        p5 = p6
        p6 = p7
        d0 = d1
        d1 = d2
        d2 = d3
        d3 = d4
        d4 = d5
        d5 = d6
        d6 = d7
        if i + 8 < BT + 3:
            p7 = _ldrow(xb, r + 8, sxt, c, 0, hi)
            d7 = _ldrow(db, r + 8, CS, cl, 0, hi)
        if HAS_SEQ:
            s3 = _ldseq(sb, r, T)
            e0 = tl.where(s0 == s3, x0, 0.0)
            e1 = tl.where(s1 == s3, x1, 0.0)
            e2 = tl.where(s2 == s3, x2, 0.0)
        else:
            e0 = x0
            e1 = x1
            e2 = x2
        z = e0 * w0 + e1 * w1 + e2 * w2 + x3 * w3
        sg = tl.sigmoid(z)
        dz = dy * sg * (1.0 + z * (1.0 - sg))
        if i < BT:
            a0 += dz * e0
            a1 += dz * e1
            a2 += dz * e2
            a3 += dz * x3
        if i >= 3:
            # dx[r-3] = sum_j dz[r-3+j] w[3-j], taps limited to the document of row r-3 (s0)
            if HAS_SEQ:
                dx = (z0 * w3 + tl.where(s1 == s0, z1, 0.0) * w2 + tl.where(s2 == s0, z2, 0.0) * w1
                      + tl.where(s3 == s0, dz, 0.0) * w0)
            else:
                dx = z0 * w3 + z1 * w2 + z2 * w1 + dz * w0
            rr = r - 3
            tl.store(dxb + rr.to(tl.int64) * sxt + c, dx.to(dxb.dtype.element_ty), mask=(rr < T) & (c >= 0))
        z0 = z1
        z1 = z2
        z2 = dz
        x0 = x1
        x1 = x2
        x2 = x3
        if HAS_SEQ:
            s0 = s1
            s1 = s2
            s2 = s3
    C = 3 * CS
    dwb = dw_ptr + ((pid_b * NT + pid_t).to(tl.int64) * C + c) * 4
    tl.store(dwb + 0, a0)
    tl.store(dwb + 1, a1)
    tl.store(dwb + 2, a2)
    tl.store(dwb + 3, a3)


_CF = int(os.environ.get("K2_CONV_CFIRST", "1"))
_FBT = int(os.environ.get("K2_CONV_BT", "32"))
_FBC = int(os.environ.get("K2_CONV_BC", "512"))
_FNW = int(os.environ.get("K2_CONV_WARPS", "2"))
_BBT = int(os.environ.get("K2_CONV_BT_BWD", "32"))
_BBC = int(os.environ.get("K2_CONV_BC_BWD", str(_FBC)))
_BNW = int(os.environ.get("K2_CONV_WARPS_BWD", str(_FNW)))


class KdaSplitConv(torch.autograd.Function):
    @staticmethod
    def forward(ctx, projected, weight, seq_idx, CS):
        S, B, W = projected.shape
        assert projected.stride(2) == 1 and projected.stride(1) == W and projected.stride(0) == B * W
        C = 3 * CS
        assert CS % _FBC == 0 and CS % _BBC == 0
        w = weight.reshape(C, -1)
        assert w.shape[1] == 4
        w = w.contiguous()
        y = torch.empty(3, B, S, CS, device=projected.device, dtype=projected.dtype)
        seq = seq_idx if seq_idx is not None else projected
        if seq_idx is not None:
            assert seq_idx.shape == (B, S) and seq_idx.is_contiguous() and seq_idx.dtype == torch.int32
        _k2_conv_fwd[(C // _FBC, triton.cdiv(S, _FBT), B) if _CF else (triton.cdiv(S, _FBT), C // _FBC, B)](
            projected, w, seq, y, S, B, sxt=B * W, sxb=W, CS=CS,
            BT=_FBT, BC=_FBC, HAS_SEQ=seq_idx is not None, CFIRST=_CF, num_warps=_FNW)
        tail = projected[..., C:].contiguous()
        ctx.save_for_backward(projected, w, seq_idx)
        ctx.CS = CS
        ctx.wshape = weight.shape
        return y[0], y[1], y[2], tail

    @staticmethod
    def backward(ctx, dq, dk, dv, dtail):
        projected, w, seq_idx = ctx.saved_tensors
        S, B, W = projected.shape
        CS = ctx.CS
        C = 3 * CS
        dproj = torch.empty_like(projected)
        NT = triton.cdiv(S, _BBT)
        dwp = torch.empty(B * NT, C, 4, device=projected.device, dtype=torch.float32)
        seq = seq_idx if seq_idx is not None else projected
        for i, dy in enumerate((dq, dk, dv)):
            dy = dy.contiguous() if dy is not None else torch.zeros(B, S, CS, device=projected.device, dtype=projected.dtype)
            _k2_conv_bwd[(CS // _BBC, NT, B) if _CF else (NT, CS // _BBC, B)](
                projected, w, seq, dy, dproj, dwp, S, i * CS, sxt=B * W, sxb=W, CS=CS,
                BT=_BBT, BC=_BBC, HAS_SEQ=seq_idx is not None, CFIRST=_CF, num_warps=_BNW)
        if dtail is not None:
            dproj[..., C:].copy_(dtail)
        else:
            dproj[..., C:].zero_()
        dw = dwp.sum(0).to(w.dtype).reshape(ctx.wshape)
        return dproj, dw, None, None


def kda_split_conv(projected, weight, seq_idx=None, CS=None):
    """projected [S, B, W] contiguous; weight [3*CS, 1, 4] or [3*CS, 4]; seq_idx [B, S] int32 or None.
    Returns q, k, v contiguous [B, S, CS] (SiLU(causal conv)), tail [S, B, W - 3*CS] contiguous."""
    if CS is None:
        CS = weight.shape[0] // 3
    return KdaSplitConv.apply(projected, weight, seq_idx, CS)


def reference(projected, weight, seq_idx=None, CS=None):
    """The fork's path (causal_conv1d CUDA channel-last, with its copies), for checks and timing."""
    from fla.modules.convolution import causal_conv1d
    S, B, W = projected.shape
    if CS is None:
        CS = weight.shape[0] // 3
    C = 3 * CS
    qkv = projected[..., :C].transpose(0, 1)
    tail = projected[..., C:]
    if qkv.stride(0) % 8 or qkv.stride(1) % 8:
        qkv = qkv.contiguous()
    kw = {}
    if seq_idx is not None:
        kw["seq_idx"] = seq_idx
    y, _ = causal_conv1d(x=qkv, weight=weight.reshape(C, -1), bias=None, activation="silu",
                         initial_state=None, output_final_state=False, backend="cuda", **kw)
    q, k, v = torch.split(y, [CS, CS, CS], dim=-1)
    return q.contiguous(), k.contiguous(), v.contiguous(), tail
