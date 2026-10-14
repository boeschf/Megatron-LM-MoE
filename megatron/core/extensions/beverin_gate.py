# Beverin (bench2 B3): fused sigmoid output gate of the global attention layers,
# y = x * sigmoid(gate) in fp32, stored in x's dtype; backward in one kernel too.
#
# Why: `_apply_output_gate` is a `jit_fuser` function; with torch.compile off on Beverin
# (TORCH_COMPILE_DISABLE=1, as round 1) it runs as five to eight eager kernels
# (copies, bf16 -> fp32, sigmoid, multiply, cast; backward likewise), about 0.5 ms per
# layer-microbatch at mid-4b on MI300A. Enabled with BEVERIN_FUSED_OUT_GATE=1.
import torch
import triton
import triton.language as tl


@triton.jit
def _gate_fwd(x_ptr, g_ptr, y_ptr, N, BLOCK: tl.constexpr):
    i = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    m = i < N
    x = tl.load(x_ptr + i, mask=m, other=0.0).to(tl.float32)
    g = tl.load(g_ptr + i, mask=m, other=0.0).to(tl.float32)
    tl.store(y_ptr + i, (x * tl.sigmoid(g)).to(y_ptr.dtype.element_ty), mask=m)


@triton.jit
def _gate_bwd(x_ptr, g_ptr, dy_ptr, dx_ptr, dg_ptr, N, BLOCK: tl.constexpr):
    i = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    m = i < N
    x = tl.load(x_ptr + i, mask=m, other=0.0).to(tl.float32)
    g = tl.load(g_ptr + i, mask=m, other=0.0).to(tl.float32)
    dy = tl.load(dy_ptr + i, mask=m, other=0.0).to(tl.float32)
    s = tl.sigmoid(g)
    tl.store(dx_ptr + i, (dy * s).to(dx_ptr.dtype.element_ty), mask=m)
    tl.store(dg_ptr + i, (dy * x * s * (1.0 - s)).to(dg_ptr.dtype.element_ty), mask=m)


_BLOCK = 4096


class _SigmoidGate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, gate):
        x = x.contiguous()
        g = gate.contiguous().view(x.shape)
        y = torch.empty_like(x)
        n = x.numel()
        _gate_fwd[(triton.cdiv(n, _BLOCK),)](x, g, y, n, BLOCK=_BLOCK, num_warps=8)
        ctx.save_for_backward(x, g)
        ctx.gshape = gate.shape
        return y

    @staticmethod
    def backward(ctx, dy):
        x, g = ctx.saved_tensors
        dy = dy.contiguous()
        dx = torch.empty_like(x)
        dg = torch.empty_like(g)
        n = x.numel()
        _gate_bwd[(triton.cdiv(n, _BLOCK),)](x, g, dy, dx, dg, n, BLOCK=_BLOCK, num_warps=8)
        return dx, dg.view(ctx.gshape)


def sigmoid_gate(x, gate):
    """x * sigmoid(gate) (computed in fp32, returned in x.dtype); gate has x's numel."""
    return _SigmoidGate.apply(x, gate)
