# beverin-rocm patch: run TEGroupedMLP expert GEMMs with Primus-Turbo CK grouped GEMM on AMD.
# TE's ROCm GroupedLinear is a loop of per-expert hipBLASLt calls whose algorithm lookup thrashes
# on the per-step varying token counts. Enabled with BEVERIN_TURBO_GG=1; the checkpoint layout
# (weight0..weightN of TE GroupedLinear) is unchanged.
import os

import torch

ENABLED = os.environ.get("BEVERIN_TURBO_GG", "0") == "1"
_CK = None


def _ops():
    global _CK
    if _CK is None:
        from primus_turbo.pytorch.core.backend import BackendType
        from primus_turbo.pytorch.kernels.grouped_gemm.grouped_gemm_fp8_impl import (
            grouped_gemm_compute_offs,
        )
        from primus_turbo.pytorch.kernels.grouped_gemm.grouped_gemm_impl import (
            grouped_gemm_impl,
            grouped_gemm_variable_k_impl,
        )

        try:  # smallscale-0004: origami raises ValueError (vector::reserve) for some shapes, but
            # _safe_rank_configs only catches RuntimeError; fall back to the default tile config.
            import primus_turbo.triton.gemm.gemm_kernel as _gk

            _orig_rank = _gk._safe_rank_configs

            def _safe_rank_configs(problem, hardware, configs):
                try:
                    return _orig_rank(problem, hardware, configs)
                except Exception:
                    return []

            _gk._safe_rank_configs = _safe_rank_configs
        except ImportError:
            pass
        # BEVERIN_TURBO_GG_BACKEND=CK|TRITON (smallscale gg_bench, job 677157: Triton 14 to 31% faster
        # than CK on varying per-expert shapes, wgrad 1.4 to 1.6x)
        _be = getattr(BackendType, os.environ.get("BEVERIN_TURBO_GG_BACKEND", "CK")).value
        _CK = (_be, grouped_gemm_compute_offs, grouped_gemm_impl,
               grouped_gemm_variable_k_impl)
    return _CK


def _stacked_view(weights):
    """[E, N, K] view of the expert weights without a copy when they are laid out back to back
    (Megatron's DDP param buffer usually does this), else a stacked copy."""
    w0 = weights[0]
    n = w0.numel()
    if all(w.is_contiguous() for w in weights):
        step = weights[1].data_ptr() - w0.data_ptr() if len(weights) > 1 else n * w0.element_size()
        if step == n * w0.element_size() and all(
            w.data_ptr() == w0.data_ptr() + i * step for i, w in enumerate(weights)
        ):
            return torch.as_strided(w0.detach(), (len(weights), *w0.shape), (n, w0.shape[1], 1))
    return torch.stack([w.detach() for w in weights], 0)


class _TurboExpertLinear(torch.autograd.Function):
    """out = x_g @ W_g^T per expert group g; W_g has TE layout [out, in]."""

    @staticmethod
    def forward(ctx, x, group_lens, *weights):
        ck, compute_offs, gg, gg_vk = _ops()
        w = _stacked_view(weights)
        offs = compute_offs(group_lens)
        out = gg(x, w, group_lens, offs, trans_a=False, trans_b=True, num_cu=None,
                 default_backend=ck, maybe_pre_sync=True)
        ctx.save_for_backward(x, group_lens, offs)
        ctx.weights = weights
        return out

    @staticmethod
    def backward(ctx, grad_out):
        ck, _, gg, gg_vk = _ops()
        x, group_lens, offs = ctx.saved_tensors
        weights = ctx.weights
        ctx.weights = None
        grad_out = grad_out.contiguous()
        w = _stacked_view(weights)
        grad_x = gg(grad_out, w, group_lens, offs, trans_a=False, trans_b=False, num_cu=None,
                    default_backend=ck)
        grad_w = gg_vk(x, grad_out, group_lens, offs, trans_a=True, trans_b=False, trans_c=True,
                       num_cu=None, default_backend=ck)
        return (grad_x, None, *grad_w.unbind(0))


def expert_linear(te_grouped_linear, x, group_lens):
    n = te_grouped_linear.num_gemms
    weights = [getattr(te_grouped_linear, f"weight{i}") for i in range(n)]
    if x.shape[0] == 0:
        return x.new_zeros((0, weights[0].shape[0])) + 0 * sum(w.sum() for w in weights)
    return _TurboExpertLinear.apply(x, group_lens, *weights)
