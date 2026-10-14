# beverin-rocm patch: run TEGroupedMLP expert GEMMs with Primus-Turbo CK grouped GEMM on AMD.
# TE's ROCm GroupedLinear is a loop of per-expert hipBLASLt calls whose algorithm lookup thrashes
# on the per-step varying token counts. Enabled with BEVERIN_TURBO_GG=1; the checkpoint layout
# (weight0..weightN of TE GroupedLinear) is unchanged.
import os

import torch

ENABLED = os.environ.get("BEVERIN_TURBO_GG", "0") == "1"
_TRITON = os.environ.get("BEVERIN_TURBO_GG_BACKEND", "CK") == "TRITON"
PTR = os.environ.get("BEVERIN_TURBO_PTR", "0") == "1" and _TRITON  # lab T4
FUSED_WGRAD = os.environ.get("BEVERIN_TURBO_FUSED_WGRAD", "0") == "1"  # lab T4, any backend (wgrad kernel is Triton)
# lab T3: honour --delay-wgrad-compute (needed by the EP-overlap schedule) by queueing the expert wgrad in the
# TE module's own WeightGradStore; TE GroupedLinear.backward_dw() then pops it and fires the DDP hooks.
DELAY_WGRAD = os.environ.get("BEVERIN_TURBO_DELAY_WGRAD", "0") == "1"
# lab2 A3: BEVERIN_TURBO_NUM_CU=n runs the expert grouped GEMMs (CK fwd/dgrad/wgrad and the fused Triton wgrad) on n
# CUs only, leaving the rest to the EP all-to-all that overlaps them (--overlap-moe-expert-parallel-comm)
NUM_CU = int(os.environ["BEVERIN_TURBO_NUM_CU"]) if os.environ.get("BEVERIN_TURBO_NUM_CU") else None
_GOFF = {}
_DUMMY = {}


def _offsets(tensors):
    """(lowest-address tensor, int64 element offsets of every tensor from it). Offsets are >= 0 (the AMD Triton
    buffer-load path miscomputes negative ones) and must fit in 2 GiB of bytes."""
    key = tuple(t.data_ptr() for t in tensors)
    hit = _GOFF.get(key)
    if hit is None:
        if len(_GOFF) > 4096:
            _GOFF.clear()
        base = min(tensors, key=lambda t: t.data_ptr())
        e = base.element_size()
        offs = [(k - base.data_ptr()) // e for k in key]
        assert all(o >= 0 for o in offs) and max(offs) * e + tensors[0].numel() * e < 2**31, "expert span > 2 GiB"
        hit = _GOFF[key] = (base, torch.tensor(offs, dtype=torch.int64, device=base.device))
    return hit


def _dummy_grad(w):
    key = (tuple(w.shape), w.dtype, w.device)
    t = _DUMMY.get(key)
    if t is None:
        t = _DUMMY[key] = torch.zeros(w.shape, dtype=w.dtype, device=w.device)
    return t
_CK = None
# bench2 B4: BEVERIN_GG_TILES=<json> pins the tile config of Primus' persistent Triton grouped GEMM (fwd/dgrad, used
# with BEVERIN_TURBO_GG_BACKEND=TRITON or BEVERIN_TURBO_PTR=1) and of the variable-K wgrad kernels (Primus bf16 and
# the fused fp32 wgrad of BEVERIN_TURBO_FUSED_WGRAD) per problem, instead of origami's pick or the fixed default.
# JSON: {"fwd": [[N, K, trans_b, G or null, [BM, BN, BK, group_m, cache_a, cache_b, stages, chunk]], ...],
#        "vk": [[OUT_M, OUT_N, G or null, [...]], ...]}. Unlisted problems keep the original choice.
_TILES_DONE = []


def _install_tiles():
    path = os.environ.get("BEVERIN_GG_TILES")
    if not path or _TILES_DONE:
        return
    _TILES_DONE.append(1)
    import json
    import primus_turbo.triton.grouped_gemm.grouped_gemm_kernel as gk
    t = json.load(open(path))
    fwd = {(int(a[0]), int(a[1]), bool(a[2]), a[3]): tuple(a[4]) for a in t.get("fwd", [])}
    vk = {(int(a[0]), int(a[1]), a[2]): tuple(a[3]) for a in t.get("vk", [])}
    of, ov = gk._get_gg_bf16_fwd_config, gk._get_gg_bf16_vk_config

    def fwd_cfg(avg_m, N, K, dtype_a, dtype_b, trans_b, G, num_sms):
        c = fwd.get((N, K, bool(trans_b), G)) or fwd.get((N, K, bool(trans_b), None))
        return c if c is not None else of(avg_m, N, K, dtype_a, dtype_b, trans_b, G, num_sms)

    def vk_cfg(OUT_M, OUT_N, avg_k, dtype_lhs, dtype_rhs, G, num_sms):
        c = vk.get((OUT_M, OUT_N, G)) or vk.get((OUT_M, OUT_N, None))
        return c if c is not None else ov(OUT_M, OUT_N, avg_k, dtype_lhs, dtype_rhs, G, num_sms)

    gk._get_gg_bf16_fwd_config, gk._get_gg_bf16_vk_config = fwd_cfg, vk_cfg
    for mod, name, fn in (("megatron.core.transformer.moe.beverin_gg", "_get_gg_bf16_fwd_config", fwd_cfg),
                          ("megatron.core.transformer.moe.beverin_vk", "_get_gg_bf16_vk_config", vk_cfg)):
        try:
            import importlib
            setattr(importlib.import_module(mod), name, fn)
        except Exception:  # noqa: BLE001
            pass
    if int(os.environ.get("RANK", "0")) == 0:
        print(f"beverin_turbo: BEVERIN_GG_TILES {path}: {len(fwd)} fwd/dgrad and {len(vk)} wgrad tile configs", flush=True)


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
        _install_tiles()
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
    def forward(ctx, x, group_lens, module, *weights):
        ck, compute_offs, gg, gg_vk = _ops()
        offs = compute_offs(group_lens)
        if PTR and all(w.is_contiguous() for w in weights):
            from megatron.core.transformer.moe.beverin_gg import grouped_gemm_ptr

            wb, goff = _offsets([w.detach() for w in weights])
            out = grouped_gemm_ptr(x, wb.unsqueeze(0), offs, goff, trans_b=True)
        else:
            w = _stacked_view(weights)
            out = gg(x, w, group_lens, offs, trans_a=False, trans_b=True, num_cu=NUM_CU,
                     default_backend=ck, maybe_pre_sync=True)
        ctx.save_for_backward(x, group_lens, offs)
        ctx.weights = weights
        ctx.module = module
        return out

    @staticmethod
    def backward(ctx, grad_out):
        ck, _, gg, gg_vk = _ops()
        x, group_lens, offs = ctx.saved_tensors
        weights = ctx.weights
        ctx.weights = None
        module, ctx.module = ctx.module, None
        grad_out = grad_out.contiguous()
        if PTR and all(w.is_contiguous() for w in weights):
            from megatron.core.transformer.moe.beverin_gg import grouped_gemm_ptr

            wb, goff = _offsets([w.detach() for w in weights])
            grad_x = grouped_gemm_ptr(grad_out, wb.unsqueeze(0), offs, goff, trans_b=False)
        else:
            w = _stacked_view(weights)
            grad_x = gg(grad_out, w, group_lens, offs, trans_a=False, trans_b=False, num_cu=NUM_CU,
                        default_backend=ck)
        store = getattr(module, "wgrad_store", None) if module is not None else None
        if DELAY_WGRAD and store is not None and store.delay_wgrad_compute():
            wl = []  # filled by _delayed_wgrad; TE reads it only when wgrad accumulation fusion is off
            store.put([x, grad_out, wl, group_lens, offs, weights, bool(getattr(module, "fuse_wgrad_accumulation", True))],
                      _delayed_wgrad)
            # zero placeholder grads: DDP's delayed hook (fired from backward_dw) asserts param.grad is set
            return (grad_x, None, None, *[_dummy_grad(w) for w in weights])
        mgs = [getattr(w, "main_grad", None) for w in weights]
        if (FUSED_WGRAD and all(m is not None and m.dtype == torch.float32 and m.is_contiguous() for m in mgs)
                and x.is_contiguous()):
            from megatron.core.transformer.moe.beverin_vk import vk_accum

            mb, moff = _offsets(mgs)
            vk_accum(grad_out, x, offs, mb, moff)
            for w in weights:
                w.grad_added_to_main_grad = True
            return (grad_x, None, None, *[_dummy_grad(w) for w in weights])
        grad_w = gg_vk(x, grad_out, group_lens, offs, trans_a=True, trans_b=False, trans_c=True,
                       num_cu=NUM_CU, default_backend=ck)
        return (grad_x, None, None, *grad_w.unbind(0))


@torch.no_grad()
def _delayed_wgrad(x, grad_out, wl, group_lens, offs, weights, fuse=True):
    """Expert wgrad for TE GroupedLinear.backward_dw (lab T3): accumulate into main_grad when it exists (TE's
    wgrad accumulation fusion), else leave the per-expert wgrads in `wl` for TE to assign to .grad."""
    ck, _, gg, gg_vk = _ops()
    if x is None:  # empty-input placeholder (autograd already gave the weights zero grads)
        if not fuse:
            wl.extend(torch.zeros_like(w) for w in weights)
        return None, [None] * len(weights), None
    mgs = [getattr(w, "main_grad", None) for w in weights] if fuse else [None] * len(weights)
    if (FUSED_WGRAD and all(m is not None and m.dtype == torch.float32 and m.is_contiguous() for m in mgs)
            and x.is_contiguous()):
        from megatron.core.transformer.moe.beverin_vk import vk_accum

        mb, moff = _offsets(mgs)
        vk_accum(grad_out, x, offs, mb, moff)
        for w in weights:
            w.grad_added_to_main_grad = True
    else:
        grad_w = gg_vk(x, grad_out, group_lens, offs, trans_a=True, trans_b=False, trans_c=True,
                       num_cu=NUM_CU, default_backend=ck)
        if all(m is not None for m in mgs):
            for w, m, g in zip(weights, mgs, grad_w.unbind(0)):
                m.add_(g)
                w.grad_added_to_main_grad = True
        else:
            wl.extend(grad_w.unbind(0))
    return None, [None] * len(weights), None


def expert_linear(te_grouped_linear, x, group_lens):
    n = te_grouped_linear.num_gemms
    weights = [getattr(te_grouped_linear, f"weight{i}") for i in range(n)]
    if x.shape[0] == 0:
        store = getattr(te_grouped_linear, "wgrad_store", None)
        if DELAY_WGRAD and store is not None and store.delay_wgrad_compute() and torch.is_grad_enabled():
            store.put([None, None, [], None, None, weights, True], _delayed_wgrad)  # keep backward_dw balanced
        return x.new_zeros((0, weights[0].shape[0])) + 0 * sum(w.sum() for w in weights)
    return _TurboExpertLinear.apply(x, group_lens, te_grouped_linear, *weights)
