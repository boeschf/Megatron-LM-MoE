"""Opt-in FLA math at the e242 SelfAttention Q/K norm call sites."""

import os

import torch
from torch.nn.modules import module as nn_module


def qknorm_fla_requested(config):
    """Enable only the pinned, non-offloaded Chonk training configuration."""
    return (
        os.environ.get("BEVERIN_QKNORM_FLA", "0") == "1"
        and config.normalization == "RMSNorm"
        and config.qk_layernorm
        and not config.qk_l2_norm
        and not config.layernorm_zero_centered_gamma
        and config.tensor_model_parallel_size == config.context_parallel_size == 1
        and not config.sequence_parallel
        and not config.deterministic_mode
        and not config.test_mode
        and not config.cpu_offloading
        and not config.fine_grained_activation_offloading
        and not config.fp8
        and not config.fp8_param
        and not config.fp4
        and not config.fp4_param
        and config.cuda_graph_impl == "none"
        # A reentrant checkpoint spanning this call would run the original
        # forward without grad and the replay with grad, selecting two routes.
        and config.recompute_granularity != "full"
        and "qkv" not in (config.recompute_modules or ())
    )


def _has_module_hooks(norm):
    # The fast path does not call Module.__call__ on the TE norm; preserve its
    # forward/backward hook semantics by using TE whenever such hooks are present.
    # Parameter hooks are different: FLA uses exactly the same leaf Parameter.
    for name in ("forward_hooks", "forward_pre_hooks", "backward_hooks", "backward_pre_hooks"):
        if getattr(norm, "_" + name) or getattr(nn_module, "_global_" + name):
            return True
    return False


def qknorm_fla_or_te(norm, x, heads):
    """Use installed FLA functional RMSNorm with the registered TE gamma."""
    if (
        not torch.version.hip
        or not torch.is_grad_enabled()
        or torch.is_autocast_enabled("cuda")
        or type(x) is not torch.Tensor
        or x.device.type != "cuda"
        or x.dtype != torch.bfloat16
        or heads not in (7, 28)
        or tuple(x.shape) != (4096, 1, heads, 128)
        or (
            not x.is_contiguous()
            and not (heads == 7 and (x.stride(0), x.stride(2), x.stride(3)) == (8960, 1280, 1))
        )
    ):
        return norm(x)

    from transformer_engine.pytorch import RMSNorm

    if (
        type(norm) is not RMSNorm
        or not norm.training
        or norm.zero_centered_gamma
        or getattr(norm, "bias", None) is not None
        or norm.eps != 1e-5
        or _has_module_hooks(norm)
    ):
        return norm(x)
    weight = norm.weight
    if (
        type(weight) is not torch.nn.Parameter
        or tuple(weight.shape) != (128,)
        or weight.device != x.device
        or weight.dtype != x.dtype
        or not weight.is_contiguous()
        or not weight.requires_grad
    ):
        return norm(x)

    # The pinned RMSNorm.forward delegates to this function.  Do not allocate a
    # new module/Parameter or catch failures of the explicitly selected backend.
    from fla.modules.layernorm import rms_norm

    return rms_norm(x, weight, None, eps=norm.eps)
