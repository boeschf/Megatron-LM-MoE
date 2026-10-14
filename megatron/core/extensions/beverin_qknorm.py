# Beverin (bench2 B3): q/k RMSNorm (head dim 128) through FLA Triton RMSNorm instead of TE.
#
# Why: TE ROCm rmsnorm backward reduces the weight gradient over 163,840 rows (8192 tokens x
# 20 heads) in rmsnorm_bwd_finalize_general_kernel, about 0.3 ms per norm per microbatch on
# MI300A; FLA's Triton RMSNorm does the same norm, forward and backward, in a fraction of
# that (global attention layer of mid-4b, MBS 1: 8.48 -> 7.92 ms fwd + bwd; outputs and grads
# agree with TE within 2e-3). Enabled with BEVERIN_QKNORM_FLA=1; same parameter name
# ("weight"), dtype and attributes, so checkpoints and the optimiser see no difference.
import torch


def maybe_replace(norm, config):
    if norm is None or not hasattr(norm, "weight"):
        return norm
    if getattr(config, "normalization", "RMSNorm") != "RMSNorm":
        return norm
    if getattr(config, "layernorm_zero_centered_gamma", False):
        return norm
    if getattr(norm, "bias", None) is not None:
        return norm
    from fla.modules import RMSNorm as FlaRMSNorm

    old = norm.weight
    new = FlaRMSNorm(old.shape[0], eps=config.layernorm_epsilon)
    new = new.to(device=old.device, dtype=old.dtype)
    with torch.no_grad():
        new.weight.copy_(old)
    for k, v in old.__dict__.items():  # Megatron attributes (sequence_parallel, etc.)
        if not hasattr(new.weight, k):
            setattr(new.weight, k, v)
    return new
