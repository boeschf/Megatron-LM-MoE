"""Opt-in initialization of CUDA TE mask-permutation kernels before EP overlap.

All ranks must enable MEGATRON_MOE_PERMUTATION_WARMUP=1 together.
Targets the pinned TE Triton no-padding BF16/FP32-probability path at TP1/CP1.
Private forwards/backwards cover row-map workspace and integer-alignment variants.
Native autotuning runs first at a declared full-geometry proxy, not a tiny case;
its choices may differ from tuning on the first real routed batch and are logged.
No model, optimizer, routing state or RNG is accessed.
Other kernel families and zero-token UCCL handling are outside this hook's scope.
"""

import json
import os
import time


def permutation_cases(max_tokens, local_experts, topk, prime_tokens):
    """Cover TE's 1024-row workspace bins and 1/divisible-by-16/other scalars."""
    if min(max_tokens, local_experts, topk, prime_tokens) < 1 or prime_tokens > max_tokens:
        raise ValueError("permutation warmup requires positive bounded token/expert counts")
    counts = {1, min(3, max_tokens), min(16, max_tokens), prime_tokens}
    # WORKSPACE_LOAD_WIDTH = next_power_of_2(local_experts * ceil(tokens / 1024)).
    # Visit each transition, including non-power-of-two local expert counts.
    width = 1 << (local_experts - 1).bit_length()
    while True:
        first_block = (width // local_experts) + 1
        first_token = (first_block - 1) * 1024 + 1
        if first_token > max_tokens:
            break
        counts.update((first_token, ((first_token + 15) // 16) * 16))
        width *= 2
    cases = set()
    for tokens in counts:
        if tokens > max_tokens:
            continue
        for capacity in (tokens, tokens + 1, ((tokens + 15) // 16) * 16):
            if capacity <= tokens * topk:
                cases.add((tokens, capacity))
    prime = (prime_tokens, prime_tokens)
    return [prime, *sorted(cases - {prime})]


def warmup(args):
    """Prime private fused permutation/autograd signatures before persistent EP work."""
    if os.environ.get("MEGATRON_MOE_PERMUTATION_WARMUP", "0") != "1":
        return
    if (args.tensor_model_parallel_size != 1 or args.context_parallel_size != 1
            or args.sequence_parallel):
        raise ValueError("permutation warmup requires TP1/CP1 without sequence parallelism")
    if not args.bf16 or args.fp8 is not None:
        raise ValueError("permutation warmup requires BF16 without global TE FP8")
    if (args.moe_token_dispatcher_type != "flex" or args.moe_flex_dispatcher_backend != "deepep"
            or not args.moe_permute_fusion or args.moe_router_padding_for_quantization):
        raise ValueError("permutation warmup requires fused no-padding flex/deepep dispatch")
    ep = args.expert_model_parallel_size
    if ep < 1 or args.num_experts < ep or args.num_experts % ep or args.moe_router_topk < 1:
        raise ValueError("permutation warmup requires positive divisible expert geometry")
    local_tokens = args.seq_length * args.micro_batch_size
    hidden = args.moe_latent_size or args.hidden_size
    if local_tokens < 1 or hidden < 1:
        raise ValueError("permutation warmup requires positive token and hidden dimensions")

    import torch
    import torch.distributed as dist
    from transformer_engine.common.triton import permutation as kernels
    from triton.runtime.autotuner import Autotuner

    from megatron.core.transformer.moe import moe_utils

    if torch.version.hip is not None or not torch.cuda.is_available() or not dist.is_initialized():
        raise RuntimeError("permutation warmup requires initialized CUDA distributed training")
    if (moe_utils.fused_permute_and_pad_with_probs is not None
            or moe_utils.fused_permute_with_probs is None):
        raise RuntimeError("permutation warmup requires the selected TE non-padding consumer")
    tuners = {name: getattr(kernels, name) for name in ("_permute_kernel", "_unpermute_kernel")}
    if not all(isinstance(tuner, Autotuner) for tuner in tuners.values()):
        raise RuntimeError("permutation warmup requires the pinned TE Triton autotuners")
    local_experts = args.num_experts // ep
    max_tokens = local_tokens * ep
    # Average routed assignments per rank provide a saturating full-geometry proxy;
    # deduplication makes the actual received row count data-dependent.
    prime_tokens = local_tokens * min(args.moe_router_topk, ep)
    cases = permutation_cases(max_tokens, local_experts, args.moe_router_topk, prime_tokens)
    started = time.monotonic()
    torch.cuda.synchronize()
    dist.barrier()
    torch.cuda.synchronize()
    with torch.enable_grad():
        for tokens, capacity in cases:
            x = torch.zeros((tokens, hidden), device="cuda", dtype=torch.bfloat16,
                            requires_grad=True)
            routing = torch.zeros((tokens, local_experts), device="cuda", dtype=torch.bool)
            routing[:, 0] = True
            probs = routing.to(torch.float32).requires_grad_()
            y, permuted_probs, row_map, pad, _ = moe_utils.permute(
                x, routing, probs=probs, num_out_tokens=capacity, fused=True)
            restored = moe_utils.unpermute(
                y, row_map, restore_shape=x.shape, routing_map=routing, fused=True, pad_offsets=pad)
            # Spare capacity is intentionally uninitialized; only mapped rows are read.
            # Zero upstream gradients still exercise both private autograd paths.
            torch.autograd.backward((restored, permuted_probs),
                                    (torch.zeros_like(restored), torch.zeros_like(permuted_probs)))
            torch.cuda.synchronize()
            del x, routing, probs, y, permuted_probs, row_map, restored
    choices = {name: {str(key): config.kwargs for key, config in tuner.cache.items()}
               for name, tuner in tuners.items()}
    dist.barrier()
    torch.cuda.synchronize()
    report = dict(rank=dist.get_rank(), max_received_tokens=max_tokens, local_experts=local_experts,
                  hidden=hidden, prime_tokens=prime_tokens, cases=cases, autotune_configs=choices,
                  seconds=time.monotonic() - started)
    print(f"[moe-permutation-warmup] {json.dumps(report, sort_keys=True)}", flush=True)
    return report
