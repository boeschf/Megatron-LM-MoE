"""Opt-in CUDA TE initialization for KDA output-projection weight gradients.

All ranks must set MEGATRON_KDA_WGRAD_WARMUP=1 together.
After model/optimizer construction, warm each local BF16-to-FP32 NT signature
before training can overlap its first use with a waiting communication kernel.
Only TP1/CP1 without sequence parallelism is supported.
Other GEMM consumers, expert kernels and dispatch are outside this hook's scope.
"""

import os
import time


def warmup(model, args):
    """Initialize local KDA wgrad paths using private tensors and no model calls."""
    if os.environ.get("MEGATRON_KDA_WGRAD_WARMUP", "0") != "1":
        return
    if (args.tensor_model_parallel_size != 1 or args.context_parallel_size != 1
            or args.sequence_parallel):
        raise ValueError("KDA wgrad warmup requires TP1/CP1 without sequence parallelism")
    if not args.bf16 or args.fp8 is not None:
        raise ValueError("KDA wgrad warmup requires BF16 without global TE FP8")

    import torch
    import torch.distributed as dist
    from transformer_engine.pytorch.cpp_extensions.gemm import general_gemm

    from megatron.core.extensions.transformer_engine import TERowParallelLinear
    from megatron.core.ssm.kimi_delta_attention import KimiDeltaAttention

    if torch.version.hip is not None or not torch.cuda.is_available() or not dist.is_initialized():
        raise RuntimeError("KDA wgrad warmup requires initialized CUDA distributed training")
    if args.main_grads_dtype != torch.float32:
        raise ValueError("KDA wgrad warmup requires FP32 main gradients")
    device = torch.device("cuda", torch.cuda.current_device())
    signatures = set()
    for chunk in model:
        for layer in chunk.modules():
            if not isinstance(layer, KimiDeltaAttention):
                continue
            projection = layer.out_proj
            weight = projection.weight
            if (not isinstance(projection, TERowParallelLinear) or projection.use_bias
                    or weight.ndim != 2 or weight.dtype != torch.bfloat16 or weight.device != device):
                raise ValueError("KDA wgrad warmup requires a local bias-free BF16 TE row projection")
            signatures.add(tuple(weight.shape))

    tokens = args.seq_length * args.micro_batch_size
    if tokens < 1:
        raise ValueError("KDA wgrad warmup requires a positive local token count")
    started = time.monotonic()
    # Finish setup and each local call before any subsequent communication starts.
    torch.cuda.synchronize()
    dist.barrier()
    torch.cuda.synchronize()
    scratch_bytes = 0
    with torch.no_grad():
        for outputs, inputs in sorted(signatures):
            x = torch.zeros((tokens, inputs), device=device, dtype=torch.bfloat16)
            dy = torch.zeros((tokens, outputs), device=device, dtype=torch.bfloat16)
            gradient = torch.empty((outputs, inputs), device=device, dtype=torch.float32)
            scratch_bytes = max(scratch_bytes, 2 * tokens * (inputs + outputs) + 4 * outputs * inputs)
            # Warm both first-microbatch replacement and subsequent accumulation.
            # These calls write only private output, never model gradients or queues.
            for accumulate in (False, True):
                general_gemm(x, dy, out_dtype=torch.float32, out=gradient,
                             layout="NT", accumulate=accumulate, grad=True)
                torch.cuda.synchronize()
            del x, dy, gradient
    dist.barrier()
    torch.cuda.synchronize()
    print(f"[kda-wgrad-warmup] rank={dist.get_rank()} tokens={tokens} "
          f"weight_shapes={sorted(signatures)} private_scratch_bytes={scratch_bytes} "
          f"seconds={time.monotonic() - started:.3f}", flush=True)
