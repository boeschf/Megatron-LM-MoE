# Copyright (c) 2026. Beverin lab T5 (warm-up).
"""Pre-warm every model chunk on every rank, in parallel, before training step 1.

Why: step 1 of a pipelined run pays first-call costs per model chunk (hipBLASLt solution search and
code-object loads, Triton compiles and autotune, HIP lazy code-object loading through comgr, torch
cpp-extension builds, lazy RCCL communicator creation, first imports). In step 1 these run one chunk
after another along the pipeline (PP x VPP chunks forward, then backward), so the warm-up is the SUM of
all chunks' first-call costs. Running one forward and backward of each local chunk on synthetic data
before step 1, on all ranks at the same time, turns the sum into roughly the per-rank maximum.

What it does (enabled with BEVERIN_PREWARM=1, default off):
- for each local chunk, Megatron's own schedules.forward_step on a synthetic microbatch (random token ids
  on the first stage, a random hidden-state input tensor elsewhere), then schedules.backward_step with a
  random output gradient (or the loss on the last stage), under DDP no_sync (no gradient communication);
- all chunks of one rank run in chunk order, so the expert-parallel all-to-alls of a stage pair up
  (all members of an EP group are on the same pipeline stage and run the same chunks);
- afterwards restores every module buffer (router expert bias, token counts, quantile-balancing beta
  accumulators, ...), the MoE logging tracker, the CPU/CUDA RNG and the model-parallel RNG tracker states,
  the rerun state machine mode, and zeroes the gradient buffers, so training step 1 sees the same state as without the pre-warm.
- with BEVERIN_PREWARM_OPT=1 (default), runs the MuonMD Newton-Schulz orthogonalisation once per distinct
  parameter shape on random input (no optimizer state is touched), so the first optimizer step does not pay
  for its GEMMs' first use either.
No data is consumed from the real iterator.
"""
import os
import time

import torch

ENABLED = os.environ.get("BEVERIN_PREWARM", "0") == "1"


class _FakeIterator:
    def __init__(self, batch):
        self.batch = batch

    def __iter__(self):
        return self

    def __next__(self):
        return {k: (v.clone() if torch.is_tensor(v) else v) for k, v in self.batch.items()}


def _fake_batch(args):
    b, s = args.micro_batch_size, args.seq_length
    g = torch.Generator().manual_seed(4321)
    tokens = torch.randint(0, args.padded_vocab_size, (b, s), generator=g, dtype=torch.int64)
    batch = {
        "tokens": tokens,
        "labels": torch.roll(tokens, -1, dims=1),
        "loss_mask": torch.ones(b, s, dtype=torch.float),
        "position_ids": torch.arange(s, dtype=torch.int64).unsqueeze(0).expand(b, s).contiguous(),
        "attention_mask": None,
    }
    if getattr(args, "create_attention_mask_in_dataloader", False):
        batch["attention_mask"] = torch.tril(torch.ones((b, 1, s, s))) < 0.5
    return batch


def _iter_optimizers(o):
    if o is None:
        return
    if hasattr(o, "chained_optimizers"):
        for c in o.chained_optimizers:
            yield from _iter_optimizers(c)
    elif not isinstance(o, torch.optim.Optimizer) and hasattr(o, "optimizer"):
        yield from _iter_optimizers(o.optimizer)
    else:
        yield o


def _prewarm_optimizer(optimizer):
    """Run the MuonMD Newton-Schulz orthogonalisation once per distinct parameter shape on random input
    (pure computation, no optimizer state touched), so its GEMMs get their hipBLASLt solutions and code
    objects before the first optimizer step. Returns (number of shapes, seconds)."""
    t0 = time.time()
    seen = set()
    for opt in _iter_optimizers(optimizer):
        if not hasattr(opt, "_orthogonalize_param"):
            continue
        import emerging_optimizers
        for group in opt.param_groups:
            if not group.get("use_orthogonal_updates", False):
                continue
            for p in group["params"]:
                if p.ndim < 2:
                    continue
                is_qkv = opt.is_qkv_fn(p)
                is_embedding = getattr(p, "is_embedding_or_output_parameter", False)
                is_router = getattr(p, "is_router", False)
                is_merged = getattr(p, "merged_offload_expert", False)
                flat_mode = opt._resolve_mode(is_embedding, is_router) == "flat"
                key = (tuple(p.shape), p.dtype, is_qkv, is_router, is_merged, flat_mode)
                if key in seen:
                    continue
                seen.add(key)
                g = torch.randn(p.shape, dtype=p.dtype, device=p.device)
                with emerging_optimizers.utils.fp32_matmul_precision(opt.fp32_matmul_prec):
                    opt._orthogonalize_param(p, g, is_qkv=is_qkv, flat_mode=flat_mode,
                                             is_merged_offload_expert=is_merged, is_router=is_router)
    torch.cuda.synchronize()
    return len(seen), time.time() - t0


def maybe_prewarm(forward_step_func, model, config, optimizer=None):
    if not ENABLED:
        return
    from contextlib import ExitStack

    from megatron.core import parallel_state, tensor_parallel
    from megatron.core.pipeline_parallel import schedules
    from megatron.core.rerun_state_machine import RerunMode, get_rerun_state_machine
    from megatron.core.transformer.moe.moe_utils import get_moe_layer_wise_logging_tracker
    from megatron.training import get_args

    args = get_args()
    rank = torch.distributed.get_rank()
    t0 = time.time()

    # snapshot state that a forward/backward may change
    rng_cpu = torch.get_rng_state()
    rng_cuda = torch.cuda.get_rng_state()
    tracker = tensor_parallel.get_cuda_rng_tracker()
    tracker_states = tracker.get_states()
    bufs = []
    for chunk in model:
        for _, b in chunk.named_buffers():
            if b is not None and torch.is_tensor(b):
                bufs.append((b, b.detach().clone()))
    log_tracker = get_moe_layer_wise_logging_tracker()
    log_vals = {k: v["values"].detach().clone() for k, v in log_tracker.items() if torch.is_tensor(v.get("values"))}

    pp_first = parallel_state.is_pipeline_first_stage(ignore_virtual=True)
    pp_last = parallel_state.is_pipeline_last_stage(ignore_virtual=True)
    cp = parallel_state.get_context_parallel_world_size()
    tp = parallel_state.get_tensor_model_parallel_world_size()
    seq = args.seq_length // cp
    if getattr(config, "sequence_parallel", False):
        seq //= tp
    shape = (seq, args.micro_batch_size, args.hidden_size)
    dtype = config.pipeline_dtype if config.pipeline_dtype is not None else torch.bfloat16
    batch = _fake_batch(args)
    passes = int(os.environ.get("BEVERIN_PREWARM_PASSES", "1"))
    times = []
    # the loss function validates its result through the rerun state machine, which only accepts that
    # inside train_step's forward-backward loop; in DISABLED mode it only checks for NaN/Inf and keeps no state
    rsm = get_rerun_state_machine()
    rsm_mode = rsm.mode
    rsm.mode = RerunMode.DISABLED
    with ExitStack() as stack:
        for chunk in model:
            stack.enter_context(chunk.no_sync())
        for _ in range(passes):
            for i, chunk in enumerate(model):
                ts = time.time()
                first = pp_first and i == 0
                last = pp_last and i == len(model) - 1
                inp = None if first else torch.randn(shape, dtype=dtype, device="cuda").requires_grad_(True)
                out, _ = schedules.forward_step(
                    forward_step_func, _FakeIterator(batch), chunk, 1, inp, [], config,
                    cp_group_size=cp, collect_non_loss_data=False, checkpoint_activations_microbatch=None,
                    is_first_microbatch=False, current_microbatch=0, vp_stage=i, is_last_stage=last,
                )
                tf = time.time()
                grad = None if last else torch.randn_like(out) * 1e-3
                if not last:  # as the schedule does after sending; custom_backward expects it
                    schedules.deallocate_output_tensor(out, config.deallocate_pipeline_outputs)
                schedules.backward_step(inp, out, grad, config)
                torch.cuda.synchronize()
                times.append((i, round(tf - ts, 1), round(time.time() - tf, 1)))
                del out, inp, grad

    # before the RNG restore below: it draws random inputs
    opt_info = ""
    if os.environ.get("BEVERIN_PREWARM_OPT", "1") == "1" and optimizer is not None:
        n, t = _prewarm_optimizer(optimizer)
        opt_info = f", optimizer {n} shapes {t:.1f} s"

    # restore state
    rsm.mode = rsm_mode
    with torch.no_grad():
        for b, v in bufs:
            b.copy_(v)
        for k, v in log_vals.items():
            log_tracker[k]["values"].copy_(v)
    for chunk in model:
        chunk.zero_grad_buffer()
    # hand the cached activation blocks back, so step 1 grows the allocator as it would without the pre-warm
    # (same memory layout as the control; new segments cost ~15 GB/s on MI300A, i.e. a few seconds)
    if os.environ.get("BEVERIN_PREWARM_EMPTY_CACHE", "1") == "1":
        import gc
        gc.collect()
        torch.cuda.empty_cache()
    torch.set_rng_state(rng_cpu)
    torch.cuda.set_rng_state(rng_cuda)
    tracker.set_states(tracker_states)
    torch.cuda.synchronize()
    dt = time.time() - t0
    print(f"[BEVERIN_PREWARM] rank {rank} chunks (id, fwd s, bwd s) {times}{opt_info} total {dt:.1f} s", flush=True)
    m = torch.tensor([dt], device="cuda")
    torch.distributed.all_reduce(m, op=torch.distributed.ReduceOp.MAX)
    if rank == 0:
        print(f"[BEVERIN_PREWARM] done, max over ranks {m.item():.1f} s", flush=True)
