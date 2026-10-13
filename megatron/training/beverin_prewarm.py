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
- with BEVERIN_PREWARM_COMM=1 (default off, added after C8 was promoted), one tiny all-reduce on each process group that the end of step 1 uses
  (pipeline, model-parallel, data-parallel, tensor+data-parallel, expert data-parallel), so these communicators
  are created here (after the dataloader workers forked) and not lazily one after another at the end of step 1.
- BEVERIN_SHARE_STORAGE_SO=<.so> (independent of the pre-warm): load the share_storage extension from a built file
  instead of load_inline, so ranks do not queue on torch's extension lock.
- with BEVERIN_PREWARM_LOG=1, training_log's MoE-metric reduction once (its process groups' communicators).
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


def _prewarm_comms():
    """One tiny all-reduce on each process group the end of step 1 uses (pipeline, model-parallel, data-parallel
    with and without context parallel, tensor+data-parallel, expert data-parallel), so their communicators are
    created now, after the dataloader workers have forked, instead of lazily one after another at the end of
    step 1. Every rank walks the same list of group kinds in the same order; each kind partitions the world."""
    from megatron.core import parallel_state as ps

    t0 = time.time()
    getters = [
        ("pp", ps.get_pipeline_model_parallel_group),
        ("mp", ps.get_model_parallel_group),
        ("dp_cp", lambda: ps.get_data_parallel_group(with_context_parallel=True)),
        ("dp", ps.get_data_parallel_group),
        ("tdp_cp", lambda: ps.get_tensor_and_data_parallel_group(with_context_parallel=True)),
        ("expt_dp", ps.get_expert_data_parallel_group),
    ]
    x = torch.ones(1, device="cuda")
    done, seen = [], set()
    for name, get in getters:
        try:
            g = get()
        except Exception:
            continue
        if g is None or id(g) in seen or torch.distributed.get_world_size(g) == 1:
            continue
        seen.add(id(g))
        torch.distributed.all_reduce(x, group=g)
        done.append(name)
    torch.cuda.synchronize()
    return done, time.time() - t0


def _prewarm_comms_intra_stage(nbytes):
    """BEVERIN_PREWARM_COMM=2: only the groups inside one pipeline stage (data-parallel with and without context
    parallel, tensor+data-parallel, expert data-parallel), so no rank waits for another stage's pre-warm, and with
    message sizes like the gradient reduce-scatter and the optimizer's parameter broadcasts, so RCCL's runtime
    connections for the large-message channels/protocols are made here rather than at the end of step 1."""
    from megatron.core import parallel_state as ps

    t0 = time.time()
    getters = [
        ("dp_cp", lambda: ps.get_data_parallel_group(with_context_parallel=True)),
        ("dp", ps.get_data_parallel_group),
        ("tdp_cp", lambda: ps.get_tensor_and_data_parallel_group(with_context_parallel=True)),
        ("expt_dp", ps.get_expert_data_parallel_group),
    ]
    small = torch.ones(1, device="cuda")
    done, seen = [], set()
    for name, get in getters:
        try:
            g = get()
        except Exception:
            continue
        if g is None or id(g) in seen or torch.distributed.get_world_size(g) == 1:
            continue
        seen.add(id(g))
        n = torch.distributed.get_world_size(g)
        numel = max(n, (nbytes // 4) // n * n)
        big = torch.ones(numel, dtype=torch.float32, device="cuda")
        out = torch.empty(numel // n, dtype=torch.float32, device="cuda")
        torch.distributed.all_reduce(small, group=g)
        torch.distributed.all_reduce(big, group=g)
        torch.distributed.reduce_scatter_tensor(out, big, group=g)
        torch.distributed.broadcast(big, src=torch.distributed.get_global_rank(g, 0), group=g)
        del big, out
        done.append(name)
    torch.cuda.synchronize()
    return done, time.time() - t0


def _prewarm_p2p(config, numel, dtype):
    """BEVERIN_PREWARM_P2P=1: one exchange with the previous and next pipeline stage through Megatron's own p2p
    helper (the overlapped `_p2p_ops` or the batched one, as the schedule uses), so the pair communicators that
    torch creates lazily for isend/irecv exist before step 1. In step 1 they are created one stage pair after the
    other along the pipeline; here all pairs connect at once."""
    from megatron.core import parallel_state as ps
    from megatron.core.pipeline_parallel import p2p_communication as p2p

    t0 = time.time()
    g = ps.get_pipeline_model_parallel_group()
    if torch.distributed.get_world_size(g) == 1:
        return 0.0
    nxt, prv = ps.get_pipeline_model_parallel_next_rank(), ps.get_pipeline_model_parallel_prev_rank()
    t = [torch.zeros(numel, dtype=dtype, device="cuda") for _ in range(4)]
    fn = p2p._p2p_ops if getattr(config, "overlap_p2p_comm", False) else p2p._batched_p2p_ops
    reqs = fn(tensor_send_prev=t[0], tensor_recv_prev=t[1], tensor_send_next=t[2], tensor_recv_next=t[3],
              group=g, prev_pipeline_rank=prv, next_pipeline_rank=nxt)
    for q in (reqs.values() if isinstance(reqs, dict) else reqs):
        q.wait()
    torch.cuda.synchronize()
    return time.time() - t0


def _moe_track_names(args):
    """The MoE metric names training_log passes to track_moe_metrics (copied from training.py)."""
    names = []
    lb = args.moe_router_load_balancing_type  # same membership tests as training.py (str or list)
    if "aux_loss" in lb:
        names.append("load_balancing_loss")
    if "seq_aux_loss" in lb:
        names.append("seq_load_balancing_loss")
    if "global_aux_loss" in lb:
        names.append("global_load_balancing_loss")
    if args.moe_z_loss_coeff is not None:
        names.append("z_loss")
    names += ["expert_max_violation", "expert_min_violation", "expert_median_violation",
              "global_expert_max_violation", "global_expert_min_violation", "global_expert_median_violation"]
    if getattr(args, "moe_router_ep_violation_metrics", False):
        names += ["ep_expert_max_violation", "ep_expert_min_violation", "ep_expert_median_violation"]
    if getattr(args, "moe_router_bias_metrics", False):
        qb = "quantile_balancing" in lb
        if args.moe_router_enable_expert_bias and not qb:
            names += ["expert_bias_mean", "expert_bias_std", "expert_bias_min", "expert_bias_max"]
        if qb:
            names += ["qb_beta_mean", "qb_beta_std", "qb_beta_min", "qb_beta_max"]
    return names


def _prewarm_log(model):
    """BEVERIN_PREWARM_LOG=1: run the MoE-metric reduction of training_log once (reduce_aux_losses_tracker_across_ranks
    with the same names and process groups), so the communicators it creates lazily (one per reduce/avg group, each
    a full RCCL init) exist before step 1; after step 1 they cost 18 s (4 nodes) to 47 s (16 nodes)."""
    from megatron.core.transformer.moe.moe_utils import (
        get_moe_layer_wise_logging_tracker, reduce_aux_losses_tracker_across_ranks)
    from megatron.core.utils import get_attr_wrapped_model
    from megatron.training import get_args

    args = get_args()
    if args.num_experts is None:
        return 0.0
    t0 = time.time()
    names = _moe_track_names(args)
    tracker = get_moe_layer_wise_logging_tracker()
    n = args.num_layers + (args.mtp_num_layers or 0)
    for key in names:  # as track_moe_metrics(force_initialize=True)
        if key not in tracker:
            tracker[key] = {"values": torch.zeros(n, device="cuda"), "reduce_group": None, "avg_group": None,
                            "reduce_group_has_dp": False}
    pgc = get_attr_wrapped_model(model[0], "pg_collection")
    reduce_aux_losses_tracker_across_ranks(names, pg_collection=pgc)
    torch.cuda.synchronize()
    return time.time() - t0


def _preload_share_storage():
    """BEVERIN_SHARE_STORAGE_SO=<path to a built share_storage_ext .so>: load the fork's random.py share_storage
    extension from that file instead of torch's load_inline. load_inline takes one file lock in the shared
    TORCH_EXTENSIONS_DIR around every call and runs ninja, so all ranks queue on it at their first recompute
    (15 s per rank on one node, py-spy 678449/678558). Independent of BEVERIN_PREWARM."""
    so = os.environ.get("BEVERIN_SHARE_STORAGE_SO")
    if not so or not os.path.exists(so):
        return False
    import importlib
    import importlib.util

    tp_random = importlib.import_module("megatron.core.tensor_parallel.random")

    if getattr(tp_random, "_share_storage_ext", None) is not None:
        return True
    spec = importlib.util.spec_from_file_location("share_storage_ext", so)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    tp_random._share_storage_ext = mod
    return True


def maybe_prewarm(forward_step_func, model, config, optimizer=None):
    _preload_share_storage()
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

    comm_info = ""
    if os.environ.get("BEVERIN_PREWARM_P2P", "0") == "1":
        comm_info += f", p2p {_prewarm_p2p(config, shape[0] * shape[1] * shape[2], dtype):.1f} s"
    if os.environ.get("BEVERIN_PREWARM_LOG", "0") == "1":
        comm_info += f", moe-log reduce {_prewarm_log(model):.1f} s"
    comm_mode = os.environ.get("BEVERIN_PREWARM_COMM", "0")
    if comm_mode == "1":
        names, t = _prewarm_comms()
        comm_info += f", comms {names} {t:.1f} s"
    elif comm_mode == "2":
        names, t = _prewarm_comms_intra_stage(int(float(os.environ.get("BEVERIN_PREWARM_COMM_MB", "256")) * 2**20))
        comm_info += f", intra-stage comms {names} {t:.1f} s"

    # restore state
    rsm.mode = rsm_mode
    with torch.no_grad():
        for b, v in bufs:
            b.copy_(v)
        for k, entry in log_tracker.items():  # entries created by the pre-warm start from zero, as without it
            if k in log_vals:
                entry["values"].copy_(log_vals[k])
            elif torch.is_tensor(entry.get("values")):
                entry["values"].zero_()
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
    print(f"[BEVERIN_PREWARM] rank {rank} chunks (id, fwd s, bwd s) {times}{opt_info}{comm_info} total {dt:.1f} s", flush=True)
    m = torch.tensor([dt], device="cuda")
    torch.distributed.all_reduce(m, op=torch.distributed.ReduceOp.MAX)
    if rank == 0:
        print(f"[BEVERIN_PREWARM] done, max over ranks {m.item():.1f} s", flush=True)
