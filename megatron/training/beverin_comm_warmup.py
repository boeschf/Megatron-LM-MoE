# beverin-rocm (smallscale): connect every RCCL communicator and p2p peer right after model-parallel
# init, while the APU still has free memory. On Beverin MI300A, creating a CXI endpoint or registering
# RCCL's buffers (cxil_map) fails with EINVAL / FI_EDOMAIN once roughly 80 GiB of an APU's 128 GiB are in
# use, and RCCL connects lazily (first PP send, first all-to-all, unbatched p2p comms), i.e. after the
# model and optimizer are allocated. Enabled with BEVERIN_COMM_WARMUP=1; best with NCCL_RUNTIME_CONNECT=0
# so that the first all-reduce on a communicator connects all its collective algorithms.
# SS_BALLAST_GB (test only) then allocates and holds that much GPU memory to emulate the full model.
import os
import time

import torch
import torch.distributed as dist

_BALLAST = []


def _groups():
    from megatron.core import parallel_state as ps

    seen, out = set(), []
    for name in sorted(vars(ps)):
        if "GLOO" in name or not name.isupper():
            continue
        val = getattr(ps, name)
        for g in val if isinstance(val, (list, tuple)) else [val]:
            if isinstance(g, dist.ProcessGroup) and id(g) not in seen:
                seen.add(id(g))
                if dist.get_backend(g) == "nccl":
                    out.append((name, g))
    return out


def _p2p(numel, batched):
    """Run Megatron's own PP p2p helpers once in both directions, so exactly the communicators the
    schedule uses get connected (for PP=2, _p2p_ops sends one direction over the WORLD group, which
    creates separate 2-rank pair communicators)."""
    from megatron.core import parallel_state as ps
    from megatron.core.pipeline_parallel import p2p_communication as p2p

    g = ps.get_pipeline_model_parallel_group()
    if dist.get_world_size(g) == 1:
        return
    nxt, prv = ps.get_pipeline_model_parallel_next_rank(), ps.get_pipeline_model_parallel_prev_rank()
    t = [torch.ones(numel, dtype=torch.bfloat16, device="cuda") for _ in range(4)]
    fn = p2p._batched_p2p_ops if batched else p2p._p2p_ops
    reqs = fn(tensor_send_prev=t[0], tensor_recv_prev=t[1], tensor_send_next=t[2], tensor_recv_next=t[3],
              group=g, prev_pipeline_rank=prv, next_pipeline_rank=nxt)
    for q in (reqs.values() if isinstance(reqs, dict) else reqs):
        q.wait()
    torch.cuda.synchronize()


def warmup(args):
    if os.environ.get("BEVERIN_COMM_WARMUP", "0") == "1":
        _connect(args)
    _ballast()


def _connect(args):
    from megatron.core import parallel_state as ps

    t0 = time.time()
    numel = args.seq_length * args.micro_batch_size * args.hidden_size  # one PP activation
    x = torch.ones(1 << 20, device="cuda")
    # BEVERIN_COMM_WARMUP_PARTS (default "ar,p2p,a2a"): which connections to make early. In the full
    # PP4 EP16 run the p2p/a2a parts are made in the first microbatch anyway (low memory); "ar" covers the
    # all-reduces that otherwise connect lazily at the end of step 1 (expert bias, aux losses, grads).
    parts = os.environ.get("BEVERIN_COMM_WARMUP_PARTS", "ar,p2p,a2a").split(",")
    groups = _groups()
    if "ar" in parts:
        for _, g in groups:
            dist.all_reduce(x, group=g)
    if "p2p" in parts:
        _p2p(numel, batched=True)
        _p2p(numel, batched=False)
    ep = ps.get_expert_model_parallel_group()
    if "a2a" in parts and dist.get_world_size(ep) > 1:
        a = torch.ones(numel * 2, dtype=torch.bfloat16, device="cuda")  # ~ a dispatch all-to-all
        b = torch.empty_like(a)
        dist.all_to_all_single(b, a, group=ep)
        dist.all_to_all_single(a, b, group=ep)
    torch.cuda.synchronize()
    dist.barrier()
    if dist.get_rank() == 0:
        print(f"[beverin] comm warm-up ({parts}): {len(groups)} NCCL groups "
              f"in {time.time() - t0:.1f} s", flush=True)


def _ballast():
    gb = float(os.environ.get("SS_BALLAST_GB", "0"))
    if gb > 0:
        while sum(t.numel() for t in _BALLAST) < gb * 2**30:
            _BALLAST.append(torch.ones(2**30, dtype=torch.uint8, device="cuda"))
        if dist.get_rank() == 0:
            free, tot = torch.cuda.mem_get_info()
            print(f"[beverin] ballast {gb} GiB held, free {free / 2**30:.1f} of {tot / 2**30:.1f} GiB", flush=True)
