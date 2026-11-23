"""Opt-in NCCL initialization before model allocation, without consuming RNG state.

All ranks must set MEGATRON_COMM_WARMUP=1 together. This warms the existing
process groups and native PP helpers, not DeepEP/UCCL dispatch or expert kernels.
The hook applies to immediate model-parallel initialization, not lazy MPU init.
"""

import os
import time


def warmup(args):
    """Connect native NCCL groups and both PP directions when explicitly enabled."""
    if os.environ.get("MEGATRON_COMM_WARMUP", "0") != "1":
        return

    import torch
    import torch.distributed as dist
    from megatron.core import parallel_state as ps
    from megatron.core.pipeline_parallel import p2p_communication as p2p

    if not dist.is_initialized() or not torch.cuda.is_available():
        raise RuntimeError("communicator warmup requires initialized GPU distributed training")

    started = time.monotonic()
    groups = [("WORLD", dist.group.WORLD)]
    # Use the registry's stable name ordering, including list-valued PP groups.
    # Aliases must not warm the same ProcessGroup twice on a rank.
    seen = {id(dist.group.WORLD)}
    for name in sorted(vars(ps)):
        if not name.isupper() or "GLOO" in name:
            continue
        value = getattr(ps, name)
        for group in value if isinstance(value, (list, tuple)) else [value]:
            if isinstance(group, dist.ProcessGroup) and id(group) not in seen:
                seen.add(id(group))
                groups.append((name, group))

    collective = torch.zeros(1 << 20, device="cuda", dtype=torch.float32)
    warmed = []
    for name, group in groups:
        if dist.get_backend(group) != "nccl" or dist.get_world_size(group) == 1:
            continue
        dist.all_reduce(collective, group=group)
        # Do not leave one group's kernels active while initializing the next.
        torch.cuda.synchronize()
        warmed.append(name)

    pp_groups = ps.get_pipeline_model_parallel_group()
    for group in pp_groups if isinstance(pp_groups, list) else [pp_groups]:
        if dist.get_world_size(group) == 1:
            continue
        ranks = dist.get_process_group_ranks(group)
        index = ranks.index(dist.get_rank())
        previous, following = ranks[index - 1], ranks[(index + 1) % len(ranks)]
        elements = args.seq_length * args.micro_batch_size * args.hidden_size
        send = torch.zeros(elements, device="cuda", dtype=torch.bfloat16)
        recv_previous = torch.empty_like(send)
        recv_following = torch.empty_like(send)
        # The native helpers own the pair-communicator selection. Warm batched
        # and unbatched variants, including the VPP wraparound, before any model
        # work can overlap their first-use initialization.
        for operation in (p2p._batched_p2p_ops, p2p._p2p_ops):
            requests = operation(
                tensor_send_prev=send, tensor_recv_prev=recv_previous,
                tensor_send_next=send, tensor_recv_next=recv_following,
                group=group, prev_pipeline_rank=previous, next_pipeline_rank=following)
            for request in requests.values() if isinstance(requests, dict) else requests:
                request.wait()
            torch.cuda.synchronize()
        del send, recv_previous, recv_following

    dist.barrier()
    torch.cuda.synchronize()
    if dist.get_rank() == 0:
        print(f"[comm-warmup] NCCL groups={warmed}; native PP peers ready "
              f"in {time.monotonic() - started:.1f}s", flush=True)
