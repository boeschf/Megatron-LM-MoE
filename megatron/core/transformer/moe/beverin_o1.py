# bench2 O1: memory-neutral overlap of the T6 EP exchange inside one MoE layer (no extra microbatch in flight).
#   BEVERIN_O1_SHARED_SIDE=1: the shared expert runs on a side stream, launched right after the combine all-to-all,
#     so its forward overlaps the forward combine and (autograd replays it on the same stream, and its nodes are
#     newer than the combine node) its backward overlaps the combine backward.
#   BEVERIN_O1_WGRAD_SIDE=1: the fused fp32 expert wgrad (beverin_vk) runs on a side stream, so the fc1 wgrad overlaps
#     the dispatch backward all-to-all; DDP's grad sync and the end of backward wait for the side stream.
#   BEVERIN_O1_<NAME>_PRIO=-1|0: priority of the side streams (lower = higher priority; default 0).
import os

import torch

SHARED_SIDE = os.environ.get("BEVERIN_O1_SHARED_SIDE", "0") == "1"
#   BEVERIN_O1_WGRAD_SIDE=2: as 1, but the expert wgrads are queued and launched on the side stream only when the next
#     backward all-to-all of the T6 exchange has been launched (fc2 and fc1 wgrad both overlap the dispatch backward).
_WMODE = os.environ.get("BEVERIN_O1_WGRAD_SIDE", "0")
WGRAD_SIDE = _WMODE in ("1", "2")
WGRAD_DEFER = _WMODE == "2"
_S = {}
_PEND = []


def defer(fn):
    """Queue fn (reads tensors it holds references to) behind the current point of the current stream."""
    _PEND.append((fn, torch.cuda.current_stream().record_event()))
    if len(_PEND) > 4:  # no all-to-all came (other dispatcher): do not hold inputs for long
        flush()


def flush():
    """Launch the queued wgrads on the wgrad side stream (call right after launching an all-to-all)."""
    if not _PEND:
        return
    s = stream("wgrad")
    with torch.cuda.stream(s):
        for fn, ev in _PEND:
            s.wait_event(ev)
            fn()
    _PEND.clear()


def stream(name):
    s = _S.get(name)
    if s is None:
        prio = int(os.environ.get(f"BEVERIN_O1_{name.upper()}_PRIO", "0"))
        s = _S[name] = torch.cuda.Stream(priority=prio)
    return s


def join():
    """Make the current stream wait for every O1 side stream (before grad sync and at the end of backward)."""
    flush()
    if _S:
        cur = torch.cuda.current_stream()
        for s in _S.values():
            if s != cur:
                cur.wait_stream(s)
