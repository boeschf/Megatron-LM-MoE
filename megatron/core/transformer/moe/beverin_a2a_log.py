"""Beverin lab: cheap per-rank, per-layer EP all-to-all timing and token-count logging.

Enabled with BEVERIN_A2A_LOG=1. Every EP all_to_all_single of the alltoall dispatcher is bracketed by
two device events on the current stream (wait-for-peers plus transfer, as seen by this rank). Events are
read once per step in flush() (after the step, when they are complete) and aggregated per
(MoE layer, kind) into one JSON line per step per rank under $BEVERIN_A2A_LOG_DIR/rank-XXXXX.jsonl.
Kinds: dp/dt = dispatch probs/tokens (forward and recompute), bp/bt = their backward,
cf = combine forward (and recompute), cb = combine backward.
Per (layer, kind): n calls, t = total ms, tmax = max ms of one call, sb/rb = bytes sent/received,
st/rt = tokens sent/received (rows), pmax = max rows to/from a single peer (imbalance).
"""
import json
import os

import torch

ENABLED = os.environ.get("BEVERIN_A2A_LOG", "0") == "1"
_state = {"layer": -1, "kind": None, "fh": None}
_recs = []


def set_layer(layer_number):
    _state["layer"] = int(layer_number) if layer_number is not None else -1


def current_layer():
    return _state["layer"]


class kind:
    """Context manager that overrides the kind tag (used by backward passes)."""

    def __init__(self, k, layer=None):
        self.k, self.layer = k, layer

    def __enter__(self):
        self.prev = (_state["kind"], _state["layer"])
        _state["kind"] = self.k
        if self.layer is not None:
            _state["layer"] = self.layer

    def __exit__(self, *a):
        _state["kind"], _state["layer"] = self.prev


def a2a(default_kind, output, input, output_split_sizes, input_split_sizes, group):
    if not ENABLED:
        return torch.distributed.all_to_all_single(
            output, input, output_split_sizes=output_split_sizes, input_split_sizes=input_split_sizes, group=group
        )
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    r = torch.distributed.all_to_all_single(
        output, input, output_split_sizes=output_split_sizes, input_split_sizes=input_split_sizes, group=group
    )
    e.record()
    row = input.numel() // max(1, input.shape[0]) * input.element_size() if input.dim() > 0 and input.shape[0] else 0
    ins = list(input_split_sizes) if input_split_sizes is not None else [input.shape[0] // group.size()] * group.size()
    outs = list(output_split_sizes) if output_split_sizes is not None else [output.shape[0] // group.size()] * group.size()
    _recs.append((_state["layer"], _state["kind"] or default_kind, s, e, row, int(sum(ins)), int(sum(outs)), int(max(ins)), int(max(outs))))
    return r


def flush(iteration):
    if not ENABLED or not _recs:
        return
    torch.cuda.synchronize()
    agg = {}
    for layer, k, s, e, row, st, rt, ps, pr in _recs:
        key = f"{layer}:{k}"
        a = agg.setdefault(key, [0, 0.0, 0.0, 0, 0, 0, 0, 0])
        t = s.elapsed_time(e)
        a[0] += 1; a[1] += t; a[2] = max(a[2], t)
        a[3] += st * row; a[4] += rt * row; a[5] += st; a[6] += rt; a[7] = max(a[7], ps, pr)
    _recs.clear()
    if _state["fh"] is None:
        d = os.environ.get("BEVERIN_A2A_LOG_DIR", ".")
        os.makedirs(d, exist_ok=True)
        _state["fh"] = open(os.path.join(d, f"rank-{torch.distributed.get_rank():05d}.jsonl"), "a")
    out = {"it": int(iteration), "rank": torch.distributed.get_rank(), "a": {
        key: {"n": a[0], "t": round(a[1], 2), "tmax": round(a[2], 2), "sb": a[3], "rb": a[4], "st": a[5], "rt": a[6], "pmax": a[7]}
        for key, a in sorted(agg.items())}}
    _state["fh"].write(json.dumps(out, separators=(",", ":")) + "\n")
    _state["fh"].flush()
