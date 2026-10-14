# beverin-rocm (lab T6): alltoall token dispatcher that moves tokens with the hierarchical, rail-aligned,
# deduplicated dispatch/combine of beverin_ep_t6.py (optional fp8 dispatch with BEVERIN_T6_FP8=1). Enabled with
# BEVERIN_T6_EP=1 (with --moe-token-dispatcher-type alltoall, expert TP 1, dropless, EP a multiple of 4 laid out
# 4 consecutive EP ranks per node). Same interface as MoEAlltoAllTokenDispatcher: the experts get tokens sorted by
# local expert, tokens_per_expert on the host and the permuted probs, and combine returns the sum of the expert
# outputs per token (probs are applied inside the experts, as in the plain path). Default off.
import os

from megatron.core.transformer.moe import beverin_ep_t6 as _t6
from megatron.core.transformer.moe.token_dispatcher import MoEAlltoAllTokenDispatcher

ENABLED = os.environ.get("BEVERIN_T6_EP", "0") == "1"


class BeverinT6AlltoAllTokenDispatcher(MoEAlltoAllTokenDispatcher):
    def __init__(self, num_local_experts, local_expert_indices, config, pg_collection=None):
        super().__init__(num_local_experts, local_expert_indices, config=config, pg_collection=pg_collection)
        assert self.tp_size == 1, "BEVERIN_T6_EP needs expert TP 1"
        assert not self.drop_and_pad and config.moe_expert_capacity_factor is None, "BEVERIN_T6_EP is dropless only"
        assert not config.moe_router_padding_for_quantization
        assert getattr(self, "shared_experts", None) is None
        hidden = config.moe_latent_size or config.hidden_size
        if os.environ.get("BEVERIN_A3_T7", "0") == "1":
            # lab2 A3 (a3-0002): same rail all-to-all, intra-node stage as Triton push/pull over symmetric memory
            # (no RCCL node all-to-all, so a global RCCL channel cap does not slow it); fp8 as BEVERIN_T6_FP8
            from megatron.core.transformer.moe import beverin_a3_t7 as _a7
            self.hier = _a7.T7EP(self.ep_group, self.num_experts, config.moe_router_topk, hidden,
                                 gpn=int(os.environ.get("BEVERIN_T6_GPN", 4)),
                                 fp8=os.environ.get("BEVERIN_T6_FP8", "0") == "1")
        else:
            self.hier = _t6.HierEP(self.ep_group, self.num_experts, config.moe_router_topk, hidden,
                                   gpn=int(os.environ.get("BEVERIN_T6_GPN", 4)))
        assert self.hier.NLE == num_local_experts
        self._pl = None
        self._tpe = None

    def set_shared_experts(self, shared_experts):
        raise AssertionError("BEVERIN_T6_EP does not support shared expert overlap")

    def dispatch_preprocess(self, hidden_states, routing_map, probs):
        self.hidden_shape = hidden_states.shape
        self.routing_map = routing_map
        self.probs = probs
        self._pl = self.hier.plan(routing_map)
        return hidden_states.view(-1, self.hidden_shape[-1]), probs

    def token_dispatch(self, hidden_states, hidden_states_sf, probs):
        assert hidden_states_sf is None
        x3, p3, tpe = self.hier.dispatch(hidden_states, probs, self._pl)
        self._tpe = tpe
        if _t6._STASH:  # lab2 A6: hand the receive buffer to the expert recompute instead of keeping x3
            r2, rows3, hier, dt = self._pl.pop("r2"), self._pl["rows3"], self.hier, x3.dtype
            self.stash_rebuild = lambda: hier.rebuild_x3(r2, rows3, dt)
        return x3, None, p3

    def dispatch_postprocess(self, global_input_tokens, global_probs):
        tpe, self._tpe = self._tpe, None
        if getattr(self.config, "overlap_moe_expert_parallel_comm", False) and not _t6._STASH:
            # The EP-overlap schedule frees the inputs of its "mlp" node after the forward (it assumes that
            # dispatch_postprocess makes a new tensor, as the plain path's sort does), but the experts save
            # their input for backward/recompute: hand them a copy.
            global_input_tokens = global_input_tokens.clone()
        return global_input_tokens, tpe, global_probs

    def combine_preprocess(self, hidden_states):
        return hidden_states

    def token_combine(self, hidden_states, async_finish=True, allocate_on_comm_stream=True):
        out = self.hier.combine(hidden_states, self._pl)
        self._pl = None
        return out

    def combine_postprocess(self, hidden_states):
        return hidden_states.view(self.hidden_shape)
