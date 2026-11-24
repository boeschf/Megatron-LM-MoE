"""CPU-only permutation-preload coverage; run directly without CUDA imports."""

import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "moe_permutation_warmup_cpu", ROOT / "megatron/training/moe_permutation_warmup.py")
WARMUP = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(WARMUP)


def signature(tokens, capacity, experts):
    blocks = (tokens + 1023) // 1024
    width = 1 << (experts * blocks - 1).bit_length()
    return width, (tokens == 1, tokens % 16 == 0), (capacity == 1, capacity % 16 == 0)


class TestPermutationWarmup(unittest.TestCase):
    def test_every_reachable_row_and_output_specialization(self):
        # Missing a workspace or scalar variant can load a new binary under persistent EP.
        for maximum, experts, topk in ((1, 16, 8), (17, 16, 8), (1025, 3, 8),
                                       (4097, 3, 1), (131072, 16, 8)):
            with self.subTest(maximum=maximum, experts=experts, topk=topk):
                cases = WARMUP.permutation_cases(maximum, experts, topk, min(65536, maximum))
                warmed = {signature(n, cap, experts) for n, cap in cases}
                self.assertTrue(all(1 <= n <= maximum and n <= cap <= n * topk
                                    for n, cap in cases))
                for tokens in range(1, maximum + 1):
                    capacities = {tokens, tokens * topk}
                    aligned = ((tokens + 15) // 16) * 16
                    if aligned <= tokens * topk:
                        capacities.add(aligned)
                    if tokens % 16 == 0 and tokens + 1 <= tokens * topk:
                        capacities.add(tokens + 1)
                    for capacity in capacities:
                        self.assertIn(signature(tokens, capacity, experts), warmed)

    def test_unsupported_paths_stop_before_gpu_initialization(self):
        args = dict(tensor_model_parallel_size=1, context_parallel_size=1,
                    sequence_parallel=False, bf16=True, fp8=None,
                    moe_token_dispatcher_type="flex", moe_flex_dispatcher_backend="deepep",
                    moe_permute_fusion=True, moe_router_padding_for_quantization=False,
                    expert_model_parallel_size=16, num_experts=256, moe_router_topk=8,
                    seq_length=4096, micro_batch_size=2, moe_latent_size=1792, hidden_size=3584)
        invalid = [dict(tensor_model_parallel_size=2), dict(context_parallel_size=2),
                   dict(sequence_parallel=True), dict(bf16=False), dict(fp8="e4m3"),
                   dict(moe_token_dispatcher_type="alltoall"), dict(moe_permute_fusion=False),
                   dict(moe_router_padding_for_quantization=True), dict(num_experts=255),
                   dict(expert_model_parallel_size=0), dict(moe_router_topk=0),
                   dict(seq_length=0), dict(moe_latent_size=-1)]
        with patch.dict(os.environ, {"MEGATRON_MOE_PERMUTATION_WARMUP": "1"}):
            for fields in invalid:
                with self.subTest(fields=fields), self.assertRaises(ValueError):
                    WARMUP.warmup(SimpleNamespace(**(args | fields)))


if __name__ == "__main__":
    unittest.main()
