# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Unit tests for the cross-document / packed-sequence batch helpers."""

import pytest
import torch

from megatron.core.utils import (
    _merge_cu_seqlens_across_micro_batch,
    flatten_batch_for_packed_sequences,
)


def _pad_row(cu_seqlens, seq_length):
    """Reproduce the dataset's right-padding of a cu_seqlens row."""
    row = torch.full((seq_length + 1,), seq_length, dtype=torch.int32)
    row[: len(cu_seqlens)] = torch.tensor(cu_seqlens, dtype=torch.int32)
    return row


class TestMergeCuSeqlens:
    def test_single_sample_strips_padding(self):
        seq_length = 16
        merged = _merge_cu_seqlens_across_micro_batch(
            _pad_row([0, 5, 11, 16], seq_length).unsqueeze(0), seq_length
        )
        assert torch.equal(merged, torch.tensor([0, 5, 11, 16], dtype=torch.int32))
        assert merged.dtype == torch.int32

    def test_single_document_sample(self):
        """A sample that is one whole document: [0, seq_length] and nothing else."""
        seq_length = 16
        merged = _merge_cu_seqlens_across_micro_batch(
            _pad_row([0, 16], seq_length).unsqueeze(0), seq_length
        )
        assert torch.equal(merged, torch.tensor([0, 16], dtype=torch.int32))

    def test_multi_sample_offsets_and_drops_leading_zero(self):
        seq_length = 16
        rows = torch.stack(
            [
                _pad_row([0, 5, 16], seq_length),
                _pad_row([0, 16], seq_length),
                _pad_row([0, 3, 9, 16], seq_length),
            ]
        )
        merged = _merge_cu_seqlens_across_micro_batch(rows, seq_length)
        # sample 0 -> 0,5,16 | sample 1 -> +16 | sample 2 -> +32
        assert torch.equal(
            merged, torch.tensor([0, 5, 16, 32, 35, 41, 48], dtype=torch.int32)
        )

    def test_merged_is_valid_cu_seqlens(self):
        """Strictly increasing, starts at 0, ends at mbs * seq_length."""
        seq_length = 32
        mbs = 4
        rows = torch.stack(
            [
                _pad_row([0, 7, 32], seq_length),
                _pad_row([0, 32], seq_length),
                _pad_row([0, 1, 2, 31, 32], seq_length),
                _pad_row([0, 16, 32], seq_length),
            ]
        )
        merged = _merge_cu_seqlens_across_micro_batch(rows, seq_length)
        assert merged[0].item() == 0
        assert merged[-1].item() == mbs * seq_length
        assert torch.all(merged[1:] - merged[:-1] > 0)


class TestFlattenBatchForPackedSequences:
    def _batch(self, mbs, seq_length, rows):
        return {
            'tokens': torch.arange(mbs * seq_length).reshape(mbs, seq_length),
            'labels': torch.arange(mbs * seq_length).reshape(mbs, seq_length),
            'loss_mask': torch.ones(mbs, seq_length),
            'position_ids': torch.zeros(mbs, seq_length, dtype=torch.long),
            'cu_seqlens': torch.stack([_pad_row(r, seq_length) for r in rows]),
            'max_seqlen': torch.tensor(
                [max(b - a for a, b in zip(r[:-1], r[1:])) for r in rows], dtype=torch.int32
            ),
        }

    def test_no_cu_seqlens_is_a_passthrough(self):
        batch = {'tokens': torch.zeros(2, 8), 'cu_seqlens': None}
        assert flatten_batch_for_packed_sequences(batch) is batch

    def test_micro_batch_size_one(self):
        seq_length = 16
        batch = flatten_batch_for_packed_sequences(
            self._batch(1, seq_length, [[0, 5, 11, 16]])
        )
        assert batch['cu_seqlens'].shape == (1, 4)
        assert torch.equal(batch['cu_seqlens'][0], torch.tensor([0, 5, 11, 16], dtype=torch.int32))
        assert batch['max_seqlen'].shape == (1,)
        assert batch['max_seqlen'].item() == 6
        assert batch['tokens'].shape == (1, seq_length)

    def test_micro_batch_size_greater_than_one(self):
        seq_length = 16
        mbs = 3
        rows = [[0, 5, 16], [0, 16], [0, 3, 9, 16]]
        original_tokens = torch.arange(mbs * seq_length)
        batch = flatten_batch_for_packed_sequences(self._batch(mbs, seq_length, rows))

        assert batch['tokens'].shape == (1, mbs * seq_length)
        assert batch['labels'].shape == (1, mbs * seq_length)
        assert batch['loss_mask'].shape == (1, mbs * seq_length)
        assert batch['position_ids'].shape == (1, mbs * seq_length)
        # Row-major flatten: sample order is preserved.
        assert torch.equal(batch['tokens'][0], original_tokens)

        assert batch['cu_seqlens'].shape[0] == 1
        assert batch['cu_seqlens'][0, -1].item() == mbs * seq_length
        # max over samples: sample 1 is one 16-token document.
        assert batch['max_seqlen'].shape == (1,)
        assert batch['max_seqlen'].item() == 16

    def test_middle_pipeline_stage_has_only_cu_seqlens(self):
        """A middle PP stage receives cu_seqlens/max_seqlen but no token tensors;
        the sequence length must then be recovered from cu_seqlens itself."""
        seq_length = 16
        mbs = 2
        batch = {
            'tokens': None,
            'labels': None,
            'loss_mask': None,
            'position_ids': None,
            'cu_seqlens': torch.stack(
                [_pad_row([0, 5, 16], seq_length), _pad_row([0, 16], seq_length)]
            ),
            'max_seqlen': torch.tensor([11, 16], dtype=torch.int32),
        }
        batch = flatten_batch_for_packed_sequences(batch)
        assert torch.equal(
            batch['cu_seqlens'][0], torch.tensor([0, 5, 16, 32], dtype=torch.int32)
        )
        assert batch['max_seqlen'].item() == 16


if __name__ == "__main__":
    pytest.main([__file__])
