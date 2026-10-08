"""Adversarial index contracts for boundary training paths."""

from __future__ import annotations

import pytest
import torch

from gliner2.models.boundary.relations import (
    RelationPairBatch,
    SparseRelationScorer,
)
from gliner2.models.boundary.targets_device import dense_targets_from_pairs
from gliner2.models.boundary.validation import (
    filter_match_indices,
    safe_query_ids,
    validate_masked_spans,
)
from gliner2.models.outputs import CandidateTensorBatch
from gliner2.utils.device_errors import is_fatal_device_error


def test_safe_query_ids_uses_smallest_indexed_query_dimension():
    ids = torch.tensor([-1, 0, 2, 5])
    safe, valid = safe_query_ids(ids, 6, 3, 4)
    assert torch.equal(safe, torch.tensor([0, 0, 2, 2]))
    assert torch.equal(valid, torch.tensor([False, True, True, False]))


def test_filter_match_indices_drops_oob_and_padded_instances():
    indices = (
        torch.tensor([0, 0, 1, 0]),
        torch.tensor([0, 0, 0, 0]),
        torch.tensor([0, 1, 0, 9]),
        torch.tensor([0, 0, 0, 0]),
    )
    instance_mask = torch.tensor([[[True, False]], [[True, True]]])
    filtered = filter_match_indices(
        indices, (2, 1, 2, 1), instance_mask=instance_mask
    )
    assert torch.equal(filtered[0], torch.tensor([0, 1]))
    assert all(
        torch.equal(value, torch.tensor([0, 0]))
        for value in filtered[1:]
    )


def test_validate_masked_spans_rejects_active_invalid_pair_only():
    spans = torch.tensor([[[[0, 2], [3, 3], [8, 9]]]])
    mask = torch.tensor([[[True, True, False]]])
    with pytest.raises(ValueError, match="invalid active span"):
        validate_masked_spans(
            spans, mask, torch.tensor([8]), name="mentions"
        )


def test_dense_targets_omit_invalid_active_pairs_without_scatter_error():
    pairs = torch.tensor([[[[0, 2], [5, 3], [0, 99]]]])
    mask = torch.ones(1, 1, 3, dtype=torch.bool)
    start, end, inside = dense_targets_from_pairs(pairs, mask, text_length=6)
    assert start.sum() == 1
    assert end.sum() == 1
    assert inside.sum() == 2


def test_relation_scorer_masks_oob_batch_and_relation_indices():
    hidden = 8
    candidates = CandidateTensorBatch(
        indices=torch.tensor([[[[0, 1]]]]),
        proposal_logits=torch.zeros(1, 1, 1),
        pair_logits=torch.zeros(1, 1, 1),
        valid_mask=torch.ones(1, 1, 1, dtype=torch.bool),
        query_mask=torch.ones(1, 1, dtype=torch.bool),
    )
    pairs = RelationPairBatch(
        batch_index=torch.tensor([0, 3]),
        relation_index=torch.tensor([0, 4]),
        head_start=torch.tensor([0, 0]),
        head_end=torch.tensor([1, 1]),
        tail_start=torch.tensor([1, 1]),
        tail_end=torch.tensor([2, 2]),
        head_prob=torch.ones(2),
        tail_prob=torch.ones(2),
        pair_mask=torch.ones(2, dtype=torch.bool),
    )
    scorer = SparseRelationScorer(hidden)
    scores = scorer(
        torch.randn(1, 3, hidden),
        torch.randn(1, 1, hidden),
        candidates,
        pairs,
    )
    assert torch.isfinite(scores).all()
    assert scores[1] == 0


def test_device_assert_is_always_fatal_but_oom_is_recoverable():
    assert is_fatal_device_error(
        RuntimeError("CUDA error: device-side assert triggered")
    )
    assert not is_fatal_device_error(torch.cuda.OutOfMemoryError("oom"))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cuda_filtered_record_matches_forward_and_backward_without_assert():
    device = torch.device("cuda")
    source = torch.randn(2, 1, 2, 1, device=device, requires_grad=True)
    indices = (
        torch.tensor([0, 0, 1, 0], device=device),
        torch.tensor([0, 0, 0, 0], device=device),
        torch.tensor([0, 1, 0, 99], device=device),
        torch.tensor([0, 0, 0, 0], device=device),
    )
    instance_mask = torch.tensor(
        [[[True, False]], [[True, True]]], device=device
    )
    safe = filter_match_indices(
        indices, source.shape, instance_mask=instance_mask
    )
    loss = source[safe].sum()
    loss.backward()
    torch.cuda.synchronize()
    assert torch.isfinite(loss)
