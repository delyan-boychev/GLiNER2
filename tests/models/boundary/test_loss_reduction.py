"""Loss reduction and imbalance-control regression tests."""

import pytest
import torch

from gliner2.configuration import BoundaryHeadSettings, validate_boundary_head
from gliner2.models.boundary.losses import (
    asymmetric_focal_loss,
    balanced_multilabel_bce,
    select_hard_negative_candidates,
)


def _two_query_loss(first_length: int, reduction: str) -> torch.Tensor:
    length = max(first_length, 2)
    logits = torch.zeros(1, 2, length)
    # Query 1 has a different mean loss from query 0 and only two valid slots.
    logits[0, 1, :2] = torch.tensor([4.0, -4.0])
    targets = torch.zeros_like(logits)
    targets[0, :, 0] = 1.0
    valid = torch.zeros_like(logits, dtype=torch.bool)
    valid[0, 0, :first_length] = True
    valid[0, 1, :2] = True
    return balanced_multilabel_bce(
        logits,
        targets,
        valid,
        query_mask=torch.ones(1, 2, dtype=torch.bool),
        reduction=reduction,
    )


def test_per_query_reduction_is_invariant_to_other_query_length():
    assert torch.allclose(_two_query_loss(4, "per_query"), _two_query_loss(64, "per_query"))
    assert not torch.allclose(_two_query_loss(4, "global"), _two_query_loss(64, "global"))


def test_sum_reduction_returns_masked_loss_sum():
    logits = torch.zeros(1, 1, 3)
    targets = torch.zeros_like(logits)
    valid = torch.tensor([[[True, False, True]]])

    loss = balanced_multilabel_bce(logits, targets, valid, reduction="sum")

    assert torch.allclose(loss, 2 * torch.log(torch.tensor(2.0)))


def test_focal_negative_weight_is_applied():
    logits = torch.zeros(1, 1, 4)
    targets = torch.tensor([[[1.0, 0.0, 0.0, 0.0]]])
    valid = torch.ones_like(targets, dtype=torch.bool)
    full = asymmetric_focal_loss(logits, targets, valid, negative_weight=1.0)
    down = asymmetric_focal_loss(logits, targets, valid, negative_weight=0.1)
    assert down < full


def test_absent_query_can_keep_every_negative():
    logits = torch.randn(1, 1, 9)
    labels = torch.zeros_like(logits)
    valid = torch.tensor([[[True, True, False, True, True, True, False, True, True]]])
    kept = select_hard_negative_candidates(
        logits,
        labels,
        valid,
        negatives_per_positive=3,
        minimum_negatives=2,
        keep_all_when_no_positive=True,
    )
    assert torch.equal(kept, valid)


@pytest.mark.parametrize(
    "bad",
    [
        {"loss_reduction": "sample"},
        {"boundary_focal_gamma_negative": -1},
        {"boundary_focal_clip": 1.0},
        {"minimum_hard_negatives": -1},
    ],
)
def test_new_loss_settings_are_validated(bad):
    with pytest.raises(ValueError):
        validate_boundary_head(bad)


def test_loss_settings_round_trip_defaults():
    values = validate_boundary_head({})
    defaults = BoundaryHeadSettings()
    assert values["loss_reduction"] == defaults.loss_reduction
    assert values["hard_negatives_per_positive"] == defaults.hard_negatives_per_positive
