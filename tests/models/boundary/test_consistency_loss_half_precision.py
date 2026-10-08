"""`marginal_pair_consistency_loss` must produce finite gradients in half precision.

The clamp in this loss used a hard `1.0 - 1e-6`, which rounds to exactly 1.0 in both bf16
(eps 7.8e-3) and fp16 (eps 9.8e-4). A saturated probability then reaches 1.0 and
`log1p(-1.0)` is `-inf`.

The forward stays finite -- the `-inf` is summed and `1 - exp(-inf)` is 1 -- so a test that
only checks the loss value cannot see this. These tests assert on the GRADIENTS.
"""

import pytest
import torch

from gliner2.models.boundary.losses import marginal_pair_consistency_loss


def _inputs(dtype):
    """One saturating pair logit beside a masked-out neighbour: the minimal trigger."""
    b, q, n, c = 1, 1, 4, 2
    pair_logits = torch.tensor([[[30.0, 30.0]]], dtype=dtype, requires_grad=True)
    indices = torch.zeros(b, q, c, 2, dtype=torch.long)
    valid_mask = torch.tensor([[[True, False]]])
    start_logits = torch.zeros(b, q, n, dtype=dtype, requires_grad=True)
    end_logits = torch.zeros(b, q, n, dtype=dtype, requires_grad=True)
    boundary_keep = torch.ones(b, q, n, dtype=torch.bool)
    return pair_logits, indices, valid_mask, start_logits, end_logits, boundary_keep


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_saturated_probability_gives_finite_gradients(dtype):
    pair_logits, indices, valid_mask, start_logits, end_logits, keep = _inputs(dtype)

    loss = marginal_pair_consistency_loss(
        pair_logits, indices, valid_mask, start_logits, end_logits, keep
    )
    assert torch.isfinite(loss), f"forward already non-finite in {dtype}"

    loss.backward()
    assert torch.isfinite(pair_logits.grad).all(), (
        f"non-finite gradient in {dtype}: the clamp epsilon is not representable in this "
        f"dtype, so log1p(-p) saw p == 1.0 exactly"
    )


def test_clamp_epsilon_is_representable_in_every_supported_dtype():
    """`1.0 - eps` must stay below 1.0 after rounding, or the clamp does nothing."""
    for dtype in (torch.float32, torch.bfloat16, torch.float16):
        eps = max(1e-6, torch.finfo(dtype).eps)
        bound = torch.tensor(1.0 - eps, dtype=dtype)
        assert bound.item() < 1.0, (
            f"{dtype}: clamp bound rounds to {bound.item()}, so the clamp is a no-op"
        )
    assert max(1e-6, torch.finfo(torch.float32).eps) == 1e-6, "float32 must be unchanged"
