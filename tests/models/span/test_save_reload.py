"""Span save/reload must preserve state-dict keys and outputs to 1e-6."""

import torch

from tests.fixtures.tiny_span_checkpoint import build_tiny_span_model
from tests.fixtures.span_golden import compute_span_signature, _formatted_close


def test_old_span_save_reload_matches(tmp_path):
    from gliner2 import GLiNER2

    model = build_tiny_span_model()
    before = compute_span_signature(model)

    model.save_pretrained(str(tmp_path))
    reloaded = GLiNER2.from_pretrained(str(tmp_path))
    after = compute_span_signature(reloaded)

    assert before["state_keys"] == after["state_keys"]
    torch.testing.assert_close(
        torch.tensor(list(before["losses"].values()), dtype=torch.float64),
        torch.tensor(list(after["losses"].values()), dtype=torch.float64),
        rtol=1e-6,
        atol=1e-6,
    )
    assert _formatted_close(before["formatted"], after["formatted"], 1e-6)


def test_saved_state_dict_keys_are_unchanged(tmp_path):
    from gliner2 import GLiNER2

    model = build_tiny_span_model()
    keys_before = sorted(model.state_dict().keys())
    model.save_pretrained(str(tmp_path))
    reloaded = GLiNER2.from_pretrained(str(tmp_path))
    keys_after = sorted(reloaded.state_dict().keys())
    assert keys_before == keys_after
