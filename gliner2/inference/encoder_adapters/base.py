"""Encoder adapters used by inference-only sequence packing.

Adapters are deliberately narrow: they only bridge an encoder's public model
contract to a block-diagonal attention mask.  Unsupported encoders fail closed
and are handled by the ordinary batching fallback in :mod:`gliner2.inference.packing`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch


class SequencePackingUnsupportedError(RuntimeError):
    """Raised when an encoder cannot safely isolate packed sequences."""


class EncoderPackingAdapter(ABC):
    """Minimal adapter contract for a parity-preserving packed encoder call."""

    @abstractmethod
    def encode(
        self,
        input_ids: torch.Tensor,
        token_attention_mask: torch.Tensor,
        pair_attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Return packed ``last_hidden_state`` with shape ``[P, S, H]``."""
