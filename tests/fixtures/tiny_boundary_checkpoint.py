"""Build a tiny, deterministic boundary-architecture model entirely offline."""

from __future__ import annotations

from pathlib import Path

import torch

from .tiny_encoder import build_tiny_encoder_config
from .tiny_tokenizer import build_tiny_tokenizer


TINY_BOUNDARY_HEAD = dict(
    boundary_dim=24,
    pair_dim=24,
    start_top_k=12,
    end_top_k=12,
    ends_per_start=6,
    starts_per_end=6,
    candidate_budget=48,
    training_candidate_budget=64,
    max_gold_per_query=16,
    end_block_size=16,
    dropout=0.0,
)


def build_tiny_boundary_model(seed: int = 17):
    """Create a deterministic tiny ``BoundaryExtractor`` on CPU in eval mode."""
    from gliner2.inference.engine import BoundaryExtractor
    from gliner2 import ExtractorConfig

    tokenizer = build_tiny_tokenizer()
    encoder_config = build_tiny_encoder_config(vocab_size=len(tokenizer))

    config = ExtractorConfig(
        model_name="tiny-bert-fixture",
        architecture="boundary",
        boundary_head=dict(TINY_BOUNDARY_HEAD),
        token_pooling="first",
    )

    torch.manual_seed(seed)
    model = BoundaryExtractor(config, encoder_config=encoder_config, tokenizer=tokenizer)
    model.eval()
    return model


def save_tiny_boundary_checkpoint(directory, seed: int = 17) -> Path:
    """Build and save a tiny boundary checkpoint, returning its path."""
    path = Path(directory)
    build_tiny_boundary_model(seed=seed).save_pretrained(str(path))
    return path
