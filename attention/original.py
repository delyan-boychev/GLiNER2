"""Self-contained reference implementation of DeBERTa-v2/v3 attention.

This module mirrors the eager disentangled self-attention computation used by
Hugging Face Transformers without importing Transformers.  It intentionally
keeps the original parameter names so a state dict can be copied directly
between this module, the optimized implementation, and the corresponding
Transformers attention module.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn


@dataclass(frozen=True)
class DebertaAttentionConfig:
    """Minimal configuration needed by disentangled self-attention."""

    hidden_size: int = 768
    num_attention_heads: int = 12
    attention_probs_dropout_prob: float = 0.1
    hidden_dropout_prob: float = 0.1
    relative_attention: bool = True
    max_relative_positions: int = -1
    max_position_embeddings: int = 512
    position_buckets: int = 256
    share_att_key: bool = True
    pos_att_type: str = "p2c|c2p"


def make_log_bucket_position(
    relative_pos: torch.Tensor,
    bucket_size: int,
    max_position: int,
) -> torch.Tensor:
    """Map signed relative distances to DeBERTa's logarithmic buckets."""

    sign = torch.sign(relative_pos)
    mid = bucket_size // 2
    abs_pos = torch.where(
        (relative_pos < mid) & (relative_pos > -mid),
        torch.full_like(relative_pos, mid - 1),
        torch.abs(relative_pos),
    )
    log_pos = (
        torch.ceil(
            torch.log(abs_pos.float() / mid)
            / math.log((max_position - 1) / mid)
            * (mid - 1)
        )
        + mid
    )
    bucket_pos = torch.where(
        torch.abs(relative_pos) < mid,
        relative_pos,
        log_pos.to(relative_pos.dtype) * sign,
    )
    return bucket_pos.to(torch.long)


def build_relative_position(
    query_layer: torch.Tensor,
    key_layer: torch.Tensor,
    bucket_size: int = -1,
    max_position: int = -1,
) -> torch.Tensor:
    """Build ``q_index - key_index`` relative positions."""

    query_size = query_layer.size(-2)
    key_size = key_layer.size(-2)
    query_ids = torch.arange(query_size, dtype=torch.long, device=query_layer.device)
    key_ids = torch.arange(key_size, dtype=torch.long, device=key_layer.device)
    relative_pos = query_ids[:, None] - key_ids[None, :]
    if bucket_size > 0 and max_position > 0:
        relative_pos = make_log_bucket_position(relative_pos, bucket_size, max_position)
    return relative_pos.unsqueeze(0)


def build_rpos(
    query_layer: torch.Tensor,
    key_layer: torch.Tensor,
    relative_pos: torch.Tensor,
    max_relative_position: int,
    bucket_size: int,
) -> torch.Tensor:
    """Match the p2c relative-position construction used by Transformers."""

    if query_layer.size(-2) != key_layer.size(-2):
        return build_relative_position(
            key_layer,
            key_layer,
            bucket_size=bucket_size,
            max_position=max_relative_position,
        )
    return relative_pos


def _parse_pos_att_type(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return tuple(part.strip().lower() for part in value.split("|") if part.strip())
    return tuple(str(part).strip().lower() for part in value if str(part).strip())


def _prepare_attention_mask(
    attention_mask: torch.Tensor,
    query_length: int,
    key_length: int,
) -> torch.Tensor:
    """Accept a base 2D mask or the expanded mask expected by HF attention."""

    if attention_mask.dim() == 2:
        mask = attention_mask.bool()
        if query_length == key_length == mask.size(-1):
            return mask[:, None, :, None] & mask[:, None, None, :]
        return mask[:, None, None, :]
    if attention_mask.dim() == 3:
        return attention_mask[:, None].bool()
    if attention_mask.dim() == 4:
        return attention_mask.bool()
    raise ValueError(
        "attention_mask must have shape [B, L], [B, Lq, Lk], or [B, 1, Lq, Lk]"
    )


class OriginalDisentangledSelfAttention(nn.Module):
    """Reference DeBERTa-v2/v3 disentangled self-attention."""

    def __init__(self, config: DebertaAttentionConfig | Any) -> None:
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"hidden_size={config.hidden_size} must be divisible by "
                f"num_attention_heads={config.num_attention_heads}"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = config.hidden_size // config.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.relative_attention = getattr(config, "relative_attention", False)
        self.share_att_key = getattr(config, "share_att_key", False)
        self.pos_att_type = _parse_pos_att_type(getattr(config, "pos_att_type", None))
        self.position_buckets = getattr(config, "position_buckets", -1)

        self.max_relative_positions = getattr(config, "max_relative_positions", -1)
        if self.max_relative_positions < 1:
            self.max_relative_positions = config.max_position_embeddings

        self.pos_ebd_size = self.max_relative_positions
        if self.position_buckets > 0:
            self.pos_ebd_size = self.position_buckets

        self.query_proj = nn.Linear(config.hidden_size, self.all_head_size, bias=True)
        self.key_proj = nn.Linear(config.hidden_size, self.all_head_size, bias=True)
        self.value_proj = nn.Linear(config.hidden_size, self.all_head_size, bias=True)

        if self.relative_attention and not self.share_att_key:
            if "c2p" in self.pos_att_type:
                self.pos_key_proj = nn.Linear(config.hidden_size, self.all_head_size, bias=True)
            if "p2c" in self.pos_att_type:
                self.pos_query_proj = nn.Linear(config.hidden_size, self.all_head_size, bias=True)

        self.pos_dropout = nn.Dropout(getattr(config, "hidden_dropout_prob", 0.0))
        self.dropout = nn.Dropout(getattr(config, "attention_probs_dropout_prob", 0.0))

    def transpose_for_scores(self, tensor: torch.Tensor, attention_heads: int) -> torch.Tensor:
        new_shape = tensor.size()[:-1] + (attention_heads, -1)
        tensor = tensor.view(new_shape)
        return tensor.permute(0, 2, 1, 3).contiguous().view(
            -1, tensor.size(1), tensor.size(-1)
        )

    def _scale(self, scale_factor: int) -> float:
        return math.sqrt(self.attention_head_size * scale_factor)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        output_attentions: bool = False,
        query_states: torch.Tensor | None = None,
        relative_pos: torch.Tensor | None = None,
        rel_embeddings: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if query_states is None:
            query_states = hidden_states

        query_layer = self.transpose_for_scores(
            self.query_proj(query_states), self.num_attention_heads
        )
        key_layer = self.transpose_for_scores(
            self.key_proj(hidden_states), self.num_attention_heads
        )
        value_layer = self.transpose_for_scores(
            self.value_proj(hidden_states), self.num_attention_heads
        )

        scale_factor = 1 + int("c2p" in self.pos_att_type) + int("p2c" in self.pos_att_type)
        scale = self._scale(scale_factor)
        attention_scores = torch.bmm(
            query_layer,
            key_layer.transpose(-1, -2) / scale,
        )

        if self.relative_attention:
            if rel_embeddings is None:
                raise ValueError("rel_embeddings is required when relative_attention=True")
            rel_embeddings = self.pos_dropout(rel_embeddings)
            attention_scores = attention_scores + self.disentangled_attention_bias(
                query_layer,
                key_layer,
                relative_pos,
                rel_embeddings,
                scale_factor,
            )

        batch_size = hidden_states.size(0)
        query_length = query_layer.size(-2)
        key_length = key_layer.size(-2)
        attention_scores = attention_scores.view(
            batch_size,
            self.num_attention_heads,
            query_length,
            key_length,
        )

        mask = _prepare_attention_mask(attention_mask, query_length, key_length)
        attention_scores = attention_scores.masked_fill(
            ~mask,
            torch.finfo(query_layer.dtype).min,
        )
        attention_probs = torch.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.bmm(
            attention_probs.view(-1, query_length, key_length),
            value_layer,
        )
        context_layer = (
            context_layer.view(
                batch_size,
                self.num_attention_heads,
                query_length,
                self.attention_head_size,
            )
            .permute(0, 2, 1, 3)
            .contiguous()
            .view(batch_size, query_length, self.all_head_size)
        )

        return context_layer, attention_probs if output_attentions else None

    def disentangled_attention_bias(
        self,
        query_layer: torch.Tensor,
        key_layer: torch.Tensor,
        relative_pos: torch.Tensor | None,
        rel_embeddings: torch.Tensor,
        scale_factor: int,
    ) -> torch.Tensor:
        if relative_pos is None:
            relative_pos = build_relative_position(
                query_layer,
                key_layer,
                bucket_size=self.position_buckets,
                max_position=self.max_relative_positions,
            )
        if relative_pos.dim() == 2:
            relative_pos = relative_pos.unsqueeze(0).unsqueeze(0)
        elif relative_pos.dim() == 3:
            relative_pos = relative_pos.unsqueeze(1)
        elif relative_pos.dim() != 4:
            raise ValueError(
                f"relative_pos must have 2, 3, or 4 dimensions, got {relative_pos.dim()}"
            )

        att_span = self.pos_ebd_size
        relative_pos = relative_pos.to(device=query_layer.device, dtype=torch.long)
        rel_embeddings = rel_embeddings[: att_span * 2].unsqueeze(0)

        pos_key_layer = None
        pos_query_layer = None
        batch_size = query_layer.size(0) // self.num_attention_heads

        if self.share_att_key:
            if "c2p" in self.pos_att_type:
                pos_key_layer = self.transpose_for_scores(
                    self.key_proj(rel_embeddings), self.num_attention_heads
                ).repeat(batch_size, 1, 1)
            if "p2c" in self.pos_att_type:
                pos_query_layer = self.transpose_for_scores(
                    self.query_proj(rel_embeddings), self.num_attention_heads
                ).repeat(batch_size, 1, 1)
        else:
            if "c2p" in self.pos_att_type:
                pos_key_layer = self.transpose_for_scores(
                    self.pos_key_proj(rel_embeddings), self.num_attention_heads
                ).repeat(batch_size, 1, 1)
            if "p2c" in self.pos_att_type:
                pos_query_layer = self.transpose_for_scores(
                    self.pos_query_proj(rel_embeddings), self.num_attention_heads
                ).repeat(batch_size, 1, 1)

        score: torch.Tensor | int = 0
        scale = self._scale(scale_factor)

        if "c2p" in self.pos_att_type:
            assert pos_key_layer is not None
            c2p_att = torch.bmm(query_layer, pos_key_layer.transpose(-1, -2))
            c2p_pos = torch.clamp(relative_pos + att_span, 0, att_span * 2 - 1)
            c2p_index = c2p_pos.squeeze(0).expand(
                query_layer.size(0),
                query_layer.size(1),
                relative_pos.size(-1),
            )
            score = score + torch.gather(c2p_att, dim=-1, index=c2p_index) / scale

        if "p2c" in self.pos_att_type:
            assert pos_query_layer is not None
            r_pos = build_rpos(
                query_layer,
                key_layer,
                relative_pos,
                self.max_relative_positions,
                self.position_buckets,
            )
            p2c_pos = torch.clamp(-r_pos + att_span, 0, att_span * 2 - 1)
            p2c_att = torch.bmm(key_layer, pos_query_layer.transpose(-1, -2))
            p2c_index = p2c_pos.squeeze(0).expand(
                query_layer.size(0),
                key_layer.size(-2),
                key_layer.size(-2),
            )
            score = score + torch.gather(
                p2c_att,
                dim=-1,
                index=p2c_index,
            ).transpose(-1, -2) / scale

        if isinstance(score, int):
            return torch.zeros(
                query_layer.size(0),
                query_layer.size(-2),
                key_layer.size(-2),
                device=query_layer.device,
                dtype=query_layer.dtype,
            )
        return score


__all__ = [
    "DebertaAttentionConfig",
    "OriginalDisentangledSelfAttention",
    "build_relative_position",
    "build_rpos",
    "make_log_bucket_position",
]
