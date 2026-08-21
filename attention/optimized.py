"""Inference-only optimized DeBERTa-v2/v3 disentangled attention.

The implementation preserves the eager attention equation while caching both
the full projected relative embeddings and their sequence-length-specific
pruned views.  ``forward_prepared`` contains no cache lookup or mutation and is
the entry point intended for ``torch.compile(..., fullgraph=True)``.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import torch

from .original import (
    DebertaAttentionConfig,
    OriginalDisentangledSelfAttention,
    _prepare_attention_mask,
    build_relative_position,
    build_rpos,
    make_log_bucket_position,
)


class PreparedPositionPlan(NamedTuple):
    """All immutable positional tensors needed by one sequence length."""

    sequence_length: int
    active_slots: torch.Tensor
    c2p_local: torch.Tensor
    p2c_local: torch.Tensor
    pos_key: torch.Tensor | None
    pos_query: torch.Tensor | None


def _invalidate_attention_cache_after_load(module: torch.nn.Module, _incompatible_keys: Any) -> None:
    module.clear_inference_cache()


class InferenceDisentangledSelfAttention(OriginalDisentangledSelfAttention):
    """Cached, active-slot-pruned DeBERTa attention for inference only.

    Parameter names are identical to :class:`OriginalDisentangledSelfAttention`,
    so weights can be copied with ``load_state_dict(..., strict=True)``.

    Typical compiled usage is::

        attention.eval().prepare_for_inference(relative_embeddings)
        plan = attention.prepare_shape(128)
        compiled = torch.compile(
            lambda hidden, mask: attention.forward_prepared(hidden, mask, plan),
            fullgraph=True,
        )
    """

    def __init__(self, config: DebertaAttentionConfig | Any) -> None:
        super().__init__(config)
        self.register_buffer("_cached_pos_key", None, persistent=False)
        self.register_buffer("_cached_pos_query", None, persistent=False)
        self._position_plan_cache: dict[tuple[int, str], PreparedPositionPlan] = {}
        self.register_load_state_dict_post_hook(_invalidate_attention_cache_after_load)

    def clear_inference_cache(self) -> None:
        self._cached_pos_key = None
        self._cached_pos_query = None
        self._position_plan_cache.clear()

    def train(self, mode: bool = True) -> "InferenceDisentangledSelfAttention":
        if mode:
            self.clear_inference_cache()
        return super().train(mode)

    def _apply(self, fn: Any, recurse: bool = True) -> "InferenceDisentangledSelfAttention":
        result = super()._apply(fn, recurse=recurse)
        self.clear_inference_cache()
        return result

    @torch.no_grad()
    def prepare_for_inference(
        self,
        rel_embeddings: torch.Tensor,
    ) -> "InferenceDisentangledSelfAttention":
        """Project and cache the complete relative embedding table."""

        if self.training:
            raise RuntimeError("prepare_for_inference() requires module.eval()")

        # Re-preparation replaces the source tensors, so every shape-specific
        # view derived from the previous projections must also be discarded.
        self.clear_inference_cache()
        if not self.relative_attention:
            return self

        att_span = self.pos_ebd_size
        rel_embeddings = self.pos_dropout(rel_embeddings[: att_span * 2]).unsqueeze(0)

        if "c2p" in self.pos_att_type:
            projection = self.key_proj if self.share_att_key else self.pos_key_proj
            self._cached_pos_key = self.transpose_for_scores(
                projection(rel_embeddings),
                self.num_attention_heads,
            )

        if "p2c" in self.pos_att_type:
            projection = self.query_proj if self.share_att_key else self.pos_query_proj
            self._cached_pos_query = self.transpose_for_scores(
                projection(rel_embeddings),
                self.num_attention_heads,
            )
        return self

    def _plan_device(self) -> torch.device:
        if self._cached_pos_key is not None:
            return self._cached_pos_key.device
        if self._cached_pos_query is not None:
            return self._cached_pos_query.device
        return self.query_proj.weight.device

    def _materialize_plan(
        self,
        sequence_length: int,
        c2p_slots: torch.Tensor,
        p2c_slots: torch.Tensor,
        device: torch.device,
    ) -> PreparedPositionPlan:
        used_slots = []
        if self.relative_attention and "c2p" in self.pos_att_type:
            used_slots.append(c2p_slots.reshape(-1))
        if self.relative_attention and "p2c" in self.pos_att_type:
            used_slots.append(p2c_slots.reshape(-1))

        if used_slots:
            active_slots = torch.unique(torch.cat(used_slots), sorted=True)
            c2p_local = torch.searchsorted(active_slots, c2p_slots)
            p2c_local = torch.searchsorted(active_slots, p2c_slots.contiguous())
        else:
            active_slots = torch.empty(0, dtype=torch.long, device=c2p_slots.device)
            c2p_local = c2p_slots
            p2c_local = p2c_slots

        active_slots = active_slots.to(device=device)
        c2p_local = c2p_local.to(device=device)
        p2c_local = p2c_local.to(device=device)

        pos_key = None
        if self.relative_attention and "c2p" in self.pos_att_type:
            if self._cached_pos_key is None:
                raise RuntimeError("call prepare_for_inference() before prepare_shape()")
            if self._cached_pos_key.device != device:
                raise ValueError("prepared relative keys and shape plan must use the same device")
            pos_key = self._cached_pos_key.index_select(1, active_slots).contiguous()

        pos_query = None
        if self.relative_attention and "p2c" in self.pos_att_type:
            if self._cached_pos_query is None:
                raise RuntimeError("call prepare_for_inference() before prepare_shape()")
            if self._cached_pos_query.device != device:
                raise ValueError("prepared relative queries and shape plan must use the same device")
            pos_query = self._cached_pos_query.index_select(1, active_slots).contiguous()

        return PreparedPositionPlan(
            sequence_length=sequence_length,
            active_slots=active_slots,
            c2p_local=c2p_local,
            p2c_local=p2c_local,
            pos_key=pos_key,
            pos_query=pos_query,
        )

    @torch.no_grad()
    def prepare_shape(
        self,
        sequence_length: int,
        device: torch.device | str | None = None,
    ) -> PreparedPositionPlan:
        """Prepare and cache all pruned positional tensors for one length."""

        if sequence_length < 1:
            raise ValueError("sequence_length must be positive")
        if self.training:
            raise RuntimeError("prepare_shape() requires module.eval()")

        resident_device = self._plan_device()
        resolved_device = torch.device(device) if device is not None else resident_device
        if resolved_device.type == resident_device.type and resolved_device.index is None:
            resolved_device = resident_device
        cache_key = sequence_length, str(resolved_device)
        cached = self._position_plan_cache.get(cache_key)
        if cached is not None:
            return cached

        positions = torch.arange(sequence_length, dtype=torch.long, device="cpu")
        relative_pos = positions[:, None] - positions[None, :]
        if self.position_buckets > 0:
            relative_pos = make_log_bucket_position(
                relative_pos,
                self.position_buckets,
                self.max_relative_positions,
            )

        att_span = self.pos_ebd_size
        relative_pos = relative_pos.unsqueeze(0).unsqueeze(0)
        c2p_slots = torch.clamp(relative_pos + att_span, 0, att_span * 2 - 1)
        p2c_slots = torch.clamp(-relative_pos + att_span, 0, att_span * 2 - 1).transpose(
            -1, -2
        )
        plan = self._materialize_plan(
            sequence_length,
            c2p_slots,
            p2c_slots,
            resolved_device,
        )
        self._position_plan_cache[cache_key] = plan
        return plan

    def _dynamic_position_plan(
        self,
        query_layer: torch.Tensor,
        key_layer: torch.Tensor,
        relative_pos: torch.Tensor,
    ) -> PreparedPositionPlan:
        if relative_pos.dim() == 2:
            relative_pos = relative_pos.unsqueeze(0).unsqueeze(0)
        elif relative_pos.dim() == 3:
            relative_pos = relative_pos.unsqueeze(1)
        elif relative_pos.dim() != 4:
            raise ValueError(
                f"relative_pos must have 2, 3, or 4 dimensions, got {relative_pos.dim()}"
            )

        relative_pos = relative_pos.to(device=query_layer.device, dtype=torch.long)
        att_span = self.pos_ebd_size
        c2p_slots = torch.clamp(relative_pos + att_span, 0, att_span * 2 - 1)
        r_pos = build_rpos(
            query_layer,
            key_layer,
            relative_pos,
            self.max_relative_positions,
            self.position_buckets,
        )
        p2c_slots = torch.clamp(-r_pos + att_span, 0, att_span * 2 - 1).transpose(
            -1, -2
        )
        return self._materialize_plan(
            query_layer.size(-2),
            c2p_slots,
            p2c_slots,
            query_layer.device,
        )

    def _validate_inference_call(self) -> None:
        if self.training:
            raise RuntimeError("InferenceDisentangledSelfAttention requires module.eval()")
        if torch.is_grad_enabled():
            raise RuntimeError(
                "InferenceDisentangledSelfAttention requires torch.no_grad() or "
                "torch.inference_mode()"
            )

    def forward_prepared(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        plan: PreparedPositionPlan,
    ) -> tuple[torch.Tensor, None]:
        """Pure tensor forward for a plan created outside the compiled graph."""

        self._validate_inference_call()
        batch_size, sequence_length = hidden_states.shape[:2]
        if sequence_length != plan.sequence_length:
            raise ValueError(
                f"prepared length {plan.sequence_length} does not match input length "
                f"{sequence_length}"
            )

        query_layer = self.transpose_for_scores(
            self.query_proj(hidden_states), self.num_attention_heads
        ).view(
            batch_size,
            self.num_attention_heads,
            sequence_length,
            self.attention_head_size,
        )
        key_layer = self.transpose_for_scores(
            self.key_proj(hidden_states), self.num_attention_heads
        ).view(
            batch_size,
            self.num_attention_heads,
            sequence_length,
            self.attention_head_size,
        )
        value_layer = self.transpose_for_scores(
            self.value_proj(hidden_states), self.num_attention_heads
        ).view(
            batch_size,
            self.num_attention_heads,
            sequence_length,
            self.attention_head_size,
        )

        scale_factor = 1 + int("c2p" in self.pos_att_type) + int("p2c" in self.pos_att_type)
        scale = self._scale(scale_factor)
        attention_scores = torch.matmul(
            query_layer,
            key_layer.transpose(-1, -2) / scale,
        )

        if self.relative_attention and "c2p" in self.pos_att_type:
            if plan.pos_key is None:
                raise ValueError("prepared plan has no content-to-position keys")
            c2p_raw = torch.matmul(query_layer, plan.pos_key.transpose(-1, -2)) / scale
            c2p_index = plan.c2p_local.expand(
                batch_size,
                self.num_attention_heads,
                sequence_length,
                sequence_length,
            )
            attention_scores = attention_scores + torch.gather(
                c2p_raw,
                dim=-1,
                index=c2p_index,
            )

        if self.relative_attention and "p2c" in self.pos_att_type:
            if plan.pos_query is None:
                raise ValueError("prepared plan has no position-to-content queries")
            p2c_raw = torch.matmul(key_layer, plan.pos_query.transpose(-1, -2)) / scale
            p2c_index = plan.p2c_local.expand(
                batch_size,
                self.num_attention_heads,
                sequence_length,
                sequence_length,
            )
            attention_scores = attention_scores + torch.gather(
                p2c_raw.transpose(-1, -2),
                dim=-2,
                index=p2c_index,
            )

        mask = _prepare_attention_mask(attention_mask, sequence_length, sequence_length)
        attention_scores = attention_scores.masked_fill(
            ~mask,
            torch.finfo(query_layer.dtype).min,
        )
        attention_probs = torch.softmax(attention_scores, dim=-1)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = (
            context_layer.permute(0, 2, 1, 3)
            .contiguous()
            .view(batch_size, sequence_length, self.all_head_size)
        )
        return context_layer, None

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        output_attentions: bool = False,
        query_states: torch.Tensor | None = None,
        relative_pos: torch.Tensor | None = None,
        rel_embeddings: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, None]:
        """Convenience wrapper with lazy preparation outside the pure hot path."""

        self._validate_inference_call()
        if output_attentions:
            raise ValueError("output_attentions=True is not supported by the inference path")
        if query_states is not None:
            raise ValueError("the optimized inference path currently supports self-attention only")

        if self.relative_attention:
            needs_key = "c2p" in self.pos_att_type and self._cached_pos_key is None
            needs_query = "p2c" in self.pos_att_type and self._cached_pos_query is None
            if needs_key or needs_query:
                if rel_embeddings is None:
                    raise ValueError(
                        "rel_embeddings is required until prepare_for_inference() has populated the cache"
                    )
                self.prepare_for_inference(rel_embeddings)

        sequence_length = hidden_states.size(1)
        if relative_pos is None:
            plan = self.prepare_shape(sequence_length, hidden_states.device)
        else:
            # Custom relative positions remain supported by the convenience
            # path, but are intentionally outside the compile-oriented API.
            query_layer = hidden_states.view(
                hidden_states.size(0), 1, sequence_length, hidden_states.size(-1)
            )
            plan = self._dynamic_position_plan(query_layer, query_layer, relative_pos)
        return self.forward_prepared(hidden_states, attention_mask, plan)


__all__ = ["InferenceDisentangledSelfAttention", "PreparedPositionPlan"]
