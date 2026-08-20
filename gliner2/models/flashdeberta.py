"""Narrow compatibility adapter for the validated FlashDeBERTa release."""

from __future__ import annotations

import types
from typing import Optional

import torch
from transformers.modeling_outputs import BaseModelOutput


def _inference_guard(module, args, kwargs) -> None:
    if module.training:
        raise RuntimeError(
            "FlashDeBERTa is inference-only in GLiNER2; call model.eval() "
            "and run under torch.inference_mode() or torch.no_grad()"
        )
    if torch.is_grad_enabled():
        raise RuntimeError(
            "FlashDeBERTa does not support gradient-enabled GLiNER2 forward calls; "
            "use torch.inference_mode() or torch.no_grad()"
        )
    requested = kwargs.get("output_attentions")
    if requested is None:
        requested = getattr(module.config, "output_attentions", False)
    if requested:
        raise NotImplementedError(
            "FlashDeBERTa does not reproduce output_attentions=True; use the "
            "Transformers encoder backend when attention tensors are required"
        )


def _flashdeberta_007_forward(
    self,
    input_ids: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    token_type_ids: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.Tensor] = None,
    inputs_embeds: Optional[torch.Tensor] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
):
    """FlashDeBERTa 0.0.7 forward without forced hidden-state retention.

    Upstream 0.0.7 always asks its encoder to retain every layer even when the
    caller only wants the final state.  GLiNER2 never uses ``z_steps``, so this
    version-gated adapter preserves requested outputs and avoids that material
    inference-memory cost.
    """

    output_attentions = (
        output_attentions
        if output_attentions is not None
        else self.config.output_attentions
    )
    if output_attentions:
        raise NotImplementedError(
            "FlashDeBERTa does not reproduce output_attentions=True; use the "
            "Transformers encoder backend when attention tensors are required"
        )
    output_hidden_states = (
        output_hidden_states
        if output_hidden_states is not None
        else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if getattr(self, "z_steps", 0) > 1:
        raise NotImplementedError(
            "FlashDeBERTa z_steps > 1 is outside GLiNER2's validated inference path"
        )
    if input_ids is not None and inputs_embeds is not None:
        raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
    if input_ids is not None:
        self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
        input_shape = input_ids.size()
        device = input_ids.device
    elif inputs_embeds is not None:
        input_shape = inputs_embeds.size()[:-1]
        device = inputs_embeds.device
    else:
        raise ValueError("You have to specify either input_ids or inputs_embeds")

    if attention_mask is None:
        attention_mask = torch.ones(input_shape, device=device)
    if token_type_ids is None:
        token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

    embedding_output = self.embeddings(
        input_ids=input_ids,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        mask=attention_mask,
        inputs_embeds=inputs_embeds,
    )
    encoder_outputs = self.encoder(
        embedding_output,
        attention_mask,
        output_hidden_states=output_hidden_states,
        output_attentions=False,
        return_dict=return_dict,
    )
    sequence_output = encoder_outputs[0]

    if not return_dict:
        return (sequence_output,) + encoder_outputs[1:]
    return BaseModelOutput(
        last_hidden_state=sequence_output,
        hidden_states=(
            encoder_outputs.hidden_states if output_hidden_states else None
        ),
        attentions=None,
    )


def build_flashdeberta_encoder(config, version: str):
    """Instantiate and guard the exact validated FlashDeBERTa model."""

    if version != "0.0.7":
        raise RuntimeError(
            f"No GLiNER2 FlashDeBERTa adapter is registered for version {version}"
        )
    from flashdeberta import FlashDebertaV2Model

    encoder = FlashDebertaV2Model(config)
    encoder.forward = types.MethodType(_flashdeberta_007_forward, encoder)
    encoder.register_forward_pre_hook(_inference_guard, with_kwargs=True)
    return encoder


__all__ = ["build_flashdeberta_encoder"]
