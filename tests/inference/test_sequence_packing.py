"""Correctness and parity tests for inference sequence packing."""

from __future__ import annotations

import copy

import pytest
import torch
from transformers import BertConfig, BertModel, DebertaV2Config, DebertaV2Model

from gliner2.inference.encoder_adapters import get_encoder_packing_adapter
from gliner2.inference.packing import (
    PackingConfig,
    PackingOverflowError,
    encode_batch_with_packing,
    estimate_baseline_cost,
    estimate_packed_cost,
    pack_requests,
    plan_packing,
    supports_sequence_packing,
    unpack_hidden_states,
)


def _deberta(*, position_biased_input: bool = False, conv_kernel_size: int = 0):
    config = DebertaV2Config(
        vocab_size=127,
        hidden_size=24,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=48,
        max_position_embeddings=1024,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        relative_attention=True,
        position_biased_input=position_biased_input,
        conv_kernel_size=conv_kernel_size,
        type_vocab_size=0,
        pad_token_id=0,
    )
    return DebertaV2Model(config).eval()


def _padded(lengths, *, seed=13):
    generator = torch.Generator().manual_seed(seed)
    width = max(lengths, default=0)
    input_ids = torch.zeros((len(lengths), width), dtype=torch.long)
    attention_mask = torch.zeros_like(input_ids)
    for index, length in enumerate(lengths):
        input_ids[index, :length] = torch.randint(
            1, 126, (length,), generator=generator
        )
        attention_mask[index, :length] = 1
    return input_ids, attention_mask


def _direct_packed_states(model, input_ids, attention_mask, config):
    packed = pack_requests(input_ids, attention_mask, config)
    adapter = get_encoder_packing_adapter(model)
    packed_hidden = adapter.encode(
        packed.input_ids,
        packed.token_attention_mask,
        packed.pair_attention_mask,
        packed.position_ids,
    )
    return unpack_hidden_states(
        packed_hidden, packed, expected_input_ids=input_ids
    ), packed


def test_packing_is_disabled_by_default():
    assert PackingConfig().enabled is False


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_packed_length": 0},
        {"max_segments_per_stream": 0},
        {"strategy": "random"},
        {"attention_backend": "flash"},
        {"length_buckets": (128, 64)},
        {"overflow": "truncate"},
        {"min_fill_ratio": 1.1},
    ],
)
def test_invalid_config_is_rejected(kwargs):
    with pytest.raises(ValueError):
        PackingConfig(**kwargs)


def test_best_fit_decreasing_is_stable_and_restorable():
    config = PackingConfig(max_packed_length=10, length_buckets=(10,))
    first = plan_packing([6, 6, 4, 4], config)
    second = plan_packing([6, 6, 4, 4], config)
    assert first == second
    segments, stream_lengths = first
    assert stream_lengths == [10, 10]
    assert segments == [
        # Stable request-order ties and stable stream-index ties.
        type(segments[0])(0, 0, 0, 6),
        type(segments[0])(2, 0, 6, 4),
        type(segments[0])(1, 1, 0, 6),
        type(segments[0])(3, 1, 6, 4),
    ]


def test_max_segments_per_stream_is_respected():
    config = PackingConfig(
        max_packed_length=16,
        max_segments_per_stream=2,
        length_buckets=(16,),
    )
    segments, stream_lengths = plan_packing([2, 2, 2, 2, 2], config)
    assert stream_lengths == [4, 4, 2]
    counts = [sum(segment.stream_index == i for segment in segments)
              for i in range(len(stream_lengths))]
    assert counts == [2, 2, 1]


def test_block_mask_positions_and_inverse_map():
    input_ids, attention_mask = _padded([7, 3, 2])
    config = PackingConfig(max_packed_length=16, length_buckets=(16,))
    packed = pack_requests(input_ids, attention_mask, config, pad_token_id=0)

    assert packed.input_ids.shape == (1, 16)
    assert packed.pair_attention_mask.dtype == torch.bool
    active = packed.token_attention_mask
    diagonal = packed.pair_attention_mask.diagonal(dim1=-2, dim2=-1)
    assert torch.equal(diagonal, active)
    assert not packed.pair_attention_mask[:, ~active[0], :].any()
    assert not packed.pair_attention_mask[:, :, ~active[0]].any()

    for segment in packed.segments:
        region = slice(segment.offset, segment.offset + segment.length)
        assert torch.equal(
            packed.position_ids[segment.stream_index, region],
            torch.arange(segment.length),
        )
        own = packed.pair_attention_mask[segment.stream_index, region, region]
        assert own.all()
        other = active[segment.stream_index] & (
            packed.segment_ids[segment.stream_index] != segment.request_index
        )
        assert not packed.pair_attention_mask[
            segment.stream_index, region, other
        ].any()

    hidden = torch.arange(16 * 3, dtype=torch.float32).reshape(1, 16, 3)
    unpacked = unpack_hidden_states(
        hidden, packed, expected_input_ids=input_ids
    )
    assert [unpacked[i].shape[0] for i in range(3)] == [7, 3, 2]


def test_empty_input_and_exact_limit_are_supported():
    config = PackingConfig(max_packed_length=8, length_buckets=(8,))
    empty = pack_requests(
        torch.empty((0, 0), dtype=torch.long),
        torch.empty((0, 0), dtype=torch.long),
        config,
    )
    assert empty.input_ids.shape == (0, 0)
    assert empty.pair_attention_mask.shape == (0, 0, 0)

    input_ids, attention_mask = _padded([8])
    exact = pack_requests(input_ids, attention_mask, config)
    assert exact.input_ids.shape == (1, 8)
    assert torch.equal(exact.input_ids[0], input_ids[0])


def test_one_token_over_limit_is_never_truncated():
    input_ids, attention_mask = _padded([513])
    config = PackingConfig(max_packed_length=512)
    with pytest.raises(PackingOverflowError, match="never truncates"):
        pack_requests(input_ids, attention_mask, config)


def test_non_prefix_attention_mask_is_rejected():
    with pytest.raises(ValueError, match="prefix-contiguous"):
        pack_requests(
            torch.tensor([[1, 2, 3]]),
            torch.tensor([[1, 0, 1]]),
            PackingConfig(max_packed_length=8, length_buckets=(8,)),
        )


def test_capability_check_rejects_convolution_and_other_families():
    safe = _deberta().config
    unsafe = copy.deepcopy(safe)
    unsafe.conv_kernel_size = 3
    assert supports_sequence_packing(safe)
    assert not supports_sequence_packing(unsafe)
    assert not supports_sequence_packing(BertConfig())


@pytest.mark.parametrize("position_biased_input", [False, True])
def test_real_deberta_encoder_states_match_individual_sequences(position_biased_input):
    torch.manual_seed(11)
    model = _deberta(position_biased_input=position_biased_input)
    input_ids, attention_mask = _padded([31, 15, 7, 2])
    config = PackingConfig(
        enabled=True,
        max_packed_length=64,
        length_buckets=(64,),
        min_fill_ratio=0.0,
    )
    unpacked, _ = _direct_packed_states(model, input_ids, attention_mask, config)

    for index, length in enumerate([31, 15, 7, 2]):
        reference = model(
            input_ids=input_ids[index:index + 1, :length],
            attention_mask=attention_mask[index:index + 1, :length],
            return_dict=True,
        ).last_hidden_state[0]
        torch.testing.assert_close(
            unpacked[index], reference, atol=1e-6, rtol=1e-5
        )


def test_reversing_request_order_does_not_change_unpacked_states():
    torch.manual_seed(17)
    model = _deberta()
    input_ids, attention_mask = _padded([31, 15, 7, 2])
    config = PackingConfig(
        max_packed_length=64, length_buckets=(64,), min_fill_ratio=0.0
    )
    forward, _ = _direct_packed_states(model, input_ids, attention_mask, config)
    order = torch.tensor([3, 2, 1, 0])
    reverse, _ = _direct_packed_states(
        model, input_ids[order], attention_mask[order], config
    )
    for forward_index, reverse_index in enumerate([3, 2, 1, 0]):
        torch.testing.assert_close(
            forward[forward_index], reverse[reverse_index], atol=1e-6, rtol=1e-5
        )


def test_cross_document_leakage_is_absent():
    torch.manual_seed(19)
    model = _deberta()
    input_ids, attention_mask = _padded([31, 15])
    config = PackingConfig(
        max_packed_length=64, length_buckets=(64,), min_fill_ratio=0.0
    )
    before, _ = _direct_packed_states(model, input_ids, attention_mask, config)

    changed = input_ids.clone()
    changed[0, :31] = (changed[0, :31] % 125) + 1
    after, _ = _direct_packed_states(model, changed, attention_mask, config)
    torch.testing.assert_close(before[1], after[1], atol=1e-6, rtol=1e-5)

    changed = input_ids.clone()
    changed[1, :15] = (changed[1, :15] % 125) + 1
    after, _ = _direct_packed_states(model, changed, attention_mask, config)
    torch.testing.assert_close(before[0], after[0], atol=1e-6, rtol=1e-5)


def test_cost_formulas_match_documented_model():
    assert estimate_baseline_cost([10, 5, 2], alpha=2, beta=3) == 690
    assert estimate_packed_cost([12, 5], token_count=17, alpha=2, beta=3) == 389


def test_cost_gate_activates_for_skew_and_preserves_parity():
    torch.manual_seed(23)
    model = _deberta()
    lengths = [50, 5, 5, 5, 5, 5, 5]
    input_ids, attention_mask = _padded(lengths)
    reference = model(
        input_ids=input_ids, attention_mask=attention_mask, return_dict=True
    ).last_hidden_state
    outputs, stats = encode_batch_with_packing(
        model,
        input_ids,
        attention_mask,
        PackingConfig(
            enabled=True,
            max_packed_length=64,
            length_buckets=(64,),
            min_fill_ratio=0.0,
        ),
    )
    assert stats.activated
    assert stats.reason == "packed"
    assert stats.packed_cost < 0.9 * stats.baseline_cost
    for index, length in enumerate(lengths):
        torch.testing.assert_close(
            outputs[index], reference[index, :length], atol=1e-6, rtol=1e-5
        )


def test_uniform_batch_stays_on_normal_path():
    model = _deberta()
    input_ids, attention_mask = _padded([20] * 8)
    outputs, stats = encode_batch_with_packing(
        model,
        input_ids,
        attention_mask,
        PackingConfig(
            enabled=True,
            max_packed_length=64,
            length_buckets=(64,),
            min_fill_ratio=0.0,
        ),
    )
    assert not stats.activated
    assert stats.reason == "estimated_cost_not_lower"
    assert all(output.shape[0] == 20 for output in outputs)


def test_overflow_uses_normal_encoder_path_without_truncation():
    torch.manual_seed(29)
    model = _deberta()
    lengths = [65, 50, 5, 5, 5, 5, 5, 5]
    input_ids, attention_mask = _padded(lengths)
    reference = model(
        input_ids=input_ids, attention_mask=attention_mask, return_dict=True
    ).last_hidden_state
    outputs, stats = encode_batch_with_packing(
        model,
        input_ids,
        attention_mask,
        PackingConfig(
            enabled=True,
            max_packed_length=64,
            length_buckets=(64,),
            overflow="fallback",
            min_fill_ratio=0.0,
        ),
    )
    assert stats.activated
    assert stats.fallback_count == 1
    assert outputs[0].shape[0] == 65
    for index, length in enumerate(lengths):
        torch.testing.assert_close(
            outputs[index], reference[index, :length], atol=1e-6, rtol=1e-5
        )


def test_overflow_error_policy_is_explicit():
    model = _deberta()
    input_ids, attention_mask = _padded([65, 5])
    with pytest.raises(PackingOverflowError, match="0:65"):
        encode_batch_with_packing(
            model,
            input_ids,
            attention_mask,
            PackingConfig(
                enabled=True,
                max_packed_length=64,
                length_buckets=(64,),
                overflow="error",
                min_fill_ratio=0.0,
            ),
        )


def test_unsupported_encoder_falls_back_to_public_forward():
    config = BertConfig(
        vocab_size=127,
        hidden_size=24,
        num_hidden_layers=1,
        num_attention_heads=4,
        intermediate_size=48,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
    )
    model = BertModel(config).eval()
    input_ids, attention_mask = _padded([20, 5, 5])
    reference = model(
        input_ids=input_ids, attention_mask=attention_mask, return_dict=True
    ).last_hidden_state
    outputs, stats = encode_batch_with_packing(
        model,
        input_ids,
        attention_mask,
        PackingConfig(enabled=True, max_packed_length=64, length_buckets=(64,)),
    )
    assert not stats.activated
    assert stats.reason == "unsupported_encoder_config"
    for index, length in enumerate([20, 5, 5]):
        torch.testing.assert_close(outputs[index], reference[index, :length])


def test_public_api_unsupported_encoder_fallback_matches(tiny_span_model):
    texts = ["apple released iphone", "google acquired company"]
    reference = tiny_span_model.batch_extract_entities(
        texts, ["company", "product"], batch_size=2,
        include_confidence=True, include_spans=True,
    )
    fallback = tiny_span_model.batch_extract_entities(
        texts, ["company", "product"], batch_size=2,
        include_confidence=True, include_spans=True,
        packing_config=PackingConfig(enabled=True),
    )
    assert fallback == reference
    assert not tiny_span_model._last_packing_stats.activated
    assert tiny_span_model._last_packing_stats.reason == "unsupported_encoder_config"


@pytest.mark.compile
@pytest.mark.parametrize("mode", ["default", "max-autotune-no-cudagraphs"])
def test_compiled_deberta_packed_graph_preserves_parity(mode):
    if not hasattr(torch, "compile"):
        pytest.skip("torch.compile is unavailable")
    torch.manual_seed(37)
    eager = _deberta()
    input_ids, attention_mask = _padded([50, 5, 5, 5, 5, 5, 5])
    reference = eager(
        input_ids=input_ids, attention_mask=attention_mask, return_dict=True
    ).last_hidden_state
    compiled = torch.compile(eager, dynamic=True, mode=mode)
    outputs, stats = encode_batch_with_packing(
        compiled,
        input_ids,
        attention_mask,
        PackingConfig(
            enabled=True,
            max_packed_length=64,
            length_buckets=(64,),
            min_fill_ratio=0.0,
        ),
    )
    assert stats.activated
    for index, length in enumerate([50, 5, 5, 5, 5, 5, 5]):
        torch.testing.assert_close(
            outputs[index], reference[index, :length], atol=1e-5, rtol=1e-4
        )


@pytest.mark.parametrize(
    "length",
    [1, 2, 7, 15, 31, 63, 64, 127, 128, 129, 255, 256, 257, 383, 384, 511, 512],
)
def test_boundary_lengths_are_planned_without_slicing(length):
    config = PackingConfig(max_packed_length=512)
    segments, stream_lengths = plan_packing([length], config)
    assert segments[0].length == length
    assert stream_lengths == [length]


def test_length_513_explicitly_overflows():
    with pytest.raises(PackingOverflowError):
        plan_packing([513], PackingConfig(max_packed_length=512))


def test_public_span_runtime_raw_heads_and_formatted_outputs_match():
    pytest.importorskip("peft")
    from gliner2 import ExtractorConfig, GLiNER2
    from gliner2.training.trainer import ExtractorCollator
    from tests.fixtures.tiny_tokenizer import build_tiny_tokenizer

    tokenizer = build_tiny_tokenizer()
    encoder_config = DebertaV2Config(
        vocab_size=len(tokenizer),
        hidden_size=24,
        num_hidden_layers=1,
        num_attention_heads=4,
        intermediate_size=48,
        max_position_embeddings=256,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        relative_attention=True,
        position_biased_input=False,
        conv_kernel_size=0,
        type_vocab_size=0,
        pad_token_id=tokenizer.pad_token_id,
    )
    model_config = ExtractorConfig(
        model_name="offline-tiny-deberta-v3",
        max_width=4,
        counting_layer="count_lstm",
        token_pooling="first",
    )
    torch.manual_seed(31)
    model = GLiNER2(
        model_config, encoder_config=encoder_config, tokenizer=tokenizer
    ).eval()

    schema = model.create_schema()
    schema.entities(["company", "product"])
    schema.classification("sentiment", ["positive", "negative", "neutral"])
    schema.relations(["acquired"])
    schema.structure("product_launch").field("company").field("product")
    texts = [
        "apple",
        "google",
        "microsoft",
        "amazon",
        "apple released iphone",
        "google acquired company",
        "the product was positive",
        ("apple released iphone 15 in location and google acquired company " * 8).strip(),
    ]
    packing_config = PackingConfig(
        enabled=True,
        max_packed_length=256,
        length_buckets=(128, 256),
        min_fill_ratio=0.0,
    )

    reference_results = model.batch_extract(
        texts,
        schema,
        batch_size=len(texts),
        include_confidence=True,
        include_spans=True,
    )
    packed_results = model.batch_extract(
        texts,
        schema,
        batch_size=len(texts),
        include_confidence=True,
        include_spans=True,
        packing_config=packing_config,
    )
    assert model._last_packing_stats.activated
    assert packed_results == reference_results

    schema_dicts, _ = model._build_schema_dicts_and_metadata([schema] * len(texts))
    batch = ExtractorCollator(
        model.processor, is_training=False, architecture="span"
    )(list(zip(texts, schema_dicts)))
    baseline_hidden = model.encoder(
        input_ids=batch.input_ids,
        attention_mask=batch.attention_mask,
        return_dict=True,
    ).last_hidden_state
    baseline_tokens, baseline_schemas = model.processor.extract_embeddings_from_batch(
        baseline_hidden, batch.input_ids, batch
    )
    exact_hidden, stats = encode_batch_with_packing(
        model.encoder,
        batch.input_ids,
        batch.attention_mask,
        packing_config,
        pad_token_id=tokenizer.pad_token_id,
    )
    assert stats.activated
    packed_tokens, packed_schemas = (
        model.processor.extract_embeddings_from_unpadded_batch(
            exact_hidden, batch.input_ids, batch
        )
    )

    for expected, actual in zip(baseline_tokens, packed_tokens):
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
    for expected_sample, actual_sample in zip(baseline_schemas, packed_schemas):
        for expected_schema, actual_schema in zip(expected_sample, actual_sample):
            assert len(expected_schema) == len(actual_schema)
            for expected, actual in zip(expected_schema, actual_schema):
                torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)

    baseline_logits, baseline_counts = model._compute_direct_span_logits_batched(
        batch, baseline_tokens, baseline_schemas
    )
    packed_logits, packed_counts = model._compute_direct_span_logits_batched(
        batch, packed_tokens, packed_schemas
    )
    assert packed_counts == baseline_counts
    for expected_sample, actual_sample in zip(baseline_logits, packed_logits):
        for expected, actual in zip(expected_sample, actual_sample):
            if expected is None:
                assert actual is None
            else:
                torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)

    # Classification logits are intentionally checked before softmax as well.
    for sample_index, task_types in enumerate(batch.task_types):
        for task_index, task_type in enumerate(task_types):
            if task_type != "classifications":
                continue
            expected = model.classifier(
                torch.stack(baseline_schemas[sample_index][task_index])[1:]
            ).squeeze(-1)
            actual = model.classifier(
                torch.stack(packed_schemas[sample_index][task_index])[1:]
            ).squeeze(-1)
            torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)

    entity_schema = model.create_schema().entities(["company", "person"])
    classification_schema = model.create_schema().classification(
        "sentiment", ["positive", "negative", "neutral"]
    )
    relation_schema = model.create_schema().relations(["acquired", "works_for"])
    structure_schema = model.create_schema()
    structure_schema.structure("launch").field("company").field("product")
    schemas = [
        entity_schema,
        classification_schema,
        relation_schema,
        structure_schema,
        schema,
        entity_schema,
        classification_schema,
        schema,
    ]
    different_schema_reference = model.batch_extract(
        texts, schemas, batch_size=len(texts), include_confidence=True,
        include_spans=True,
    )
    different_schema_packed = model.batch_extract(
        texts, schemas, batch_size=len(texts), include_confidence=True,
        include_spans=True, packing_config=packing_config,
    )
    assert model._last_packing_stats.activated
    assert different_schema_packed == different_schema_reference
