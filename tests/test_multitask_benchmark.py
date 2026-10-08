"""Unit tests for the GLiNER2 multi-task benchmark harness."""

from __future__ import annotations

from pathlib import Path

import pytest

from benchmarks.multitask.adapters import (
    Example,
    LoadedSplit,
    _as_index,
    _as_records,
    _id_entity_relations,
    _ner_from_tokens,
    _offset_pair,
    _span_tokens,
    _token_pos,
    _translations_by_language,
)
from benchmarks.multitask.catalog import DATASETS, MODELS, DatasetSpec
from benchmarks.multitask.inference import (
    classification_schema,
    model_capabilities,
    parse_classification,
    parse_entities,
    parse_relations,
    supports_task,
)
from benchmarks.multitask.finetune import training_config
from benchmarks.multitask.io import shard_complete, shard_path, write_json_atomic
from benchmarks.multitask.run import _adaptive_batch_size, _run_batched
from benchmarks.multitask.metrics import (
    classification_scores,
    filter_by_threshold,
    gold_spans_set,
    gold_triples_set,
    map_mentions_to_clusters,
    select_threshold,
    span_scores,
    spans_from_predictions,
    triple_scores,
    triples_from_predictions,
)
from benchmarks.multitask.normalize import (
    canonical_label,
    decode_tag,
    display_label,
    find_span,
    tags_to_spans,
    token_span_to_char,
    tokens_to_text,
)
from benchmarks.multitask.sampling import cap_examples
from benchmarks.multitask.train_corpus import (
    to_training_example,
    training_specs,
    validate_train_split,
)


def test_display_and_canonical_labels():
    assert display_label("Sci/Tech") == "sci tech"
    assert display_label("card_arrival") == "card arrival"
    assert display_label("CelestialObjectRegion") == "celestial object region"
    assert display_label("DIRECT-REGULATOR") == "direct regulator"
    assert display_label("PER") == "person"
    assert display_label("GPE_LOC") == "geopolitical location"
    assert display_label("en") == "english"
    assert display_label("CPR:3") == "upregulator"
    assert canonical_label("Card_Arrival") == "card arrival"


def test_training_specs_filter_tasks_and_reject_unknown():
    specs = training_specs(["classification", "ner"])
    assert len(specs) == 30
    assert {spec.task for spec in specs} == {"classification", "ner"}
    assert [spec.id for spec in specs] == [
        spec.id
        for spec in DATASETS.values()
        if spec.task in {"classification", "ner"}
    ]
    with pytest.raises(ValueError, match="unknown training task"):
        training_specs(["generation"])


def test_tokens_to_text_offsets_roundtrip():
    text, offsets = tokens_to_text(["Tim", "Cook", "runs", "Apple"])
    assert text == "Tim Cook runs Apple"
    assert text[offsets[0][0]:offsets[0][1]] == "Tim"
    assert text[offsets[1][0]:offsets[1][1]] == "Cook"
    assert text[offsets[3][0]:offsets[3][1]] == "Apple"


def test_bio_to_spans():
    text, offsets = tokens_to_text(["Barack", "Obama", "visits", "Paris"])
    spans = tags_to_spans(["B-PER", "I-PER", "O", "B-LOC"], offsets)
    assert spans == [
        (0, 12, "person"),
        (20, 25, "location"),
    ]
    assert text[0:12] == "Barack Obama"
    assert text[20:25] == "Paris"


def test_flat_tags_and_type_map():
    _, offsets = tokens_to_text(["the", "pers", "org"])
    spans = tags_to_spans(
        ["O", "pers", "org"],
        offsets,
        style="flat",
        type_map={"pers": "person", "org": "organization"},
    )
    assert [span[2] for span in spans] == ["person", "organization"]


def _toy_ner_spec(**extra):
    payload = {"tokens": "tokens", "tags": "ner_tags", "style": "bio"}
    payload.update(extra)
    return DatasetSpec(
        id="toy_ner",
        task="ner",
        path="toy",
        loader="ner_bio",
        test_split="test",
        language="en",
        license="other",
        caveats="",
        extra=payload,
    )


def test_ner_drops_misc_and_keeps_person():
    example = _ner_from_tokens(
        _toy_ner_spec(),
        {
            "tokens": ["Barack", "Obama", "likes", "jazz"],
            "ner_tags": ["B-PER", "I-PER", "O", "B-MISC"],
        },
        None,
    )
    assert example.gold_spans == [(0, 12, "person")]


def test_ner_drops_miscellaneous_and_dataset_excludes():
    example = _ner_from_tokens(
        _toy_ner_spec(exclude_types=["trigger"]),
        {
            "tokens": ["Paris", "flood", "jazz"],
            "ner_tags": ["B-LOC", "B-trigger", "B-miscellaneous"],
        },
        None,
    )
    assert example.gold_spans == [(0, 5, "location")]


def test_find_span_casefold():
    assert find_span("Alice founded Acme", "acme") == (14, 18)


def test_sampling_is_deterministic_and_stratified():
    examples = [{"k": "a"}] * 8 + [{"k": "b"}] * 2
    first = cap_examples(examples, 5, seed=42, key_fn=lambda item: item["k"])
    second = cap_examples(examples, 5, seed=42, key_fn=lambda item: item["k"])
    assert first == second
    assert len(first) == 5
    assert {item["k"] for item in first} == {"a", "b"}


def test_sampling_without_key_shuffles_stably():
    items = list(range(20))
    assert cap_examples(items, 5, seed=1) == cap_examples(items, 5, seed=1)
    assert cap_examples(items, 5, seed=1) != cap_examples(items, 5, seed=2)


def test_twenty_shot_sampling_is_deterministic_and_stratified():
    examples = [{"label": str(index % 4), "id": index} for index in range(100)]
    first = cap_examples(examples, 20, seed=42, key_fn=lambda item: item["label"])
    second = cap_examples(examples, 20, seed=42, key_fn=lambda item: item["label"])
    assert first == second
    assert len(first) == 20
    assert {item["label"] for item in first} == {"0", "1", "2", "3"}


def test_single_and_multilabel_classification_scores():
    labels = ["negative", "positive"]
    single = classification_scores(
        [["positive"], ["negative"]],
        [["Positive"], ["neutral"]],
        labels,
        multi_label=False,
    )
    assert single["accuracy"] == 0.5
    assert single["unknown_predictions"] == 1

    multi = classification_scores(
        [["joy", "love"], ["anger"]],
        [["love", "joy"], ["anger", "fear"]],
        ["joy", "love", "anger", "fear"],
        multi_label=True,
    )
    assert multi["exact_match"] == 0.5
    assert 0.0 < multi["micro_f1"] <= 1.0


def test_span_and_triple_matching_is_directional():
    gold_spans = [gold_spans_set([(0, 4, "PER"), (8, 12, "ORG")])]
    pred_spans = [spans_from_predictions([
        {"start": 0, "end": 4, "type": "per", "confidence": 0.9},
        {"start": 8, "end": 12, "type": "org", "confidence": 0.4},
    ])]
    scores = span_scores(gold_spans, pred_spans)
    assert scores["matched"] == 2

    gold = [gold_triples_set([(0, 4, "works for", 8, 12)])]
    pred_ok = [triples_from_predictions([{
        "head_start": 0, "head_end": 4, "type": "works_for",
        "tail_start": 8, "tail_end": 12, "confidence": 0.8,
    }])]
    pred_rev = [triples_from_predictions([{
        "head_start": 8, "head_end": 12, "type": "works_for",
        "tail_start": 0, "tail_end": 4, "confidence": 0.8,
    }])]
    assert triple_scores(gold, pred_ok)["f1"] == 1.0
    assert triple_scores(gold, pred_rev)["f1"] == 0.0


def test_threshold_selection_uses_validation_only():
    gold = [gold_spans_set([(0, 2, "PER")])]
    preds = [[{"start": 0, "end": 2, "type": "PER", "confidence": 0.7}]]
    best, curve = select_threshold(gold, preds, [0.5, 0.8], kind="ner")
    assert best == 0.5
    assert curve[0.5] == 1.0
    assert curve[0.8] == 0.0
    kept = filter_by_threshold(preds[0], 0.8)
    assert kept == []


def test_redocred_cluster_mapping():
    clusters = [[(0, 4), (20, 24)], [(10, 15)]]
    mapped = map_mentions_to_clusters([(20, 24, "founded", 10, 15)], clusters)
    assert mapped == {(0, "founded", 1)}


def test_catalog_has_forty_datasets():
    assert len(DATASETS) == 40
    assert len(MODELS) == 5
    assert sum(spec.task == "classification" for spec in DATASETS.values()) == 13
    assert sum(spec.task == "ner" for spec in DATASETS.values()) == 17
    assert sum(spec.task == "relation" for spec in DATASETS.values()) == 10
    assert DATASETS["wikineural"].train_split == "train_en"
    assert DATASETS["tweet_topic_multi"].train_split == "train_2021"
    assert DATASETS["sst2"].test_split == "validation"
    assert DATASETS["sst2"].extra["text"] == "sentence"
    for domain, config in [
        ("ai", "ai"),
        ("literature", "literature"),
        ("music", "music"),
        ("politics", "politics"),
        ("science", "science"),
    ]:
        spec = DATASETS[f"crossner_{domain}"]
        assert spec.path == "DFKI-SLT/cross_ner"
        assert spec.config == config
        assert spec.loader == "ner_bio"
        assert spec.train_split == "train"
        assert spec.test_split == "test"
        assert spec.extra.get("trust_remote_code") is True
    assert all(spec.train_split != spec.test_split for spec in DATASETS.values())
    assert all(
        spec.train_split != spec.validation_split
        for spec in DATASETS.values()
        if spec.validation_split
    )


def test_train_split_isolation_guard():
    safe = DATASETS["imdb"]
    validate_train_split(safe)
    unsafe = DatasetSpec(
        id="unsafe",
        task="classification",
        path="example/unsafe",
        loader="classification_classlabel",
        test_split="test",
        train_split="test",
        language="en",
        license="test",
        caveats="test fixture",
    )
    try:
        validate_train_split(unsafe)
    except ValueError as exc:
        assert "train split equals test split" in str(exc)
    else:
        raise AssertionError("expected split-isolation failure")


def test_benchmark_examples_convert_to_training_examples():
    cls_spec = DATASETS["imdb"]
    cls_loaded = LoadedSplit(
        cls_spec,
        "train",
        [],
        ["negative", "positive"],
        multi_label=False,
    )
    cls = to_training_example(
        Example("Excellent film.", "classification", gold_labels=["positive"]),
        cls_loaded,
    )
    assert cls.classifications[0].task == "imdb"
    assert cls.classifications[0].to_dict()["true_label"] == ["positive"]

    ner_spec = DATASETS["conll2003"]
    ner_loaded = LoadedSplit(ner_spec, "train", [], ["person", "organization"])
    ner = to_training_example(
        Example(
            "Ada joined Fastino.",
            "ner",
            gold_spans=[(0, 3, "person"), (11, 18, "organization")],
        ),
        ner_loaded,
    )
    assert ner.entities == {"person": ["Ada"], "organization": ["Fastino"]}

    relation_spec = DATASETS["conll04"]
    relation_loaded = LoadedSplit(relation_spec, "train", [], ["work for"])
    relation = to_training_example(
        Example(
            "Ada works for Fastino.",
            "relation",
            gold_spans=[
                (0, 3, "person"),
                (14, 21, "organization"),
                (4, 9, ""),
            ],
            gold_triples=[(0, 3, "work for", 14, 21)],
        ),
        relation_loaded,
    )
    assert relation.relations[0].to_dict() == {
        "work for": {"head": "Ada", "tail": "Fastino"}
    }
    assert "" not in relation.entities
    assert relation.validate() == []


def test_capability_detection_and_parsers():
    class Boundary:
        architecture = "boundary"
        enable_relations = False
        config = type("C", (), {"architecture": "boundary"})()

    caps = model_capabilities(Boundary())
    assert caps["relations"] is False
    assert supports_task(caps, "relation") is False
    assert supports_task(caps, "ner") is True
    schema = classification_schema(["negative", "positive"], multi_label=False)
    assert schema == {"category": ["negative", "positive"]}
    assert parse_classification({"category": {"label": "Positive"}}, ["negative", "positive"], multi_label=False) == ["positive"]
    entities = parse_entities({"entities": {"person": [{"text": "Ada", "start": 0, "end": 3, "confidence": 0.9}]}})
    assert entities[0]["start"] == 0
    relations = parse_relations({
        "relation_extraction": {"works_for": [("Ada", "Fastino")]}
    })
    assert relations[0]["type"] == "works for"


def test_nested_hf_records_and_offsets():
    records = _as_records({
        "id": ["T1", "T2"],
        "type": ["CHEMICAL", "GENE"],
        "offsets": [[0, 4], [11, 19]],
    })
    assert [item["id"] for item in records] == ["T1", "T2"]
    assert _offset_pair([0, 4]) == (0, 4)
    assert _offset_pair([[11, 19]]) == (11, 19)
    assert _as_index(3) == 3
    assert _span_tokens([0, 2]) == (0, 2)
    text, offsets = tokens_to_text(["Acadia", "University", "in", "Wolfville"])
    start, end = token_span_to_char(0, 2, offsets, end_exclusive=True)
    assert text[start:end] == "Acadia University"
    assert decode_tag(8, ["O", "B-PER"]) == "O"


def test_xnli_translation_lists_are_keyed_by_language():
    assert _translations_by_language(
        {"language": ["en", "fr"], "translation": ["Hello", "Bonjour"]}
    ) == {"en": "Hello", "fr": "Bonjour"}


def test_chemprot_dict_of_lists_row():
    example = _id_entity_relations(
        {
            "text": "Drug binds protein.",
            "entities": {
                "id": ["T1", "T2"],
                "type": ["CHEMICAL", "GENE"],
                "text": ["Drug", "protein"],
                "offsets": [[0, 4], [11, 18]],
            },
            "relations": {"type": ["CPR:3"], "arg1": ["T1"], "arg2": ["T2"]},
        },
        DATASETS["chemprot"],
    )
    assert example.gold_spans[0] == (0, 4, "chemical")
    assert example.gold_triples == [(0, 4, "upregulator", 11, 18)]


def test_instruct_uie_character_offsets():
    text = "We study neural networks in papers."
    _, offsets = tokens_to_text(text.split())
    start, end = find_span(text, "neural networks")
    span = _token_pos([start, end], offsets, text, "neural networks")
    assert span == (start, end)
    assert _token_pos([112, 130], offsets, text, "neural networks") == (start, end)


def test_adaptive_batch_and_oom_retry():
    assert _adaptive_batch_size(8, ["r"] * 200, ["short"]) == 1
    calls: list[int] = []

    def predict(batch):
        calls.append(len(batch))
        if len(batch) > 1:
            raise RuntimeError("CUDA out of memory. Tried to allocate 2.19 GiB")
        return [{"ok": True} for _ in batch]

    outputs, elapsed = _run_batched(predict, ["a", "b", "c", "d"], chunk_size=4)
    assert len(outputs) == 4
    assert elapsed >= 0
    assert 1 in calls


def test_finetune_config_is_full_and_step_bounded(tmp_path: Path):
    config = training_config(
        tmp_path / "run",
        max_steps=250,
        seed=42,
        batch_size=1,
        gradient_accumulation_steps=8,
    )
    assert config.max_steps == 250
    assert config.effective_batch_size == 8
    assert not config.use_lora
    assert not config.save_adapter_only
    assert config.allow_invalid_samples
    assert not config.fp16 and not config.bf16


def test_unload_model_accepts_none():
    from benchmarks.multitask.run import _unload_model

    _unload_model(None)


def test_resume_skips_complete_shards(tmp_path: Path):
    path = shard_path(tmp_path, "fastino/gliner2-base-v1", "conll2003")
    write_json_atomic(path, {"status": "complete", "dataset": "conll2003"})
    assert shard_complete(path)
    write_json_atomic(path.with_name("partial.json"), {"status": "running"})
    assert not shard_complete(path.with_name("partial.json"))
