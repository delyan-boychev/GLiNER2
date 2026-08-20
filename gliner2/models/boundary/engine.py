"""Public boundary extractor class = shared runtime + boundary model core."""

from __future__ import annotations

import bisect
import logging
from collections import OrderedDict
from typing import Any, Dict, List, Tuple

import torch

logger = logging.getLogger(__name__)

from gliner2.inference.candidate_decoder import token_boundaries_to_character_offsets
from gliner2.inference.runtime import ExtractorRuntimeMixin
from gliner2.inference.packing import PackingConfig, PackingStats
from gliner2.models.boundary.model import (
    BoundaryExtractorModel,
    _group_scored_candidates,
)
from gliner2.models.base import QueryLayout
from gliner2.models.boundary.records import decode_group


def _resolve_flat_spans(
    scored: List[Tuple[float, int, int]]
) -> List[Tuple[float, int, int]]:
    """Maximum-total-score non-overlapping subset via interval scheduling."""
    if not scored:
        return []
    spans = sorted(scored, key=lambda item: (item[2], item[1], -item[0]))
    ends = [end for _, _, end in spans]
    predecessors = [
        bisect.bisect_right(ends, start, 0, index) - 1
        for index, (_, start, _) in enumerate(spans)
    ]
    best = [0.0] * (len(spans) + 1)
    choose = [False] * len(spans)
    for index, (score, _, _) in enumerate(spans):
        with_span = score + best[predecessors[index] + 1]
        without_span = best[index]
        if with_span > without_span:
            best[index + 1] = with_span
            choose[index] = True
        else:
            best[index + 1] = without_span
    kept: List[Tuple[float, int, int]] = []
    index = len(spans) - 1
    while index >= 0:
        if choose[index] and (
            spans[index][0] + best[predecessors[index] + 1] > best[index]
        ):
            kept.append(spans[index])
            index = predecessors[index]
        else:
            index -= 1
    return sorted(kept, key=lambda item: (-item[0], item[1], item[2]))


def _resolve_spans(
    scored: List[Tuple[float, int, int]], policy: str
) -> List[Tuple[float, int, int]]:
    if policy == "flat":
        return _resolve_flat_spans(scored)
    ranked = sorted(scored, key=lambda item: (-item[0], item[1], item[2]))
    if policy == "nested":
        return ranked
    if policy == "longest":
        kept = []
        for candidate in ranked:
            _, start, end = candidate
            if any(
                kept_start <= start
                and end <= kept_end
                and (kept_start < start or end < kept_end)
                for _, kept_start, kept_end in kept
            ):
                continue
            kept.append(candidate)
        return kept
    raise ValueError(f"unknown boundary overlap policy {policy!r}")


class BoundaryExtractor(ExtractorRuntimeMixin, BoundaryExtractorModel):
    """Boundary architecture with the shared public extraction runtime.

    Overrides ``_extract_from_batch`` with the sparse candidate path: encode →
    boundary head → threshold + flat-span resolution → exact half-open
    token→character conversion. Entities, classification, and enabled
    record/event schemas and enabled sparse relation decoding are supported.
    """

    architecture = "boundary"

    def _extract_from_batch(
        self,
        batch,
        threshold: float,
        metadata_list: List[Dict],
        include_confidence: bool,
        include_spans: bool,
        packing_config: PackingConfig = None,
    ) -> List[Dict[str, Any]]:
        if packing_config is not None and packing_config.enabled:
            if not getattr(self, "_packing_unsupported_warned", False):
                logger.warning(
                    "sequence packing is currently limited to the span architecture; "
                    "using normal batching for the boundary architecture"
                )
                self._packing_unsupported_warned = True
            self._last_packing_stats = PackingStats(
                activated=False,
                reason="unsupported_architecture_boundary",
                original_count=len(batch),
                fallback_count=len(batch),
            )
        core = self._encode_core(batch)
        has_queries = core["query_states"].shape[1] > 0
        candidates = None
        probs = None
        grouped_candidates = None
        null_probs = None
        if has_queries:
            out = self.boundary_head(
                core["text_states"], core["text_mask"],
                core["query_states"], core["query_mask"],
                return_candidates=True,
            )
            candidates = out.candidates
            probs = torch.sigmoid(
                candidates.pair_logits
                / self.boundary_settings.pair_temperature
            )
            grouped_candidates = _group_scored_candidates(
                candidates,
                threshold=threshold,
                probabilities=probs,
                count_log_rates=out.count_log_rates,
                adaptive_threshold=self.boundary_settings.adaptive_threshold,
            )
            if out.null_logits is not None:
                null_probs = torch.sigmoid(out.null_logits).float().cpu()

        results: List[Dict[str, Any]] = []
        for i in range(len(batch)):
            sample: Dict[str, Any] = {}
            overlap_policy = (
                metadata_list[i].get("_overlap_policy")
                or self.boundary_settings.overlap_policy
            )
            specs = core["ext_specs"][i] if has_queries else []
            offset = core["word_offsets"][i]
            start_map = batch.start_mappings[i]
            end_map = batch.end_mappings[i]
            text = batch.original_texts[i]
            text_len = len(start_map)

            record_results = self._decode_records(
                batch, i, core, candidates, offset, start_map, end_map,
                text, text_len, include_confidence, include_spans,
            )
            for name, instances in record_results.items():
                if instances:
                    sample[name] = instances

            relation_results = self._decode_relations(
                i, core, candidates, metadata_list[i], threshold, offset,
                start_map, end_map, text, text_len, include_confidence, include_spans,
            )
            sample.update(relation_results)

            entity_results: "OrderedDict[str, Any]" = OrderedDict()
            for qid, spec in enumerate(specs):
                if spec["task_type"] != "entities":
                    continue  # non-entity queries are decoded by the record head
                scored = grouped_candidates[i][qid]
                if (
                    null_probs is not None
                    and float(null_probs[i, qid])
                    > self.boundary_settings.abstention_threshold
                ):
                    entity_results[spec["field_name"]] = []
                    continue
                spans: List[Tuple[str, float, int, int]] = []
                for p, s, e in _resolve_spans(scored, overlap_policy):
                    ts, te = s - offset, e - offset
                    if ts < 0 or te > text_len or te <= ts:
                        continue
                    char_start, char_end = token_boundaries_to_character_offsets(
                        ts, te, start_map, end_map
                    )
                    surface = text[char_start:char_end].strip()
                    if surface:
                        spans.append((surface, p, char_start, char_end))

                entity_results[spec["field_name"]] = self._format_spans(
                    spans, include_confidence, include_spans, already_finalized=True
                )

            if entity_results:
                sample["entities"] = [entity_results]

            schema = batch.original_schemas[i]
            for cls in core["cls_specs"][i]:
                self._extract_classification_result(
                    sample, cls["task_name"], schema,
                    cls["group_embs"], cls["schema_tokens"],
                    temperature=self.boundary_settings.classification_temperature,
                )

            results.append(sample)

        return results

    def _decode_relations(
        self,
        sample_index: int,
        core: Dict[str, Any],
        candidates,
        metadata: Dict[str, Any],
        threshold: float,
        offset: int,
        start_map,
        end_map,
        text: str,
        text_len: int,
        include_confidence: bool,
        include_spans: bool,
    ) -> Dict[str, Any]:
        """Decode sparse relation pairs for one sample."""
        if not getattr(self, "enable_relations", False) or candidates is None:
            return {}
        rel_specs = core["rel_specs"][sample_index]
        if not rel_specs:
            return {}
        sample_candidates = self._single_sample_candidates(candidates, sample_index)
        pairs = self.relation_pair_generator.generate(
            sample_candidates,
            [QueryLayout(queries=())],
            [entry["spec"] for entry in rel_specs],
        )
        if not len(pairs):
            return {}
        query_states = torch.stack(
            [entry["query_state"] for entry in rel_specs]
        ).unsqueeze(0)
        logits = self.relation_scorer(
            core["text_states"][sample_index:sample_index + 1],
            query_states,
            sample_candidates,
            pairs,
        )
        probabilities = torch.sigmoid(
            logits / self.boundary_settings.relation_temperature
        )
        out: Dict[str, Any] = {}
        relation_metadata = metadata.get("relation_metadata", {})
        for pair_index, probability in enumerate(probabilities):
            relation_type = pairs.relation_types[pair_index]
            relation_threshold = relation_metadata.get(relation_type, {}).get(
                "threshold", threshold
            )
            if relation_threshold is None:
                relation_threshold = threshold
            score = float(probability.detach())
            if score < relation_threshold:
                continue
            hs = int(pairs.head_start[pair_index]) - offset
            he = int(pairs.head_end[pair_index]) - offset
            ts = int(pairs.tail_start[pair_index]) - offset
            te = int(pairs.tail_end[pair_index]) - offset
            if not (0 <= hs < he <= text_len and 0 <= ts < te <= text_len):
                continue
            h0, h1 = token_boundaries_to_character_offsets(hs, he, start_map, end_map)
            t0, t1 = token_boundaries_to_character_offsets(ts, te, start_map, end_map)
            head, tail = text[h0:h1].strip(), text[t0:t1].strip()
            if not head or not tail:
                continue
            if include_spans:
                value = {
                    "head": {"text": head, "start": h0, "end": h1},
                    "tail": {"text": tail, "start": t0, "end": t1},
                }
                if include_confidence:
                    value["head"]["confidence"] = score
                    value["tail"]["confidence"] = score
            elif include_confidence:
                value = {
                    "head": {"text": head, "confidence": score},
                    "tail": {"text": tail, "confidence": score},
                }
            else:
                value = (head, tail)
            out.setdefault(relation_type, []).append(value)
        return out

    def _decode_records(
        self,
        batch,
        sample_index: int,
        core: Dict[str, Any],
        candidates,
        offset: int,
        start_map,
        end_map,
        text: str,
        text_len: int,
        include_confidence: bool,
        include_spans: bool,
    ) -> Dict[str, Any]:
        """Decode record/event groups into public structure output shapes."""
        if not getattr(self, "enable_records", False):
            return {}
        if candidates is None or candidates.candidate_states is None:
            return {}
        record_specs = getattr(batch, "record_specs", ())
        if sample_index >= len(record_specs) or not record_specs[sample_index]:
            return {}

        settings = self.boundary_settings
        query_states_i = core["query_states"][sample_index]
        out: Dict[str, Any] = {}

        def _format_field(spans, is_scalar):
            formatted: List[Tuple[str, float, int, int]] = []
            for (ts_raw, te_raw) in spans:
                ts, te = ts_raw - offset, te_raw - offset
                if ts < 0 or te > text_len or te <= ts:
                    continue
                cs, ce = token_boundaries_to_character_offsets(ts, te, start_map, end_map)
                surface = text[cs:ce].strip()
                if surface:
                    formatted.append((surface, 1.0, cs, ce))
            if is_scalar:
                if not formatted:
                    return None
                s, conf, cs, ce = formatted[0]
                if include_spans and include_confidence:
                    return {"text": s, "confidence": conf, "start": cs, "end": ce}
                if include_spans:
                    return {"text": s, "start": cs, "end": ce}
                if include_confidence:
                    return {"text": s, "confidence": conf}
                return s
            return self._format_spans(
                formatted, include_confidence, include_spans, already_finalized=True
            )

        for task_index, spec in record_specs[sample_index].items():
            group = self.record_decoder.forward_group(
                spec, query_states_i, candidates, sample_index
            )
            decoded = decode_group(
                group,
                anchor_threshold=settings.record_anchor_threshold,
                field_threshold=settings.record_field_threshold,
                object_threshold=settings.record_anchor_threshold,
                temperature=settings.record_temperature,
            )
            instances = []
            for rec in decoded:
                inst: "OrderedDict[str, Any]" = OrderedDict()
                # Emit every declared field in schema order (legacy shape):
                # scalar -> str/None, list -> list[str] (possibly empty).
                for fspec in spec.fields:
                    spans = rec.fields.get(fspec.query_id, [])
                    value = _format_field(spans, fspec.cardinality.is_scalar)
                    if not fspec.cardinality.is_scalar and value is None:
                        value = []
                    inst[fspec.name] = value
                if any(v is not None and v != [] for v in inst.values()):
                    instances.append(inst)
            if instances:
                out[spec.task_name] = instances
        return out


__all__ = ["BoundaryExtractor"]
