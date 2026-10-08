"""Joint IE integration through sparse boundary candidates."""

import torch

from gliner2 import ExtractorConfig
from gliner2.inference.engine import BoundaryExtractor
from gliner2.joint_ie import JointIE, JointIEConfig
from gliner2.joint_ie.candidate_scores import CandidateScoreSet
from gliner2.joint_ie.schema import JointSchema
from tests.fixtures.tiny_boundary_checkpoint import TINY_BOUNDARY_HEAD
from tests.fixtures.tiny_encoder import build_tiny_encoder_config
from tests.fixtures.tiny_tokenizer import build_tiny_tokenizer


def _relation_model():
    tokenizer = build_tiny_tokenizer()
    encoder_config = build_tiny_encoder_config(vocab_size=len(tokenizer))
    head = dict(TINY_BOUNDARY_HEAD)
    head.update(
        enable_relations=True,
        relation_heads_per_type=8,
        relation_tails_per_type=8,
        relation_pair_cap=16,
        relation_argument_proposal_threshold=0.0,
    )
    torch.manual_seed(3)
    return BoundaryExtractor(
        ExtractorConfig(
            model_name="tiny-bert-fixture",
            architecture="boundary",
            boundary_head=head,
            token_pooling="first",
        ),
        encoder_config=encoder_config,
        tokenizer=tokenizer,
    ).eval()


def test_boundary_joint_ie_scores_entities_relations_and_constraints(
    monkeypatch,
):
    engine = JointIE(_relation_model())
    seen = {}
    original_collate = engine.model.processor.collate_fn_inference

    def collate(*args, **kwargs):
        seen["architecture"] = kwargs.get("architecture")
        return original_collate(*args, **kwargs)

    monkeypatch.setattr(
        engine.model.processor, "collate_fn_inference", collate
    )
    schema = (
        JointSchema()
        .entity("person", threshold=0.1, candidate_threshold=0.0)
        .entity("org", threshold=0.1, candidate_threshold=0.0)
        .relation("works_for", "person", "org", threshold=0.1)
    )
    config = JointIEConfig(candidate_threshold=0.0, optimizer="greedy")

    scores = engine.score("Alice works at Acme", schema, config=config)
    assert isinstance(scores, CandidateScoreSet)
    assert scores.mentions
    assert scores.edges
    assert scores.start_mappings == (0, 6, 12, 15)
    assert seen["architecture"] == "boundary"

    result = engine.extract("Alice works at Acme", schema, config=config)
    assert result.feasible
    entity_types = {entity.id: entity.type for entity in result.entities}
    for relation in result.relations:
        assert relation.type == "works_for"
        assert entity_types[relation.head] == "person"
        assert entity_types[relation.tail] == "org"
