"""Constrained-classification integration with the boundary architecture."""

from gliner2.classification import (
    ClassificationConfig,
    ClassificationSchema,
    Classifier,
)
from gliner2.classification import constraints as C
from tests.fixtures.tiny_boundary_checkpoint import build_tiny_boundary_model


def test_boundary_model_runs_cross_task_constrained_classification():
    model = build_tiny_boundary_model()
    schema = (
        ClassificationSchema()
        .single("intent", ["read", "delete"])
        .multi("effects", ["read_only", "delete"], min_labels=1)
        .constrain(
            C.implies(("intent", "delete"), ("effects", "delete"))
        )
    )

    result = Classifier(model).classify(
        "Delete the temporary file",
        schema,
        config=ClassificationConfig(decoder="exact"),
    )

    assert result.feasible
    if result.value("intent") == "delete":
        assert "delete" in result.selected("effects")


def test_classifier_from_pretrained_uses_architecture_dispatch(monkeypatch):
    import gliner2

    model = build_tiny_boundary_model()
    captured = {}

    class Loader:
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            captured.update(path=path, kwargs=kwargs)
            return model

    monkeypatch.setattr(gliner2, "AutoExtractor", Loader)
    classifier = Classifier.from_pretrained(
        "boundary-checkpoint",
        map_location="cpu",
        revision="release",
        use_flashdeberta=False,
    )

    assert classifier.model is model
    assert captured == {
        "path": "boundary-checkpoint",
        "kwargs": {
            "map_location": "cpu",
            "revision": "release",
            "use_flashdeberta": False,
        },
    }
