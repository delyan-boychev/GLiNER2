"""Public schema/annotation contract for Instance Formation record metadata."""

from __future__ import annotations

import pytest

from gliner2.inference.schema import Schema
from gliner2.processing.records import (
    FieldCardinality,
    compile_record_specs,
    normalize_record_metadata,
)
from gliner2.models.base import QueryLayout, QuerySpec


def _purchase_schema():
    s = Schema()
    (
        s.structure("purchase", mode="natural", anchor="buyer")
        .field("buyer", dtype="str", cardinality="required_one")
        .field("item", dtype="list", cardinality="zero_or_more", exclusive=True)
    )
    return s


def test_build_emits_record_metadata():
    built = _purchase_schema().build()
    meta = built["record_metadata"]["purchase"]
    assert meta["mode"] == "natural"
    assert meta["anchor"] == "buyer"
    assert meta["fields"]["buyer"]["cardinality"] == "required_one"
    assert meta["fields"]["item"]["exclusive"] is True


def test_to_dict_from_dict_round_trip_preserves_records():
    d = _purchase_schema().to_dict()
    assert d["structures"]["purchase"]["mode"] == "natural"
    assert d["structures"]["purchase"]["anchor"] == "buyer"
    fields = {f["name"]: f for f in d["structures"]["purchase"]["fields"]}
    assert fields["buyer"]["cardinality"] == "required_one"
    assert fields["item"]["exclusive"] is True

    rebuilt = Schema.from_dict(d).build()
    assert rebuilt["record_metadata"]["purchase"]["mode"] == "natural"
    assert rebuilt["record_metadata"]["purchase"]["anchor"] == "buyer"


def test_missing_metadata_is_legacy():
    s = Schema()
    s.structure("plain").field("a").field("b")
    built = s.build()
    assert "record_metadata" not in built


def test_natural_requires_valid_anchor():
    s = Schema()
    s.structure("bad", mode="natural", anchor="missing").field("a")
    with pytest.raises(ValueError, match="anchor"):
        s.build()


def test_latent_must_not_declare_anchor():
    s = Schema()
    with pytest.raises(ValueError, match="must not set an anchor"):
        s.structure("x", mode="latent", anchor="a").field("a").build()


def test_normalize_rejects_unknown_mode():
    with pytest.raises(ValueError, match="mode"):
        normalize_record_metadata({"r": {"mode": "weird"}})


def test_compile_specs_respects_layout_order_and_dtype():
    # Two fields declared in one order; layout presents them shuffled.
    layout = QueryLayout(queries=(
        QuerySpec(query_id=0, task_index=0, task_type="json_structures",
                  task_name="purchase", role_index=0, role_name="item"),
        QuerySpec(query_id=1, task_index=0, task_type="json_structures",
                  task_name="purchase", role_index=1, role_name="buyer"),
    ))
    meta = {"purchase": {"mode": "natural", "anchor": "buyer"}}
    dtypes = {"purchase": {"buyer": "str", "item": "list"}}
    specs = compile_record_specs(
        query_layout=layout, record_metadata=meta, field_dtypes=dtypes
    )
    spec = specs[0]
    assert spec.mode == "natural"
    assert spec.anchor_query_id == 1
    buyer = spec.field_for_query(1)
    item = spec.field_for_query(0)
    assert buyer.is_anchor and buyer.cardinality == FieldCardinality.REQUIRED_ONE
    assert item.cardinality == FieldCardinality.ZERO_OR_MORE


def test_compile_specs_uses_dtype_for_implicit_scalar_cardinality():
    schema = Schema()
    (
        schema.structure("transaction", mode="natural", anchor="merchant")
        .field("merchant", dtype="str")
        .field("date", dtype="str")
        .field("tags", dtype="list")
    )
    built = schema.build()
    layout = QueryLayout(queries=(
        QuerySpec(
            query_id=0,
            task_index=0,
            task_type="json_structures",
            task_name="transaction",
            role_index=0,
            role_name="merchant",
        ),
        QuerySpec(
            query_id=1,
            task_index=0,
            task_type="json_structures",
            task_name="transaction",
            role_index=1,
            role_name="date",
        ),
        QuerySpec(
            query_id=2,
            task_index=0,
            task_type="json_structures",
            task_name="transaction",
            role_index=2,
            role_name="tags",
        ),
    ))

    spec = compile_record_specs(
        query_layout=layout,
        record_metadata=built["record_metadata"],
        field_dtypes={
            "transaction": {
                "merchant": "str",
                "date": "str",
                "tags": "list",
            }
        },
    )[0]

    assert spec.field_for_query(0).cardinality == FieldCardinality.REQUIRED_ONE
    assert spec.field_for_query(1).cardinality == FieldCardinality.OPTIONAL_ONE
    assert spec.field_for_query(2).cardinality == FieldCardinality.ZERO_OR_MORE


def test_compile_specs_excludes_deferred_relation_groups():
    layout = QueryLayout(queries=(
        QuerySpec(query_id=0, task_index=0, task_type="relations",
                  task_name="employment", role_index=0, role_name="head"),
        QuerySpec(query_id=1, task_index=0, task_type="relations",
                  task_name="employment", role_index=1, role_name="tail"),
    ))
    specs = compile_record_specs(
        query_layout=layout,
        record_metadata={"employment": {"mode": "natural", "anchor": "head"}},
    )
    assert specs == {}
