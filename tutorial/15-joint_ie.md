# GLiNER2.5 Joint Information Extraction Tutorial

Learn how to extract **entities and relations together** under typed endpoint and graph constraints. Independent `extract_entities` + `extract_relations` can produce an org that never appears as a person-employer pair. `JointIE` scores candidates, then searches a globally consistent graph.

GLiNER2.5 boundary checkpoints with `enable_relations=True` (including `fastino/gliner2.5-multi-v1`) supply sparse mention and relation candidates to the same greedy/beam decoder.

## Table of Contents
- [Setup](#setup)
- [Independent Extraction vs Joint IE](#independent-extraction-vs-joint-ie)
- [Declaring a Schema](#declaring-a-schema)
- [Relation Options](#relation-options)
- [Graph Constraints](#graph-constraints)
- [Running Extraction](#running-extraction)
- [Reading the Graph](#reading-the-graph)
- [Decoding Config](#decoding-config)
- [Feasibility](#feasibility)
- [Long Documents](#long-documents)
- [Batch Extraction](#batch-extraction)
- [Best Practices](#best-practices)

## Setup

```python
from gliner2.joint_ie import JointIE, JointIEConfig

joint = JointIE.from_pretrained("fastino/gliner2.5-multi-v1")
```

Other sizes: `gliner2.5-base-v1`, `gliner2.5-small-v1`. See the [README model catalog](../README.md#-available-models).

`JointIE.from_pretrained` loads through `AutoExtractor`. Prediction knobs belong in `JointIEConfig` on each `extract` call, not on `from_pretrained`.

Relations require a boundary checkpoint trained with `enable_relations=True`. Entity-only Joint IE still works without the relation head.

For unconstrained relation tuples (no typed endpoints, no uniqueness), see [Relation Extraction](6-relation_extraction.md).

## Independent Extraction vs Joint IE

```python
from gliner2 import AutoExtractor

model = AutoExtractor.from_pretrained("fastino/gliner2.5-multi-v1")
text = "Alice works for Acme in Paris."

# Independent: entities and edges decoded separately
ents = model.extract_entities(text, ["person", "organization", "location"], include_spans=True)
rels = model.extract_relations(text, ["works_for", "located_in"], include_spans=True)
```

Independent decoding cannot guarantee:

- `works_for` heads are `person` and tails are `organization`
- a person has at most one employer
- no self-loops
- inverse pairs stay consistent

Joint IE applies those rules while choosing the graph.

## Declaring a Schema

`joint.create_schema()` returns a `JointSchema`. Declare entities first; relations may only reference known types.

```python
schema = (
    joint.create_schema()
    .entities(["person", "organization", "location"])
    .relation("works_for", "person", "organization")
    .relation("located_in", "organization", "location")
)
```

Entity descriptions and thresholds:

```python
schema = (
    joint.create_schema()
    .entities({
        "person": "Named people",
        "organization": {
            "description": "Companies, agencies, or teams",
            "threshold": 0.3,
            "max_candidates": 16,
        },
        "location": "Cities, countries, or addresses",
    })
    .relation(
        "works_for",
        head="person",
        tail="organization",
        description="Employment or affiliation",
        threshold=0.25,
    )
)
```

A relation head or tail can be several entity types:

```python
.relation("affiliated_with", head="person", tail=["organization", "location"])
```

## Relation Options

```python
.relation(
    "works_for",
    "person",
    "organization",
    directed=True,          # default
    allow_self=False,       # reject head == tail
    unique_head=True,       # at most one works_for per person (max_per_head=1)
    unique_tail=False,
    max_per_head=1,
    max_per_tail=None,
    symmetric=False,
    inverse=None,           # name of the inverse relation, if declared
    acyclic=False,
    threshold=None,
    candidate_threshold=None,
)
```

| Option | Effect |
| --- | --- |
| `unique_head=True` | Each head entity has at most one edge of this type |
| `unique_tail=True` | Each tail entity has at most one incoming edge of this type |
| `symmetric=True` | Undirected; head and tail type sets must match |
| `inverse="employed"` | Decoder may realize the inverse pair |
| `acyclic=True` | Forbids cycles on this relation (also `.acyclic(name)`) |
| `allow_self=True` | Permits a span linked to itself |

`symmetric` and `inverse` cannot be set together.

## Graph Constraints

Chain methods after relations exist:

```python
schema = (
    joint.create_schema()
    .entities(["person", "organization", "location"])
    .relation("works_for", "person", "organization", unique_head=True)
    .relation("located_in", "organization", "location")
    .relation("knows", "person", "person", symmetric=True, allow_self=False)
    .no_self_loops()
    .at_most("located_in", per_head=1)
    .acyclic("works_for")  # only if the type should not form cycles
)
```

| Method | Meaning |
| --- | --- |
| `.no_self_loops()` / `.no_self_loops("knows")` | Forbid head == tail (all types or one type) |
| `.at_most("works_for", per_head=1)` | Cap edges per head (and/or `per_tail=`) |
| `.acyclic("reports_to")` | No directed cycles |
| `.constraint(obj)` | Attach a constraint instance directly |

Built-in constraint classes include `NoSelfLoops`, `MaxRelationsPerHead`, `MaxRelationsPerTail`, `AcyclicRelation`, `TypedEndpoints`, `UniqueRelationPair`, `SymmetricRelation`, and `InverseRelation`. Typed endpoints from `.relation(head, tail)` are already compiled in.

## Running Extraction

```python
text = "Alice works for Acme in Paris. Bob joined Acme last year."

result = joint.extract(
    text,
    schema,
    config=JointIEConfig(optimizer="beam", beam_size=32),
)

print(result.feasible)
print(result.to_dict())
# {
#   "entities": [
#     {"id": "e1", "type": "person", "text": "Alice", "start": 0, "end": 5, "confidence": 0.94},
#     {"id": "e2", "type": "organization", "text": "Acme", "start": 16, "end": 20, "confidence": 0.91},
#     {"id": "e3", "type": "location", "text": "Paris", "start": 24, "end": 29, "confidence": 0.89},
#     {"id": "e4", "type": "person", "text": "Bob", "start": 31, "end": 34, "confidence": 0.90},
#   ],
#   "relations": [
#     {"type": "works_for", "head": "e1", "tail": "e2", "confidence": 0.88},
#     {"type": "works_for", "head": "e4", "tail": "e2", "confidence": 0.81},
#     {"type": "located_in", "head": "e2", "tail": "e3", "confidence": 0.86},
#   ],
# }
```

Character offsets are half-open `[start, end)` into `result.text`.

```python
alice = result.entity("e1")
assert text[alice.start:alice.end] == alice.text
```

Optional lattice for debugging:

```python
result, lattice = joint.extract(text, schema, return_lattice=True)
```

## Reading the Graph

`extract` returns a `JointResult`.

```python
result.entities                      # List[JointEntity]
result.relations                     # List[JointRelation]
result.entity("e2")                  # lookup by id
result.entities_by_type("person")
result.relations_by_type("works_for")
result.outgoing("e1")                # edges from Alice
result.incoming("e2")                # edges into Acme
result.neighbors("e2")               # adjacent entities
result.relations_of("e1")            # incident edges
result.to_dict(include_text=True)
```

Entity IDs are stable within one result. Relation `head` / `tail` are those IDs, not surface strings.

```python
for rel in result.relations:
    head = result.entity(rel.head)
    tail = result.entity(rel.tail)
    print(f"{head.text} -{rel.type}-> {tail.text}")
```

Optional NetworkX export:

```python
graph = result.to_networkx()  # requires networkx
```

## Decoding Config

```python
config = JointIEConfig(
    optimizer="beam",          # "beam" | "greedy" | "auto"
    beam_size=32,
    candidate_threshold=0.05,
    relation_role_threshold=0.05,
    top_k_entities=32,
    top_k_roles=12,
    relation_pair_cap=128,
    max_edges_per_type=256,
    include_confidence=True,
    include_spans=True,
    entity_threshold=None,
    max_len=None,
    batch_size=8,
)

result = joint.extract(text, schema, config=config)
```

| Field | Role |
| --- | --- |
| `optimizer="greedy"` | Faster, locally chooses high-scoring edges |
| `optimizer="beam"` | Searches a beam of partial graphs (default) |
| `candidate_threshold` | Mention candidate floor |
| `relation_role_threshold` | Head/tail role floor |
| `top_k_entities` / `top_k_roles` | Caps how many mentions/roles enter the ILP-style search |
| `relation_pair_cap` | Max scored pairs per relation type per sample |
| `entity_threshold` | Optional extra cut on entity probability |

Lower thresholds and larger `top_k_*` improve recall and cost more. Tune on a small labeled set.

Score without decoding if you want raw lattices:

```python
lattice = joint.score(text, schema, config=config)
```

Schemas are compiled and cached. Reuse the same `JointSchema` across calls.

## Feasibility

`result.feasible` is `False` when the decoder **could not satisfy the declared hard constraints** and fell back to an empty assignment. That is distinct from “the text contains no entities”.

```python
if not result.feasible:
    # constraints too tight, or candidates missing required arguments
    ...
elif not result.entities:
    # genuine empty extraction
    ...
```

If many documents come back infeasible, relax uniqueness, raise `top_k_entities`, or lower `candidate_threshold`.

## Long Documents

`extract_long` chunks the text, runs Joint IE per chunk, then remaps spans to **document** offsets.

```python
result = joint.extract_long(
    open("biography.txt").read(),
    schema,
    chunk_size=384,
    chunk_overlap=64,
    config=JointIEConfig(optimizer="beam"),
)
```

Important limits:

- A mention is kept only if its start and end fall in the **same chunk**.
- A relation is kept only if **both endpoints were extracted in the same chunk**. Cross-window edges are never synthesized.
- Duplicate `(type, start, end)` mentions across overlaps keep the higher-confidence copy.

Increase `chunk_overlap` when subject and object often sit on either side of a boundary. See [Long-Context Extraction](12-long_context.md).

## Batch Extraction

```python
texts = [
    "Alice works for Acme in Paris.",
    "Bob founded Globex.",
]
results = joint.batch_extract(texts, schema, config=JointIEConfig(batch_size=8))
```

Per-document schemas:

```python
results = joint.batch_extract(texts, [schema_a, schema_b])
```

## Best Practices

- Declare the closed set of entity types you actually want; Joint IE will not invent extra types.
- Put typing in `.relation(head, tail)` rather than filtering edges after the fact.
- Use `unique_head` / `at_most(..., per_head=1)` for 1-to-N facts (employer, capital, headquarters).
- Keep `no_self_loops()` on unless the domain truly has reflexive edges.
- Start with `optimizer="beam"` and modest `top_k_entities`; scale up if gold edges are missing.
- Always check `result.feasible` before treating an empty graph as “no facts”.
- Verify `text[e.start:e.end] == e.text` while bringing up a new schema.
- Prefer Joint IE when graph consistency matters; prefer `extract_relations` when you only need unlabeled head/tail strings.
