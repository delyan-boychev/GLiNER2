# GLiNER2.5 Constrained Classification Tutorial

Learn how to classify text with **hard cross-task constraints**. The convenience method `classify_text()` scores each task independently. `Classifier` scores the same GLiNER2.5 encoder, then decodes a globally consistent assignment.

Use this API when one label legally implies, forbids, or caps another — intents vs effects, topic vs audience, severity vs action, and similar structured taxonomies.

## Table of Contents
- [Setup](#setup)
- [When to Use Constrained Classification](#when-to-use-constrained-classification)
- [Tasks: Single, Multi, and Ordinal](#tasks-single-multi-and-ordinal)
- [Constraint DSL](#constraint-dsl)
- [End-to-End Example](#end-to-end-example)
- [Reading Results](#reading-results)
- [Decoding Controls](#decoding-controls)
- [Infeasible Assignments](#infeasible-assignments)
- [Label Descriptions, Instructions, and Examples](#label-descriptions-instructions-and-examples)
- [Batch and Long Documents](#batch-and-long-documents)
- [Best Practices](#best-practices)

## Setup

```python
from gliner2.classification import (
    Classifier,
    ClassificationSchema,
    ClassificationConfig,
)
from gliner2.classification import constraints as C
from gliner2.classification.errors import InfeasibleError, SchemaError

clf = Classifier.from_pretrained("fastino/gliner2.5-multi-v1")
```

Other sizes: `gliner2.5-base-v1`, `gliner2.5-small-v1`. See the [README model catalog](../README.md#-available-models).

`Classifier.from_pretrained` loads through `AutoExtractor`, so GLiNER2.5 boundary checkpoints work. Prediction knobs belong in `ClassificationConfig`, not in `from_pretrained`.

Independent (unconstrained) classification remains:

```python
from gliner2 import AutoExtractor

model = AutoExtractor.from_pretrained("fastino/gliner2.5-multi-v1")
model.classify_text(
    "Delete the temporary file",
    {"intent": {"labels": ["read", "write", "delete"]}},
)
```

That path does **not** enforce `implies` / `excludes` rules.

## When to Use Constrained Classification

| Use `classify_text` | Use `Classifier` |
| --- | --- |
| One task, or several independent tasks | Labels on task A legally constrain task B |
| You only need argmax / threshold | You need a feasible joint assignment |
| Quick prototypes | Policies, tool routing, compliance taxonomies |

Example invariants:

- If intent is `delete`, effects must include `delete`.
- If intent is `read`, effects cannot include `delete`.
- Pick at most two topics.
- Severity `critical` requires action `page_oncall`.

## Tasks: Single, Multi, and Ordinal

`ClassificationSchema` is a mutable builder. Declare tasks **before** constraints that mention them.

```python
schema = (
    ClassificationSchema()
    .single("intent", ["read", "write", "delete"])
    .multi("effects", ["read_only", "create", "modify", "delete"], min_labels=1)
    .ordinal("severity", ["low", "medium", "high", "critical"])
)
```

| Builder | Meaning | Defaults |
| --- | --- | --- |
| `.single(name, labels)` | Exactly one label | `min_labels=1`, `max_labels=1` |
| `.multi(name, labels, min_labels=0)` | Zero or more labels | Unbounded max unless you set `max_labels` |
| `.ordinal(name, labels)` | Ordered exclusive scale | Same cardinality as single; `ordered=True` |

Labels can be a list of strings or a `{label: description}` mapping:

```python
.single(
    "intent",
    {
        "read": "Inspect or preview without changing data",
        "write": "Create or update records",
        "delete": "Permanently remove data",
    },
)
```

Optional task kwargs: `threshold`, `instruction`, `examples`, `activation` (`"auto"`, `"sigmoid"`, `"softmax"`), `temperature`, `default`.

## Constraint DSL

Import the helpers from `gliner2.classification.constraints`.

A condition is usually a `(task, label)` pair, meaning “this label is selected on this task”.

```python
schema.constrain(
    C.implies(("intent", "delete"), ("effects", "delete")),
    C.excludes(("intent", "read"), ("effects", "delete")),
    C.at_most("effects", 2),
)
```

### Boolean

| Helper | Meaning |
| --- | --- |
| `C.implies(A, B)` | If A then B |
| `C.iff(A, B)` | A if and only if B |
| `C.excludes(A, B)` | A and B cannot both hold |
| `C.not_(A)` | A must not hold |
| `C.all_of(...)` | Conjunction |
| `C.any_of(...)` | Disjunction |
| `C.exactly_one_of(...)` | Exactly one of the expressions |

### Cardinality (any task)

| Helper | Meaning |
| --- | --- |
| `C.at_least(task, k)` | At least `k` labels on `task` |
| `C.at_most(task, k)` | At most `k` labels |
| `C.exactly(task, k)` | Exactly `k` labels |

### Ordinal (requires `.ordinal`)

| Helper | Meaning |
| --- | --- |
| `C.at_level(task, label)` | Severity is exactly this level |
| `C.min_level(task, label)` | At least this level |
| `C.max_level(task, label)` | At most this level |
| `C.between_level(task, lo, hi)` | Inclusive band |

### Selection predicates

| Helper | Meaning |
| --- | --- |
| `C.any_selected(task)` | At least one label is on |
| `C.any_other_selected(task)` | Some label other than the implied one is on |

Constraints are validated against tasks declared **so far**. This raises:

```python
ClassificationSchema().constrain(C.at_most("intent", 1))
# SchemaError: constraint references undeclared task 'intent'
```

## End-to-End Example

Route a file-operation request so intent and side effects stay consistent.

```python
schema = (
    ClassificationSchema()
    .single("intent", ["read", "write", "delete"])
    .multi(
        "effects",
        ["read_only", "create", "modify", "delete"],
        min_labels=1,
        max_labels=2,
    )
    .constrain(
        C.implies(("intent", "delete"), ("effects", "delete")),
        C.implies(("intent", "read"), ("effects", "read_only")),
        C.excludes(("intent", "read"), ("effects", "delete")),
        C.excludes(("intent", "read"), ("effects", "modify")),
    )
)

result = clf.classify("Delete the temporary file from /tmp", schema)

print(result.value("intent"))      # "delete"
print(result.selected("effects"))  # ("delete",)
print(result.feasible)             # True
print(result.to_dict())
# {
#   "intent": {
#     "value": "delete",
#     "confidence": 0.93,
#     "probabilities": {"read": 0.02, "write": 0.05, "delete": 0.93},
#   },
#   "effects": {
#     "value": ["delete"],
#     "confidence": 0.88,
#     "probabilities": {...},
#   },
#   "_meta": {"feasible": True, "decoder": "exact", "exact": True, ...},
# }
```

Without the constraint decoder, independent argmax could pick `intent=delete` and `effects=["read_only"]`. The constrained path forbids that.

## Reading Results

`classify` returns a `ClassificationResult`.

```python
result.value("intent")         # exclusive task → str
result.value("effects")        # multi task → tuple/list of labels
result.selected("effects")     # always the selected label tuple
result.confidence("intent")    # scalar confidence
result.probabilities("intent") # mapping label → probability
result.feasible                # False if constraints could not be fully met
result.violations              # constraint objects still broken, if any
result.to_dict(include_confidence=False)
# {"intent": "delete", "effects": ["delete"]}
```

`result["intent"]` is the per-task `TaskResult`. Unknown task names raise `SchemaError`.

## Decoding Controls

Pass a `ClassificationConfig` per call. Do not put these kwargs on `from_pretrained`.

```python
config = ClassificationConfig(
    decoder="auto",          # "auto" | "independent" | "exact" | "beam"
    beam_size=16,
    exact_node_budget=200_000,
    candidate_threshold=0.5,
    max_candidates_per_task=64,
    include_confidence=True,
    on_infeasible="relax",   # "relax" | "min_violations" | "raise"
    max_len=None,
    batch_size=8,
)

result = clf.classify(text, schema, config=config)
```

| `decoder` | Behavior |
| --- | --- |
| `"independent"` | Ignore cross-task constraints (same spirit as `classify_text`) |
| `"exact"` | Search a feasible assignment within `exact_node_budget` |
| `"beam"` | Beam search of size `beam_size` |
| `"auto"` | Prefer exact when the search is small enough, else beam |

Reuse the same schema object or an equivalent builder: compile results are cached by fingerprint (LRU, cap 128).

You can also split scoring and decoding:

```python
scores = clf.score(text, schema, config=config)
result = clf.decode(scores, schema, config=config)
```

## Infeasible Assignments

Sometimes no label combination satisfies every constraint (contradictory rules, or scores that leave no legal candidate).

```python
config = ClassificationConfig(on_infeasible="raise")
try:
    clf.classify(text, schema, config=config)
except InfeasibleError:
    ...
```

| `on_infeasible` | Behavior |
| --- | --- |
| `"relax"` (default) | Return the best effort assignment; check `result.feasible` |
| `"min_violations"` | Minimize broken constraints |
| `"raise"` | Raise `InfeasibleError` |

Always inspect `result.feasible` in production if you keep the default.

## Label Descriptions, Instructions, and Examples

```python
schema = (
    ClassificationSchema()
    .single(
        "ticket",
        {
            "billing": "Invoices, refunds, payment failures",
            "technical": "Bugs, outages, login failures",
            "sales": "Pricing questions and plan upgrades",
        },
        instruction="Classify the support ticket. Prefer technical when a bug is described.",
        examples=(
            ("I was double charged last month", "billing"),
            ("The app crashes on launch", "technical"),
        ),
        threshold=0.4,
    )
    .multi("product_area", ["ios", "android", "web", "api"])
    .constrain(
        C.excludes(("ticket", "sales"), ("product_area", "api")),
    )
)
```

`examples` are `(text, label)` pairs injected into the prompt. Reserved marker tokens such as `[P]`, `[C]`, `[E]` are banned in names and descriptions.

## Batch and Long Documents

```python
results = clf.batch_classify(
    ["Show the invoice", "Remove the cache files"],
    schema,
    config=ClassificationConfig(batch_size=8),
)
```

For documents longer than the encoder window, logits are aggregated across overlapping chunks, then decoded **once**:

```python
result = clf.classify_long(
    open("runbook.txt").read(),
    schema,
    chunk_size=384,
    chunk_overlap=64,
    aggregate="max",  # per-label aggregation across chunks
)
```

That keeps constraints global. Chunk-wise independent `classify_text_long` can disagree across pages and cannot enforce `implies`.

## Best Practices

- Declare every task before `.constrain(...)`.
- Keep exclusive decisions on `.single` / `.ordinal`; use `.multi` only when several labels can be true together.
- Encode policy in constraints instead of post-filtering independent argmax.
- Start with `decoder="auto"` and `on_infeasible="relax"`, then tighten.
- Use `{label: description}` maps for domain jargon.
- Compile is cached: rebuild the schema object only when tasks or constraints change.
- For a single unconstrained head, `model.classify_text` is enough and cheaper to reason about.
