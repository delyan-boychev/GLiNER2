# GLiNER2.5 Span Attributes Tutorial

Learn how to attach labels such as **sentiment** to extracted entity spans. Attributes are span-conditioned: the model first finds entities, then scores attribute labels at those exact spans.

This is different from document-level classification (`classify_text`). A review can be mixed overall while individual products are positive or negative.

## Table of Contents
- [Setup](#setup)
- [How Attributes Work](#how-attributes-work)
- [Sentiment on Product Mentions](#sentiment-on-product-mentions)
- [Restricting Attributes to Entity Types](#restricting-attributes-to-entity-types)
- [Single-Label vs Multi-Label Attributes](#single-label-vs-multi-label-attributes)
- [Multiple Attribute Groups](#multiple-attribute-groups)
- [Qualifying Labels](#qualifying-labels)
- [Combining with Other Schema Tasks](#combining-with-other-schema-tasks)
- [Reading the Output](#reading-the-output)
- [Best Practices](#best-practices)

## Setup

GLiNER2.5 is the boundary architecture. Load it with `AutoExtractor` so the checkpoint's `architecture` field selects `BoundaryExtractor`.

```python
from gliner2 import AutoExtractor, AttributeGroup

model = AutoExtractor.from_pretrained("fastino/gliner2.5-multi-v1")
```

Other GLiNER2.5 sizes: `gliner2.5-small-v1` (English, fast) and `gliner2.5-base-v1` (English). See the [README model catalog](../README.md#-available-models).

`entity_attributes()` must be called **after** `entities()`. Attribute group names cannot be `text`, `confidence`, `start`, or `end`.

## How Attributes Work

1. You declare content entity types (`product`, `company`, …).
2. You declare one or more `AttributeGroup`s (for example `sentiment`).
3. Attribute labels are encoded as internal queries. They are **not** returned as extra entity types.
4. After entity decoding, each retained span is force-scored for every applicable attribute group.
5. The chosen label(s) are attached to that entity object.

Single-label groups use softmax (one value). Multi-label groups use sigmoid plus `threshold`.

## Sentiment on Product Mentions

```python
text = "The new iPhone camera is excellent, but the battery life is disappointing."

schema = (
    model.create_schema()
    .entities(["product"])
    .entity_attributes({
        "sentiment": AttributeGroup(
            ["positive", "negative", "neutral"],
            applies_to=["product"],
            qualify_labels=True,
        )
    })
)

result = model.extract(
    text,
    schema,
    include_spans=True,
    include_confidence=True,
)
print(result)
# {
#   "entities": {
#     "product": [
#       {
#         "text": "iPhone camera",
#         "start": 8,
#         "end": 21,
#         "confidence": 0.91,
#         "sentiment": {"label": "positive", "confidence": 0.87},
#       },
#       {
#         "text": "battery life",
#         "start": 35,
#         "end": 47,
#         "confidence": 0.88,
#         "sentiment": {"label": "negative", "confidence": 0.84},
#       },
#     ]
#   }
# }
```

Always request spans while developing so you can check `text[start:end] == entity["text"]`.

```python
for entity in result["entities"]["product"]:
    assert text[entity["start"]:entity["end"]] == entity["text"]
    print(entity["text"], "→", entity["sentiment"]["label"])
```

## Restricting Attributes to Entity Types

`applies_to` limits which entity types receive the group. Company names in the example below are extracted without a sentiment field.

```python
schema = (
    model.create_schema()
    .entities(["product", "company"])
    .entity_attributes({
        "sentiment": AttributeGroup(
            ["positive", "negative", "neutral"],
            applies_to=["product"],
        )
    })
)

result = model.extract(
    "Apple's new AirPods are fantastic.",
    schema,
    include_spans=True,
    include_confidence=True,
)
# company spans have no "sentiment" key
# product spans include sentiment
```

If `applies_to` is omitted, the group is scored on **every** declared entity type. Unknown names in `applies_to` raise `ValueError`.

## Single-Label vs Multi-Label Attributes

### Single-label (default)

One mutually exclusive value, via softmax. Use this for sentiment, polarity, or status.

```python
AttributeGroup(
    ["positive", "negative", "neutral"],
    multi_label=False,  # default
)
```

Without `include_confidence`, the attribute is still a dict with `label` (and `confidence` is omitted from the parent entity, but attribute objects from the span path typically still carry a score when confidence is requested).

### Multi-label

Independent sigmoid decisions. Several labels can fire on the same span.

```python
schema = (
    model.create_schema()
    .entities(["product"])
    .entity_attributes({
        "aspects": AttributeGroup(
            ["price", "quality", "design", "battery", "support"],
            multi_label=True,
            threshold=0.4,
            applies_to=["product"],
        )
    })
)

result = model.extract(
    "The Pixel fold is expensive but the build quality and design are outstanding.",
    schema,
    include_spans=True,
    include_confidence=True,
)
# "aspects": [
#   {"label": "price", "confidence": 0.71},
#   {"label": "quality", "confidence": 0.82},
#   {"label": "design", "confidence": 0.79},
# ]
```

Lower `threshold` to recall more aspects; raise it to reduce noise.

## Multiple Attribute Groups

A span can carry several independent groups. Labels must be unique across groups. If a label string would collide with an entity type or another group, set `qualify_labels=True`.

```python
schema = (
    model.create_schema()
    .entities({
        "product": "Consumer devices or software products",
        "company": "Company or brand names",
    })
    .entity_attributes({
        "sentiment": AttributeGroup(
            ["positive", "negative", "neutral"],
            applies_to=["product"],
            qualify_labels=True,
        ),
        "urgency": AttributeGroup(
            ["low", "medium", "high"],
            applies_to=["product"],
            qualify_labels=True,
        ),
    })
)

result = model.extract(
    "The charger failed on day one and we need a replacement immediately.",
    schema,
    include_spans=True,
    include_confidence=True,
)
```

Each product span then includes both `sentiment` and `urgency`.

## Qualifying Labels

`qualify_labels=True` prefixes model-facing labels with the group name (`sentiment: positive`) while returning the short label (`positive`) in the result. Use it when:

- the same word is both an entity type and an attribute (`status`, `type`, `label`)
- two groups share similar vocabulary
- you want the encoder to see an unambiguous query

```python
AttributeGroup(
    ["positive", "negative", "neutral"],
    qualify_labels=True,
)
```

If an unqualified attribute label collides with an entity type, schema construction raises:

```text
Attribute labels collide with entity labels: ...; use qualify_labels=True
```

## Combining with Other Schema Tasks

Attributes compose with classification, relations, and structures in one `extract` call.

```python
schema = (
    model.create_schema()
    .entities(["product", "company"])
    .entity_attributes({
        "sentiment": AttributeGroup(
            ["positive", "negative", "neutral"],
            applies_to=["product"],
            qualify_labels=True,
        )
    })
    .classification("review_type", ["unboxing", "complaint", "comparison", "praise"])
)

result = model.extract(
    "Unboxing the new Surface Laptop: the keyboard is a joy, Microsoft nailed it.",
    schema,
    include_spans=True,
    include_confidence=True,
)
```

Document-level `review_type` is independent of per-span `sentiment`.

For documents longer than the model window, use `extract_long` with the same schema. See [Long-Context Extraction](12-long_context.md).

## Reading the Output

| `include_spans` | `include_confidence` | Entity object |
| --- | --- | --- |
| `True` | `True` | `{text, start, end, confidence, sentiment: {label, confidence}}` |
| `True` | `False` | `{text, start, end, sentiment: ...}` |
| `False` | `True` | `{text, confidence, sentiment: ...}` |

Single-label attribute value:

```python
entity["sentiment"]["label"]          # "positive"
entity["sentiment"]["confidence"]     # 0.87
```

Multi-label attribute value:

```python
[item["label"] for item in entity["aspects"]]
```

Attributes are only attached to **retained** entities. If a mention is dropped by threshold or overlap policy, it is not attributed.

## Best Practices

- Call `entities()` first, then `entity_attributes()`.
- Use `applies_to` so companies, dates, and locations are not given product sentiment.
- Prefer `qualify_labels=True` in multi-task schemas.
- Keep attribute vocabularies short and mutually exclusive for single-label groups.
- Request `include_spans=True` until offsets are verified against the source text.
- Do not reuse reserved names (`text`, `start`, `end`, `confidence`) as group names.
- For document-level sentiment of the whole review, use classification instead of (or in addition to) span attributes.
- Boundary checkpoints score attributes at decoded spans; they are not a second NER pass and will not invent extra entity types in the public output.
