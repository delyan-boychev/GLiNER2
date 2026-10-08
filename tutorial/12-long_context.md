# GLiNER2.5 Long-Context Extraction Tutorial

Learn how to extract information from documents that are longer than the model's encoded window. Standard `extract(...)` with `max_len` **truncates** to the first N word tokens. Long-context APIs **scan** the full document with overlapping chunks and remap spans to global character offsets.

This applies to GLiNER2 span checkpoints and GLiNER2.5 boundary checkpoints. Boundary models can extract arbitrarily long spans **inside one encoded window**; they still cannot join a span whose start and end never co-occur in the same chunk.

## Table of Contents
- [Why Use Long-Context Extraction](#why-use-long-context-extraction)
- [Setup](#setup)
- [How Chunking Works](#how-chunking-works)
- [Entity Extraction](#entity-extraction)
- [Choosing Chunk Settings](#choosing-chunk-settings)
- [Overlap Policy](#overlap-policy)
- [Full Schema Extraction](#full-schema-extraction)
- [Task-Specific Long APIs](#task-specific-long-apis)
- [Span Attributes on Long Documents](#span-attributes-on-long-documents)
- [Classification on Long Documents](#classification-on-long-documents)
- [Relations and Joint IE](#relations-and-joint-ie)
- [Batch Long-Document Extraction](#batch-long-document-extraction)
- [Limits You Should Not Ignore](#limits-you-should-not-ignore)
- [Best Practices](#best-practices)

## Why Use Long-Context Extraction

Use long-context extraction for reports, contracts, support tickets, transcripts, PDFs converted to text, logs, and other multi-page documents.

It:

1. Splits the document into overlapping **word** chunks (`chunk_size`, `chunk_overlap`).
2. Runs normal GLiNER inference on each chunk.
3. Remaps chunk-local character spans back to the original document.
4. Merges duplicate detections from overlapping regions.

Do **not** use `extract_entities(..., max_len=512)` for this. `max_len` drops the rest of the file.

## Setup

```python
from gliner2 import AutoExtractor, AttributeGroup

# GLiNER2.5 boundary multi-task checkpoint (see README for small/base/multi sizes)
model = AutoExtractor.from_pretrained("fastino/gliner2.5-multi-v1")
```

Span-architecture models work the same way:

```python
from gliner2 import GLiNER2

extractor = GLiNER2.from_pretrained("fastino/gliner2-base-v1")
```

The examples below use `model` from `AutoExtractor`.

## How Chunking Works

Chunks are counted in **whitespace word tokens**, not characters or subwords.

```
words:     0        64              384             448
chunk 1:   |------------------------|
chunk 2:            |------------------------------|
           overlap (64 words)
```

- `chunk_size=384` — each window has at most 384 words.
- `chunk_overlap=64` — the next window starts 320 words later.
- Keep `chunk_overlap < chunk_size`.

Offsets in the result are **global**. If `include_spans=True`:

```python
assert long_text[entity["start"]:entity["end"]] == entity["text"]
```

Classification predictions are aggregated across chunks (higher confidence wins when scores exist). Span tasks are merged by position.

By default, chunks are counted with the `"whitespace"` word splitter (the same strategy used to train public checkpoints). Long-document APIs use the splitter attached to the loaded model, so `word_splitter="char"` also changes chunk boundaries. That character-level splitter is suitable for languages without whitespace-delimited words, such as Chinese:

```python
model = AutoExtractor.from_pretrained(
    "fastino/gliner2.5-multi-v1",
    word_splitter="char",
)
# or after loading
model.set_word_splitter("char")
```

Custom callables must yield `(token, start, end)` with exclusive-end offsets into the original text. Changing a pretrained model's word boundaries can affect quality unless training used the same splitter.

## Entity Extraction

```python
long_text = """
Annual report background and financial commentary...

Apple CEO Tim Cook announced Vision Pro updates in Cupertino
during September 2025.

More report text continues for many pages...
"""

result = model.extract_entities_long(
    long_text,
    ["company", "person", "product", "location", "date"],
    chunk_size=384,
    chunk_overlap=64,
    include_spans=True,
    include_confidence=True,
)

print(result)
# {
#     "entities": {
#         "company": [{"text": "Apple", "confidence": 0.99, "start": 52, "end": 57}],
#         "person": [{"text": "Tim Cook", "confidence": 0.99, "start": 62, "end": 70}],
#         "product": [{"text": "Vision Pro", "confidence": 0.99, "start": 81, "end": 91}],
#         "location": [{"text": "Cupertino", "confidence": 0.99, "start": 103, "end": 112}],
#         "date": [{"text": "September 2025", "confidence": 0.99, "start": 120, "end": 134}],
#     }
# }
```

Descriptions still help on noisy long text:

```python
entity_types = {
    "contract_party": "Companies or people that are parties to the agreement",
    "effective_date": "Dates when the agreement starts or becomes valid",
    "termination_clause": "Text describing when or how the agreement can end",
}

result = model.extract_entities_long(
    contract_text,
    entity_types,
    chunk_size=512,
    chunk_overlap=96,
    include_spans=True,
)
```

## Choosing Chunk Settings

| Goal | Starting point |
| --- | --- |
| General prose | `chunk_size=384`, `chunk_overlap=64` |
| Mentions often sit on boundaries | raise overlap to 96–128 |
| Faster inference | smaller `chunk_size`, smaller overlap |
| Long noun phrases / GLiNER2.5 wide spans | keep the whole phrase inside one chunk; overlap still cannot invent a span that never fits |

GLiNER2.5 can extract a span of any length **inside one chunk**. A 200-word clause is fine if `chunk_size` is at least 200. A clause that starts in chunk 1 and ends only in chunk 2 will be missed.

## Overlap Policy

All local long-document methods accept `overlap_policy`. `None` keeps the architecture default.

| Policy | Behavior |
| --- | --- |
| `allow` | Keep distinct overlapping spans |
| `nested` | Permit containment |
| `flat` / `disallow` | Remove overlaps deterministically |
| `longest` | Drop strictly contained shorter spans |

```python
result = model.extract_entities_long(
    long_text,
    ["person", "organization"],
    chunk_size=384,
    chunk_overlap=64,
    include_spans=True,
    overlap_policy="nested",
)
```

GLiNER2 deduplicates **overlap artifacts** from adjacent chunks. Distinct mentions at different document positions are kept even if the surface string repeats.

## Full Schema Extraction

Use `extract_long` with any schema the short API accepts.

```python
schema = (
    model.create_schema()
    .entities({
        "company": "Company names",
        "person": "Names of people",
        "product": "Product names",
    })
    .classification("document_type", ["report", "contract", "email", "support_ticket"])
)

result = model.extract_long(
    long_text,
    schema,
    chunk_size=384,
    chunk_overlap=64,
    include_spans=True,
    include_confidence=True,
)
```

Span outputs are merged. Classification is aggregated across chunk predictions.

## Task-Specific Long APIs

```python
classification = model.classify_text_long(
    long_text,
    {"document_type": ["report", "contract", "email"]},
)

relations = model.extract_relations_long(
    long_text,
    ["works_for", "located_in"],
    chunk_size=384,
    chunk_overlap=64,
    include_spans=True,
)

structured = model.extract_json_long(
    long_text,
    {
        "invoice": [
            "vendor::str",
            "invoice_number::str",
            "line_item::list",
        ]
    },
    include_spans=True,
)
```

Matching batch methods:

- `batch_extract_entities_long`
- `batch_extract_long`
- `batch_classify_text_long`
- `batch_extract_relations_long`
- `batch_extract_json_long`

`format_results=True` is required for the long batch APIs.

## Span Attributes on Long Documents

Attribute groups work on `extract_long` because each chunk runs the same schema. Attributes stay attached to globally remapped spans. See [Span Attributes](13-span_attributes.md).

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
)

result = model.extract_long(
    long_text,
    schema,
    chunk_size=384,
    chunk_overlap=64,
    include_spans=True,
    include_confidence=True,
)

for item in result["entities"].get("product", []):
    print(item["text"], item.get("sentiment"))
    assert long_text[item["start"]:item["end"]] == item["text"]
```

## Classification on Long Documents

### Independent heads

`classify_text_long` / schema classification in `extract_long` aggregates per-chunk scores (prefers higher confidence when available). Good for document type, language, or overall topic.

### Constrained classification

Do **not** stitch independent chunk labels if tasks constrain each other. Use `Classifier.classify_long`, which aggregates logits then decodes **once**:

```python
from gliner2.classification import Classifier, ClassificationSchema, ClassificationConfig
from gliner2.classification import constraints as C

clf = Classifier.from_pretrained("fastino/gliner2.5-multi-v1")
schema = (
    ClassificationSchema()
    .single("intent", ["read", "write", "delete"])
    .multi("effects", ["read_only", "create", "modify", "delete"], min_labels=1)
    .constrain(C.implies(("intent", "delete"), ("effects", "delete")))
)

result = clf.classify_long(
    long_text,
    schema,
    chunk_size=384,
    chunk_overlap=64,
    aggregate="max",
    config=ClassificationConfig(),
)
```

See [Constrained Classification](14-constrained_classification.md).

## Relations and Joint IE

### Unconstrained relations

```python
rels = model.extract_relations_long(
    long_text,
    ["works_for", "located_in", "founded"],
    chunk_size=384,
    chunk_overlap=96,  # more overlap helps head/tail near boundaries
    include_spans=True,
)
```

A pair is found only if head and tail appear in the **same chunk**.

### Joint IE

```python
from gliner2.joint_ie import JointIE, JointIEConfig

joint = JointIE.from_pretrained("fastino/gliner2.5-multi-v1")
schema = (
    joint.create_schema()
    .entities(["person", "organization", "location"])
    .relation("works_for", "person", "organization", unique_head=True)
    .relation("located_in", "organization", "location")
    .no_self_loops()
)

result = joint.extract_long(
    long_text,
    schema,
    chunk_size=384,
    chunk_overlap=96,
    config=JointIEConfig(optimizer="beam"),
)

for rel in result.relations:
    head = result.entity(rel.head)
    tail = result.entity(rel.tail)
    print(head.text, rel.type, tail.text)
```

`JointIE.extract_long` never creates cross-chunk edges. Duplicate mentions across overlaps keep the higher-confidence copy. See [Joint Information Extraction](15-joint_ie.md).

## Batch Long-Document Extraction

Same schema, many files:

```python
documents = [
    open("report_2024.txt").read(),
    open("report_2025.txt").read(),
]

results = model.batch_extract_entities_long(
    documents,
    ["company", "person", "product", "location", "date"],
    batch_size=8,
    chunk_size=384,
    chunk_overlap=64,
    include_spans=True,
)

for doc, result in zip(documents, results):
    for entities in result["entities"].values():
        for entity in entities:
            assert doc[entity["start"]:entity["end"]] == entity["text"]
```

Per-document schemas:

```python
schemas = [
    model.create_schema().entities(["company", "date"]),
    model.create_schema().entities(["person", "location"]),
]

results = model.batch_extract_long(
    documents,
    schemas,
    chunk_size=384,
    chunk_overlap=64,
    include_spans=True,
)
```

## Limits You Should Not Ignore

| Supported | Not supported |
| --- | --- |
| Arbitrary span length **inside one chunk** (GLiNER2.5) | A span whose start and end never share a chunk |
| Relations whose both arguments sit in one chunk | Relations that cross chunk boundaries |
| Global `[start, end)` offsets into the original string | Treating chunk-local offsets as document offsets |
| Deduping overlap copies of the same mention | Assuming repeated surface forms are collapsed document-wide |

If a fact is systematically split (name at the end of a page, employer at the start of the next), increase `chunk_overlap` or pre-segment into sentences/paragraphs that contain both arguments.

## Best Practices

- Prefer `*_long` methods over `max_len` truncation whenever the file can exceed the window.
- Start with `chunk_size=384`, `chunk_overlap=64`; raise overlap for relations and multi-token names.
- Request `include_spans=True` until every offset slices back to the expected substring.
- Use label descriptions on long, repetitive documents to cut generic false positives.
- For constrained labels, aggregate then decode (`Classifier.classify_long`), do not majority-vote chunk labels.
- For graphs, use `JointIE.extract_long` and accept that edges are intra-chunk.
- Keep long-context calls explicit in production so latency (chunks × inference) stays visible.
- Tune `threshold`, `chunk_size`, and `chunk_overlap` on a small domain sample, not only on short sentences.
