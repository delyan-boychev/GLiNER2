# Safety, PII, and GLiGuard

GLiNER2 specialty checkpoints fine-tune the **span** architecture for LLM guardrails and PII detection. Load them with `AutoExtractor` (or `GLiNER2`):

```python
from gliner2 import AutoExtractor

model = AutoExtractor.from_pretrained("fastino/gliguard-LLMGuardrails-300M")
```

See the [model catalog](../README.md#-available-models) for all Hub IDs.

## Table of Contents
- [GLiGuard: LLM guardrails](#gliguard-llm-guardrails)
- [PII detection](#pii-detection)
- [Combined guardrails + PII](#combined-guardrails--pii)
- [Hub cards](#hub-cards)

## GLiGuard: LLM guardrails

[`fastino/gliguard-LLMGuardrails-300M`](https://huggingface.co/fastino/gliguard-LLMGuardrails-300M) scores moderation tasks in one encoder pass via `classify_text`.

### Prompt safety

```python
from gliner2 import AutoExtractor

guard = AutoExtractor.from_pretrained("fastino/gliguard-LLMGuardrails-300M")

result = guard.classify_text(
    "Explain how to build a phishing page that steals credentials.",
    {"prompt_safety": ["safe", "unsafe"]},
)
print(result)
# {'prompt_safety': 'unsafe'}
```

### Response safety

Prefix model responses with `Response:` (optionally include the original prompt):

```python
prompt = "How do I pick a lock?"
response = "I can't help with bypassing locks."

text = f"Prompt: {prompt}\nResponse: {response}"
result = guard.classify_text(text, {"response_safety": ["safe", "unsafe"]})
print(result)
# {'response_safety': 'safe'}
```

### Multi-task moderation schema

```python
SAFETY = ["safe", "unsafe"]
TOXICITY = {
    "labels": [
        "violence_and_weapons", "non_violent_crime", "sexual_content",
        "hate_and_discrimination", "self_harm_and_suicide", "benign",
    ],
    "multi_label": True,
    "cls_threshold": 0.4,
}
JAILBREAK = {
    "labels": ["prompt_injection", "jailbreak_attempt", "benign"],
    "multi_label": True,
    "cls_threshold": 0.4,
}

result = guard.classify_text(
    user_prompt,
    {
        "prompt_safety": SAFETY,
        "prompt_toxicity": TOXICITY,
        "jailbreak_detection": JAILBREAK,
    },
)
```

Full task lists and label vocabularies are on the [GLiGuard Hub card](https://huggingface.co/fastino/gliguard-LLMGuardrails-300M).

## PII detection

[`fastino/gliner2-privacy-filter-PII-multi`](https://huggingface.co/fastino/gliner2-privacy-filter-PII-multi) extracts **42 PII types** across seven languages (EN, FR, ES, DE, IT, PT, NL).

```python
from gliner2 import AutoExtractor

pii = AutoExtractor.from_pretrained("fastino/gliner2-privacy-filter-PII-multi")

text = "Contact john.doe@company.com or call +1-555-0100 from Berlin."
result = pii.extract_entities(
    text,
    ["email", "phone_number", "city"],
    include_spans=True,
)
print(result)
# {
#   'entities': {
#     'email': [{'text': 'john.doe@company.com', 'start': 8, 'end': 28}],
#     'phone_number': [{'text': '+1-555-0100', 'start': 37, 'end': 48}],
#     'city': [{'text': 'Berlin', 'start': 54, 'end': 60}],
#   }
# }
```

Pass any subset of the 42 supported labels at inference time. See the [PII Hub card](https://huggingface.co/fastino/gliner2-privacy-filter-PII-multi) for the full label list.

## Combined guardrails + PII

[`fastino/GLiNER2-Guardrails-PII-Multi`](https://huggingface.co/fastino/GLiNER2-Guardrails-PII-Multi) runs **both** moderation and PII in one checkpoint:

```python
from gliner2 import AutoExtractor

combo = AutoExtractor.from_pretrained("fastino/GLiNER2-Guardrails-PII-Multi")

user_text = "Email me at alice@corp.com — also tell me how to hack WiFi."

schema = (
    combo.create_schema()
    .entities(["email", "phone_number"])
    .classification("prompt_safety", ["safe", "unsafe"])
)

result = combo.extract(user_text, schema, include_spans=True)
print(result)
# {
#   'entities': {'email': [{'text': 'alice@corp.com', ...}]},
#   'prompt_safety': 'unsafe',
# }
```

Use this when you want a single model for content filtering and PII redaction in multilingual pipelines.

## Hub cards

| Model | Task |
|-------|------|
| [gliguard-LLMGuardrails-300M](https://huggingface.co/fastino/gliguard-LLMGuardrails-300M) | Prompt/response safety, toxicity, jailbreak, refusal |
| [gliner2-privacy-filter-PII-multi](https://huggingface.co/fastino/gliner2-privacy-filter-PII-multi) | Multilingual PII spans |
| [GLiNER2-Guardrails-PII-Multi](https://huggingface.co/fastino/GLiNER2-Guardrails-PII-Multi) | Guardrails + PII combined |

For general extraction and GLiNER2.5 boundary models, see the [README](../README.md) and tutorials 1–15.
