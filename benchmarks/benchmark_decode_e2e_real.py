#!/usr/bin/env python3
"""End-to-end real-text benchmark for accelerator decode synchronization.

The default run uses 512 natural-language documents, heterogeneous schemas,
variable document and encoder lengths, CUDA FP16, and the repository's public
``compile=True`` loading option.  It compares:

* ``default``: F.linear scoring and the original per-value synchronized decoder;
* ``optimized``: the same scoring with synchronization-collapsed decoding.

Every warmup and measured run must produce exactly identical formatted output.
No synthetic tensors or repeated-token padding are used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence, Tuple

import torch

from gliner2 import GLiNER2
from gliner2.inference.schema import AttributeGroup
from gliner2.training.trainer import ExtractorCollator


REAL_PARAGRAPHS = (
    "Apple chief executive Tim Cook introduced a new generation of iPhone hardware at the company's Cupertino campus. The presentation focused on battery life, camera improvements, and satellite safety features, while executives confirmed that the devices would reach stores later in September.",
    "Microsoft opened a cloud engineering center in Warsaw after two years of planning with local universities. The company said the facility will employ software engineers, security researchers, and customer-support specialists serving organizations across Central Europe.",
    "A regional hospital in Manchester expanded its cardiology unit with three new operating rooms and an outpatient diagnostics center. Dr. Helen Murray said the project should shorten waiting times and allow more patients to receive follow-up care close to home.",
    "NVIDIA presented its latest data-center processors during a developer conference in San Jose. Several research laboratories said they plan to use the systems for climate modeling, medical imaging, and large-scale language-model experiments.",
    "The European Central Bank left interest rates unchanged following its meeting in Frankfurt. President Christine Lagarde said officials would continue to review inflation, wage growth, and business investment before making another policy decision.",
    "Toyota will add a battery assembly line to its manufacturing plant in Kentucky. Construction is expected to begin in October, and the automaker estimates that the expansion will create more than four hundred permanent jobs.",
    "Researchers at Stanford University published a study of coastal groundwater levels in California. The team combined satellite observations with measurements from local wells and found that seasonal changes were larger than previous models had predicted.",
    "Amazon Web Services announced a new cloud region for customers in Thailand. The project includes three availability zones and is intended to support banks, retailers, public agencies, and technology startups that must keep data within the country.",
    "A passenger train traveling from Paris to Lyon was delayed after heavy rain damaged signaling equipment outside Dijon. Rail operator SNCF arranged replacement buses and advised passengers to check updated departure times before traveling.",
    "The city of Toronto approved funding for two hundred electric buses and charging equipment at three depots. Transit officials expect the first vehicles to enter service next spring, beginning with routes that pass schools and major hospitals.",
    "OpenAI and a group of independent publishers announced a research program examining how readers discover cited material in conversational systems. The participants will evaluate attribution formats, link placement, and methods for correcting outdated references.",
    "A conservation team relocated twelve sea turtles from a rehabilitation center in Queensland to a protected beach near Cairns. Veterinarians had treated the animals for injuries caused by fishing lines and plastic debris.",
    "Samsung launched a compact foldable phone in Seoul with an upgraded hinge and a brighter exterior display. The company will offer the device in four colors, with prices starting at nine hundred and ninety-nine dollars.",
    "The University of Edinburgh created a scholarship fund for students studying renewable energy and environmental engineering. Initial donations from alumni and local businesses will support thirty undergraduate awards during the next academic year.",
    "Pfizer began a late-stage clinical study of an experimental influenza vaccine at sites in the United States, Brazil, and South Africa. Investigators plan to enroll more than ten thousand adults before the northern hemisphere flu season.",
    "The British Museum placed a collection of restored Roman coins on public display in London. Conservators spent eighteen months removing corrosion and documenting inscriptions that identify several previously unknown regional mints.",
    "A farming cooperative near Valencia installed solar panels above irrigation canals to generate electricity while reducing water loss. Engineers estimate that the pilot system can supply nearly half of the energy used by nearby pumping stations.",
    "Netflix acquired worldwide distribution rights to an independent documentary filmed in Iceland. The production follows rescue teams, geologists, and residents during a year of volcanic activity on the Reykjanes peninsula.",
    "SpaceX launched a communications satellite from Cape Canaveral early Tuesday morning. The Falcon 9 first stage returned to a drone ship in the Atlantic Ocean, while the payload continued toward geostationary transfer orbit.",
    "A Berlin software company raised twenty-five million euros to expand its fraud-detection platform. The financing round was led by Northbridge Capital and included existing investors from Germany and the Netherlands.",
    "Japan's national weather agency issued heat warnings for Tokyo, Osaka, and several surrounding prefectures. Officials asked residents to limit outdoor activity, drink water regularly, and check on elderly neighbors living alone.",
    "Ford recalled a group of sport utility vehicles after engineers identified a fault in the rear camera wiring. Owners will receive letters explaining how dealerships can inspect and replace the affected component without charge.",
    "Archaeologists working near Alexandria uncovered part of a residential district dating to the second century. The excavation revealed painted walls, ceramic workshops, storage rooms, and a street leading toward the ancient harbor.",
    "The World Health Organization delivered emergency medical supplies to clinics affected by flooding in northern Mozambique. The shipment contained antibiotics, water-purification tablets, protective equipment, and treatment kits for severe dehydration.",
)


ENTITY_TYPES = (
    "person", "organization", "location", "date", "product", "event",
    "money", "percentage", "job title", "facility", "law", "language",
    "nationality", "medical condition", "medication", "chemical", "vehicle",
    "artwork", "book", "film", "award", "academic institution",
    "government agency", "sports team", "political party", "technology",
    "scientific concept", "email address", "phone number", "URL", "address",
    "quantity", "duration", "time", "country", "city", "state or province",
    "river", "mountain", "airport", "train station", "company division",
    "research laboratory", "university", "hospital", "currency", "stock symbol",
    "disease", "medical procedure", "gene", "protein", "software",
    "programming language", "hardware", "cloud service", "dataset", "publication",
    "legal case", "contract", "government program", "energy source", "animal",
    "plant", "food",
)


VARIANTS = {
    "default": False,
    "optimized": True,
}


def parse_int_list(value: str) -> List[int]:
    parsed = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not parsed or any(item <= 0 for item in parsed):
        raise argparse.ArgumentTypeError("expected comma-separated positive integers")
    return parsed


def percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(fraction * (len(ordered) - 1)))
    return ordered[index]


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def load_corpus(path: Path) -> List[str]:
    documents = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        if line.startswith("{"):
            value = json.loads(line)
            text = value.get("text")
            if not isinstance(text, str):
                raise ValueError(f"{path}:{line_number} has no string 'text' field")
            documents.append(text.strip())
        else:
            documents.append(line)
    if not documents:
        raise ValueError(f"no documents found in {path}")
    return documents


def build_documents(count: int, corpus: Sequence[str]) -> List[str]:
    """Build varied documents from whole natural-language paragraphs."""
    documents = []
    paragraph_counts = (1, 2, 1, 3, 2, 4, 1, 5)
    for index in range(count):
        paragraph_count = paragraph_counts[index % len(paragraph_counts)]
        start = (index * 5 + index // len(paragraph_counts)) % len(corpus)
        paragraphs = [
            corpus[(start + offset) % len(corpus)]
            for offset in range(paragraph_count)
        ]
        documents.append("\n\n".join(paragraphs))
    return documents


def build_schema_profiles(model) -> List[Tuple[str, Any]]:
    entity_4 = model.create_schema().entities(list(ENTITY_TYPES[:4]))
    entity_16 = model.create_schema().entities(list(ENTITY_TYPES[:16]))
    entity_32 = model.create_schema().entities(list(ENTITY_TYPES[:32]))
    entity_64 = model.create_schema().entities(list(ENTITY_TYPES[:64]))

    classifications = (
        model.create_schema()
        .classification(
            "topic",
            ["technology", "business", "health", "science", "public policy", "culture"],
        )
        .classification(
            "signals",
            ["announcement", "financial event", "research finding", "public warning"],
            multi_label=True,
            cls_threshold=0.35,
        )
    )
    relations = model.create_schema().relations(
        {
            "works_for": {"threshold": 0.4},
            "located_in": {"threshold": 0.4},
            "announced": {"threshold": 0.4},
            "acquired": {"threshold": 0.4},
        }
    )
    structures = (
        model.create_schema()
        .structure("announcement")
        .field("organization", dtype="str")
        .field("subject", dtype="str")
        .field("location", dtype="list")
        .field("date", dtype="str")
        .field("amount", dtype="list")
        .field(
            "status", dtype="str",
            choices=["planned", "ongoing", "completed", "cancelled"],
        )
    )
    mixed = (
        model.create_schema()
        .entities(list(ENTITY_TYPES[:12]))
        .classification(
            "document type",
            ["company news", "research", "health", "transport", "government"],
        )
        .relations(["works_for", "located_in", "announced"], threshold=0.4)
        .structure("event summary")
        .field("actor", dtype="str")
        .field("action", dtype="str")
        .field("place", dtype="list")
        .field("time", dtype="str")
    )
    attributes = (
        model.create_schema()
        .entities(["person", "organization", "product"])
        .entity_attributes({
            "role": AttributeGroup(
                ["executive", "researcher", "official", "participant"],
                qualify_labels=True,
            ),
            "prominence": AttributeGroup(
                ["primary", "secondary"],
                multi_label=True,
                threshold=0.35,
                qualify_labels=True,
            ),
        })
    )
    return [
        ("entities_q4", entity_4),
        ("entities_q16", entity_16),
        ("entities_q32", entity_32),
        ("entities_q64", entity_64),
        ("classifications", classifications),
        ("relations", relations),
        ("structures", structures),
        ("mixed", mixed),
        ("entity_attributes", attributes),
    ]


def digest(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    return hashlib.sha256(payload).hexdigest()


def assert_exact(reference: List[Dict], candidate: List[Dict], context: str) -> None:
    if reference == candidate:
        return
    if len(reference) != len(candidate):
        raise AssertionError(
            f"{context}: result count differs "
            f"(default={len(reference)}, optimized={len(candidate)})"
        )
    mismatch = next(
        index
        for index, (left, right) in enumerate(zip(reference, candidate))
        if left != right
    )
    raise AssertionError(
        f"{context}: formatted output mismatch at document {mismatch}\n"
        f"default={reference[mismatch]}\n"
        f"candidate={candidate[mismatch]}"
    )


def measure(
    function: Callable[[], List[Dict]],
    device: torch.device,
) -> Tuple[List[Dict], Dict[str, float]]:
    synchronize(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    output = function()
    synchronize(device)
    wall_ms = (time.perf_counter() - started) * 1_000
    row = {"wall_ms": wall_ms}
    if device.type == "cuda":
        row.update({
            "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 2**20,
            "peak_reserved_mib": torch.cuda.max_memory_reserved(device) / 2**20,
        })
    return output, row


def summarize(samples: Sequence[Dict[str, float]], document_count: int) -> Dict[str, float]:
    wall = [sample["wall_ms"] for sample in samples]
    result = {
        "median_ms": statistics.median(wall),
        "p90_ms": percentile(wall, 0.90),
        "p95_ms": percentile(wall, 0.95),
        "documents_per_second": document_count * 1_000 / statistics.median(wall),
        "samples_ms": wall,
    }
    if "peak_allocated_mib" in samples[0]:
        result.update({
            "peak_allocated_mib": max(sample["peak_allocated_mib"] for sample in samples),
            "peak_reserved_mib": max(sample["peak_reserved_mib"] for sample in samples),
        })
    return result


def length_summary(values: Sequence[int]) -> Dict[str, float]:
    return {
        "min": min(values),
        "median": statistics.median(values),
        "p90": percentile(values, 0.90),
        "p95": percentile(values, 0.95),
        "max": max(values),
        "mean": statistics.mean(values),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--model", default="fastino/gliner2-base-v1")
    parser.add_argument("--device", choices=("cuda", "mps", "cpu"), default="cuda")
    parser.add_argument("--documents", type=int, default=512)
    parser.add_argument("--batch-sizes", type=parse_int_list, default=parse_int_list("8,16,32"))
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--execution-mode", choices=("compile", "eager"), default="compile")
    parser.add_argument(
        "--corpus", type=Path,
        help="optional JSONL with a text field, or plain text with one document per line",
    )
    parser.add_argument("--output", type=Path, default=Path("decode_e2e_cuda_fp16.json"))
    args = parser.parse_args()

    if args.documents <= 0 or args.warmup <= 0 or args.iterations <= 0:
        parser.error("documents, warmup, and iterations must be positive")
    if not 0 <= args.threshold <= 1:
        parser.error("threshold must be between zero and one")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable")
    if args.device == "mps" and not torch.backends.mps.is_available():
        parser.error("MPS is unavailable")

    device = torch.device(args.device)
    compile_model = args.execution_mode == "compile"
    print(
        f"loading model={args.model} device={device} dtype=fp16 "
        f"compile={compile_model}"
    )
    model = GLiNER2.from_pretrained(
        args.model,
        map_location=args.device,
        quantize=True,
        compile=compile_model,
    ).eval()

    corpus = load_corpus(args.corpus) if args.corpus else list(REAL_PARAGRAPHS)
    texts = build_documents(args.documents, corpus)
    profiles = build_schema_profiles(model)
    profile_names = [profiles[index % len(profiles)][0] for index in range(len(texts))]
    schemas = [profiles[index % len(profiles)][1] for index in range(len(texts))]
    profile_counts = dict(Counter(profile_names))

    tokenizer_output = model.processor.tokenizer(
        texts, add_special_tokens=False, truncation=False
    )
    document_token_lengths = [len(ids) for ids in tokenizer_output["input_ids"]]

    schema_dicts, _ = model._build_schema_dicts_and_metadata(schemas)
    collator = ExtractorCollator(
        model.processor, is_training=False, architecture=model.architecture
    )
    encoder_lengths = []
    inspection_batch_size = max(args.batch_sizes)
    for offset in range(0, len(texts), inspection_batch_size):
        batch = collator(list(zip(
            texts[offset:offset + inspection_batch_size],
            schema_dicts[offset:offset + inspection_batch_size],
        )))
        encoder_lengths.extend(batch.attention_mask.sum(dim=1).tolist())

    print("schema profiles:", profile_counts)
    print("document token lengths:", length_summary(document_token_lengths))
    print("encoder sequence lengths:", length_summary(encoder_lengths))

    def configure(name: str) -> None:
        model.sync_collapsed_decode = VARIANTS[name]

    def run(name: str, batch_size: int, run_texts=texts, run_schemas=schemas):
        configure(name)
        return model.batch_extract(
            run_texts,
            run_schemas,
            batch_size=batch_size,
            threshold=args.threshold,
            num_workers=0,
            format_results=True,
            include_confidence=True,
            include_spans=True,
        )

    benchmark_results = {}
    with torch.inference_mode():
        for batch_size in args.batch_sizes:
            print(f"\nbenchmarking batch_size={batch_size}")
            warm_count = min(len(texts), max(72, batch_size * 4))
            warm_texts = texts[:warm_count]
            warm_schemas = schemas[:warm_count]

            def run_warm(name: str):
                configure(name)
                return model.batch_extract(
                    warm_texts,
                    warm_schemas,
                    batch_size=batch_size,
                    threshold=args.threshold,
                    num_workers=0,
                    format_results=True,
                    include_confidence=True,
                    include_spans=True,
                )

            for warmup_index in range(args.warmup):
                warm_outputs = {name: run_warm(name) for name in VARIANTS}
                assert_exact(
                    warm_outputs["default"], warm_outputs["optimized"],
                    f"batch={batch_size} warmup={warmup_index}",
                )

            samples = {name: [] for name in VARIANTS}
            reference_digest = None
            names = list(VARIANTS)
            for iteration in range(args.iterations):
                order = names[iteration % len(names):] + names[:iteration % len(names)]
                outputs = {}
                for name in order:
                    outputs[name], timing = measure(
                        lambda name=name: run(name, batch_size), device
                    )
                    samples[name].append(timing)
                assert_exact(
                    outputs["default"], outputs["optimized"],
                    f"batch={batch_size} iteration={iteration}",
                )
                current_digest = digest(outputs["default"])
                if reference_digest is None:
                    reference_digest = current_digest
                elif current_digest != reference_digest:
                    raise AssertionError(
                        f"default output changed across iterations for batch {batch_size}"
                    )

            summary = {
                name: summarize(values, len(texts))
                for name, values in samples.items()
            }
            summary["speedup"] = (
                summary["default"]["median_ms"]
                / summary["optimized"]["median_ms"]
            )
            summary["formatted_parity"] = "exact"
            summary["output_sha256"] = reference_digest
            benchmark_results[str(batch_size)] = summary

            for name in VARIANTS:
                row = summary[name]
                memory = (
                    f" peak={row['peak_allocated_mib']:.1f}MiB"
                    if "peak_allocated_mib" in row else ""
                )
                print(
                    f"{name:<12} median={row['median_ms']:.2f}ms "
                    f"p90={row['p90_ms']:.2f}ms "
                    f"docs/s={row['documents_per_second']:.2f}{memory}"
                )
            print(f"speedup: {summary['speedup']:.3f}x")
            print(f"formatted parity: exact ({reference_digest})")

    result = {
        "model": args.model,
        "device": (
            torch.cuda.get_device_name(device)
            if device.type == "cuda" else str(device)
        ),
        "dtype": "fp16",
        "execution_mode": args.execution_mode,
        "documents": len(texts),
        "batch_sizes": args.batch_sizes,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "threshold": args.threshold,
        "schema_profiles": profile_counts,
        "document_token_lengths": length_summary(document_token_lengths),
        "encoder_sequence_lengths": length_summary(encoder_lengths),
        "results": benchmark_results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(f"\nJSON: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
