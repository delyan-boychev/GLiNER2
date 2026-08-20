#!/usr/bin/env python3
"""End-to-end real-text benchmark for accelerator decode synchronization.

The default run uses 9 schema profiles (task families), each with 512
documents. Documents are natural-language units drawn from an expanded
hand-written corpus and vary in length from a single sentence to a single
paragraph up to five paragraphs. Within every profile the per-document schema
is sampled with a seeded RNG, so entity/relation/structure/classification
queries differ across documents while staying inside a defined band for each
group. Runs use CUDA FP16 and the repository's public ``compile=True`` loading
option. It compares:

* ``default``: F.linear scoring and the original per-value synchronized decoder;
* ``optimized``: the same scoring with synchronization-collapsed decoding.

Every warmup and measured run must produce exactly identical formatted output.
No synthetic tensors or repeated-token padding are used. Timing is reported
per profile (per task family), not only aggregated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
import time
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence, Tuple

import torch

from gliner2 import GLiNER2
from gliner2.inference.schema import AttributeGroup
from gliner2.training.trainer import ExtractorCollator


# Hand-written natural sentences used for the shortest documents.
SENTENCES = (
    "Apple's chief executive introduced a new iPhone generation at the company's Cupertino campus.",
    "Microsoft opened a cloud engineering center in Warsaw after two years of planning with local universities.",
    "A regional hospital in Manchester expanded its cardiology unit with three new operating rooms.",
    "NVIDIA presented its latest data-center processors during a developer conference in San Jose.",
    "The European Central Bank left interest rates unchanged following its meeting in Frankfurt.",
    "Toyota will add a battery assembly line to its manufacturing plant in Kentucky.",
    "Researchers at Stanford University published a study of coastal groundwater levels in California.",
    "Amazon Web Services announced a new cloud region for customers in Thailand.",
    "A passenger train traveling from Paris to Lyon was delayed after heavy rain damaged signaling equipment.",
    "The city of Toronto approved funding for two hundred electric buses and charging equipment.",
    "OpenAI and a group of independent publishers announced a research program on citation discovery.",
    "A conservation team relocated twelve sea turtles to a protected beach near Cairns.",
    "Samsung launched a compact foldable phone in Seoul with an upgraded hinge and brighter display.",
    "The University of Edinburgh created a scholarship fund for students in renewable energy.",
    "Pfizer began a late-stage clinical study of an experimental influenza vaccine.",
    "The British Museum placed a collection of restored Roman coins on public display.",
    "A farming cooperative near Valencia installed solar panels above irrigation canals.",
    "Netflix acquired worldwide distribution rights to an independent documentary filmed in Iceland.",
    "SpaceX launched a communications satellite from Cape Canaveral early Tuesday morning.",
    "A Berlin software company raised twenty-five million euros to expand its fraud-detection platform.",
    "Japan's national weather agency issued heat warnings for Tokyo, Osaka, and nearby prefectures.",
    "Ford recalled a group of sport utility vehicles after engineers found a wiring fault.",
    "Archaeologists working near Alexandria uncovered part of a residential district dating to the second century.",
    "The World Health Organization delivered emergency medical supplies to clinics affected by flooding.",
    "A Dutch startup began trials of a delivery drone that can carry packages across urban districts.",
    "Engineers in Oslo completed a carbon-neutral office tower that heats itself with seawater pumps.",
    "Chilean astronomers detected a fast radio burst coming from a dwarf galaxy eight billion light-years away.",
    "The port of Rotterdam tested automated cranes that load containers onto ships without human operators.",
    "A Swiss watchmaker introduced a limited series of timepieces assembled from recycled aerospace alloys.",
    "Teachers in Nairobi launched an after-school program teaching students to repair household electronics.",
    "Brazilian officials announced new rules requiring banks to report cryptocurrency transactions.",
    "A Canadian film festival awarded its top prize to a documentary about ice-core climate research.",
)


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
    "A Dutch startup began trials of a delivery drone designed to carry packages across dense urban districts. City authorities granted a temporary license for routes between distribution hubs and neighborhood pickup points.",
    "Engineers in Oslo completed a carbon-neutral office tower that draws heat from seawater pumps and recycles rainwater for cooling. The building's facade is covered in glass panels that adjust their tint automatically with the sun.",
    "Chilean astronomers detected a fast radio burst coming from a dwarf galaxy roughly eight billion light-years away. The signal repeats every sixteen days, a pattern the team says is difficult to explain with current models.",
    "The port of Rotterdam tested automated cranes that load containers onto ships without direct human control. Union leaders asked for guarantees that experienced operators would train the new systems rather than lose their posts.",
    "A Swiss watchmaker introduced a limited series of timepieces assembled from recycled aerospace alloys. Each model carries a certificate documenting the origin of the metal and the number of units produced.",
    "Teachers in Nairobi launched an after-school program where students learn to repair household electronics. The initiative recycles donated devices and plans to supply refurbished computers to local schools.",
    "Brazilian officials announced new rules requiring banks to report cryptocurrency transactions above a fixed threshold. The measure is part of a broader effort to close tax loopholes and monitor cross-border capital flows.",
    "A Canadian film festival awarded its top prize to a documentary about ice-core research in the Arctic. The jury praised the production for combining archival footage with new measurements collected during a two-year expedition.",
    "Volkswagen unveiled plans for an entry-level electric hatchback to be built at a plant in eastern Germany. Executives said the model would target a lower price point to compete with imported compact cars in Europe.",
    "The Australian Securities and Investments Commission fined a trading platform for failing to disclose order-execution costs. Regulators said the penalty reflected repeated violations over a two-year inspection period.",
    "Researchers at the University of Zurich published a study linking urban tree cover with lower summer temperatures in residential neighborhoods. The analysis used satellite data from more than a thousand cities across six continents.",
    "A consortium of six European airlines agreed to share real-time weather data through a common platform. The arrangement is designed to reduce fuel burn by allowing pilots to request reroutes around developing storms.",
    "The state of Kerala announced a five-year plan to install rooftop solar systems on every public school building. Officials estimate the program will cut electricity spending by forty percent while providing backup power.",
    "A Korean shipbuilder delivered the first of eight methanol-powered container vessels ordered by a shipping line. The company says the engines cut carbon emissions by more than half compared with conventional heavy fuel.",
    "Linguists at a research institute in Paris completed a digital archive of endangered dialects spoken along the Pyrenees. The collection includes thousands of audio recordings, grammatical notes, and bilingual storybooks.",
    "A financial watchdog in Singapore proposed guidelines for banks using generative models in credit decisions. The draft rules require institutions to document how models are tested and to allow customers to contest automated denials.",
    "The city of Copenhagen opened a pedestrian bridge connecting its central station to a redeveloped harbor district. Engineers designed the structure to tilt upward during storms so that rising water levels cannot damage its bearings.",
    "A nonprofit organization in Lagos distributed solar-powered refrigerators to health clinics without reliable electricity. Health workers said the units allow them to store vaccines and blood products at stable temperatures.",
    "The European Space Agency selected a mission to study the magnetic field of an unexplored moon of Saturn. The orbiter will measure surface composition and search for plumes of water vapor escaping through cracks in the ice.",
    "A British pharmacy chain tested a subscription service that delivers prescription refills on a fixed weekly schedule. The company says the program improves adherence for patients managing chronic conditions.",
    "Researchers in Japan demonstrated a robotic arm that can sort plastic waste by touch, using pressure sensors to recognize material stiffness. The prototype sorts about two hundred items per hour with an accuracy above ninety percent.",
    "The federal railroad administration opened an investigation after a freight train derailed near a river crossing in Ohio. Investigators are checking the condition of the rails and whether heavy rain had weakened the embankment.",
    "A technology consortium published a standard for transferring patient records between hospitals and mobile health applications. The group says the new format preserves privacy while making records easier to share in emergencies.",
    "The government of New Zealand proposed a carbon price floor for agricultural emissions, with a rebate for farmers who adopt low-emission feeding systems. The plan would take effect after two seasons of pilot trials.",
    "A museum in Vienna unveiled a reconstruction of a medieval trading ship recovered from the Danube. Visitors can walk through the hull and examine replicas of the tools used to build it a thousand years ago.",
    "An airline in the Middle East ordered forty wide-body aircraft and agreed to purchase sustainable aviation fuel from a producer in Spain. The deal includes an option to expand the order depending on route growth.",
    "Researchers at a marine laboratory in Australia tagged forty reef sharks to study their movement around tourist diving sites. The data will help park managers decide where to limit boat traffic during breeding season.",
    "The city council in Montreal approved a pilot program allowing food trucks to operate in parks on a rotating schedule. Vendors must report their waste and energy use so the council can evaluate the environmental impact.",
    "A semiconductor company in Taiwan began construction of a new research center dedicated to advanced packaging. The facility will employ engineers working on chip designs that stack memory and logic in a single package.",
    "Physicists at a laboratory near Geneva published results from a detector upgrade that measures the mass of a rare particle with greater precision. The measurement confirms a prediction made thirty years ago by two theorists.",
    "A cooperative of coffee growers in Guatemala formed an export alliance with a roasting company in the United States. The agreement guarantees minimum prices for the next four harvests in exchange for direct trade.",
    "The transit authority in Mexico City launched a bike-sharing expansion that adds stations near metro lines and university campuses. Officials say the program aims to reduce congestion during peak commuting hours.",
    "A team of agronomists in Kenya tested drought-tolerant maize varieties across twelve demonstration farms. Yields improved by a third in the driest plots, and farmers reported that the new seeds required less fertilizer.",
    "A Norwegian energy company commissioned a floating wind platform designed to operate in deep water far from the coast. The platform will be tested for a year before the company decides whether to scale production.",
    "The national statistics office published revised figures showing stronger manufacturing growth in the second quarter. Economists said the revision reflected new data on small businesses that earlier surveys had missed.",
    "A hospital network in Spain deployed software that flags early signs of sepsis from patient monitoring data. Clinicians review the alerts during routine rounds and say the system has reduced response times.",
    "The Federal Communications Commission proposed a rule requiring internet providers to display broadband speed data on a searchable map. Consumer groups supported the measure, while providers warned about the cost of compliance.",
    "A publisher in Nigeria began printing low-cost science textbooks in four local languages. The first run of fifty thousand copies will be distributed to secondary schools free of charge.",
    "Engineers in Denmark finished a trial of wireless charging roads that power electric buses while they drive. The test section recharges vehicles at stops and intersections, and the city plans to expand it to a full route.",
    "The World Bank approved a loan for a water-recycling project serving two million residents of a coastal city. The investment will upgrade treatment plants and install sensors that detect leaks in the distribution network.",
    "A research station in Antarctica reported a record low sea-ice extent for the month of August. Scientists say the trend matches projections from climate models that anticipate increasingly open water in the southern winter.",
    "An e-commerce platform in India launched a same-day delivery network for groceries in three metropolitan areas. The company operates a chain of neighborhood warehouses that stock fast-moving items close to customers.",
    "The International Olympic Committee selected a host city for the winter games after two rounds of voting. The winning bid emphasized the reuse of existing venues and a compact layout across a single valley.",
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

CLASSIFICATION_LABELS = (
    "technology", "business", "health", "science", "public policy", "culture",
    "transport", "government", "research", "sports", "finance", "energy",
    "education", "entertainment", "law", "environment",
)

RELATION_POOL = {
    "works_for": 0.4,
    "located_in": 0.4,
    "announced": 0.4,
    "acquired": 0.4,
}

STRUCTURE_FIELDS = (
    ("organization", "str"),
    ("subject", "str"),
    ("location", "list"),
    ("date", "str"),
    ("amount", "list"),
    ("status", "str", ("planned", "ongoing", "completed", "cancelled")),
    ("actor", "str"),
    ("action", "str"),
    ("place", "list"),
    ("time", "str"),
    ("reason", "str"),
    ("outcome", "list"),
    ("participants", "list"),
    ("budget", "str"),
)

ATTRIBUTE_GROUPS = {
    "role": ("executive", "researcher", "official", "participant", "spokesperson"),
    "prominence": ("primary", "secondary"),
    "sentiment": ("positive", "negative", "neutral"),
    "scale": ("local", "regional", "national", "global"),
}


VARIANTS = {
    "default": False,
    "optimized": True,
}

# Entity query-count bands per entities_qN profile (min, max inclusive).
ENTITY_BANDS = {
    "entities_q4": (3, 5),
    "entities_q16": (12, 20),
    "entities_q32": (26, 38),
    "entities_q64": (56, 64),
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


def build_documents(
    count: int,
    paragraphs: Sequence[str],
    sentences: Sequence[str],
    seed: int,
    sentence_fraction: float = 0.2,
    single_paragraph_fraction: float = 0.2,
) -> List[str]:
    """Build realistic documents of varied length.

    About ``sentence_fraction`` of documents are a single sentence, about
    ``single_paragraph_fraction`` are one paragraph, and the remainder are two
    to five paragraphs joined with blank lines. A per-call RNG seeded with
    ``seed`` makes the corpus reproducible while giving each profile a distinct
    set of documents.
    """
    rng = random.Random(seed)
    documents = []
    for index in range(count):
        roll = rng.random()
        if roll < sentence_fraction:
            documents.append(sentences[index % len(sentences)])
        elif roll < sentence_fraction + single_paragraph_fraction:
            documents.append(paragraphs[index % len(paragraphs)])
        else:
            paragraph_count = rng.randint(2, 5)
            start = rng.randrange(len(paragraphs))
            documents.append("\n\n".join(
                paragraphs[(start + offset) % len(paragraphs)]
                for offset in range(paragraph_count)
            ))
    return documents


# ─── Per-document schema builders ─────────────────────────────────────────

def _entities_builder(model, band: Tuple[int, int]):
    low, high = band

    def builder(rng: random.Random) -> Any:
        count = rng.randint(low, high)
        types = rng.sample(list(ENTITY_TYPES), min(count, len(ENTITY_TYPES)))
        return model.create_schema().entities(types)

    return builder


def _classifications_builder(model):
    def builder(rng: random.Random) -> Any:
        schema = model.create_schema()
        task_count = rng.randint(1, 2)
        for index in range(task_count):
            labels = rng.sample(
                list(CLASSIFICATION_LABELS),
                rng.randint(2, min(6, len(CLASSIFICATION_LABELS))),
            )
            if rng.random() < 0.5:
                schema.classification(
                    f"task{index + 1}", labels, multi_label=True, cls_threshold=0.35
                )
            else:
                schema.classification(f"task{index + 1}", labels)
        return schema

    return builder


def _relations_builder(model):
    def builder(rng: random.Random) -> Any:
        names = rng.sample(list(RELATION_POOL), rng.randint(1, len(RELATION_POOL)))
        return model.create_schema().relations(
            {name: {"threshold": RELATION_POOL[name]} for name in names}
        )

    return builder


def _structures_builder(model):
    def builder(rng: random.Random) -> Any:
        schema = model.create_schema().structure("announcement")
        fields = rng.sample(list(STRUCTURE_FIELDS), rng.randint(2, 6))
        for field in fields:
            if len(field) == 3:
                schema.field(field[0], dtype=field[1], choices=list(field[2]))
            else:
                schema.field(field[0], dtype=field[1])
        return schema

    return builder


def _mixed_builder(model):
    def builder(rng: random.Random) -> Any:
        schema = model.create_schema()
        schema.entities(rng.sample(list(ENTITY_TYPES), rng.randint(4, 16)))
        if rng.random() < 0.7:
            labels = rng.sample(
                list(CLASSIFICATION_LABELS),
                rng.randint(2, min(5, len(CLASSIFICATION_LABELS))),
            )
            schema.classification("document type", labels)
        if rng.random() < 0.7:
            names = rng.sample(list(RELATION_POOL), rng.randint(1, 3))
            schema.relations(
                {name: {"threshold": RELATION_POOL[name]} for name in names}
            )
        if rng.random() < 0.7:
            structure = schema.structure("event summary")
            fields = rng.sample(list(STRUCTURE_FIELDS), rng.randint(2, 5))
            for field in fields:
                if len(field) == 3:
                    structure.field(field[0], dtype=field[1], choices=list(field[2]))
                else:
                    structure.field(field[0], dtype=field[1])
        return schema

    return builder


def _attributes_builder(model):
    def builder(rng: random.Random) -> Any:
        schema = model.create_schema().entities(["person", "organization", "product"])
        group_names = rng.sample(list(ATTRIBUTE_GROUPS), rng.randint(1, 2))
        attributes = {}
        for name in group_names:
            labels = rng.sample(
                list(ATTRIBUTE_GROUPS[name]),
                rng.randint(2, len(ATTRIBUTE_GROUPS[name])),
            )
            attributes[name] = AttributeGroup(
                labels,
                qualify_labels=rng.random() < 0.5,
                multi_label=rng.random() < 0.3,
                threshold=0.35,
            )
        return schema.entity_attributes(attributes)

    return builder


def build_schema_profiles(model) -> List[Tuple[str, Callable[[random.Random], Any]]]:
    """Return ``(name, builder)`` pairs; each builder yields a per-document schema."""
    profiles: List[Tuple[str, Callable[[random.Random], Any]]] = []
    for name, band in ENTITY_BANDS.items():
        profiles.append((name, _entities_builder(model, band)))
    profiles.append(("classifications", _classifications_builder(model)))
    profiles.append(("relations", _relations_builder(model)))
    profiles.append(("structures", _structures_builder(model)))
    profiles.append(("mixed", _mixed_builder(model)))
    profiles.append(("entity_attributes", _attributes_builder(model)))
    return profiles


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
    parser.add_argument("--documents", type=int, default=512,
                        help="documents per schema profile")
    parser.add_argument("--batch-sizes", type=parse_int_list, default=parse_int_list("8,16,32"))
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--execution-mode", choices=("compile", "eager"), default="compile")
    parser.add_argument("--seed", type=int, default=0,
                        help="deterministic seed for document and schema sampling")
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

    paragraphs = load_corpus(args.corpus) if args.corpus else list(REAL_PARAGRAPHS)
    sentences = list(SENTENCES)
    profiles = build_schema_profiles(model)
    print(f"schema profiles: {len(profiles)} x {args.documents} documents each")

    # Build a distinct, reproducible 512-document corpus per profile.
    groups: "OrderedDict[str, Tuple[List[str], List[Any]]]" = OrderedDict()
    for index, (name, builder) in enumerate(profiles):
        group_seed = args.seed + index * 1_000_003
        texts = build_documents(
            args.documents, paragraphs, sentences, seed=group_seed
        )
        rng = random.Random(group_seed + 7)
        schemas = [builder(rng) for _ in range(args.documents)]
        groups[name] = (texts, schemas)

    tokenizer = model.processor.tokenizer
    collator = ExtractorCollator(
        model.processor, is_training=False, architecture=model.architecture
    )
    inspection_batch_size = max(args.batch_sizes)
    length_summaries: "OrderedDict[str, Dict[str, Dict[str, float]]]" = OrderedDict()
    for name, (texts, schemas) in groups.items():
        tokenizer_output = tokenizer(texts, add_special_tokens=False, truncation=False)
        document_token_lengths = [len(ids) for ids in tokenizer_output["input_ids"]]
        schema_dicts, _ = model._build_schema_dicts_and_metadata(schemas)
        encoder_lengths = []
        for offset in range(0, len(texts), inspection_batch_size):
            batch = collator(list(zip(
                texts[offset:offset + inspection_batch_size],
                schema_dicts[offset:offset + inspection_batch_size],
            )))
            encoder_lengths.extend(batch.attention_mask.sum(dim=1).tolist())
        length_summaries[name] = {
            "document_token_lengths": length_summary(document_token_lengths),
            "encoder_sequence_lengths": length_summary(encoder_lengths),
        }
        print(
            f"  {name:<16} tokens={length_summaries[name]['document_token_lengths']['mean']:.0f}"
            f" mean | enc={length_summaries[name]['encoder_sequence_lengths']['max']:.0f} max"
        )

    def configure(variant: str) -> None:
        model.sync_collapsed_decode = VARIANTS[variant]

    def run(
        variant: str,
        batch_size: int,
        texts: Sequence[str],
        schemas: Sequence[Any],
    ) -> List[Dict]:
        configure(variant)
        return model.batch_extract(
            texts,
            schemas,
            batch_size=batch_size,
            threshold=args.threshold,
            num_workers=0,
            format_results=True,
            include_confidence=True,
            include_spans=True,
        )

    benchmark_results: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
    with torch.inference_mode():
        for batch_size in args.batch_sizes:
            print(f"\nbenchmarking batch_size={batch_size}")
            batch_results: "OrderedDict[str, Any]" = OrderedDict()
            for name, (texts, schemas) in groups.items():
                warm_count = min(len(texts), max(72, batch_size * 4))
                warm_texts = texts[:warm_count]
                warm_schemas = schemas[:warm_count]

                for warmup_index in range(args.warmup):
                    warm_outputs = {
                        variant: run(variant, batch_size, warm_texts, warm_schemas)
                        for variant in VARIANTS
                    }
                    assert_exact(
                        warm_outputs["default"], warm_outputs["optimized"],
                        f"batch={batch_size} profile={name} warmup={warmup_index}",
                    )

                samples: Dict[str, List[Dict[str, float]]] = {
                    variant: [] for variant in VARIANTS
                }
                reference_digest = None
                variants = list(VARIANTS)
                for iteration in range(args.iterations):
                    order = (
                        variants[iteration % len(variants):]
                        + variants[:iteration % len(variants)]
                    )
                    outputs: Dict[str, List[Dict]] = {}
                    for variant in order:
                        outputs[variant], timing = measure(
                            lambda variant=variant: run(
                                variant, batch_size, texts, schemas
                            ),
                            device,
                        )
                        samples[variant].append(timing)
                    assert_exact(
                        outputs["default"], outputs["optimized"],
                        f"batch={batch_size} profile={name} iteration={iteration}",
                    )
                    current_digest = digest(outputs["default"])
                    if reference_digest is None:
                        reference_digest = current_digest
                    elif current_digest != reference_digest:
                        raise AssertionError(
                            f"default output changed across iterations for "
                            f"batch {batch_size} profile {name}"
                        )

                summary = {
                    variant: summarize(values, len(texts))
                    for variant, values in samples.items()
                }
                summary["speedup"] = (
                    summary["default"]["median_ms"]
                    / summary["optimized"]["median_ms"]
                )
                summary["formatted_parity"] = "exact"
                summary["output_sha256"] = reference_digest
                batch_results[name] = summary

                for variant in VARIANTS:
                    row = summary[variant]
                    memory = (
                        f" peak={row['peak_allocated_mib']:.1f}MiB"
                        if "peak_allocated_mib" in row else ""
                    )
                    print(
                        f"  {name:<16} {variant:<10} "
                        f"median={row['median_ms']:.2f}ms "
                        f"p90={row['p90_ms']:.2f}ms "
                        f"docs/s={row['documents_per_second']:.2f}{memory}"
                    )
                print(f"  {name:<16} speedup: {summary['speedup']:.3f}x")
                print(f"  {name:<16} formatted parity: exact ({reference_digest})")

            benchmark_results[str(batch_size)] = batch_results

    result = {
        "model": args.model,
        "device": (
            torch.cuda.get_device_name(device)
            if device.type == "cuda" else str(device)
        ),
        "dtype": "fp16",
        "execution_mode": args.execution_mode,
        "seed": args.seed,
        "documents_per_profile": args.documents,
        "schema_profiles": {name: len(texts) for name, (texts, _) in groups.items()},
        "length_summaries": dict(length_summaries),
        "results": dict(benchmark_results),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(f"\nJSON: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())