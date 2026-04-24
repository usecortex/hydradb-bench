import json
import sys
import uuid

try:
    import chromadb  # noqa: F401
except ImportError:
    print(
        "ERROR: chromadb is required for test data generation but is not installed.\n"
        "Install it with:  pip install -e '.[datagen]'\n"
        "See CONTRIBUTING.md for details.",
        file=sys.stderr,
    )
    sys.exit(1)

from deepeval.synthesizer import Synthesizer
from deepeval.synthesizer.config import ContextConstructionConfig, Evolution, EvolutionConfig, FiltrationConfig
from dotenv import load_dotenv

load_dotenv()

# DeepEval doesn't have gpt-5.4 in its pricing table so cost comes back as
# None, crashing on `synthesis_cost += None`. Patch the class with a descriptor
# that returns a float subclass where `+ None` safely returns 0 instead.
import deepeval.synthesizer.synthesizer as _ds  # noqa: E402


class _SafeCost(float):
    def __add__(self, other):
        return _SafeCost(float.__add__(self, other if other is not None else 0.0))

    __radd__ = __add__


class _SafeCostDescriptor:
    def __get__(self, obj, objtype=None):
        return _SafeCost(getattr(obj, "_synthesis_cost", 0.0)) if obj else self

    def __set__(self, obj, value):
        object.__setattr__(obj, "_synthesis_cost", float(value) if value is not None else 0.0)


_ds.Synthesizer.synthesis_cost = _SafeCostDescriptor()

evolution_config = EvolutionConfig(
    evolutions={
        Evolution.REASONING: 1 / 4,
        Evolution.MULTICONTEXT: 1 / 4,
        Evolution.CONCRETIZING: 1 / 4,
        Evolution.CONSTRAINED: 1 / 4,
    },
    num_evolutions=4,
)

filtration_config = FiltrationConfig(
    critic_model="gpt-5.4",
    synthetic_input_quality_threshold=0.7,
)


synthesizer = Synthesizer(
    model="gpt-5.4",
    evolution_config=evolution_config,
    filtration_config=filtration_config,
)

context_construction_config = ContextConstructionConfig(
    critic_model="gpt-4o",  # gpt-5.4 scores return unparseable by DeepEval → all chunks fail threshold
)

goldens = synthesizer.generate_goldens_from_docs(
    document_paths=[
        "data/privacy_qa/mee.txt",
        "data/privacy_qa/Fiverr.txt",
        "data/privacy_qa/Groupon.txt",
        "data/privacy_qa/Keep.txt",
        "data/privacy_qa/TickTick_ To Do List with Reminder, Day Planner.txt",
        "data/privacy_qa/Viber Messenger.txt",
        "data/privacy_qa/Wordscapes.txt",
    ],
    include_expected_output=True,
    max_goldens_per_context=2,
    context_construction_config=context_construction_config,
)

synthesizer.save_as(file_type="json", directory="./synthetic_data")

samples = []
for g in goldens:
    question = getattr(g, "input", "") or ""
    if not question.strip():
        continue
    context = getattr(g, "context", None) or []
    if isinstance(context, str):
        context = [context]
    samples.append(
        {
            "id": str(uuid.uuid4()),
            "question": question.strip(),
            "reference_answer": (getattr(g, "expected_output", "") or "").strip(),
            "reference_contexts": [c for c in context if c and c.strip()],
        }
    )

with open("data/privacy_qa.json", "w", encoding="utf-8") as f:
    json.dump(samples, f, indent=2, ensure_ascii=False)

print(f"Generated {len(goldens)} golden(s) → {len(samples)} sample(s) saved to data/privacy_qa.json")
