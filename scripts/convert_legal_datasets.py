"""
Convert legal benchmark JSON files to HydraDB TestSample format.

Supports: contractnli, cuad, maud, privacy_qa

Input format:
  {"tests": [{"query": "...", "snippets": [{"file_path": "...", "span": [...], "answer": "..."}]}]}

Output format (TestSample):
  [{"id": "...", "question": "...", "reference_answer": "...", "reference_contexts": [...]}]

Usage:
  python scripts/convert_legal_datasets.py
  python scripts/convert_legal_datasets.py --max-samples 50
  python scripts/convert_legal_datasets.py --datasets cuad maud
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

DATASETS = {
    "contractnli": {
        "input":  "data/contractnli.json",
        "output": "data/legal_contractnli_samples.json",
        "prefix": "cnli",
    },
    "cuad": {
        "input":  "data/cuad.json",
        "output": "data/legal_cuad_samples.json",
        "prefix": "cuad",
    },
    "maud": {
        "input":  "data/maud.json",
        "output": "data/legal_maud_samples.json",
        "prefix": "maud",
    },
    "privacy_qa": {
        "input":  "data/privacy_qa.json",
        "output": "data/legal_privacyqa_samples.json",
        "prefix": "pvqa",
    },
}


def convert(name: str, cfg: dict, max_samples: int | None) -> list[dict]:
    input_path = Path(cfg["input"])
    if not input_path.exists():
        print(f"  [SKIP] {input_path} not found.")
        return []

    with open(input_path, encoding="utf-8") as f:
        raw = json.load(f)

    tests = raw.get("tests", [])
    if max_samples:
        tests = tests[:max_samples]

    samples = []
    skipped = 0
    for i, test in enumerate(tests):
        query = (test.get("query") or "").strip()
        if not query:
            skipped += 1
            continue

        snippets = test.get("snippets") or []

        # reference_contexts: each snippet answer is one context chunk
        reference_contexts = [
            s["answer"].strip()
            for s in snippets
            if s.get("answer", "").strip()
        ]

        # reference_answer: join all snippet answers (they are the ground-truth evidence)
        reference_answer = "\n\n".join(reference_contexts)

        if not reference_answer:
            skipped += 1
            continue

        samples.append({
            "id": f"{cfg['prefix']}_{i + 1:04d}",
            "question": query,
            "reference_answer": reference_answer,
            "reference_contexts": reference_contexts,
        })

    print(f"  {name}: {len(samples)} samples converted ({skipped} skipped, {len(tests)} read)")
    return samples


def main():
    parser = argparse.ArgumentParser(description="Convert legal benchmark JSON to TestSample format")
    parser.add_argument("--max-samples", type=int, default=50,
                        help="Max samples per dataset (default: 50, use 0 for all)")
    parser.add_argument("--datasets", nargs="+", choices=list(DATASETS.keys()),
                        default=list(DATASETS.keys()),
                        help="Which datasets to convert (default: all)")
    args = parser.parse_args()

    max_s = args.max_samples if args.max_samples > 0 else None

    print(f"Converting legal datasets (max_samples={max_s or 'all'})...")
    print()

    for name in args.datasets:
        cfg = DATASETS[name]
        samples = convert(name, cfg, max_s)
        if not samples:
            continue

        output_path = Path(cfg["output"])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(samples, f, indent=2, ensure_ascii=False)
        print(f"  Saved -> {output_path}")
        print()


if __name__ == "__main__":
    main()
