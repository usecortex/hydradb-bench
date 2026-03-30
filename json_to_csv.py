#!/usr/bin/env python3
"""Convert a HydraDB DeepEval benchmark JSON report to CSV.

Usage:
    python json_to_csv.py reports/sample01.json
    python json_to_csv.py reports/sample01.json --output reports/sample01.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


def convert(json_path: Path, output_path: Path) -> None:
    data = json.loads(json_path.read_text(encoding="utf-8"))

    per_sample = data.get("per_sample", [])
    if not per_sample:
        print("No per_sample data found in JSON — nothing to convert.")
        sys.exit(1)

    run_id = data.get("run_id", "")

    # Flatten each sample into a plain dict — auto-discovers all metric columns
    fixed = ["run_id", "sample_id", "question", "answer", "reference_answer",
             "context_string", "context_tokens", "latency_ms"]
    metrics = list(per_sample[0].get("scores", {}).keys())
    fieldnames = fixed + metrics + [f"{m}_reason" for m in metrics]

    flat_rows: list[dict] = []
    for ss in per_sample:
        row: dict = {
            "run_id": run_id,
            "sample_id": ss.get("sample_id", ""),
            "question": ss.get("question", ""),
            "answer": ss.get("answer", ""),
            "reference_answer": ss.get("reference_answer", ""),
            "context_string": ss.get("context_string", ""),
            "context_tokens": ss.get("context_tokens", 0),
            "latency_ms": round(ss.get("latency_ms", 0), 1),
        }
        for m, score in ss.get("scores", {}).items():
            row[m] = score if score is not None else ""
        for m, reason in ss.get("reasons", {}).items():
            row[f"{m}_reason"] = reason or ""
        flat_rows.append(row)

    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(flat_rows)

    print(f"CSV written to: {output_path.resolve()}")
    print(f"  {len(per_sample)} rows, {len(fieldnames)} columns")


def main() -> None:
    p = argparse.ArgumentParser(description="Convert benchmark JSON report to CSV")
    p.add_argument("json", metavar="JSON_FILE", help="Path to benchmark JSON report")
    p.add_argument("--output", "-o", metavar="CSV_FILE",
                   help="Output CSV path (default: same name as JSON with .csv extension)")
    args = p.parse_args()

    json_path = Path(args.json)
    if not json_path.exists():
        print(f"File not found: {json_path}")
        sys.exit(1)

    output_path = Path(args.output) if args.output else json_path.with_suffix(".csv")
    convert(json_path, output_path)


if __name__ == "__main__":
    main()
