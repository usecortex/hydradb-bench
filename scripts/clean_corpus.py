#!/usr/bin/env python3
"""
Clean messy PDF-extracted .txt files in a corpus directory.

Fixes common artifacts from PDF-to-text extraction of financial documents:
  1. Broken words          — "Cash Flow s"     → "Cash Flows"
  2. Whitespace-only lines — lines with just spaces removed
  3. Orphan $ signs        — "$\n5,363"         → "$5,363"
  4. Trailing $ on numbers — "34,229 $"         → "34,229"
  5. Label + values        — joins a text label with the numeric
                             values that immediately follow it:
                             "Cash and cash equivalents\n$4,258\n$3,655"
                             → "Cash and cash equivalents: $4,258 | $3,655"
  6. Multiple blank lines  — 3+ consecutive blanks → 1 blank

Usage:
    python scripts/clean_corpus.py ./data/financebench_corpus
    python scripts/clean_corpus.py ./data/financebench_corpus --dry-run
    python scripts/clean_corpus.py ./data/financebench_corpus --output ./data/financebench_corpus_clean
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_numeric_line(line: str) -> bool:
    """True if a line looks like a financial number (possibly with $, commas, parens, %)."""
    s = line.strip().rstrip('$').strip()
    # e.g. "34,229", "(1,749)", "19.7", "1.2 %", "$ 3,655"
    return bool(re.fullmatch(r'[\$\(\)\-\d,\.\s%]+', s)) and bool(re.search(r'\d', s))


def _is_value_line(line: str) -> bool:
    """True if line is a number OR an orphan '$'."""
    s = line.strip()
    return s == '$' or _is_numeric_line(s)


def _normalise_number(line: str) -> str:
    """Strip trailing '$ ' artefacts from a number token like '34,229 $'."""
    return re.sub(r'\s*\$\s*$', '', line.strip())


# ---------------------------------------------------------------------------
# Core cleaning pipeline
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    lines = text.split('\n')

    # ── Pass 1: fix broken words and strip whitespace-only lines ──────────
    cleaned: list[str] = []
    for line in lines:
        # Whitespace-only → empty
        if line.strip() == '':
            cleaned.append('')
            continue

        # Fix broken words: "Cash Flow s" → "Cash Flows", "Balance Shee t" → "Balance Sheet"
        # Only collapse when a SINGLE orphan char follows a space — these are always
        # broken word-endings from PDF ligature/hyphenation issues. Never collapse
        # 2+ char tokens because those are likely real words ("of", "in", "and" etc.).
        line = re.sub(r'(\w) ([a-z])\b', r'\1\2', line)

        cleaned.append(line.rstrip())

    lines = cleaned

    # ── Pass 2: join orphan '$' with the number on the next line ──────────
    merged: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.strip() == '$' and i + 1 < len(lines):
            next_line = lines[i + 1].strip()
            if next_line and not _is_value_line(next_line) is False:
                merged.append('$' + _normalise_number(next_line))
                i += 2
                continue
        merged.append(line)
        i += 1

    lines = merged

    # ── Pass 3: strip trailing '$' artefacts from number lines ────────────
    lines = [
        _normalise_number(l) if _is_numeric_line(l) and l.strip().endswith('$') else l
        for l in lines
    ]

    # ── Pass 4: join label lines with their following value tokens ─────────
    # A label is a non-empty, non-numeric line. If it is immediately followed
    # by one or more value lines, merge them: "Label: $v1 | $v2 | $v3"
    result: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]

        # Skip separator and heading lines as-is
        if line.startswith('#') or line.strip() == '---':
            result.append(line)
            i += 1
            continue

        # Collect value lines after a label, skipping blank lines between them
        if line.strip() and not _is_value_line(line):
            values: list[str] = []
            j = i + 1
            # Allow skipping over blank lines to find values
            while j < len(lines):
                if lines[j].strip() == '':
                    j += 1
                    continue
                if _is_value_line(lines[j]):
                    values.append(_normalise_number(lines[j].strip()))
                    j += 1
                else:
                    break

            if values:
                result.append(f"{line.strip()}: {' | '.join(values)}")
                i = j
                continue

        result.append(line)
        i += 1

    # ── Pass 5: collapse 3+ consecutive blank lines into one ──────────────
    final_text = '\n'.join(result)
    final_text = re.sub(r'\n{3,}', '\n\n', final_text)

    return final_text.strip()


# ---------------------------------------------------------------------------
# File processing
# ---------------------------------------------------------------------------

def process_file(path: Path, output_path: Path, dry_run: bool) -> dict:
    original = path.read_text(encoding='utf-8')
    cleaned = clean_text(original)

    original_lines = original.count('\n')
    cleaned_lines = cleaned.count('\n')
    reduction = original_lines - cleaned_lines

    if not dry_run:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(cleaned, encoding='utf-8')

    return {
        'file': path.name,
        'original_lines': original_lines,
        'cleaned_lines': cleaned_lines,
        'reduction': reduction,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('input_dir', help='Directory containing .txt files to clean')
    parser.add_argument('--output', default=None,
                        help='Output directory (default: overwrite in-place)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show stats without writing files')
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        print(f'Error: {input_dir} is not a directory', file=sys.stderr)
        sys.exit(1)

    output_dir = Path(args.output) if args.output else input_dir
    txt_files = sorted(input_dir.glob('*.txt'))

    if not txt_files:
        print(f'No .txt files found in {input_dir}')
        sys.exit(0)

    print(f"{'Cleaning' if not args.dry_run else 'Dry-run for'} {len(txt_files)} files "
          f"in {input_dir}/")
    if args.output:
        print(f"Output -> {output_dir}/")
    print()

    total_orig = total_clean = 0
    for f in txt_files:
        out = output_dir / f.name
        stats = process_file(f, out, args.dry_run)
        total_orig += stats['original_lines']
        total_clean += stats['cleaned_lines']
        print(f"  {stats['file']:<50} {stats['original_lines']:>5} -> {stats['cleaned_lines']:>5} lines  "
              f"(-{stats['reduction']})")

    print()
    print(f"Total: {total_orig} -> {total_clean} lines  "
          f"(-{total_orig - total_clean} lines, "
          f"{(total_orig - total_clean) / total_orig * 100:.1f}% reduction)")
    if args.dry_run:
        print('\n[dry-run] No files written.')
    else:
        print(f'\nDone. Files written to {output_dir}/')


if __name__ == '__main__':
    main()
