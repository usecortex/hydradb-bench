"""
Convert all company PDFs in the data/ folder to Markdown using PyMuPDF4LLM.
Output .md files are saved alongside the source PDFs in each company folder.

Usage:
    python convert_pdfs_to_markdown.py                  # convert all
    python convert_pdfs_to_markdown.py --company APPLE  # one company
    python convert_pdfs_to_markdown.py --workers 2      # parallelism
    python convert_pdfs_to_markdown.py --skip-existing  # skip already converted
"""

import argparse
import logging
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent / "data"


def convert_pdf(pdf_path: Path, skip_existing: bool) -> tuple[Path, bool, str]:
    """
    Convert a single PDF to Markdown using PyMuPDF4LLM.
    Returns (pdf_path, success, message).
    Must be top-level for multiprocessing pickling.
    """
    md_path = pdf_path.with_suffix(".md")

    if skip_existing and md_path.exists():
        return pdf_path, True, "skipped (already exists)"

    try:
        import pymupdf4llm
        md_text = pymupdf4llm.to_markdown(
            str(pdf_path),
            show_progress=False,
            write_images=False,
            embed_images=False,
        )
        md_path.write_text(md_text, encoding="utf-8")
        return pdf_path, True, f"-> {md_path.name}"
    except Exception as exc:
        return pdf_path, False, f"ERROR: {exc}"


def gather_pdfs(company: str | None) -> list[Path]:
    if company:
        folder = DATA_DIR / company.upper()
        if not folder.is_dir():
            log.error("Company folder not found: %s", folder)
            sys.exit(1)
        folders = [folder]
    else:
        folders = sorted(p for p in DATA_DIR.iterdir() if p.is_dir())

    pdfs: list[Path] = []
    for folder in folders:
        pdfs.extend(sorted(folder.glob("*.pdf")))
    return pdfs


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert company PDFs to Markdown via PyMuPDF4LLM")
    parser.add_argument("--company", help="Convert only this company (e.g. APPLE)")
    parser.add_argument("--workers", type=int, default=1, help="Parallel worker processes (default: 1)")
    parser.add_argument("--skip-existing", action="store_true", help="Skip PDFs that already have a .md file")
    parser.add_argument("--limit", type=int, default=None, help="Max number of PDFs to convert (e.g. 30)")
    args = parser.parse_args()

    pdfs = gather_pdfs(args.company)
    if args.limit:
        pdfs = pdfs[: args.limit]
    total = len(pdfs)

    if total == 0:
        log.info("No PDFs found.")
        return

    log.info("Found %d PDF(s) to convert (workers=%d, skip_existing=%s)", total, args.workers, args.skip_existing)

    success_count = 0
    skip_count = 0
    fail_count = 0
    failures: list[tuple[Path, str]] = []

    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(convert_pdf, pdf, args.skip_existing): pdf for pdf in pdfs
            }
            done = 0
            for future in as_completed(futures):
                done += 1
                pdf_path, ok, msg = future.result()
                label = f"[{done}/{total}] {pdf_path.parent.name}/{pdf_path.name}"
                if ok:
                    if "skipped" in msg:
                        skip_count += 1
                    else:
                        success_count += 1
                    log.info("%s  %s", label, msg)
                else:
                    fail_count += 1
                    failures.append((pdf_path, msg))
                    log.error("%s  %s", label, msg)
    else:
        for idx, pdf in enumerate(pdfs, 1):
            label = f"[{idx}/{total}] {pdf.parent.name}/{pdf.name}"
            pdf_path, ok, msg = convert_pdf(pdf, args.skip_existing)
            if ok:
                if "skipped" in msg:
                    skip_count += 1
                else:
                    success_count += 1
                log.info("%s  %s", label, msg)
            else:
                fail_count += 1
                failures.append((pdf_path, msg))
                log.error("%s  %s", label, msg)

    log.info(
        "Done. converted=%d  skipped=%d  failed=%d",
        success_count, skip_count, fail_count,
    )

    if failures:
        log.warning("Failed files:")
        for p, msg in failures:
            log.warning("  %s  %s", p, msg)
        sys.exit(1)


if __name__ == "__main__":
    main()
