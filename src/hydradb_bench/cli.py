"""CLI entry point — delegates to run_benchmark.main()."""

import sys
import os

# Ensure the repo root is on sys.path so run_benchmark can be imported
# regardless of how the package was installed.
_repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)


def main() -> None:
    import run_benchmark
    run_benchmark.main()
