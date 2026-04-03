#!/usr/bin/env bash
# scripts/bootstrap.sh — Zero-to-running setup for hydradb-bench
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV_DIR="${REPO_ROOT}/.venv"
REQUIRED_PYTHON_MAJOR=3
REQUIRED_PYTHON_MINOR=10

# ── Colours (disabled when not a terminal) ──────────────────────────────────
if [ -t 1 ]; then
  GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; RESET='\033[0m'
else
  GREEN=''; YELLOW=''; RED=''; RESET=''
fi

info()  { echo -e "${GREEN}[✓]${RESET} $*"; }
warn()  { echo -e "${YELLOW}[!]${RESET} $*"; }
error() { echo -e "${RED}[✗]${RESET} $*"; }

# ── 1. Check Python ─────────────────────────────────────────────────────────
if ! command -v python3 &>/dev/null; then
  error "python3 is not installed. Please install Python ${REQUIRED_PYTHON_MAJOR}.${REQUIRED_PYTHON_MINOR}+."
  exit 1
fi

PYTHON_VERSION="$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
PYTHON_MAJOR="$(echo "$PYTHON_VERSION" | cut -d. -f1)"
PYTHON_MINOR="$(echo "$PYTHON_VERSION" | cut -d. -f2)"

if [ "$PYTHON_MAJOR" -lt "$REQUIRED_PYTHON_MAJOR" ] || \
   { [ "$PYTHON_MAJOR" -eq "$REQUIRED_PYTHON_MAJOR" ] && [ "$PYTHON_MINOR" -lt "$REQUIRED_PYTHON_MINOR" ]; }; then
  error "Python ${REQUIRED_PYTHON_MAJOR}.${REQUIRED_PYTHON_MINOR}+ is required (found ${PYTHON_VERSION})."
  exit 1
fi
info "Python ${PYTHON_VERSION} detected"

# ── 2. Create virtual environment ───────────────────────────────────────────
if [ ! -d "$VENV_DIR" ]; then
  info "Creating virtual environment at ${VENV_DIR}"
  python3 -m venv "$VENV_DIR"
else
  info "Virtual environment already exists at ${VENV_DIR}"
fi

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"
info "Activated virtual environment"

# ── 3. Install dependencies ─────────────────────────────────────────────────
info "Upgrading pip"
pip install --quiet --upgrade pip

# Note: pyproject.toml uses a non-standard build backend (setuptools.backends.legacy:build)
# that prevents `pip install -e .`. Install dependencies directly instead.
info "Installing dependencies from pyproject.toml"
pip install --quiet httpx pydantic pyyaml python-dotenv rich tiktoken deepeval openai

# Add the src directory to the path so imports work
SITE_PACKAGES=$("${VENV_DIR}/bin/python" -c "import site; print(site.getsitepackages()[0])")
echo "${REPO_ROOT}/src" > "${SITE_PACKAGES}/hydradb-bench.pth"
info "Added src/ to Python path"

# ── 4. Environment file ─────────────────────────────────────────────────────
if [ ! -f "${REPO_ROOT}/.env" ]; then
  if [ ! -f "${REPO_ROOT}/.env.example" ]; then
    warn ".env.example not found — skipping .env creation. Create it manually."
  else
    cp "${REPO_ROOT}/.env.example" "${REPO_ROOT}/.env"
    warn "Created .env from .env.example — edit it with your API keys"
  fi
else
  info ".env already exists"
fi

# ── 5. Summary ───────────────────────────────────────────────────────────────
echo ""
info "Bootstrap complete! Next steps:"
echo "  1. Activate the venv:  source .venv/bin/activate"
echo "  2. Add your API keys:  \$EDITOR .env"
echo "  3. Run the benchmark:  python run_benchmark.py --provider hydradb"
echo ""
echo "  Run 'make help' to see all available targets."
