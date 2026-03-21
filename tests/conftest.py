"""Load .env before pytest collects/runs tests."""
import sys
from pathlib import Path

# Ensure src/ is on the path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Load .env so OPENAI_API_KEY etc. are available during collection
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env", override=False)
