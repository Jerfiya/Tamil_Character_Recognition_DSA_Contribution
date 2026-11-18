"""Shared pytest setup: make src/ importable and provide tiny configs."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))


@pytest.fixture
def cfg():
    from config import load_config
    return load_config(None, ["threads=2"])


def has_processed() -> bool:
    return (ROOT / "data" / "processed" / "manifest.csv").exists()


needs_data = pytest.mark.skipif(not has_processed(), reason="run prepare_data.py first")
