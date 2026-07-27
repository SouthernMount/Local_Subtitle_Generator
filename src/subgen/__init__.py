"""Local subtitle generation CLI."""

from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_HF_HOME = PROJECT_ROOT / ".cache" / "huggingface"
os.environ.setdefault("HF_HOME", str(DEFAULT_HF_HOME))

__version__ = "0.1.0"
