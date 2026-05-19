from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.meta_path = [
    finder
    for finder in sys.meta_path
    if "MesonpyMetaFinder" not in type(finder).__name__
    or "ForgeFF" not in repr(finder)
    and "motep" not in repr(finder)
]
