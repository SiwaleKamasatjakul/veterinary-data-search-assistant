"""Single source of truth for filesystem locations.

Everything in this project is anchored to BASE_DIR, which is derived from this
file's own location. Nothing is hardcoded to a user's home directory, so the
project runs from any folder on any machine — that is the core portability fix
over the Linux original.
"""

from __future__ import annotations

from pathlib import Path

# src/tools/paths.py -> src/tools -> src -> <project root>
BASE_DIR = Path(__file__).resolve().parents[2]

CONFIG_DIR = BASE_DIR / "config"
DATA_DIR = BASE_DIR / "data"
VECTORDB_DIR = BASE_DIR / "src" / "vectordb"
LOG_DIR = BASE_DIR / "logs"


def resolve(path_like: str | Path) -> Path:
    """Resolve a config value to an absolute path.

    Absolute values are respected as-is; relative values are treated as relative
    to the project root.
    """
    p = Path(path_like)
    return p if p.is_absolute() else (BASE_DIR / p)
