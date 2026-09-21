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

ENV_FILE = BASE_DIR / ".env"

'''
def load_env() -> bool:
    """Load .env from the PROJECT ROOT, not the current working directory.

    This lives here — the module with no dependencies that everything else
    imports — so that env vars are available no matter what the entry point is:
    uvicorn, pytest, or a standalone script in scripts/.

    Anchoring to BASE_DIR matters. Bare ``load_dotenv()`` searches upward from
    the cwd, so running a script from another directory would silently find no
    .env and leave OPENAI_API_KEY unset.

    Returns True if a .env file was found.
    """
    try:
        from dotenv import load_dotenv
    except ImportError:  # python-dotenv is optional at import time
        return False
    if ENV_FILE.exists():
        # override=False: a real environment variable beats the file, which is
        # what you want in CI and containers.
        load_dotenv(ENV_FILE, override=False)
        return True
    return False
'''
def load_env() -> bool:
    """Load .env from the PROJECT ROOT, not the current working directory."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return False
    if ENV_FILE.exists():
        # override=False: a real environment variable beats the file
        load_dotenv(ENV_FILE, override=False)
        return True
    return False

# Run on import. Every entry point reaches this module via config_loader.
load_env()

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
