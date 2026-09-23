"""Load config.yaml and resolve paths relative to the project root."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"


@lru_cache(maxsize=None)
def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(relative: str | Path) -> Path:
    """Resolve a config path (relative to the project root) to an absolute Path."""
    p = Path(relative)
    return p if p.is_absolute() else PROJECT_ROOT / p
