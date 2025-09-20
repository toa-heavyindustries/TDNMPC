"""Configuration and output directory helpers."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a JSON or YAML configuration file.

    Parameters
    ----------
    path:
        Path or string pointing to a ``.json`` or ``.yaml``/``.yml`` file.

    Returns
    -------
    dict[str, Any]
        Parsed configuration dictionary. Empty configs yield an empty dict.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the suffix is unsupported.
    """

    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")

    suffix = cfg_path.suffix.lower()
    with cfg_path.open("r", encoding="utf-8") as handle:
        if suffix == ".json":
            return json.load(handle)
        if suffix in {".yaml", ".yml"}:
            data = yaml.safe_load(handle)
            return data if isinstance(data, dict) else {}

    raise ValueError(f"Unsupported config format: {cfg_path.suffix}")


def ensure_run_dir(tag: str | None = None, base: str | Path = "runs") -> Path:
    """Create and return a run directory under ``base``.

    Parameters
    ----------
    tag:
        Optional tag name for the run directory. A timestamp-based tag is used if omitted.
    base:
        Root directory for run outputs. Created if missing.

    Returns
    -------
    Path
        Path to the ensured run directory.
    """

    base_path = Path(base)
    base_path.mkdir(parents=True, exist_ok=True)

    slug = _slugify(tag) if tag else datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_dir = base_path / slug
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _slugify(value: str) -> str:
    """Normalize a tag for filesystem usage."""

    stripped = value.strip().replace(" ", "-")
    safe_chars = [c for c in stripped if c.isalnum() or c in {"-", "_"}]
    fallback = "run"
    return "".join(safe_chars) or fallback

