"""Repository-relative path constants."""

from __future__ import annotations

from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parents[1]
CONFIG_DIR = REPO_ROOT / "config"
DEFAULT_CONFIG = CONFIG_DIR / "default.toml"
DATA_DIR = REPO_ROOT / "data"
ARTIFACTS_DIR = REPO_ROOT / "artifacts"
NOTEBOOKS_DIR = REPO_ROOT / "notebooks"
TRAINING_DIR = REPO_ROOT / "training"
DOCS_DIR = REPO_ROOT / "docs"
