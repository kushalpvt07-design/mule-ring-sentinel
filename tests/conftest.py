"""
tests/conftest.py
─────────────────
Shared fixtures, and an honest report header.

WHY THE HEADER MATTERS
──────────────────────
Every data-dependent test in this suite skips when the generated artefacts are
absent, which is correct behaviour for a fresh clone — but a run that skips 40
tests and prints "all passed" is exactly the kind of vacuous signal this suite
was rewritten to remove. `pytest_report_header` therefore states, at the top of
every run, which artefacts are present and which are not, so nobody mistakes a
skipped suite for a green one.

The fixtures are session-scoped: the val graph alone takes a few seconds to build
and several tests need it, and re-reading 180,000 edges per test would make the
suite slow enough that people stop running it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

# ──────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models" / "saved_models"

# pytest's `prepend` import mode already puts the repo root on sys.path because
# tests/ is a package and the root is not. Doing it explicitly anyway costs
# nothing and keeps `pytest tests/test_features.py` working from any cwd and
# under `--import-mode=importlib`.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SPLITS = ("train", "val", "test")

EDGE_PATHS = {s: RAW_DIR / f"{s}_edges.csv" for s in SPLITS}
FEATURE_PATHS = {s: PROCESSED_DIR / f"{s}_features.csv" for s in SPLITS}
CONTEXT_PATH = RAW_DIR / "serving_context_edges.csv"

REGENERATE = "python -m data.generator && python -m data.extractor"


# ──────────────────────────────────────────────────────────────────
# Report header
# ──────────────────────────────────────────────────────────────────

def pytest_report_header(config) -> list[str]:
    """State what is on disk, so a skipped suite cannot be read as a passing one."""
    def tick(paths) -> str:
        missing = [p.name for p in paths if not p.exists()]
        return "present" if not missing else f"MISSING ({', '.join(missing)})"

    from models.features import MODEL_NAME  # noqa: PLC0415  (cheap, no deps)

    lines = [
        f"sentinel: edge files      {tick(EDGE_PATHS.values())}",
        f"sentinel: feature tables  {tick(FEATURE_PATHS.values())}",
        f"sentinel: serving context {tick([CONTEXT_PATH])}",
        f"sentinel: trained model   {tick([MODEL_DIR / MODEL_NAME])}",
    ]
    if any("MISSING" in line for line in lines):
        lines.append(f"sentinel: tests needing the missing artefacts will SKIP "
                     f"— regenerate with `{REGENERATE}`")
    return lines


# ──────────────────────────────────────────────────────────────────
# Loaders
# ──────────────────────────────────────────────────────────────────

def _read_edges(path: Path) -> pd.DataFrame:
    if not path.exists():
        pytest.skip(f"{path.relative_to(PROJECT_ROOT)} not found. Run `{REGENERATE}`.")
    return pd.read_csv(path, parse_dates=["timestamp"])


def _read_features(path: Path) -> pd.DataFrame:
    if not path.exists():
        pytest.skip(f"{path.relative_to(PROJECT_ROOT)} not found. Run `{REGENERATE}`.")
    return pd.read_csv(path)


@pytest.fixture(scope="session")
def raw_edges() -> dict[str, pd.DataFrame]:
    """The three split edge files, timestamps parsed. Keyed 'train'/'val'/'test'."""
    return {s: _read_edges(p) for s, p in EDGE_PATHS.items()}


@pytest.fixture(scope="session")
def node_features() -> dict[str, pd.DataFrame]:
    """The three processed node feature tables, as models/train.py loads them."""
    return {s: _read_features(p) for s, p in FEATURE_PATHS.items()}


@pytest.fixture(scope="session")
def serving_context() -> pd.DataFrame:
    """The historical graph the API scores against."""
    return _read_edges(CONTEXT_PATH)


@pytest.fixture(scope="session")
def val_graph(raw_edges):
    """
    The val split as a DiGraph, built once.

    val rather than train because it is the graph the API actually serves against
    (`serving_context_edges.csv` spans the same window), so stability results on
    it describe production behaviour rather than a training artefact.
    """
    networkx = pytest.importorskip(
        "networkx", reason="networkx is required to build a graph")
    del networkx
    from data.extractor import build_graph
    return build_graph(raw_edges["val"])


@pytest.fixture(scope="session")
def frozen_partition(val_graph):
    """
    The community partition computed once from the val graph.

    This is exactly what api/main.py freezes at startup and passes into
    `compute_node_features`, and the reason design rule 5 holds at all — see
    tests/test_features.py.
    """
    pytest.importorskip("community",
                        reason="python-louvain is required to partition")
    from data.extractor import compute_louvain_communities
    return compute_louvain_communities(val_graph.to_undirected())


@pytest.fixture(scope="session")
def metrics() -> dict:
    """
    Parsed metrics.json, or a skip.

    Deliberately NOT a pytest.skip on a stale file: tests/test_baselines.py
    fails on a version mismatch, because a metrics.json describing a different
    model is a published claim about a model that no longer exists.
    """
    import json
    path = MODEL_DIR / "metrics.json"
    if not path.exists():
        pytest.skip("models/saved_models/metrics.json not found. "
                    "Run `python -m models.train`.")
    return json.loads(path.read_text(encoding="utf-8"))
