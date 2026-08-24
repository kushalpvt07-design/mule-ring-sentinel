"""
tests/test_contract.py
──────────────────────
The feature contract, and train/serve parity.

models/features.py cites this file by name:

    This function [assert_feature_contract] existed at the time and was never
    called. It is now called from the API's startup path, and covered by
    tests/test_contract.py.

data/generator.py cites it too, for the window contract on the emitted data.

─────────────────────────────────────────────────────────────────────────────
THE BUG THIS FILE EXISTS TO PREVENT
─────────────────────────────────────────────────────────────────────────────
Before v3, api/main.py carried its own hard-coded list of 12 feature names,
matching the retired `sentinel_v1.xgb`. It therefore loaded a stale model without
complaint and applied the CURRENT model's threshold to it. Nothing raised. Every
score was wrong. `assert_feature_contract` was already written and simply never
called.

Two lessons, and a test for each:

  1. A guard that is not invoked is not a guard. `test_api_startup_invokes_the
     _contract_check` parses api/main.py and fails if the call is not there,
     which is cheap, needs no dependencies, and would have caught the original
     defect on day one.

  2. A second copy of a list is a second source of truth. `test_no_module_hard
     _codes_a_feature_list` walks the AST of every module and fails on any list
     or tuple literal that looks like a feature list and is not one of the four
     declared subsets. Declared subsets are then themselves checked for being
     subsets, because a typo in `GRAPH_FEATURES` would silently mis-specify the
     graph-ablation baseline.

Usage:
    pytest tests/test_contract.py -v
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from models.features import (
    FEATURE_COLS,
    FEATURE_DESCRIPTIONS,
    LABEL_META_COLS,
    METADATA_COLS,
    MODEL_NAME,
    MODEL_VERSION,
    TARGET_COL,
    assert_feature_contract,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SOURCE_PACKAGES = ("api", "data", "models", "dashboard")

# Modules exempt from the "no hard-coded feature list" scan.
#   models/features.py  IS the single source of truth.
#   tests/              asserts against the contract by definition.
CONTRACT_OWNER = PROJECT_ROOT / "models" / "features.py"

# Names allowed to hold a list of feature names, each a DECLARED SUBSET of
# FEATURE_COLS and checked as such below. Anything else that looks like a feature
# list is a second source of truth and fails the scan.
DECLARED_SUBSETS = {
    "FEATURE_COLS",          # the contract itself
    "GRAPH_FEATURES",        # models/train.py — graph-ablation baseline
    "NON_GRAPH_FEATURES",    # models/train.py — its complement
    "LOG1P_COLS",            # models/train.py — heavy-tailed columns
    "TOOLTIP_FEATURES",      # dashboard/components/graph_viz.py — display subset
}

# A literal trips the scan when it has at least this many string elements and at
# least this many of them are feature names. Tuned so the four declared subsets
# (5, 7 and 13 names) are the only things in the repo that reach it.
SCAN_MIN_STRINGS = 5
SCAN_MIN_FEATURE_NAMES = 4


def _python_files() -> list[Path]:
    files: list[Path] = []
    for package in SOURCE_PACKAGES:
        files += sorted((PROJECT_ROOT / package).rglob("*.py"))
    return [f for f in files if "__pycache__" not in f.parts]


# ══════════════════════════════════════════════════════════════════
# 1. assert_feature_contract actually rejects what it must
# ══════════════════════════════════════════════════════════════════

class TestContractEnforcement:
    """
    Silently serving a model whose columns don't line up is the single most
    expensive failure mode in an ML service: nothing raises, every score is
    wrong, and the bug is invisible until someone audits outcomes.
    """

    def test_exact_match_passes(self):
        assert_feature_contract(list(FEATURE_COLS))

    def test_missing_names_are_rejected(self):
        """A booster with no feature names cannot be checked, so it is refused."""
        with pytest.raises(RuntimeError, match="UNVERIFIABLE"):
            assert_feature_contract(None)

    def test_reordering_is_rejected(self):
        """
        The subtle one. Same names, same count, same set — and every score wrong,
        because XGBoost matches columns positionally.
        """
        swapped = list(FEATURE_COLS)
        swapped[0], swapped[1] = swapped[1], swapped[0]
        with pytest.raises(RuntimeError, match="DIFFERENT ORDER"):
            assert_feature_contract(swapped)

    def test_a_missing_feature_is_rejected_and_named(self):
        short = [c for c in FEATURE_COLS if c != "cycle_participation"]
        with pytest.raises(RuntimeError, match="cycle_participation"):
            assert_feature_contract(short)

    def test_an_extra_feature_is_rejected_and_named(self):
        long = list(FEATURE_COLS) + ["louvain_community"]
        with pytest.raises(RuntimeError, match="louvain_community"):
            assert_feature_contract(long)

    def test_the_historical_v1_list_is_rejected(self):
        """
        The exact regression. This is the 12-name list api/main.py used to carry,
        with `net_flow` and `community_size` — both since dropped — and no
        `cycle_participation`, `reciprocity`, `burst_ratio` or
        `counterparty_amount_cv`.
        """
        v1_list = [
            "in_degree", "out_degree", "degree_ratio", "in_amount_sum",
            "out_amount_sum", "net_flow", "pagerank", "clustering_coefficient",
            "fan_in_concentration", "txn_velocity", "amount_cv", "community_size",
        ]
        with pytest.raises(RuntimeError, match="MISMATCH"):
            assert_feature_contract(v1_list)

    def test_api_startup_invokes_the_contract_check(self):
        """
        A guard nobody calls is not a guard.

        Parsed rather than grepped so a mention inside a docstring or a comment
        cannot satisfy it — only a real call site does.
        """
        tree = ast.parse((PROJECT_ROOT / "api" / "main.py").read_text(
            encoding="utf-8"))
        called = {
            node.func.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        assert "assert_feature_contract" in called, (
            "api/main.py never CALLS assert_feature_contract. This is exactly how "
            "a stale sentinel_v1.xgb was served against v3's threshold: the check "
            "was written, imported, and never invoked."
        )


# ══════════════════════════════════════════════════════════════════
# 2. One source of truth for the feature list
# ══════════════════════════════════════════════════════════════════

class TestSingleSourceOfTruth:
    """Three copies of a feature list is three chances for train and serve to drift."""

    def test_no_module_hard_codes_a_feature_list(self):
        feature_set = set(FEATURE_COLS)
        offenders: list[str] = []

        for path in _python_files():
            if path == CONTRACT_OWNER:
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"))

            # Map each list/tuple literal to the name it is assigned to, if any,
            # so a declared subset can be recognised.
            assigned_to: dict[int, str] = {}
            for node in ast.walk(tree):
                if isinstance(node, (ast.Assign, ast.AnnAssign)):
                    value = node.value
                    if value is None:
                        continue
                    targets = (node.targets if isinstance(node, ast.Assign)
                               else [node.target])
                    for target in targets:
                        if isinstance(target, ast.Name):
                            assigned_to[id(value)] = target.id

            for node in ast.walk(tree):
                if not isinstance(node, (ast.List, ast.Tuple)):
                    continue
                strings = [e.value for e in node.elts
                           if isinstance(e, ast.Constant) and isinstance(e.value, str)]
                if len(strings) < SCAN_MIN_STRINGS:
                    continue
                overlap = feature_set & set(strings)
                if len(overlap) < SCAN_MIN_FEATURE_NAMES:
                    continue
                name = assigned_to.get(id(node))
                if name in DECLARED_SUBSETS:
                    continue
                offenders.append(
                    f"{path.relative_to(PROJECT_ROOT)}:{node.lineno} "
                    f"({name or 'unnamed literal'}, {len(overlap)} feature names)")

        assert not offenders, (
            "feature names are hard-coded outside models/features.py:\n  "
            + "\n  ".join(offenders)
            + "\nImport FEATURE_COLS instead. A second copy is how api/main.py "
              "came to serve a 12-feature v1 model against v3's threshold.\n"
              "If this really is a legitimate declared subset, add its name to "
              "DECLARED_SUBSETS in this test — which will then require it to be a "
              "subset of FEATURE_COLS."
        )

    @pytest.mark.parametrize("module,name", [
        ("models.train", "GRAPH_FEATURES"),
        ("models.train", "NON_GRAPH_FEATURES"),
        ("models.train", "LOG1P_COLS"),
    ])
    def test_declared_subsets_really_are_subsets(self, module, name):
        """
        A typo in `GRAPH_FEATURES` would not raise anywhere — it would silently
        change which features the ablation baseline removes, and the reported
        "graph features are worth X" number would be measuring something else.
        """
        pytest.importorskip("xgboost", reason="models.train imports xgboost")
        pytest.importorskip("sklearn", reason="models.train imports scikit-learn")
        import importlib

        values = getattr(importlib.import_module(module), name)
        unknown = [v for v in values if v not in FEATURE_COLS]
        assert not unknown, (
            f"{module}.{name} names features that are not in FEATURE_COLS: "
            f"{unknown}"
        )

    def test_graph_and_non_graph_features_partition_the_contract(self):
        """The ablation is only interpretable if the two halves cover everything once."""
        pytest.importorskip("xgboost", reason="models.train imports xgboost")
        pytest.importorskip("sklearn", reason="models.train imports scikit-learn")
        from models.train import GRAPH_FEATURES, NON_GRAPH_FEATURES

        assert set(GRAPH_FEATURES) | set(NON_GRAPH_FEATURES) == set(FEATURE_COLS)
        assert not set(GRAPH_FEATURES) & set(NON_GRAPH_FEATURES)


# ══════════════════════════════════════════════════════════════════
# 3. Internal consistency of the contract itself
# ══════════════════════════════════════════════════════════════════

class TestContractIsWellFormed:
    """Cheap invariants on models/features.py that nothing else would notice."""

    def test_no_duplicate_feature_names(self):
        duplicates = [c for c in set(FEATURE_COLS) if FEATURE_COLS.count(c) > 1]
        assert not duplicates, f"FEATURE_COLS repeats {duplicates}"

    def test_every_feature_has_an_analyst_description(self):
        """
        The API returns these strings to investigators. A missing one degrades an
        explanation to a raw column name; a stale one describes a feature the
        model no longer has.
        """
        missing = [c for c in FEATURE_COLS if c not in FEATURE_DESCRIPTIONS]
        stale = [c for c in FEATURE_DESCRIPTIONS if c not in FEATURE_COLS]
        assert not missing, f"FEATURE_DESCRIPTIONS is missing {missing}"
        assert not stale, (
            f"FEATURE_DESCRIPTIONS describes dropped feature(s) {stale}"
        )

    def test_metadata_and_features_do_not_overlap(self):
        assert not set(METADATA_COLS) & set(FEATURE_COLS), (
            f"columns are both metadata and features: "
            f"{sorted(set(METADATA_COLS) & set(FEATURE_COLS))}"
        )
        assert not set(LABEL_META_COLS) & set(FEATURE_COLS), (
            f"label metadata is being fed to the model: "
            f"{sorted(set(LABEL_META_COLS) & set(FEATURE_COLS))}"
        )
        assert TARGET_COL not in FEATURE_COLS, (
            "the target is in FEATURE_COLS — the model would be trained on the "
            "answer"
        )

    def test_louvain_community_is_metadata_not_a_feature(self):
        """
        Design rule 1: structural, not nominal. XGBoost was splitting on
        "community id < 17.5", which is meaningless, and the numbering has no
        correspondence between graphs.
        """
        assert "louvain_community" in METADATA_COLS
        assert "louvain_community" not in FEATURE_COLS

    def test_dropped_v2_features_have_not_returned(self):
        """`net_flow` was redundant; `community_size` scored test AUC 0.10."""
        for dropped in ("net_flow", "community_size"):
            assert dropped not in FEATURE_COLS, (
                f"'{dropped}' was dropped in v3 and is back in FEATURE_COLS"
            )

    def test_model_filename_and_version_agree(self):
        """
        api/main.py derives its path from MODEL_NAME. If the two ever disagree the
        service loads one model and reports the version of another.
        """
        assert MODEL_NAME == f"{MODEL_VERSION}.xgb", (
            f"MODEL_NAME {MODEL_NAME!r} does not match MODEL_VERSION "
            f"{MODEL_VERSION!r}"
        )


# ══════════════════════════════════════════════════════════════════
# 4. Train/serve parity
# ══════════════════════════════════════════════════════════════════

class TestTrainServeParity:
    """
    The training and serving paths must compute the same features from the same
    edges — checked on real data, not by reading the code.
    """

    def test_processed_tables_carry_the_contract_in_order(self, node_features):
        """The columns the model is fitted on come from these files."""
        for split, frame in node_features.items():
            present = [c for c in frame.columns if c in set(FEATURE_COLS)]
            assert present == list(FEATURE_COLS), (
                f"{split}_features.csv holds the features in a different order "
                f"than FEATURE_COLS declares.\n  file:     {present}\n"
                f"  contract: {list(FEATURE_COLS)}"
            )

    def test_serving_context_loader_never_returns_label_columns(self):
        """
        The context file on disk DOES carry ground truth — it is a copy of a
        labelled split. The loader must drop it, or the serving path has the
        answer sitting next to the question.
        """
        pytest.importorskip("fastapi", reason="api.main imports FastAPI")
        pytest.importorskip("xgboost", reason="api.main imports xgboost")
        from api.main import CONTEXT_COLS, LABEL_COLS_NEVER_READ, load_context_edges

        edges = load_context_edges()
        if edges is None:
            pytest.skip("data/raw/serving_context_edges.csv not found")

        assert list(edges.columns) == list(CONTEXT_COLS), (
            f"load_context_edges returned {list(edges.columns)}, expected "
            f"{list(CONTEXT_COLS)}"
        )
        leaked = [c for c in LABEL_COLS_NEVER_READ if c in edges.columns]
        assert not leaked, f"ground-truth columns reached the serving path: {leaked}"

    def test_trained_window_is_measured_from_the_training_edges(self, raw_edges):
        """
        The API reports `trained_window_days` so a caller can tell whether the
        scores are on the same scale as the metrics. It must measure the real
        thing.
        """
        pytest.importorskip("fastapi", reason="api.main imports FastAPI")
        pytest.importorskip("xgboost", reason="api.main imports xgboost")
        from api.main import measure_trained_window_days

        measured = measure_trained_window_days()
        if measured is None:
            pytest.skip("data/raw/train_edges.csv not found")

        expected = float(
            (raw_edges["train"]["timestamp"].max()
             - raw_edges["train"]["timestamp"].min()).total_seconds() / 86_400.0)
        assert abs(measured - expected) < 0.01, (
            f"measure_trained_window_days reported {measured:.3f}d against an "
            f"actual training span of {expected:.3f}d"
        )

    @pytest.mark.slow
    def test_merge_with_context_reassembles_the_graph_exactly(
        self, raw_edges, frozen_partition, val_graph
    ):
        """
        THE PARITY TEST.

        Split the val edges into "history" and "50 submitted transactions", push
        them back through the serving merge, and recompute. Every feature must
        come out bit-identical to the training-time computation on the whole file
        — because the merge is supposed to reassemble exactly the same edge set.

        An off-by-one in the window trim, a duplicated concat, a lost dtype or a
        re-sort that changes node insertion order would all show up here as a
        feature that moved, and nowhere else at all.
        """
        pytest.importorskip("fastapi", reason="api.main imports FastAPI")
        pytest.importorskip("xgboost", reason="api.main imports xgboost")
        import numpy as np

        from api.main import merge_with_context
        from data.extractor import build_graph, compute_node_features

        cols = ["sender", "receiver", "amount", "timestamp"]
        ordered = (raw_edges["val"][cols]
                   .sort_values("timestamp", kind="mergesort")
                   .reset_index(drop=True))
        submitted = ordered.tail(50).reset_index(drop=True)
        context = ordered.head(len(ordered) - 50).reset_index(drop=True)

        span_days = float(
            (ordered["timestamp"].max() - ordered["timestamp"].min())
            .total_seconds() / 86_400.0)
        # A hair wider than the true span so the trim keeps everything; a window
        # exactly equal to the span would sit on the boundary of the >= comparison.
        merged, diagnostics = merge_with_context(
            submitted, context, window_days=span_days + 0.01)

        assert len(merged) == len(ordered), (
            f"merge produced {len(merged)} edges from {len(ordered)} "
            f"({diagnostics['n_context_used']} of {len(context)} context rows "
            f"survived the window). The serving path is not seeing the graph the "
            f"model was trained against."
        )

        reference = (compute_node_features(val_graph, partition=frozen_partition)
                     .set_index("node").sort_index())
        served = (compute_node_features(build_graph(merged),
                                        partition=frozen_partition)
                  .set_index("node").sort_index())

        assert list(served.index) == list(reference.index), (
            "the reassembled graph has a different account set"
        )
        drifted = {
            c: float(np.abs(served[c].to_numpy(dtype=float)
                            - reference[c].to_numpy(dtype=float)).max())
            for c in FEATURE_COLS
            if not np.array_equal(served[c].to_numpy(dtype=float),
                                  reference[c].to_numpy(dtype=float))
        }
        assert not drifted, (
            f"TRAIN/SERVE SKEW: {len(drifted)} feature(s) differ between the "
            f"training-time computation and the same edges routed through "
            f"merge_with_context: {drifted}"
        )
