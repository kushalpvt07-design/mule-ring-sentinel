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

Both are static, need no dependencies, and pass on a bare checkout — which is why
one more check has been given a home here that does not obviously belong to the
feature contract. The defense-only guarantee is the one whose failure disqualifies
this project outright, and every test of it lived in tests/test_responder.py behind
a module-level `pytest.importorskip("pydantic")`. Without pydantic installed, that
suite skips in full and nothing checks the constraint at all. Section 5 parses the
action names out of the source instead, so the floor of that guarantee is always
enforced.

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
#
# `FEATURE_COLS` is deliberately NOT here. It used to be — "the contract itself" —
# but models/features.py, the one place that name legitimately holds a literal, is
# already exempt as CONTRACT_OWNER below and never reaches this set. Listing the
# name here instead granted the exemption to EVERY module: a 12-name v1 list
# re-added to api/main.py under `FEATURE_COLS = [...]` would have been waved
# through, which is the precise defect this scan exists to catch. Every other
# module gets the contract by `from models.features import FEATURE_COLS` or
# `list(FEATURE_COLS)` — an import or a call, neither of which is a list literal —
# so removing the exemption cannot make a legitimate module fail.
DECLARED_SUBSETS = {
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

# What each subset's module needs before it can be imported at all. Held per
# module because the subset test used to skip on xgboost and scikit-learn whatever
# it was checking — neither of which graph_viz has ever imported, while the three
# it does need went unmentioned. A test that skips on the wrong dependency is
# either skipping when it could have run or erroring when it should have skipped.
SUBSET_IMPORT_REQUIREMENTS: dict[str, tuple[tuple[str, str], ...]] = {
    "models.train": (
        ("xgboost", "models.train imports xgboost"),
        ("sklearn", "models.train imports scikit-learn"),
    ),
    "dashboard.components.graph_viz": (
        ("streamlit", "graph_viz is a Streamlit page"),
        ("networkx", "graph_viz builds the graph with networkx"),
        ("community", "graph_viz imports data.extractor, which uses python-louvain"),
    ),
}


def _python_files() -> list[Path]:
    """
    Every module the hard-coded-list scan covers: the four packages, plus the
    root-level modules.

    Root modules were missed entirely before — `console.py` lives outside every
    package, so a walk driven only by SOURCE_PACKAGES never opened it, and a scan
    that can be evaded by moving a file one directory up is not a guard. Globbed
    rather than named so the next root module is covered the day it lands.
    """
    files: list[Path] = sorted(PROJECT_ROOT.glob("*.py"))
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

        AND the call site has to be on the path startup actually runs. The earlier
        form collected every `Call` in the module and asked only whether
        `assert_feature_contract` appeared among them, so a call left behind in a
        dead helper — `def _unused(): assert_feature_contract(...)` — or parked
        under `if False:` would satisfy it while the model loaded unchecked. That
        is the same shape of defect the whole file guards: a guard that is present
        but not reached. So this pins two things instead: the call is lexically
        inside the `lifespan` coroutine, and `lifespan` is the one handed to
        `FastAPI(lifespan=...)`, i.e. the code the server runs on the way up.
        """
        tree = ast.parse((PROJECT_ROOT / "api" / "main.py").read_text(
            encoding="utf-8"))

        lifespan = next(
            (n for n in ast.walk(tree)
             if isinstance(n, (ast.AsyncFunctionDef, ast.FunctionDef))
             and n.name == "lifespan"),
            None)
        assert lifespan is not None, (
            "api/main.py has no `lifespan` function. Startup is where the feature "
            "contract has to be enforced — a model whose columns do not line up "
            "with FEATURE_COLS scores every account wrong without raising."
        )

        called_in_lifespan = {
            node.func.id
            for node in ast.walk(lifespan)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        assert "assert_feature_contract" in called_in_lifespan, (
            "api/main.py never CALLS assert_feature_contract on the startup path. "
            "This is exactly how a stale sentinel_v1.xgb was served against v3's "
            "threshold: the check was written, imported, and never invoked. A call "
            "elsewhere in the module — a dead helper, an `if False:` — does not "
            "count, because the server does not run it."
        )

        def _is_fastapi(func: ast.expr) -> bool:
            return ((isinstance(func, ast.Name) and func.id == "FastAPI")
                    or (isinstance(func, ast.Attribute) and func.attr == "FastAPI"))

        registered = any(
            _is_fastapi(node.func)
            and any(kw.arg == "lifespan"
                    and isinstance(kw.value, ast.Name)
                    and kw.value.id == "lifespan"
                    for kw in node.keywords)
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
        )
        assert registered, (
            "api/main.py defines `lifespan` but does not pass it to "
            "FastAPI(lifespan=lifespan). An unregistered lifespan never runs, so "
            "the contract check inside it would not fire at startup."
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
        ("dashboard.components.graph_viz", "TOOLTIP_FEATURES"),
    ])
    def test_declared_subsets_really_are_subsets(self, module, name):
        """
        A typo in `GRAPH_FEATURES` would not raise anywhere — it would silently
        change which features the ablation baseline removes, and the reported
        "graph features are worth X" number would be measuring something else.

        `TOOLTIP_FEATURES` was named in DECLARED_SUBSETS and then never checked, so
        the exemption granted it was doing the opposite of its job: the scan above
        stopped complaining about the literal and nothing took over. Its failure
        mode is quieter but the same shape — a mistyped name reaches `frame[col]`
        on the graph page and a tooltip is missing a value, or the page raises a
        KeyError in front of whoever is demoing it.
        """
        for dependency, reason in SUBSET_IMPORT_REQUIREMENTS[module]:
            pytest.importorskip(dependency, reason=reason)
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

    def test_the_score_path_threads_the_frozen_partition(self):
        """
        The serving path must score against the partition frozen at startup, not
        one recomputed per request.

        This is the api/main.py half of design rule 5, and it had no test. The
        stability guarantee in tests/test_features.py is asserted by handing
        `compute_node_features(partition=frozen_partition)` the frozen partition on
        BOTH sides of the perturbation — which proves the extractor RESPECTS a
        frozen partition, but says nothing about whether the serving code still
        PASSES one. Deleting `partition=STATE.reference_partition` from
        score_transactions, or replacing it with a per-request
        `compute_louvain_communities(...)`, would silently reintroduce the exact
        defect rule 5 exists to prevent — two strangers in the same request batch
        moving an unrelated account's community feature — and every test in
        test_features.py would stay green, because none of them run this function.

        Static rather than behavioural for the reason section 5 gives: exercising
        the real path needs a loaded booster, the context graph and a populated
        STATE, i.e. standing the service up. The deletion worth catching is
        structural and visible in the AST, so it is caught here where it needs
        nothing installed.
        """
        tree = ast.parse((PROJECT_ROOT / "api" / "main.py").read_text(
            encoding="utf-8"))

        score_fn = next(
            (n for n in ast.walk(tree)
             if isinstance(n, (ast.AsyncFunctionDef, ast.FunctionDef))
             and n.name == "score_transactions"),
            None)
        assert score_fn is not None, (
            "api/main.py has no `score_transactions` — the serving entry point "
            "this check pins the frozen partition to."
        )

        def _is_frozen_partition(value: ast.expr) -> bool:
            # STATE.reference_partition — the partition computed once at startup.
            return (isinstance(value, ast.Attribute)
                    and value.attr == "reference_partition"
                    and isinstance(value.value, ast.Name)
                    and value.value.id == "STATE")

        threaded = any(
            (isinstance(node.func, ast.Name)
             and node.func.id == "compute_node_features")
            and any(kw.arg == "partition" and _is_frozen_partition(kw.value)
                    for kw in node.keywords)
            for node in ast.walk(score_fn)
            if isinstance(node, ast.Call)
        )
        assert threaded, (
            "score_transactions does not call compute_node_features(partition="
            "STATE.reference_partition). Without the frozen partition threaded in, "
            "the serving path repartitions per request, and two unrelated accounts "
            "submitted together can move each other's community_internal_ratio — "
            "the defect design rule 5 and tests/test_features.py exist to prevent, "
            "which those tests cannot see because they never run this function."
        )

        repartitions = [
            node.func.id
            for node in ast.walk(score_fn)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id == "compute_louvain_communities"
        ]
        assert not repartitions, (
            "score_transactions calls compute_louvain_communities — it "
            "repartitions on the request. The partition is frozen ONCE at startup "
            "(STATE.reference_partition) precisely so that the accounts in one "
            "request cannot change one another's community feature."
        )

    def test_startup_freezes_the_partition_on_the_projection_not_the_multigraph(
        self
    ):
        """
        The startup half of finding 4. The reference partition is frozen on the
        WEIGHTED UNDIRECTED PROJECTION of the context graph —
        `undirected_projection(build_graph(ctx))` — because that is the graph
        `compute_node_features` scores against. The test above pins the request
        half (that the frozen partition is threaded into scoring); this pins that
        the thing frozen is the right graph in the first place.

        The defect is `.to_undirected()` in place of `undirected_projection`. On
        the context MultiDiGraph networkx resolves a reciprocal pair by letting one
        direction's attribute dict overwrite the other's, and `best_partition`
        optimises WEIGHTED modularity — so the frozen partition became an optimum of
        a graph that discarded ~5.7% of the context file's transaction weight and
        that nothing else in the system ever built. Every request then scored
        against a partition fitted to the wrong graph.

        Static, and by AST rather than substring: the comment on the fixed line
        quotes `.to_undirected()` verbatim to say what the code must NOT do, so a
        text search would match the very comment documenting the fix. The AST
        carries no comments, so a `.to_undirected()` call is visible here only if
        the code actually makes one.
        """
        tree = ast.parse((PROJECT_ROOT / "api" / "main.py").read_text(
            encoding="utf-8"))

        lifespan = next(
            (n for n in ast.walk(tree)
             if isinstance(n, (ast.AsyncFunctionDef, ast.FunctionDef))
             and n.name == "lifespan"),
            None)
        assert lifespan is not None, (
            "api/main.py has no `lifespan` — the startup path this check pins the "
            "frozen partition's construction to."
        )

        def _sets_reference_partition(assign: ast.Assign) -> bool:
            return any(
                (isinstance(t, ast.Attribute) and t.attr == "reference_partition")
                or (isinstance(t, ast.Name) and t.id == "reference_partition")
                for t in assign.targets)

        partition_assigns = [
            n for n in ast.walk(lifespan)
            if isinstance(n, ast.Assign) and _sets_reference_partition(n)]
        assert partition_assigns, (
            "lifespan never assigns reference_partition — the partition is no "
            "longer frozen at startup at all."
        )

        on_projection = any(
            any(isinstance(c, ast.Call) and isinstance(c.func, ast.Name)
                and c.func.id == "undirected_projection"
                for c in ast.walk(a.value))
            for a in partition_assigns)
        assert on_projection, (
            "the reference partition is frozen on something other than "
            "undirected_projection(...). Built on build_graph(ctx) directly or on "
            "`.to_undirected()`, Louvain optimises weighted modularity over a graph "
            "the scoring path never uses, and the frozen partition is an optimum of "
            "the wrong graph."
        )

        to_undirected = [
            n for n in ast.walk(lifespan)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
            and n.func.attr == "to_undirected"]
        assert not to_undirected, (
            "lifespan calls `.to_undirected()`. On the context MultiDiGraph that "
            "collapses reciprocal pairs by last-write-wins and discards transaction "
            "weight; the projection the features are computed on is "
            "`undirected_projection`, which sums parallel edges. Freezing the "
            "partition on the wrong one is finding 4."
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

    def test_a_late_batch_neither_moves_the_window_nor_trims_the_context(
        self, raw_edges
    ):
        """
        FINDING 5. The observation window is anchored to the latest CONTEXT
        transaction, so a batch dated after the context end still scores — its own
        edges are never trimmed — but it cannot slide the window forward and delete
        the oldest context edges, which belong to accounts with nothing to do with
        the request.

        The defect was `window_end = max(submitted.max(), context.max())`. Against
        the shipped context a four-transaction batch dated 20 days late dropped a
        third of the graph, every request still reporting the window comparable.
        This submits a batch dated well past the context end and asserts the window
        did not move and no context was trimmed — while also asserting the batch
        really did fall outside, so the test cannot pass by the batch quietly
        landing in-window.
        """
        pytest.importorskip("fastapi", reason="api.main imports FastAPI")
        pytest.importorskip("xgboost", reason="api.main imports xgboost")
        import pandas as pd

        from api.main import merge_with_context

        cols = ["sender", "receiver", "amount", "timestamp"]
        context = (raw_edges["val"][cols]
                   .sort_values("timestamp", kind="mergesort")
                   .reset_index(drop=True))
        ctx_max = context["timestamp"].max()
        span_days = float(
            (ctx_max - context["timestamp"].min()).total_seconds() / 86_400.0)

        # A batch dated 20 days AFTER the context ends. Under the old anchor this
        # slides window_end to ctx_max + 20d and trims 20 days off the FRONT of the
        # context.
        late = pd.DataFrame({
            "sender": ["LATE_A", "LATE_B"],
            "receiver": ["LATE_C", "LATE_D"],
            "amount": [11.0, 22.0],
            "timestamp": [ctx_max + pd.Timedelta(days=20),
                          ctx_max + pd.Timedelta(days=20)],
        })
        # Window a hair wider than the true span, so on the fixed code the trim
        # keeps every context edge and the only thing that could drop one is the
        # window sliding.
        _, diag = merge_with_context(late, context, window_days=span_days + 0.01)

        assert diag["window_end"] == ctx_max, (
            f"the window moved to {diag['window_end']} — a submitted batch dated "
            f"after the context end ({ctx_max}) slid it forward. The window must be "
            f"anchored to the context, not to the merged maximum."
        )
        assert diag["n_context_used"] == len(context), (
            f"{len(context) - diag['n_context_used']} context edges were trimmed by "
            f"a batch that postdates the context. Nothing the caller sends may drop "
            f"an unrelated account's history."
        )
        # Prove the batch genuinely fell outside the window — otherwise the two
        # assertions above would hold vacuously for an in-window batch.
        assert diag["n_submitted_outside_window"] == len(late), (
            "the late batch was not recognised as outside the window; this test is "
            "not exercising the anchoring rule it claims to."
        )
        assert diag["submitted_days_after_window_end"] > 0

    def test_an_exact_replay_of_a_context_edge_is_de_duplicated_and_counted(
        self, raw_edges
    ):
        """
        FINDING 8. `build_graph` aggregates weight and total_amount, so a
        transaction present in BOTH the request and the context file would be
        counted twice, inflating in_amount_sum, out_amount_sum, repeat_ratio,
        txn_velocity and burst_ratio for both endpoints. The 422 for an
        out-of-window batch tells the caller to date transactions inside the
        context window, and the context file ships in the repo — so replaying a
        context row is the obvious first request, and it was silently mis-scored.

        Exact matches on (sender, receiver, amount, timestamp) are dropped keeping
        the context copy, and the count is reported. This submits three exact
        replays of context rows alongside two genuinely new edges, and asserts the
        three were counted and dropped while the two new ones survived.
        """
        pytest.importorskip("fastapi", reason="api.main imports FastAPI")
        pytest.importorskip("xgboost", reason="api.main imports xgboost")
        import pandas as pd

        from api.main import merge_with_context

        cols = ["sender", "receiver", "amount", "timestamp"]
        context = (raw_edges["val"][cols]
                   .sort_values("timestamp", kind="mergesort")
                   .reset_index(drop=True))
        # The de-dup count is only unambiguous if the context has no internal
        # 4-field duplicate to begin with — which the serving context is documented
        # not to. Assert it, so a future context file that breaks the assumption
        # explains itself here instead of skewing the count.
        assert not context.duplicated(subset=cols).any(), (
            "the context split already contains rows identical on "
            "(sender, receiver, amount, timestamp); the replay count below would "
            "not be attributable to the submitted batch alone."
        )

        replays = context.head(3).copy()          # three exact replays
        ctx_max = context["timestamp"].max()
        fresh = pd.DataFrame({                      # two genuinely new, in-window
            "sender": ["NEW_A", "NEW_B"],
            "receiver": ["NEW_C", "NEW_D"],
            "amount": [1.0, 2.0],
            "timestamp": [ctx_max, ctx_max],
        })
        submitted = pd.concat([replays, fresh], ignore_index=True)

        span_days = float(
            (ctx_max - context["timestamp"].min()).total_seconds() / 86_400.0)
        merged, diag = merge_with_context(
            submitted, context, window_days=span_days + 0.01)

        assert diag["n_duplicates_dropped"] == 3, (
            f"expected 3 exact replays to be counted, got "
            f"{diag['n_duplicates_dropped']}. A bare concat reports none and "
            f"double-counts every replayed edge's weight and amount."
        )
        assert len(merged) == len(context) + 2, (
            f"merged graph has {len(merged)} edges; expected {len(context) + 2} "
            f"(the full context plus the two new edges, the three replays dropped). "
            f"Without de-duplication it would be {len(context) + 5}."
        )

    def test_the_comparability_warning_names_the_v4_rescaled_features(
        self, raw_edges
    ):
        """
        FINDING 9. When the effective window drifts from the trained one the
        warning must describe the v4 residual, not the v3 one. v3 named the three
        MAGNITUDE features as the risk; data/extractor.py now rescales
        in_amount_sum, out_amount_sum and repeat_ratio to a 60-day reference
        window, so magnitude is no longer the problem and what remains is graph
        STRUCTURE. An earlier draft named `txn_velocity` as the third scaling
        feature, which was backwards — txn_velocity divides by the account's own
        active span, so it is already a rate.

        This forces the not-comparable branch with a trained window far from the
        observed span (no trim, no replay, no out-of-window edge, so this is the
        only warning raised) and asserts the message names repeat_ratio and does
        NOT name txn_velocity — the exact wording error the v3 text carried.
        """
        pytest.importorskip("fastapi", reason="api.main imports FastAPI")
        pytest.importorskip("xgboost", reason="api.main imports xgboost")
        import pandas as pd

        from api.main import merge_with_context

        cols = ["sender", "receiver", "amount", "timestamp"]
        context = (raw_edges["val"][cols]
                   .sort_values("timestamp", kind="mergesort")
                   .reset_index(drop=True))
        ctx_max = context["timestamp"].max()
        span_days = float(
            (ctx_max - context["timestamp"].min()).total_seconds() / 86_400.0)

        one_in_window = pd.DataFrame({
            "sender": ["Q1"], "receiver": ["Q2"], "amount": [5.0],
            "timestamp": [ctx_max],
        })
        # Trained window far longer than the observed span: drift is large, so the
        # comparability check trips. window_start lands well before the context
        # begins, so nothing is trimmed and no other warning fires.
        _, diag = merge_with_context(
            one_in_window, context, window_days=span_days + 50.0)

        assert diag["comparable"] is False, (
            "the not-comparable branch did not fire; this test is not exercising "
            "the comparability warning."
        )
        # Isolate the comparability warning by its unique closing phrase, so the
        # de-dup warning (which legitimately names txn_velocity) can never be the
        # one inspected.
        warnings = [w for w in diag["warnings"] if "indicative only" in w]
        assert len(warnings) == 1, (
            f"expected exactly one comparability warning, got {len(warnings)}: "
            f"{diag['warnings']}"
        )
        message = warnings[0]
        assert "repeat_ratio" in message, (
            "the comparability warning does not name repeat_ratio among the "
            "rescaled magnitude features. The v3 text named txn_velocity here, "
            "which is a rate, not a level."
        )
        assert "txn_velocity" not in message, (
            "the comparability warning still names txn_velocity as a window-scaling "
            "feature. It is normalised by the account's active span and does not "
            "scale with the observation window; naming it here is the v3 error."
        )


# ══════════════════════════════════════════════════════════════════
# 5. Defense-only, checked without importing anything
# ══════════════════════════════════════════════════════════════════

# Enforcement verbs, written out here and read from nowhere. Deliberately NOT
# imported from api/responder.py: cases generated from the code under test can only
# cover what that code already knows, which is precisely how `LOCK_ACCOUNT` passed
# both rails for as long as it did — "BLOCK" is not a substring of "LOCK_ACCOUNT"
# and nothing in the module said LOCK.
ENFORCEMENT_VERBS = (
    "BAN", "BLOCK", "FREEZE", "SUSPEND", "TERMINATE", "DISABLE", "REVOKE",
    "SEIZE", "CLOSE", "LOCK", "QUARANTINE", "RESTRICT", "HALT", "BLACKLIST",
    "DENY", "LIMIT", "REVERSE",
)


def _tree(relative_path: str) -> ast.Module:
    return ast.parse((PROJECT_ROOT / relative_path).read_text(encoding="utf-8"))


def _literal_strings(value: ast.expr | None) -> list[str] | None:
    """
    The string elements of a list/tuple/set literal, unwrapping one layer of
    `frozenset(...)` or `set(...)` so an allowlist written as
    `frozenset({"ALLOW", ...})` reads the same as a plain tuple.
    """
    if (isinstance(value, ast.Call) and isinstance(value.func, ast.Name)
            and value.func.id in ("frozenset", "set", "tuple", "list")
            and value.args):
        return _literal_strings(value.args[0])
    if isinstance(value, (ast.List, ast.Tuple, ast.Set)):
        return [e.value for e in value.elts
                if isinstance(e, ast.Constant) and isinstance(e.value, str)]
    return None


def _collection_named(tree: ast.Module, name: str) -> list[str]:
    """
    Every string in the collection assigned to `name`.

    Raises when the name is absent rather than returning an empty list: a static
    check with nothing to look at otherwise reports success, which is the failure
    mode this whole section exists to avoid.
    """
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
        else:
            continue
        if not any(isinstance(t, ast.Name) and t.id == name for t in targets):
            continue
        strings = _literal_strings(node.value)
        if strings is not None:
            return strings
    raise AssertionError(
        f"no collection literal named {name} found — this test cannot check a "
        f"rail that has been renamed or computed instead of written out"
    )


def _enum_members(tree: ast.Module, class_name: str) -> dict[str, str]:
    """Member name → value for a `class X(str, Enum)` body, without importing it."""
    for node in ast.walk(tree):
        if not (isinstance(node, ast.ClassDef) and node.name == class_name):
            continue
        members: dict[str, str] = {}
        for stmt in node.body:
            if (isinstance(stmt, ast.Assign) and len(stmt.targets) == 1
                    and isinstance(stmt.targets[0], ast.Name)
                    and isinstance(stmt.value, ast.Constant)
                    and isinstance(stmt.value.value, str)):
                members[stmt.targets[0].id] = stmt.value.value
        assert members, f"{class_name} has no string members"
        return members
    raise AssertionError(f"class {class_name} not found")


class TestActionsAreDefenseOnlyStatically:
    """
    "Anything offense-capable is disqualified" — checked on the source text, so it
    holds on a bare checkout.

    tests/test_responder.py covers the same constraint far more thoroughly, but its
    first statement is `pytest.importorskip("pydantic")`, and a skipped module skips
    every test in it. On a machine without pydantic installed the ONLY project
    guarantee that carries a disqualification penalty went entirely unchecked, and
    the suite still printed green. Parsing costs nothing and cannot be skipped, so
    the floor of that guarantee lives here: no action name the repo ships reads as
    enforcement, and the two places the permitted set is written agree.
    """

    def test_no_shipped_action_name_reads_as_enforcement(self):
        """
        Both the enum members and the responder's allowlist, names and values
        alike — `LOCK_ACCOUNT = "REVIEW"` would be an enforcement action with an
        innocent value, and vice versa.
        """
        members = _enum_members(_tree("api/schemas.py"), "ActionCode")
        permitted = _collection_named(_tree("api/responder.py"),
                                      "PERMITTED_ACTIONS")

        candidates = set(members) | set(members.values()) | set(permitted)
        offenders = [
            (name, verb) for name in sorted(candidates)
            for verb in ENFORCEMENT_VERBS if verb in name.upper()
        ]
        assert not offenders, (
            "action names that read as enforcement, which is a disqualification "
            f"criterion for this project: {offenders}. The Sentinel recommends "
            "review; it does not act on accounts."
        )

    def test_the_enum_and_the_allowlist_name_the_same_actions(self):
        """
        The runtime sweep in api/responder.py requires these to match exactly, and
        that sweep needs pydantic to run. Checked here as text so a stale allowlist
        entry — still permitting a string the enum renamed away — is caught on any
        checkout.
        """
        members = _enum_members(_tree("api/schemas.py"), "ActionCode")
        permitted = _collection_named(_tree("api/responder.py"),
                                      "PERMITTED_ACTIONS")
        assert set(members.values()) == set(permitted), (
            f"only in ActionCode: {sorted(set(members.values()) - set(permitted))}; "
            f"only in PERMITTED_ACTIONS: "
            f"{sorted(set(permitted) - set(members.values()))}"
        )

    def test_the_action_rail_still_fails_closed(self):
        """
        That `_validate_action` tests membership of the allowlist at all.

        Shallow on purpose — it looks for the `not in PERMITTED_ACTIONS` comparison
        and no further. The behaviour is tested properly in test_responder.py; what
        cannot be tested there is the case where pydantic is missing, and the defect
        worth catching without it is structural: reverting to blocklists alone,
        which permits by default and let `LOCK_ACCOUNT` through.
        """
        tree = _tree("api/responder.py")
        function = next(
            (n for n in ast.walk(tree)
             if isinstance(n, ast.FunctionDef) and n.name == "_validate_action"),
            None)
        assert function is not None, "api/responder.py has no _validate_action"

        checks_allowlist = any(
            isinstance(node, ast.Compare)
            and any(isinstance(op, ast.NotIn) for op in node.ops)
            and any(isinstance(c, ast.Name) and c.id == "PERMITTED_ACTIONS"
                    for c in node.comparators)
            for node in ast.walk(function)
        )
        assert checks_allowlist, (
            "_validate_action no longer refuses actions outside PERMITTED_ACTIONS. "
            "A blocklist permits by default and cannot enumerate every way of "
            "saying 'stop this account'; this rail has to fail closed."
        )

