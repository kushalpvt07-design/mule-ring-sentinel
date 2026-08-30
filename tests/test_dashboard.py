"""
tests/test_dashboard.py
──────────────────────
The demo surface, tested — because it is the part a reviewer actually clicks.

WHY THIS FILE EXISTS
────────────────────
The README concedes that the Streamlit dashboard and the FastAPI scoring endpoint
carry no automated tests, which made them the only code in the project whose
claims nobody checked. That is the wrong place to have a gap: the dashboard is
where a viewer reads "financial impact" off a screen, and v2 computed that screen
from a simulated score containing the ground-truth label:

    y_proba = np.clip(raw_scores + noise + y_true * 0.3, 0.01, 0.99)

Every precision, recall and rupee figure on that page was fiction, and no test
would have caught it. `dashboard/scoring.py` was written to make that impossible;
this file is the part that verifies it stayed impossible.

WHAT IS TESTED HERE, AND WHAT DELIBERATELY IS NOT
─────────────────────────────────────────────────
`dashboard/scoring.py` is pure and importable — no Streamlit, no booster needed
for most of it — so its guarantees are tested behaviourally: the threshold has no
default, a stale metrics.json stops the page, a label cannot reach the model, and
the importance panel names which of the two importances is on screen.

`dashboard/app.py` cannot be imported without Streamlit, and importing it runs
`st.set_page_config` at module scope. Rather than fight that, the claims about it
that matter are checked against its SOURCE: the API endpoint is configurable, and
the score path is not reachable from anything that knows a label. A source test is
weaker than a behavioural one and is used only where a behavioural one would
require standing up a server; where the code is importable, it is imported.

Everything that needs the trained booster, the processed CSVs, or Streamlit skips
cleanly, so this file stays green on a fresh clone — see conftest's report header,
which states what is on disk so a skipped run cannot be mistaken for a passing one.

Usage:
    pytest tests/test_dashboard.py -v
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dashboard.scoring import (
    METRICS_PATH,
    ModelUnavailable,
    feature_importance,
    load_metrics,
    resolve_threshold,
    score_frame,
)
from models.features import FEATURE_COLS, MODEL_VERSION, TARGET_COL

PROJECT_ROOT = Path(__file__).resolve().parent.parent
APP_SOURCE = PROJECT_ROOT / "dashboard" / "app.py"
SCORING_SOURCE = PROJECT_ROOT / "dashboard" / "scoring.py"
COMPONENT_SOURCES = (
    PROJECT_ROOT / "dashboard" / "components" / "cost_slider.py",
    PROJECT_ROOT / "dashboard" / "components" / "graph_viz.py",
)


def _source(path: Path) -> str:
    if not path.exists():                       # pragma: no cover — repo layout
        pytest.skip(f"{path.name} not found")
    return path.read_text(encoding="utf-8")


def _tree(path: Path) -> ast.Module:
    """Parse a module. Cannot be skipped and needs nothing installed."""
    return ast.parse(_source(path), filename=str(path))


def _attribute_path(node: ast.Attribute) -> str | None:
    """`np.random.default_rng` from the attribute chain, or None if not a chain."""
    parts: list[str] = []
    current: ast.expr = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if not isinstance(current, ast.Name):
        return None
    parts.append(current.id)
    return ".".join(reversed(parts))


def _called_name(node: ast.Call) -> str | None:
    """The dotted name being called, or None for a call on an expression."""
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return _attribute_path(node.func) or node.func.attr
    return None


class _RecordingModel:
    """
    A stand-in booster that records exactly which columns it was handed.

    Not a mock of convenience: the v2 defect was a label reaching the scorer, and
    the only way to prove it cannot is to look at what the model actually
    received. A real booster would score the frame and tell us nothing about
    where the numbers came from.
    """

    def __init__(self):
        self.saw: list[str] | None = None

    def predict_proba(self, X):
        self.saw = list(X.columns)
        n = len(X)
        return np.column_stack([np.full(n, 0.9), np.full(n, 0.1)])


def _frame_with_everything(n: int = 8) -> pd.DataFrame:
    """A feature frame shaped like the real CSV — labels and metadata included."""
    rng = np.random.default_rng(4)
    frame = pd.DataFrame({c: rng.uniform(size=n) for c in FEATURE_COLS})
    frame["node"] = [f"acct_{i}" for i in range(n)]
    frame[TARGET_COL] = (rng.uniform(size=n) < 0.4).astype(int)
    frame["ring_id"] = -1
    frame["ring_type"] = "none"
    frame["split"] = "test"
    return frame


# ══════════════════════════════════════════════════════════════════
# 1. The page cannot invent a score
# ══════════════════════════════════════════════════════════════════

class TestTheDashboardCannotInventScores:
    """
    The one guarantee the whole module exists for. Everything else on the page is
    a presentation choice; this is a correctness property.
    """

    def test_the_label_never_reaches_the_model(self):
        """
        `score_frame` selects FEATURE_COLS explicitly, so `is_mule` is dropped by
        construction rather than by remembering to drop it. This is the direct
        regression test for the v2 defect: the label was in the score, and the
        page reported the result as the model's performance.
        """
        model = _RecordingModel()
        frame = _frame_with_everything()
        score_frame(model, frame)
        assert model.saw == list(FEATURE_COLS), (
            f"the model was handed {model.saw}, not the feature contract "
            f"{list(FEATURE_COLS)}."
        )
        for leaked in (TARGET_COL, "ring_id", "ring_type", "split", "node"):
            assert leaked not in (model.saw or []), (
                f"{leaked!r} reached predict_proba. On a scoring path that feeds "
                f"a page labelled 'financial impact', a label column in the "
                f"input makes every figure on screen unfalsifiable."
            )

    def test_column_order_is_part_of_the_contract(self):
        """
        XGBoost matches columns POSITIONALLY. A frame whose columns are in a
        different order scores plausibly and wrongly, with nothing raised — so
        the selection must reorder rather than merely subset.
        """
        model = _RecordingModel()
        frame = _frame_with_everything()
        shuffled = frame[list(reversed(frame.columns))]
        score_frame(model, shuffled)
        assert model.saw == list(FEATURE_COLS), (
            f"a column-shuffled frame reached the model as {model.saw}. "
            f"Positional matching means those scores would be computed from the "
            f"wrong features."
        )

    def test_a_missing_feature_raises_instead_of_scoring(self):
        """
        The extractor and the feature contract can drift. Scoring on whatever
        columns happen to be present is the failure that produces a full page of
        confident numbers from the wrong inputs.
        """
        frame = _frame_with_everything().drop(columns=[FEATURE_COLS[0]])
        with pytest.raises(ModelUnavailable, match=FEATURE_COLS[0]):
            score_frame(_RecordingModel(), frame)

    def test_no_simulated_score_has_come_back(self):
        """
        A structural guard on the specific defect. `_simulate_predictions` was
        deleted, and the way it would return is as a well-intentioned "demo mode"
        for a machine with no trained model — the honest answer there is an
        explanation, which is what `ModelUnavailable` carries.

        Parsed rather than grepped, and that is not fussiness: BOTH
        dashboard/scoring.py and dashboard/components/cost_slider.py name
        `_simulate_predictions` in their docstrings, because explaining the defect
        is how the fix stays understood. A substring test would forbid the
        documentation and pass the moment someone deleted it. So this looks at
        definitions and calls, where the defect would actually live.
        """
        for path in (APP_SOURCE, SCORING_SOURCE, *COMPONENT_SOURCES):
            tree = _tree(path)
            defined = [n.name for n in ast.walk(tree)
                       if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                       and "simulat" in n.name.lower()]
            assert not defined, (
                f"{path.name} defines {defined}. Scores on this page come from "
                f"the booster or they do not come at all — see the module "
                f"docstring in dashboard/scoring.py for what a simulated score "
                f"did to the v2 cost page."
            )
            called = sorted({
                _called_name(n) for n in ast.walk(tree)
                if isinstance(n, ast.Call)
                and "simulat" in (_called_name(n) or "").lower()
            })
            assert not called, f"{path.name} calls {called}."

    def test_no_random_number_reaches_a_displayed_figure(self):
        """
        The other half of the same defect: v2's simulated score was
        `features + noise + label * 0.3`. A seeded RNG anywhere on the scoring or
        rendering path means at least one number on a page labelled "financial
        impact" was drawn rather than measured.

        Sample data for a *demo input form* is a different thing, so this checks
        the modules that compute and display results, not the whole package.

        `APP_SOURCE` IS IN THE LIST NOW. It was omitted while the constant was
        defined in this file and used by four of its siblings, which left the one
        module where the rupee figures are actually computed out of the scan — the
        most likely home for the defect, unchecked. Nothing in app.py touches
        `random` today, so adding it costs nothing; the point is that it can no
        longer start to without failing here. The sibling above
        (`test_no_simulated_score_has_come_back`) already scans all four, so the
        omission was this test's alone.
        """
        for path in (APP_SOURCE, SCORING_SOURCE, *COMPONENT_SOURCES):
            uses = sorted({
                _attribute_path(n) for n in ast.walk(_tree(path))
                if isinstance(n, ast.Attribute)
                and "random" in (_attribute_path(n) or "")
            })
            assert not uses, (
                f"{path.name} uses {uses}. Every figure on the results pages has "
                f"to be traceable to the booster's output on real features."
            )

    def test_the_scorer_has_no_fallback_that_returns_numbers(self):
        """
        `ModelUnavailable` only helps if nothing swallows it. A broad handler in
        the scoring path can only end in re-raising (so write the narrow type) or
        substituting a number nobody measured — and the second is what turns a
        missing model into a page of plausible fiction.
        """
        broad = []
        for node in ast.walk(_tree(SCORING_SOURCE)):
            if not isinstance(node, ast.ExceptHandler):
                continue
            if node.type is None:
                broad.append("bare except")
            elif isinstance(node.type, ast.Name) and node.type.id in (
                    "Exception", "BaseException"):
                broad.append(f"except {node.type.id}")
        assert not broad, (
            f"dashboard/scoring.py has {broad}. Narrow the handler: the only "
            f"failures this module is allowed to absorb are the ones it can "
            f"explain to the viewer as a ModelUnavailable."
        )


# ══════════════════════════════════════════════════════════════════
# 2. The threshold has no safe default
# ══════════════════════════════════════════════════════════════════

class TestTheThresholdHasNoSafeDefault:
    """
    v2 used `metrics.get("optimal_threshold", 0.5)`. With a miss priced at 13.3x
    a false alert the real operating point is 0.1836 on the shipped model, so that
    default throws away most of the recall the model was tuned for — and reports
    the result as the model's performance. The threshold is an economic quantity;
    there is no value to guess.

    (This paragraph used to say the operating point "sits near 0.07", i.e. at the
    break-even probability p*. That is where the optimum would sit if the booster's
    scores were calibrated, and they are not; the fitted minimiser on validation is
    2.6x higher. The distinction is the whole subject of the last test below.)
    """

    def test_a_missing_threshold_stops_the_page(self):
        with pytest.raises(ModelUnavailable, match="no safe default"):
            resolve_threshold({"model_version": MODEL_VERSION})

    def test_the_published_threshold_is_used_verbatim(self):
        """No rounding, no clamping: the number on screen is the number chosen."""
        assert resolve_threshold({"optimal_threshold": 0.0712345}) == 0.0712345

    @pytest.mark.parametrize("bad", [0.0, -0.1, 1.5, 2.0])
    def test_an_out_of_range_threshold_is_refused(self, bad):
        """
        A threshold outside (0, 1] is not a probability. 0.0 in particular flags
        every account and would render a page whose queue is the entire network,
        described as the cost-optimal operating point.
        """
        with pytest.raises(ModelUnavailable, match="out-of-range"):
            resolve_threshold({"optimal_threshold": bad})

    # How far the fitted operating point may sit from the break-even probability
    # before the dashboard is presumed to be reading the wrong file. Two-sided and
    # deliberately wide: the ratio is the output of a fitting run, so a retrain is
    # allowed to move it. What it is not allowed to do is leave the neighbourhood.
    #
    # Measured on the shipped metrics.json: threshold 0.183564, p* 0.069767,
    # ratio 2.631. The ceiling is set at 6.0 because a file carrying v2's
    # cost-insensitive 0.5 default lands at 0.5 / 0.069767 = 7.17x, so 6.0 catches
    # that with room to spare on either side of the real value. The floor at 0.25x
    # catches the opposite failure — a threshold collapsed far below anything the
    # cost matrix justifies, which would render a queue of most of the network.
    P_STAR_RATIO_FLOOR = 0.25
    P_STAR_RATIO_CEILING = 6.0

    def test_the_shipped_threshold_sits_in_the_neighbourhood_of_break_even(
            self, metrics):
        """
        Not a re-test of the model — a check that what the dashboard will resolve
        is the cost-sensitive operating point and not something near 0.5.

        THIS TEST USED TO BE CALLED `test_the_shipped_threshold_is_below_the_break_even`
        AND ITS NAME WAS FALSE. It loaded `p_star` into a local and then never
        mentioned it again, so the claim in the name — and the matching sentence in
        the docstring, "the optimum sits well below break-even p*" — was never
        compared against anything. Measured, the shipped threshold is 0.183564 and
        p* is 0.069767: the operating point is **2.631x ABOVE** p*, the opposite of
        what the test was named for, and the test was green.

        WHY IT IS ABOVE, WHICH IS NOT A DEFECT. p* = fp / (fp + fn) is the
        cost-minimising cut for a *calibrated* probability — the point where the
        expected cost of alerting equals the expected cost of not. XGBoost's
        sigmoid output is not calibrated, so the empirically cost-minimising cut on
        validation has no reason to land on p*, and here it lands higher: at the
        margin the booster is more confident than a calibrated score would be, so
        the same expected cost is reached at a higher number. p* remains the floor
        of the step-up band in api/responder.py, which is a different job — it is
        where the cost model says "stop calling this benign", not where the fitted
        model says "alert".

        WHAT IS ASSERTED INSTEAD. Two things that survive a retrain:

          1. p* recomputed from the two prices equals the p* metrics.json
             publishes. It is an identity over fn_cost and fp_cost, so a file where
             they disagree is internally inconsistent, and every band in the
             responder is derived from the published side.
          2. The threshold sits within a bounded factor of p* (see the constants
             above) and below 0.5. A ratio, not a direction: pinning the direction
             would be pinning one fitting run, which is exactly the failure this
             test already committed once.

        The two assertions this replaces did no work. `assert 0.0 < threshold < 1.0`
        restates the out-of-range refusal that `resolve_threshold` raises on — and
        which `test_an_out_of_range_threshold_is_refused` above already covers — so
        it can only fire on a value that never got this far. `assert p_star > 0.0`
        could not fail on any file with a positive fn_cost, and p* is now a divisor,
        so it is load-bearing rather than decorative.

        MUTATION RESULT, since that is the bar. Six mutants: threshold 0.5, 0.49,
        0.01; fn_cost halved, fp_cost doubled, and break_even_probability set to 0.5,
        each with the other side left stale. The body above catches all six. The
        body it replaces caught ONE — threshold 0.5, and only because 0.5 is not
        strictly below 0.5; nudging the mutant to 0.49 defeated it. The band admits
        0.25x through 5.88x p* and rejects from 6.02x, so the shipped 2.631x has
        room on both sides.
        """
        threshold = resolve_threshold(metrics)
        cost = metrics["cost_config"]
        p_star = cost["break_even_probability"]

        fn_cost, fp_cost = cost["fn_cost"], cost["fp_cost"]
        recomputed = fp_cost / (fp_cost + fn_cost)
        assert recomputed == pytest.approx(p_star, abs=1e-6), (
            f"metrics.json publishes break_even_probability={p_star} but its own "
            f"prices give fp/(fp+fn) = {fp_cost}/({fp_cost}+{fn_cost}) = "
            f"{recomputed:.6f}. The published value is what api/responder.py uses "
            f"as the floor of the step-up band, so an inconsistent file moves the "
            f"tier boundaries away from the cost model they claim to implement.\n"
            f"  Fix: python -m models.train"
        )

        assert threshold < 0.5, (
            f"the dashboard would operate at {threshold}, near the default a "
            f"cost-insensitive model would use. At {cost['fn_fp_ratio']}:1 the "
            f"optimum is far below 0.5 — check which metrics.json this is."
        )

        ratio = threshold / p_star
        assert self.P_STAR_RATIO_FLOOR <= ratio <= self.P_STAR_RATIO_CEILING, (
            f"the shipped threshold is {threshold:.6f}, which is {ratio:.3f}x the "
            f"break-even probability {p_star:.6f} — outside the "
            f"[{self.P_STAR_RATIO_FLOOR}, {self.P_STAR_RATIO_CEILING}] band this "
            f"test allows. A fitted operating point is allowed to move away from "
            f"p* (uncalibrated scores; the shipped value is 2.631x above it), but "
            f"this far out means either the dashboard is resolving a threshold "
            f"from a different model's file, or the cost matrix and the training "
            f"run no longer describe the same problem.\n"
            f"  Fix: confirm models/saved_models/metrics.json is the file the "
            f"current `python -m models.train` wrote."
        )


# ══════════════════════════════════════════════════════════════════
# 3. A stale metrics.json stops the page rather than mislabelling it
# ══════════════════════════════════════════════════════════════════

class TestAStaleMetricsFileStopsThePage:
    """
    metrics.json is the dashboard's only source for the threshold and every
    headline number. A file describing a different model is worse than no file:
    the page renders, looks finished, and attributes another model's results to
    this one.
    """

    def test_a_version_mismatch_refuses_to_load(self, tmp_path, monkeypatch):
        stale = tmp_path / "metrics.json"
        stale.write_text(json.dumps({"model_version": "sentinel_v1",
                                     "optimal_threshold": 0.07}))
        monkeypatch.setattr("dashboard.scoring.METRICS_PATH", stale)
        with pytest.raises(ModelUnavailable, match="sentinel_v1"):
            load_metrics()

    def test_a_missing_file_names_the_command_that_makes_one(self, tmp_path,
                                                             monkeypatch):
        """
        Also pins the error path against a second failure mode it used to have:
        the message was built with `METRICS_PATH.parent.relative_to(
        PROJECT_ROOT)`, which raises `ValueError` for any path outside the repo.
        Pointing METRICS_PATH somewhere else — as a test must, and as a
        relocated deployment would — replaced the sentence naming the command
        with a stack trace from inside the error handler.
        """
        monkeypatch.setattr("dashboard.scoring.METRICS_PATH",
                            tmp_path / "absent.json")
        with pytest.raises(ModelUnavailable, match="models.train"):
            load_metrics()

    def test_the_current_file_loads(self):
        """
        The positive case, against whatever is actually on disk. Skips on a fresh
        clone; fails loudly if the file is present and describes another model,
        which is exactly the state a viewer must not be shown.
        """
        if not METRICS_PATH.exists():
            pytest.skip("no metrics.json on disk; run `python -m models.train`")
        loaded = load_metrics()
        assert loaded.get("model_version") == MODEL_VERSION


# ══════════════════════════════════════════════════════════════════
# 4. The importance panel says which importance it is showing
# ══════════════════════════════════════════════════════════════════

class TestFeatureImportanceSaysWhichKindItIs:
    """
    Gain and mean |SHAP| answer different questions — how much the split
    criterion improved during TRAINING versus how much a feature moves
    predictions on TEST. A viewer reading "feature importance" assumes the
    second. Presenting either without naming it invites the wrong reading, so the
    caption is part of the contract and is tested like one.
    """

    def test_shap_is_preferred_and_named(self):
        frame, source = feature_importance({
            "feature_importance_mean_abs_shap": {"pagerank": 0.4,
                                                 "cycle_participation": 0.2},
            "feature_importance": {"pagerank": 99.0},
        })
        assert "SHAP" in source
        assert len(frame) == 2

    def test_gain_is_used_only_as_a_fallback_and_is_flagged_as_training_time(self):
        _, source = feature_importance({"feature_importance": {"pagerank": 9.0}})
        assert "gain" in source.lower()
        assert "training" in source.lower(), (
            f"the gain caption reads {source!r} and does not say the numbers are "
            f"from training. Unlabelled beside a test-set results page, gain "
            f"reads as test-time influence."
        )

    def test_neither_present_raises_rather_than_rendering_an_empty_panel(self):
        with pytest.raises(ModelUnavailable, match="importance"):
            feature_importance({"model_version": MODEL_VERSION})

    def test_the_shipped_importance_covers_the_feature_contract(self, metrics):
        """
        A panel listing features the model does not use, or omitting ones it
        does, is a page describing a different model.
        """
        frame, _ = feature_importance(metrics)
        listed = set(frame["feature"]) if "feature" in frame.columns \
            else set(frame.index)
        assert listed <= set(FEATURE_COLS), (
            f"the importance panel would show {sorted(listed - set(FEATURE_COLS))}"
            f", which are not in the feature contract."
        )


# ══════════════════════════════════════════════════════════════════
# 5. The demo can reach an API that is not on this machine
# ══════════════════════════════════════════════════════════════════

class TestTheApiEndpointIsConfigurable:
    """
    The Score Demo posts to a URL. Hardcoded to localhost it works only on the
    machine running `uvicorn`, which makes the one interactive part of the
    project undemonstrable anywhere else — a container, a remote backend, a
    reviewer's laptop with the API on another host.

    Source-level rather than behavioural because importing `dashboard/app.py`
    runs `st.set_page_config` at module scope. The behavioural half is the import
    smoke test below, which runs when Streamlit is installed.
    """

    def test_the_url_comes_from_the_environment(self):
        source = _source(APP_SOURCE)
        assert 'os.getenv("API_URL"' in source, (
            "dashboard/app.py does not read API_URL from the environment. "
            "Hardcoded, the Score Demo only works on the machine running the "
            "API."
        )

    def test_the_hardcoded_url_survives_only_as_the_default(self):
        """
        One occurrence, inside the `os.getenv` default. A second one somewhere
        else means a code path that ignores the override and silently posts to
        localhost.
        """
        source = _source(APP_SOURCE)
        code = "\n".join(line for line in source.splitlines()
                         if not line.lstrip().startswith("#"))
        assert code.count("http://localhost:8000") <= 1, (
            "dashboard/app.py mentions http://localhost:8000 more than once "
            "outside comments. Every request must go through API_URL, or "
            "overriding it fixes only some of them."
        )

    def test_requests_is_imported_at_module_scope(self):
        """
        `import requests` inside the handler re-runs the import on every click
        and hides a missing dependency until a user presses the button. At the
        top, a broken environment fails at start-up where it can be read.
        """
        source = _source(APP_SOURCE)
        top = source.split("def ")[0]
        assert "\nimport requests" in top, (
            "dashboard/app.py does not import requests at module scope."
        )
        body = source[len(top):]
        assert "import requests" not in body, (
            "dashboard/app.py still imports requests inside a function."
        )

    def test_the_error_path_tells_the_viewer_which_url_failed(self):
        """
        With the URL configurable, "cannot reach the API" is no longer actionable
        on its own — the whole point is that it might not be localhost.
        """
        source = _source(APP_SOURCE)
        assert "Cannot reach the API at {API_URL}" in source, (
            "the connection-error message does not name the URL it tried. Once "
            "the endpoint is configurable, the address is the useful half of "
            "the message."
        )


# ══════════════════════════════════════════════════════════════════
# 6. The page imports at all
# ══════════════════════════════════════════════════════════════════

class TestTheDashboardImports:
    """
    The cheapest test that would have caught a syntax error, a bad import or a
    module-scope crash in the one file nobody runs under pytest. Skips without
    Streamlit, so a fresh clone stays green.
    """

    def test_app_imports_with_streamlit_present(self, monkeypatch):
        pytest.importorskip("streamlit",
                            reason="streamlit is required to import the page")
        monkeypatch.setenv("API_URL", "http://sentinel.example:9000/score")
        import importlib

        module = importlib.import_module("dashboard.app")
        importlib.reload(module)
        assert module.API_URL == "http://sentinel.example:9000/score", (
            f"API_URL resolved to {module.API_URL!r} with the environment "
            f"variable set. The override is not taking effect."
        )

    def test_the_components_import(self):
        pytest.importorskip("streamlit")
        import importlib

        for name in ("dashboard.components.cost_slider",
                     "dashboard.components.graph_viz"):
            importlib.import_module(name)
