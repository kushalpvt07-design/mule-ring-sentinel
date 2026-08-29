"""
tests/test_baselines.py
───────────────────────
The adversarial baseline: is the label learnable, and does the model earn its keep?

models/features.py rule 4, generator.py:103 and train.py:163 all cite this file
by name. It carries the two questions a fraud model can fail silently:

  1. IS THE DATASET HONEST?  If one feature alone separates the classes at
     AUC ≥ 0.99, the generator planted the label — the model is memorising a
     watermark, and every downstream metric is theatre. `test_no_single_feature
     _solves_the_task` recomputes the strongest direction-corrected single-feature
     AUC straight from the test table and fails at the ceiling. This is measured
     on the shipped data, not read from a file that a stale run could have written.

  2. DOES THE MODEL BEAT THE ONE-LINE RULE?  A graph pipeline, a gradient-boosted
     ensemble and a cost matrix have to beat `if cycle_participation >= t: review`
     and the trivial flag-all / flag-none policies, priced on the SAME cost model,
     before any of the complexity is justified. `TestModelEarnsItsComplexity`
     checks exactly that against the published metrics.json.

Direction correction is the subtle part. A feature at raw AUC 0.13 is not weak —
it is strongly inverted, and any model that can flip a sign has that separating
power in hand. So the ceiling is applied to `max(auc, 1 - auc)`, never the raw
value. `test_the_auc_instrument_is_itself_correct` proves the AUC helper reports
1.0 / 0.5 / 0.0 on perfect / constant / inverted input, because a ceiling test
built on a broken ruler is worse than none.

Usage:
    pytest tests/test_baselines.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

from models.cost_matrix import roc_auc
from models.features import FEATURE_COLS, MODEL_VERSION, TARGET_COL

# The leakage ceiling. Must match models.train.LEAKAGE_AUC_CEILING; that module
# imports xgboost, so the value is stated here (dependency-free) and the agreement
# is asserted separately under an importorskip guard.
LEAKAGE_AUC_CEILING = 0.99

# The strongest single feature on the shipped test split measured 0.8708
# (cycle_participation). A comfortable margin below the ceiling is expected; a
# jump toward 0.99 means a generator change started leaking the label.
EXPECTED_STRONGEST_AUC_APPROX = 0.87


def _require_current_metrics(metrics) -> None:
    """
    Guard for every metrics-dependent test EXCEPT the version check itself.

    A stale metrics.json is reported once — loudly — by
    `test_metrics_describe_the_current_model_version`. The remaining tests would
    only add redundant tracebacks (or raw KeyErrors, since a retired payload
    predates the `baselines` block), so they skip and defer to that one signal.
    """
    version = metrics.get("model_version")
    if version != MODEL_VERSION:
        pytest.skip(
            f"metrics.json is for {version!r}, code is at {MODEL_VERSION!r}; "
            f"see test_metrics_describe_the_current_model_version. "
            f"Regenerate with `python -m models.train`."
        )


def _directional_auc(y: np.ndarray, x: np.ndarray) -> float:
    """The separating power available to a model free to choose the sign."""
    auc = roc_auc(y, x)
    return max(auc, 1.0 - auc)


def _strongest_single_feature(frame) -> tuple[str, float]:
    y = frame[TARGET_COL].to_numpy()
    best_feature, best_auc = "", 0.0
    for feature in FEATURE_COLS:
        auc = _directional_auc(y, frame[feature].to_numpy(dtype=float))
        if auc > best_auc:
            best_feature, best_auc = feature, auc
    return best_feature, best_auc


# ══════════════════════════════════════════════════════════════════
# 0. The ruler, before anything is measured with it
# ══════════════════════════════════════════════════════════════════

class TestTheInstrumentIsCorrect:
    """
    A leakage ceiling is only as trustworthy as the AUC it compares. These four
    cases pin the helper to sklearn's definition on the corners that matter.
    """

    def test_perfect_separation_is_one(self):
        y = np.array([0, 0, 1, 1])
        s = np.array([0.1, 0.2, 0.8, 0.9])
        assert roc_auc(y, s) == pytest.approx(1.0)

    def test_inverted_separation_is_zero(self):
        """The case direction-correction exists for: perfectly wrong is AUC 0."""
        y = np.array([0, 0, 1, 1])
        s = np.array([0.9, 0.8, 0.2, 0.1])
        assert roc_auc(y, s) == pytest.approx(0.0)
        assert _directional_auc(y, s) == pytest.approx(1.0)

    def test_a_constant_score_is_one_half(self):
        """All ties average to rank n/2, so a useless feature reads 0.5, not NaN."""
        y = np.array([0, 1, 0, 1])
        s = np.array([0.5, 0.5, 0.5, 0.5])
        assert roc_auc(y, s) == pytest.approx(0.5)

    def test_a_single_class_slice_is_nan_not_a_crash(self):
        """Per-archetype loops hit single-class slices; NaN is the useful answer."""
        assert np.isnan(roc_auc(np.array([1, 1, 1]), np.array([0.2, 0.5, 0.9])))


# ══════════════════════════════════════════════════════════════════
# 1. Is the dataset honest?  (measured on the shipped data)
# ══════════════════════════════════════════════════════════════════

class TestDatasetIsNotWatermarked:
    """
    The single most important test in the suite. If it ever fails, nothing else
    in the repo means anything: the generator is writing the answer into a column.
    """

    def test_no_single_feature_solves_the_task(self, node_features):
        feature, auc = _strongest_single_feature(node_features["test"])
        assert auc < LEAKAGE_AUC_CEILING, (
            f"'{feature}' alone reaches direction-corrected AUC {auc:.4f} on test, "
            f"at or above the {LEAKAGE_AUC_CEILING} leakage ceiling. A single "
            f"column this separating was planted by data/generator.py, not learned "
            f"— fix the generator; every metric in the repo is meaningless until "
            f"you do."
        )

    def test_the_strongest_feature_is_the_intended_one(self, node_features):
        """
        Not a leakage guard — a description check. The design intends the ring
        signal (cycle participation / amount dispersion / centrality) to be the
        strongest single cue. If some arithmetic column overtakes it, the feature
        set drifted from what the write-up claims.
        """
        feature, auc = _strongest_single_feature(node_features["test"])
        assert auc == pytest.approx(EXPECTED_STRONGEST_AUC_APPROX, abs=0.08), (
            f"strongest single feature is '{feature}' at AUC {auc:.4f}; the write-up "
            f"expects roughly {EXPECTED_STRONGEST_AUC_APPROX}. Regenerate the data or "
            f"update the documented figure."
        )

    def test_ceiling_agrees_with_the_training_module(self):
        """The value stated here must be the one the training run actually enforces."""
        pytest.importorskip("xgboost", reason="models.train imports xgboost")
        pytest.importorskip("sklearn", reason="models.train imports scikit-learn")
        from models.train import LEAKAGE_AUC_CEILING as TRAIN_CEILING

        assert LEAKAGE_AUC_CEILING == TRAIN_CEILING, (
            f"tests say the ceiling is {LEAKAGE_AUC_CEILING}, models/train.py "
            f"enforces {TRAIN_CEILING} — one of them is stale."
        )


# ══════════════════════════════════════════════════════════════════
# 2. The published metrics.json describes THIS model
# ══════════════════════════════════════════════════════════════════

class TestMetricsAreNotStale:
    """
    conftest skips these when metrics.json is absent (a fresh clone). It does NOT
    skip on a version mismatch: a metrics.json describing another model is a
    published claim about a model that no longer exists, and that is a failure,
    not a missing artefact.

    Exactly ONE test reports that failure —
    `test_metrics_describe_the_current_model_version`. Every other
    metrics-dependent test defers to it via `_require_current_metrics`, so a stale
    file produces one actionable red line instead of five tracebacks that all mean
    "re-run models.train".
    """

    def test_metrics_describe_the_current_model_version(self, metrics):
        assert metrics.get("model_version") == MODEL_VERSION, (
            f"metrics.json reports model_version {metrics.get('model_version')!r} "
            f"but the code is at {MODEL_VERSION!r}. Re-run `python -m models.train`; "
            f"do not ship metrics for a retired model."
        )

    def test_metrics_feature_list_matches_the_contract(self, metrics):
        """The metrics were computed on some feature set; it must be the current one."""
        _require_current_metrics(metrics)
        assert metrics.get("feature_cols") == list(FEATURE_COLS), (
            "metrics.json was computed against a different feature list than "
            "FEATURE_COLS declares — the numbers describe a different model."
        )

    def test_published_leakage_check_matches_the_data(self, metrics, node_features):
        """
        metrics.json publishes the strongest single-feature AUC. Recompute it from
        the test table and require the published figure to match, so a stale run
        cannot advertise a clean dataset that the shipped data no longer is.
        """
        _require_current_metrics(metrics)
        block = (metrics.get("baselines", {})
                 .get("best_single_feature_rule_by_cost", {})
                 .get("strongest_single_feature_test_auc"))
        if not block:
            pytest.skip("metrics.json predates the strongest-single-feature block")

        _, live = _strongest_single_feature(node_features["test"])
        assert block["auc"] == pytest.approx(live, abs=0.01), (
            f"metrics.json advertises strongest single-feature AUC {block['auc']} "
            f"but the shipped test data gives {live:.4f}. The published leakage "
            f"claim does not describe the data in the repo."
        )
        assert block["auc"] < LEAKAGE_AUC_CEILING


# ══════════════════════════════════════════════════════════════════
# 3. Does the model earn its complexity?
# ══════════════════════════════════════════════════════════════════

class TestModelEarnsItsComplexity:
    """
    The honest-lift claim. All costs below are TEST costs priced on the one cost
    model, so the comparison is like for like — the reason the baseline rule stores
    its threshold rather than a fixed cost.
    """

    def test_the_model_beats_the_cheapest_trivial_policy(self, metrics):
        """Flag-everyone and flag-no-one are the floor; failing this is pointless."""
        _require_current_metrics(metrics)
        trivial = metrics["baselines"]["trivial"]["cheapest_trivial_policy_cost"]
        model_cost = metrics["total_cost"]
        assert model_cost < trivial, (
            f"the model's test cost ₹{model_cost:,.0f} is not below the cheapest "
            f"trivial policy ₹{trivial:,.0f}. A model that cannot beat "
            f"flag-everything / flag-nothing has no reason to exist."
        )

    def test_the_model_beats_the_best_single_feature_rule(self, metrics):
        """
        The bar the whole pipeline exists to clear. If the one-line rule wins, the
        graph features, the ensemble and the cost matrix are unjustified — and this
        failing is the honest signal to hear before submitting, not after.
        """
        _require_current_metrics(metrics)
        rule = metrics["baselines"]["best_single_feature_rule_by_cost"]
        model_cost = metrics["total_cost"]
        assert model_cost <= rule["test_total_cost"], (
            f"the full model's test cost ₹{model_cost:,.0f} does not beat the best "
            f"single-feature rule ({rule['rule']}) at ₹{rule['test_total_cost']:,.0f}. "
            f"The complexity is not paying for itself."
        )

    def test_graph_features_have_measured_value(self, metrics):
        """
        The graph-ablation gap is a claim in the README ("graph features are worth
        ₹X"). It must be present and non-negative, or the claim is unsupported.
        """
        _require_current_metrics(metrics)
        ablation = metrics["baselines"].get("xgboost_without_graph_features", {})
        if "test_total_cost" not in ablation:
            pytest.skip("graph-ablation baseline was skipped (scikit-learn absent)")

        model_cost = metrics["total_cost"]
        assert model_cost <= ablation["test_total_cost"] + 1e-6, (
            f"the full model (₹{model_cost:,.0f}) costs more than the same model "
            f"without graph features (₹{ablation['test_total_cost']:,.0f}). The "
            f"graph pipeline is a net negative on test."
        )


# ══════════════════════════════════════════════════════════════════
# 4. The sensitivity table is not secretly an oracle
# ══════════════════════════════════════════════════════════════════

class TestSensitivityTableDoesNotPeekAtTest:
    """
    The regression guard for a defect that shipped once.

    `sensitivity_to_cost_ratio` used to take one split and call
    find_optimal_threshold() on it, and train.py passed it TEST. Every row
    therefore chose its threshold with test labels, which made the row at the
    shipped ratio a silent duplicate of `oracle_threshold_diagnostic` — it
    published ₹39,39,300 where the honest cost is ₹55,00,000.

    That is the worst class of bug in this repo: not a wrong number, but a
    dishonest one, sitting inside the artefact whose whole job is to prove the
    operating point was not cherry-picked. The README's fourth honesty claim is
    literally "the threshold is never selected on test", so nothing here may
    contradict it.
    """

    @staticmethod
    def _row_nearest_shipped_ratio(metrics) -> dict:
        table = metrics.get("cost_ratio_sensitivity") or []
        if not table:
            pytest.skip("metrics.json has no cost_ratio_sensitivity block")
        shipped = float(metrics["cost_config"]["fn_fp_ratio"])
        row = min(table, key=lambda r: abs(float(r["fn_fp_ratio"]) - shipped))
        # Fail with the diagnosis rather than a bare KeyError three tests deep.
        assert "val_threshold" in row and "test_total_cost" in row, (
            f"sensitivity rows use the legacy schema {sorted(row)}. That schema "
            f"is the defect: its 'threshold'/'total_cost' columns were produced "
            f"by optimising on TEST. Re-run `python -m models.train` to "
            f"regenerate metrics.json with split-tagged columns."
        )
        return row

    def test_every_row_records_which_split_chose_its_threshold(self, metrics):
        """
        Provenance is structural, not documentary. A bare `threshold` column is
        the old, ambiguous schema — the one that let a test-selected number pass
        as a reported result. Names must say which split produced them.
        """
        _require_current_metrics(metrics)
        table = metrics.get("cost_ratio_sensitivity") or []
        if not table:
            pytest.skip("metrics.json has no cost_ratio_sensitivity block")

        for row in table:
            assert "val_threshold" in row, (
                f"sensitivity row for ratio {row.get('fn_fp_ratio')} has no "
                f"'val_threshold'. Keys present: {sorted(row)}. A threshold "
                f"column without a split in its name is how the test-peeking "
                f"bug hid — every figure here must carry its provenance."
            )
            assert "threshold" not in row, (
                "sensitivity rows still carry the ambiguous 'threshold' key; "
                "use 'val_threshold' so the selection split is unmistakable."
            )

    def test_shipped_ratio_reuses_the_validation_threshold(self, metrics):
        """
        At the ratio the project actually ships, the sensitivity table must land
        on the very threshold the headline uses — because both are selected the
        same way, on validation. Under the bug this was 0.2665 against a
        validation threshold of 0.5908.
        """
        _require_current_metrics(metrics)
        row = self._row_nearest_shipped_ratio(metrics)
        val_threshold = float(metrics["validation"]["cost_optimal"]["threshold"])
        got = float(row["val_threshold"])
        # Tolerance covers the 13.33-vs-13.3333 rounding in the ratio grid,
        # which can nudge the plateau; it is far tighter than the 0.32 gap the
        # oracle bug produced.
        assert abs(got - val_threshold) < 0.05, (
            f"the sensitivity row at ratio {row['fn_fp_ratio']} reports "
            f"threshold {got:.4f}, but the validation-selected threshold is "
            f"{val_threshold:.4f}. A gap this size means the row optimised on "
            f"a different split — almost certainly test."
        )

    def test_no_row_undercuts_the_honest_headline_cost(self, metrics):
        """
        The fingerprint of test-peeking is a cost that beats the honest one. A
        threshold frozen from validation cannot outperform, on test, a threshold
        that was chosen using test labels — so if the table's shipped-ratio cost
        comes in below the published total, it was not frozen.
        """
        _require_current_metrics(metrics)
        row = self._row_nearest_shipped_ratio(metrics)
        headline = float(metrics["total_cost"])
        reported = float(row["test_total_cost"])
        # 2% slack for the ratio-grid rounding (fn_cost 199,950 vs 200,000);
        # the shipped defect understated cost by 28%.
        assert reported >= headline * 0.98, (
            f"sensitivity at ratio {row['fn_fp_ratio']} reports test cost "
            f"₹{reported:,.0f}, undercutting the published headline "
            f"₹{headline:,.0f} by {100 * (1 - reported / headline):.1f}%. A "
            f"validation-frozen threshold cannot beat the headline on test; "
            f"this row was optimised on test labels."
        )

    def test_shipped_row_is_not_the_oracle_row(self, metrics):
        """
        The bug stated directly: the sensitivity row had become a copy of the
        explicitly-labelled test-peeking diagnostic, minus its warning label.
        """
        _require_current_metrics(metrics)
        oracle = (metrics.get("test", {}) or {}).get("oracle_threshold_diagnostic")
        if not oracle:
            pytest.skip("metrics.json has no oracle_threshold_diagnostic block")

        row = self._row_nearest_shipped_ratio(metrics)
        assert abs(float(row["val_threshold"])
                   - float(oracle["threshold"])) > 1e-6, (
            f"the sensitivity row at ratio {row['fn_fp_ratio']} uses threshold "
            f"{row['val_threshold']}, which is exactly "
            f"oracle_threshold_diagnostic's test-selected threshold. That block "
            f"carries the warning {oracle.get('warning','')!r} — the sensitivity "
            f"table inherited its numbers without inheriting its caveat."
        )
