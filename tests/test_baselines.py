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

Three further sections guard the blocks added to answer the README's stated
limitations, each of which publishes a number a reader could otherwise take on
trust:

  3. PREVALENCE PROJECTION.  The headline precision is measured at an elevated
     ~4-7% base rate. `TestPrevalenceProjectionIsArithmeticNotOptimism` checks the
     published projection is the actual identity — it must reproduce the measured
     precision at the observed rate — and that recall stays FIXED down the table,
     since TPR is a within-class rate no base rate can move.

  4. CAPACITY-FAIR COMPARISON.  The uncapped baseline table lets each policy raise
     unlimited alerts, which at 13.3:1 is an advantage and not a fair fight.
     `TestEveryPolicyIsPricedUnderTheSameBudget` requires each policy's threshold
     to respect the budget on the split that chose it.

  5. CALIBRATION.  The scores are a good ranking and a bad probability, on purpose.
     `TestCalibrationDoesNotChangeTheRanking` measures the gap and — the
     load-bearing part — proves the reported calibrator cannot reorder the alert
     queue, which is what makes it a diagnostic rather than a silent model change.

Usage:
    pytest tests/test_baselines.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

from models.cost_matrix import (
    CostEvaluator,
    ThresholdReport,
    brier_score,
    expected_calibration_error,
    fit_platt_scaling,
    precision_at_prevalence,
    prevalence_for_precision,
    reliability_table,
    roc_auc,
)
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


def _block_or_skip(metrics, key: str) -> list | dict:
    """
    Fetch a metrics.json block added after the last recorded training run.

    Skips rather than fails when the key is absent or null. A metrics.json written
    before these blocks existed is STALE, not wrong, and
    `test_metrics_describe_the_current_model_version` is the one test that should
    say so — the rest deferring to it is the same convention
    `_require_current_metrics` follows. The distinction matters in practice: this
    file has to stay green on a checkout whose metrics.json predates the retrain.
    """
    block = metrics.get(key)
    if not block:
        pytest.skip(
            f"metrics.json has no {key!r} block; regenerate with "
            f"`python -m models.train`."
        )
    return block


def _shipped_test_report(metrics) -> ThresholdReport:
    """
    Rebuild the published test operating point as a ThresholdReport.

    The confusion counts are the source of truth here, not the published
    precision: the point of the projection tests is to re-derive precision from
    tp/fp and check it against what was published, so reading the published
    figure back in would make the check circular.
    """
    at = (metrics.get("test") or {}).get("at_selected_threshold")
    if not at or "tp" not in at:
        pytest.skip("metrics.json has no test confusion counts to project from")
    tp, fp, tn, fn = (int(at["tp"]), int(at["fp"]),
                      int(at["tn"]), int(at["fn"]))
    return ThresholdReport(
        threshold=float(at["threshold"]), tp=tp, fp=fp, tn=tn, fn=fn,
        precision=at.get("precision"), recall=at.get("recall"),
        f1=at.get("f1"), total_cost=float(at["total_cost"]),
        cost_per_prediction=0.0,
        alert_rate=(tp + fp) / max(tp + fp + tn + fn, 1),
        alerts_per_1000=1000.0 * (tp + fp) / max(tp + fp + tn + fn, 1),
    )


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


# ══════════════════════════════════════════════════════════════════
# 5. The prevalence projection is arithmetic, not optimism
# ══════════════════════════════════════════════════════════════════

class TestPrevalenceProjectionIsArithmeticNotOptimism:
    """
    The README concedes that precision and rupee cost are measured at an elevated
    ~4-7% base rate. The projection block answers "so what would they be lower
    down", and the danger in publishing it is that a projection is trivially easy
    to make flattering: pick the wrong identity, or quietly let recall drift up as
    prevalence falls, and the table becomes a sales document.

    So these tests pin it to the identity it claims to be. The anchor is that at
    the OBSERVED base rate the projection must reproduce the number actually
    measured on test — a projection that cannot recover the measurement it starts
    from is not describing the same operating point.
    """

    def test_the_identity_recovers_the_measured_precision(self, metrics):
        """
        P(pi) = pi*TPR / (pi*TPR + (1-pi)*FPR), evaluated at the observed pi, must
        equal tp/(tp+fp) as measured. This is the check that the whole table hangs
        on and it is exact, not approximate — same counts, same arithmetic.
        """
        _require_current_metrics(metrics)
        report = _shipped_test_report(metrics)
        n = report.tp + report.fp + report.tn + report.fn
        tpr = report.tp / (report.tp + report.fn)
        fpr = report.fp / (report.fp + report.tn)
        observed = (report.tp + report.fn) / n

        projected = precision_at_prevalence(tpr, fpr, observed)
        measured = report.tp / (report.tp + report.fp)
        assert projected == pytest.approx(measured, abs=1e-9), (
            f"projecting the operating point back onto its own base rate gives "
            f"precision {projected:.6f}, but the measured precision is "
            f"{measured:.6f}. The projection is not describing this operating "
            f"point, so every row below the observed rate is fiction."
        )

    def test_recall_is_the_same_at_every_base_rate(self, metrics):
        """
        The pedagogical point of the table, and a real failure mode: TPR is a
        WITHIN-CLASS rate, so no change of prevalence can move it. A table whose
        recall column drifts has re-derived recall from projected counts and is
        smuggling in an improvement that base rate cannot buy.
        """
        _require_current_metrics(metrics)
        table = _block_or_skip(metrics, "prevalence_projection")
        recalls = {round(float(r["recall"]), 9) for r in table
                   if r.get("recall") is not None}
        assert len(recalls) == 1, (
            f"the projection reports {len(recalls)} distinct recall values "
            f"({sorted(recalls)}) across base rates. Recall is TPR, a "
            f"within-class rate; prevalence cannot change it."
        )

    def test_precision_falls_as_prevalence_falls(self, metrics):
        """
        Monotonicity. With TPR and FPR fixed, precision is strictly increasing in
        prevalence, so sorting by base rate must sort by precision too. A
        non-monotone column means rows were computed at inconsistent rates.
        """
        _require_current_metrics(metrics)
        table = _block_or_skip(metrics, "prevalence_projection")
        rows = sorted(
            (r for r in table if r.get("projected_precision") is not None),
            key=lambda r: float(r["prevalence"]),
        )
        precisions = [float(r["projected_precision"]) for r in rows]
        assert precisions == sorted(precisions), (
            f"projected precision is not monotone in prevalence: {precisions} "
            f"at base rates {[float(r['prevalence']) for r in rows]}."
        )

    def test_exactly_one_row_is_the_measured_one(self, metrics):
        """
        The observed rate must be flagged and must appear once. Two `is_observed`
        rows means the de-duplication against the default ladder failed and the
        measured point is being double-counted; none means a reader cannot tell
        which row is evidence and which is arithmetic.
        """
        _require_current_metrics(metrics)
        table = _block_or_skip(metrics, "prevalence_projection")
        flagged = [r for r in table if r.get("is_observed")]
        assert len(flagged) == 1, (
            f"{len(flagged)} rows are marked is_observed; exactly one row — the "
            f"base rate the model was actually measured at — may be."
        )

    def test_the_break_even_claim_is_consistent_with_the_table(self, metrics):
        """
        `clears_break_even` must agree with comparing projected precision against
        p* = fp/(fp+fn). This is the one column a reader will act on — it is the
        difference between "the queue pays for itself" and "it does not" — so it
        may not be a hand-set flag that drifted from the arithmetic beside it.

        p* is re-derived from the published costs rather than read from
        `break_even_probability`, which is rounded to six places: the flag was
        computed against the unrounded value, and comparing against the rounded
        one would put a 4e-7-wide band of precisions on the wrong side of the
        boundary. The rounded figure is checked against its own inputs instead.
        """
        _require_current_metrics(metrics)
        table = _block_or_skip(metrics, "prevalence_projection")
        config = metrics["cost_config"]
        fn_cost, fp_cost = float(config["fn_cost"]), float(config["fp_cost"])
        p_star = fp_cost / (fp_cost + fn_cost)
        assert float(config["break_even_probability"]) == pytest.approx(
            p_star, abs=5e-7), (
            f"the published break_even_probability "
            f"{config['break_even_probability']} does not match fp/(fp+fn) for "
            f"the published costs (₹{fp_cost:,.0f} / ₹{fn_cost:,.0f} = "
            f"{p_star:.6f})."
        )
        for row in table:
            precision = row.get("projected_precision")
            clears = row.get("clears_break_even")
            if precision is None:
                assert clears is None, (
                    f"row at prevalence {row['prevalence']} has no computable "
                    f"precision but reports clears_break_even={clears!r}. With "
                    f"an empty queue there is nothing to compare to p*, and "
                    f"False would read as a measured failure."
                )
                continue
            assert clears is (float(precision) >= p_star), (
                f"row at prevalence {row['prevalence']} reports precision "
                f"{precision} and clears_break_even={clears!r}, but p* is "
                f"{p_star}. The flag contradicts the number printed next to it."
            )

    def test_the_table_shows_where_the_queue_stops_paying_for_itself(self, metrics):
        """
        The finding worth publishing: somewhere down the ladder projected
        precision crosses p*, and below that the alert queue costs more to work
        than to ignore. If every row clears, the table concedes nothing and the
        crossing point should be added to the ladder — a projection with no bad
        news in it is not doing the job the Limitations section asked of it.
        """
        _require_current_metrics(metrics)
        table = _block_or_skip(metrics, "prevalence_projection")
        report = _shipped_test_report(metrics)
        tpr, fpr = (report.tp / (report.tp + report.fn),
                    report.fp / (report.fp + report.tn))
        config = metrics["cost_config"]
        p_star = float(config["fp_cost"]) / (float(config["fp_cost"])
                                             + float(config["fn_cost"]))
        crossing = prevalence_for_precision(tpr, fpr, p_star)
        assert crossing is not None, "the model catches nothing on test"

        flags = [bool(r["clears_break_even"]) for r in table
                 if r.get("clears_break_even") is not None]
        if all(flags):
            lowest = min(float(r["prevalence"]) for r in table)
            assert crossing < lowest, (
                f"every row clears break-even, yet the crossing is at "
                f"{crossing:.6%} — inside the ladder. The table is claiming the "
                f"queue always pays for itself while its own inputs say it stops "
                f"at {crossing:.6%}."
            )
        for row in table:
            if row.get("clears_break_even") is None:
                continue
            # Above the crossing the queue pays; below it, it does not. The flag
            # and the crossing are two views of one number and must agree.
            assert bool(row["clears_break_even"]) is (
                float(row["prevalence"]) >= crossing - 1e-12), (
                f"row at prevalence {row['prevalence']} reports "
                f"clears_break_even={row['clears_break_even']!r}, but the "
                f"break-even base rate for this operating point is "
                f"{crossing:.8f}."
            )

    def test_the_inverse_round_trips(self):
        """
        Pure math, no metrics: `prevalence_for_precision` must invert
        `precision_at_prevalence`. Published together, so an inconsistency would
        let the README quote a break-even base rate the projection table denies.
        """
        tpr, fpr = 0.8710, 0.07805
        for target in (0.05, 0.0698, 0.10, 0.25, 0.50):
            pi = prevalence_for_precision(tpr, fpr, target)
            assert pi is not None
            assert precision_at_prevalence(tpr, fpr, pi) == pytest.approx(
                target, abs=1e-12), (
                f"round trip failed at target precision {target}: prevalence "
                f"{pi} projects back to "
                f"{precision_at_prevalence(tpr, fpr, pi)}."
            )

    def test_a_queue_that_flags_nothing_has_no_precision(self):
        """
        The null-versus-zero convention, at the degenerate corner. TPR = FPR = 0
        raises no alerts, so there is no precision — 0.0 there would read as a
        model that flagged accounts and got them all wrong.
        """
        assert precision_at_prevalence(0.0, 0.0, 0.04) is None
        # A model that catches nothing has no base rate at which it clears any
        # positive precision target, and None says that; 0.0 would claim the
        # target is met at a zero base rate.
        assert prevalence_for_precision(0.0, 0.1, 0.07) is None
        # A model with no false positives is perfectly precise at any base rate
        # above zero, so the answer is a real 0.0, not a null.
        assert prevalence_for_precision(0.9, 0.0, 0.07) == 0.0

    def test_a_single_class_split_refuses_to_be_projected(self):
        """
        Projection needs both a TPR and an FPR. On a split with no positives one
        of them does not exist, and inventing it would produce a table of
        confident numbers about a measurement never made.
        """
        evaluator = CostEvaluator()
        no_positives = ThresholdReport(
            threshold=0.5, tp=0, fp=10, tn=90, fn=0, precision=0.0,
            recall=None, f1=None, total_cost=150_000.0,
            cost_per_prediction=1500.0, alert_rate=0.1, alerts_per_1000=100.0,
        )
        with pytest.raises(ValueError, match="single-class|no positives"):
            evaluator.project_to_prevalence(no_positives)


# ══════════════════════════════════════════════════════════════════
# 6. One budget, applied to everybody
# ══════════════════════════════════════════════════════════════════

class TestEveryPolicyIsPricedUnderTheSameBudget:
    """
    The regression guard for an unfair comparison this project published.

    `TestModelEarnsItsComplexity` above prices every policy UNCAPPED, which is the
    right like-for-like on cost alone. But the README also states a capacity
    constraint of 20 alerts per 1,000 accounts, and the capped model was being
    compared against baselines still free to flood the queue. At 13.3:1 misses
    dominate, so unlimited alerting is an advantage: the one-line rule bought its
    recall with roughly twelve times the queue the cap allows, and then "the rule
    beats the model under a cap" was reported as though the cap applied to both.

    That was a defect in the measurement, not in the model, and the fix is a table
    where the budget binds on everyone. These tests keep it binding.
    """

    def test_every_policy_respects_the_budget_where_it_was_selected(self, metrics):
        """
        The budget must bind on VALIDATION, the split each threshold was chosen
        on. This is the constraint the table exists to impose; a row over the
        budget here means `threshold_for_alert_budget` was not what produced it.
        """
        _require_current_metrics(metrics)
        table = _block_or_skip(metrics, "capacity_fair_comparison")
        budget = float(metrics["cost_config"]["alert_budget_per_1000"])
        for row in table:
            if row.get("val_threshold") is None:
                continue                    # trivial policies have no threshold
            assert float(row["val_alerts_per_1000"]) <= budget + 1e-6, (
                f"policy {row['policy']!r} raises "
                f"{row['val_alerts_per_1000']} alerts per 1,000 on validation, "
                f"over the stated budget of {budget}. The point of this table is "
                f"that the constraint binds on every policy, not just the model."
            )

    def test_a_test_side_overflow_is_reported_not_hidden(self, metrics):
        """
        A validation-frozen threshold CAN overflow the budget on test — the score
        distribution moved and the threshold did not. That must be visible per
        row. The tempting fix is to re-solve the threshold on test until the
        constraint holds, which is selection on test wearing a capacity argument
        as a disguise, so the honest schema carries the flag.
        """
        _require_current_metrics(metrics)
        table = _block_or_skip(metrics, "capacity_fair_comparison")
        budget = float(metrics["cost_config"]["alert_budget_per_1000"])
        for row in table:
            assert "test_within_budget" in row, (
                f"policy {row['policy']!r} does not report test_within_budget. "
                f"Keys present: {sorted(row)}. Without it a reader cannot tell a "
                f"threshold that held on test from one that spilled over."
            )
            assert row["test_within_budget"] is (
                float(row["test_alerts_per_1000"]) <= budget + 1e-9), (
                f"policy {row['policy']!r} reports "
                f"test_within_budget={row['test_within_budget']!r} with "
                f"{row['test_alerts_per_1000']} alerts per 1,000 against a "
                f"budget of {budget}. The flag contradicts the count."
            )

    def test_the_comparison_is_not_vacuous(self, metrics):
        """
        A "fair comparison" containing only the model and the two trivial policies
        proves nothing. At least one real baseline must be priced under the cap,
        and exactly one row may be the model.
        """
        _require_current_metrics(metrics)
        table = _block_or_skip(metrics, "capacity_fair_comparison")
        models = [r for r in table if r.get("is_model")]
        assert len(models) == 1, (
            f"{len(models)} rows are flagged is_model; exactly one must be."
        )
        contenders = [r for r in table
                      if not r.get("is_model")
                      and r.get("val_threshold") is not None]
        assert contenders, (
            "the capacity-fair table prices no non-trivial baseline under the "
            "budget, so it does not compare anything. Re-run training with "
            "baselines enabled."
        )

    def test_flag_everything_is_marked_infeasible(self, metrics):
        """
        Flag-everything is cheap on paper under a 13.3:1 cost model and impossible
        to staff at 1,000 alerts per 1,000 accounts. It stays in the table so a
        reader sees the excluded policy rather than wondering whether it was
        quietly dropped — which only works if its infeasibility is recorded.
        """
        _require_current_metrics(metrics)
        table = _block_or_skip(metrics, "capacity_fair_comparison")
        budget = float(metrics["cost_config"]["alert_budget_per_1000"])
        rows = [r for r in table if r["policy"] == "flag everything"]
        if not rows:
            pytest.skip("no flag-everything row in the capacity-fair table")
        row = rows[0]
        assert float(row["test_alerts_per_1000"]) == 1000.0
        assert row["feasible_under_budget"] is (budget >= 1000.0), (
            f"flag-everything reports "
            f"feasible_under_budget={row['feasible_under_budget']!r} against a "
            f"budget of {budget} alerts per 1,000. Flagging every account is "
            f"1,000 per 1,000 and cannot be feasible below that."
        )

    def test_flag_nothing_keeps_the_null_versus_zero_convention(self, metrics):
        """
        The convention `_prf` exists to protect, at the row most likely to break
        it. Flag-nothing opens no queue, so precision is null; the split does have
        positives and all were missed, so recall is a measured 0.0. Publishing
        0.0 precision here would flatter the model in the one table meant to keep
        it honest.
        """
        _require_current_metrics(metrics)
        table = _block_or_skip(metrics, "capacity_fair_comparison")
        rows = [r for r in table if r["policy"] == "flag nothing"]
        if not rows:
            pytest.skip("no flag-nothing row in the capacity-fair table")
        row = rows[0]
        assert row["test_precision"] is None, (
            f"flag-nothing publishes precision {row['test_precision']!r}. It "
            f"raised no alerts, so there is no queue whose purity could be "
            f"measured — that is null, not zero."
        )
        assert float(row["test_recall"]) == 0.0, (
            "flag-nothing's recall is a measurement (every positive was missed) "
            "and must be 0.0, not null."
        )

    def test_the_model_still_wins_when_the_budget_binds_on_everyone(self, metrics):
        """
        The claim the fair table exists to support. The model already beats the
        one-line rule uncapped (`TestModelEarnsItsComplexity`); under a shared
        budget it must still win, or the capacity constraint — not the cap being
        applied unevenly — is what the pipeline cannot survive.

        This failing is a real result to hear before submitting, not a test to
        loosen: it would mean the honest headline is that a one-line rule is the
        better policy at this queue size.
        """
        _require_current_metrics(metrics)
        table = _block_or_skip(metrics, "capacity_fair_comparison")
        model = [r for r in table if r.get("is_model")]
        if not model:
            pytest.skip("no model row in the capacity-fair table")
        model_cost = float(model[0]["test_total_cost"])
        rivals = [r for r in table
                  if not r.get("is_model") and r.get("feasible_under_budget")
                  and r.get("val_threshold") is not None]
        if not rivals:
            pytest.skip("no feasible non-trivial rival under the budget")
        best = min(rivals, key=lambda r: float(r["test_total_cost"]))
        assert model_cost <= float(best["test_total_cost"]) + 1e-6, (
            f"under the same {metrics['cost_config']['alert_budget_per_1000']} "
            f"alerts/1,000 budget the model costs ₹{model_cost:,.0f} against "
            f"₹{float(best['test_total_cost']):,.0f} for {best['policy']!r}. "
            f"With the comparison made fair, the simpler policy wins — that is "
            f"the honest headline and the README must say so."
        )


# ══════════════════════════════════════════════════════════════════
# 7. The calibrator is a diagnostic, and provably not a model change
# ══════════════════════════════════════════════════════════════════

class TestCalibrationDoesNotChangeTheRanking:
    """
    `scale_pos_weight` inflates predicted probabilities on purpose, so the scores
    are a good ORDERING and a bad PROBABILITY. That is measured and published
    rather than left for a reviewer to find.

    The risk in publishing a calibrator is that someone wires it in. The shipped
    threshold was selected on the raw scale, so rescaling scores underneath it
    moves the operating point without saying so. These tests hold the reported map
    to being what it claims: a measurement of the gap that cannot touch the queue.
    """

    def test_the_reported_calibrator_cannot_reorder_the_queue(self, metrics):
        """
        The load-bearing check. A monotone map leaves ROC-AUC identical; the
        published block carries both AUCs and its own tolerance so this is
        verifiable from the artefact without a retrain.
        """
        _require_current_metrics(metrics)
        block = _block_or_skip(metrics, "probability_calibration")
        invariance = block["ranking_invariance"]
        assert invariance["abs_delta"] <= invariance["tolerance"], (
            f"calibration moved test ROC-AUC from "
            f"{invariance['test_auc_raw']} to "
            f"{invariance['test_auc_calibrated']} "
            f"(delta {invariance['abs_delta']:.2e}, tolerance "
            f"{invariance['tolerance']}). A calibrator that re-ranks accounts is "
            f"not a calibrator — it is an undocumented change to the model."
        )

    def test_the_calibrator_is_fitted_off_the_test_split(self, metrics):
        """
        Fitting on test and reporting the improvement on test would measure what a
        calibrator can memorise. The block must state validation, for the same
        reason every threshold in this project does.
        """
        _require_current_metrics(metrics)
        block = _block_or_skip(metrics, "probability_calibration")
        assert block["fitted_on"] == "validation", (
            f"the calibrator reports fitted_on={block['fitted_on']!r}. Fitting "
            f"it on test makes the published improvement unearned."
        )
        assert block["scaler"]["fit_on"] == "validation"
        assert block["measured_on"] == "test"

    def test_the_calibration_gap_is_actually_reported(self, metrics):
        """
        The limitation is only answered if the size of it is on the page. Brier
        and ECE must both be present, raw and calibrated, as real numbers.
        """
        _require_current_metrics(metrics)
        block = _block_or_skip(metrics, "probability_calibration")
        for side in ("test_raw", "test_calibrated"):
            for key in ("brier_score", "expected_calibration_error"):
                value = block[side].get(key)
                assert value is not None, (
                    f"{side}.{key} is null in the calibration block; the "
                    f"limitation is stated but not quantified."
                )
                assert 0.0 <= float(value) <= 1.0

    def test_calibration_does_not_claim_to_be_applied_at_serving_time(self, metrics):
        """
        A structural guard on the honest framing. If someone later wires the map
        into the serving path, this block's stated purpose becomes false — and the
        published operating point would have silently moved.
        """
        _require_current_metrics(metrics)
        block = _block_or_skip(metrics, "probability_calibration")
        assert "diagnostic" in block["purpose"].lower(), (
            f"the calibration block's purpose reads {block['purpose']!r}, which "
            f"no longer says it is a diagnostic. If the calibrator is now "
            f"applied at serving time, the threshold must be re-selected on the "
            f"calibrated scale and every published operating point re-derived."
        )

    def test_a_monotone_map_leaves_auc_alone(self):
        """
        Pure math, no metrics: the property the whole diagnostic rests on, proved
        on a case where the raw scores are deliberately over-confident.
        """
        rng = np.random.default_rng(20260829)
        y = (rng.uniform(size=4000) < 0.045).astype(int)
        raw = 1.0 / (1.0 + np.exp(-(2.4 * (rng.normal(size=4000) + 1.7 * y - 1.2))))

        scaler = fit_platt_scaling(y, raw)
        calibrated = scaler.transform(raw)
        assert roc_auc(y, calibrated) == pytest.approx(roc_auc(y, raw), abs=1e-9)

        # Monotone NON-decreasing is the precise claim: compressing logits can
        # round two adjacent doubles onto the same value, so the map may create a
        # tie. It may never invert a pair, and that is what an argsort comparison
        # would wrongly flag while this check states exactly the guarantee.
        walk = np.diff(calibrated[np.argsort(raw, kind="mergesort")])
        assert not (walk < 0).any(), (
            "the calibrated scores step DOWN somewhere along the raw score "
            "order, so the map inverted a pair of accounts."
        )

    def test_calibration_improves_the_probabilities_it_was_fitted_for(self):
        """
        A calibrator that does not reduce Brier or ECE on the data it was fitted
        to is broken, and would make the published before/after meaningless.
        """
        rng = np.random.default_rng(7)
        truth = rng.uniform(0.01, 0.99, 20_000)
        y = (rng.uniform(size=truth.size) < truth).astype(int)
        # Over-confident and biased high, the direction scale_pos_weight produces.
        skewed = 1.0 / (1.0 + np.exp(-(2.0 * np.log(truth / (1 - truth)) + 1.5)))

        scaler = fit_platt_scaling(y, skewed)
        fixed = scaler.transform(skewed)
        assert scaler.converged
        assert brier_score(y, fixed) < brier_score(y, skewed)
        assert (expected_calibration_error(y, fixed)
                < expected_calibration_error(y, skewed))
        # Slope below 1 is the signature of over-confidence: the calibrator has to
        # shrink the logits back toward the evidence.
        assert scaler.slope < 1.0

    def test_a_calibrator_cannot_be_fitted_on_one_class(self):
        """
        On a single-class split the fitted map encodes the target-smoothing
        constants and nothing about the model, so it must refuse rather than
        return a confident-looking artefact.
        """
        with pytest.raises(ValueError, match="single-class"):
            fit_platt_scaling(np.zeros(50, dtype=int), np.full(50, 0.3))

    def test_an_empty_reliability_bin_reports_no_frequency(self):
        """
        Null-versus-zero again, in the reliability table. A bin no account landed
        in has no observed frequency, and 0.0 there would draw a reliability curve
        diving to the floor through regions where nothing was ever predicted.
        """
        table = reliability_table(np.array([0, 0, 1, 1]),
                                 np.array([0.05, 0.05, 0.95, 1.0]), n_bins=10)
        assert int(table["n"].sum()) == 4, "a score fell out of the table"
        # p = 1.0 must land in the last bin, not past the final edge.
        assert int(table.iloc[-1]["n"]) == 2
        empty = table[table["n"] == 0]
        assert empty["observed_frequency"].isna().all()
        assert empty["mean_predicted"].isna().all()
