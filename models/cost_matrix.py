"""
models/cost_matrix.py
─────────────────────
Cost-sensitive evaluation. This is the module that makes the metrics honest:
precision and recall alone let you hide a bad trade-off, and the track bar asks
explicitly for "honest metrics including false-positive cost".

─────────────────────────────────────────────────────────────────────────────
THE COST MODEL, AND ITS ASSUMPTIONS STATED PLAINLY
─────────────────────────────────────────────────────────────────────────────
Both costs are per account, over the same 30-day horizon:

  False negative — a mule account we failed to flag.        ₹2,00,000
      One mule hop keeps laundering for another cycle. This is a stand-in for
      recoverable value plus the regulatory and reputational tail, and it is an
      ASSUMPTION, not a measurement.

  False positive — a legitimate account we flagged.            ₹15,000
      Manual review time, customer friction, and the risk of losing a merchant.
      Note what this cost does NOT include: this system never blocks anyone
      (see api/responder.py — the strongest action is REVIEW), so a false
      positive costs an analyst's attention, not a frozen account.

Only the RATIO matters for choosing a threshold: 13.3 : 1. The absolute
rupee figures matter for reporting totals. Because both numbers are assumptions,
`sensitivity_to_cost_ratio()` exists and its output belongs in the README — a
threshold that only looks good at one assumed ratio is not a result.

─────────────────────────────────────────────────────────────────────────────
BREAK-EVEN PRECISION: THE NUMBER THAT MAKES THE COST MATRIX MEAN SOMETHING
─────────────────────────────────────────────────────────────────────────────
Flagging an account is worth it when the expected cost of flagging is below the
expected cost of not flagging. For an account with probability p of being a mule:

    flag:       (1 - p) · fp_cost          we pay fp_cost when we're wrong
    don't flag:      p  · fn_cost          we pay fn_cost when we're wrong

    flag iff    (1 - p) · fp_cost  <  p · fn_cost
           iff        p  >  fp_cost / (fp_cost + fn_cost)

    p*  =  15,000 / 215,000  =  0.0698

So on a well-calibrated model the cost-optimal cutoff is p ≈ 0.07, and
equivalently the BREAK-EVEN PRECISION of an alert queue is 6.98%: any queue
whose precision beats 7% is cheaper to work than to ignore. That is a far more
defensible framing than "we tuned the threshold to 0.42", and it is derived from
the cost assumptions rather than fitted to a dataset.

`find_optimal_threshold` still sweeps empirically, because the model is not
perfectly calibrated and the empirical minimum is what actually happened. When
the two disagree badly, the model's calibration is the problem, not the sweep —
`summary()` prints both so the gap is visible.

─────────────────────────────────────────────────────────────────────────────
v2 → v3 CHANGES
─────────────────────────────────────────────────────────────────────────────
1. THE SWEEP IS NOW EXACT, AND OVER OBSERVED SCORES.
   v2 swept `np.linspace(0.01, 0.99, 200)`. Total cost is a STEP function of
   the threshold — it only changes when the threshold crosses an actual
   predicted score — so a fixed grid is both wasteful (200 evaluations to
   discover a few dozen distinct outcomes) and unsound: if the model's scores
   all sit inside [0.001, 0.05], the grid steps straight over every decision
   that matters and returns a threshold chosen from nothing. The grid also
   could not express "flag nothing", so it could never report that a model is
   worthless. The sweep now enumerates every distinct achievable confusion
   matrix exactly, in one vectorised pass.

2. THRESHOLD SELECTION IS PLATEAU-AWARE.
   Because cost is a step function, the minimum is almost always a PLATEAU of
   thresholds that produce identical predictions. v2's `if cost < best_cost`
   kept the first one seen, i.e. the LEFT EDGE of that plateau — the single most
   fragile point available, one hair away from a different decision set. We now
   locate the widest minimum-cost run, take its MIDPOINT, and report the
   plateau's bounds and width so a reviewer can see how much slack there is.
   A wide plateau is a robust operating point; a plateau of width 1e-4 is a
   warning that the threshold is fitted to noise.

3. NO sklearn DEPENDENCY.
   Precision, recall, F1 and ROC-AUC are four lines of arithmetic here, and
   computing them directly (a) removes a heavyweight import from the dashboard's
   hot path, (b) makes the sweep O(n log n) once instead of O(n · steps), and
   (c) means this module and its tests run in any environment with numpy —
   which is what let it be verified. Values match sklearn's, including its
   tie-averaged ROC-AUC and `zero_division=0` conventions.

4. OPERATIONAL LOAD IS REPORTED, AND SO IS THE CAPACITY-CONSTRAINED POINT.
   `alert_rate` and `alerts_per_1000` are on every report, because an alert
   queue's real constraint is analyst hours and "precision 0.71" does not tell
   you whether that is 12 alerts a day or 1,200.

   This turned out to matter more than expected. With FN priced at 13.3x FP, the
   cost-MINIMISING threshold flags roughly one account in five: correct
   arithmetic, unstaffable queue. So `threshold_for_alert_budget()` reports the
   best recall purchasable within a stated capacity, and `summary()` prints both
   points side by side with the cost difference between them labelled as what it
   is — the price of a queue someone can actually work.

Usage:
    from models.cost_matrix import CostEvaluator

    ev = CostEvaluator(fn_cost=200_000, fp_cost=15_000)
    choice = ev.find_optimal_threshold(y_val, p_val)   # select on VALIDATION
    final  = ev.evaluate_at_threshold(y_test, p_test, choice.threshold)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# 30-day horizon; see the module docstring for what each number stands in for.
DEFAULT_FN_COST = 200_000.0
DEFAULT_FP_COST = 15_000.0


@dataclass
class CostConfig:
    """Cost parameters, per account, over a 30-day horizon (₹)."""
    fn_cost: float = DEFAULT_FN_COST
    fp_cost: float = DEFAULT_FP_COST

    def __post_init__(self) -> None:
        if self.fn_cost <= 0 or self.fp_cost <= 0:
            raise ValueError(
                f"Costs must be positive; got fn_cost={self.fn_cost}, "
                f"fp_cost={self.fp_cost}. A zero or negative cost makes the "
                "optimal threshold degenerate (flag everything / flag nothing)."
            )

    @property
    def ratio(self) -> float:
        """How many false positives one missed mule is worth."""
        return self.fn_cost / self.fp_cost

    @property
    def break_even_probability(self) -> float:
        """
        p* = fp_cost / (fp_cost + fn_cost).

        The cost-optimal cutoff for a calibrated model, and equivalently the
        break-even precision of an alert queue. Derived, not tuned — see the
        module docstring.
        """
        return self.fp_cost / (self.fp_cost + self.fn_cost)


@dataclass
class ThresholdReport:
    """
    Evaluation at one threshold.

    `plateau_lo`/`plateau_hi`/`plateau_width` are populated only by
    `find_optimal_threshold`, which knows the cost-equivalent range the chosen
    threshold sits inside. A report produced by evaluating a threshold someone
    else handed us has no plateau to describe, so they stay None.
    """
    threshold: float
    tp: int
    fp: int
    tn: int
    fn: int
    precision: float
    recall: float
    f1: float
    total_cost: float
    cost_per_prediction: float
    alert_rate: float
    alerts_per_1000: float
    plateau_lo: float | None = None
    plateau_hi: float | None = None
    n_equivalent: int = 1

    @property
    def plateau_width(self) -> float | None:
        if self.plateau_lo is None or self.plateau_hi is None:
            return None
        return self.plateau_hi - self.plateau_lo

    def as_dict(self) -> dict:
        """Flat, JSON-serialisable form for metrics.json."""
        return {
            "threshold": round(float(self.threshold), 6),
            "tp": int(self.tp),
            "fp": int(self.fp),
            "tn": int(self.tn),
            "fn": int(self.fn),
            "precision": round(float(self.precision), 4),
            "recall": round(float(self.recall), 4),
            "f1": round(float(self.f1), 4),
            "total_cost": round(float(self.total_cost), 2),
            "cost_per_prediction": round(float(self.cost_per_prediction), 2),
            "alert_rate": round(float(self.alert_rate), 4),
            "alerts_per_1000_accounts": round(float(self.alerts_per_1000), 1),
            "plateau_lo": (None if self.plateau_lo is None
                           else round(float(self.plateau_lo), 6)),
            "plateau_hi": (None if self.plateau_hi is None
                           else round(float(self.plateau_hi), 6)),
            "plateau_width": (None if self.plateau_width is None
                              else round(float(self.plateau_width), 6)),
            "n_cost_equivalent_thresholds": int(self.n_equivalent),
        }


# ══════════════════════════════════════════════════════════════════
# Metric primitives (sklearn-equivalent, numpy only)
# ══════════════════════════════════════════════════════════════════

def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    """Precision, recall, F1 with sklearn's `zero_division=0` convention."""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)
    return float(precision), float(recall), float(f1)


def roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """
    ROC-AUC via the Mann-Whitney U statistic, ties given average rank.

    Identical to sklearn.metrics.roc_auc_score for binary labels. Returns NaN
    when one class is absent, where sklearn raises — NaN is the more useful
    behaviour inside a per-archetype metrics loop, which will hit single-class
    slices.

    Fully vectorised, including the tie-averaging, because models/train.py calls
    this 1,000 times to bootstrap a confidence interval; a per-element Python
    loop over the tie groups made that the slowest step in the training run.
    """
    y = np.asarray(y_true).astype(int).ravel()
    s = np.asarray(y_score, dtype=float).ravel()
    n_pos = int(y.sum())
    n_neg = y.size - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    # Average ranks for ties, matching scipy.stats.rankdata(method="average").
    order = np.argsort(s, kind="mergesort")
    sorted_s = s[order]
    # First index of each tie group, and (derived) its last index.
    first = np.flatnonzero(np.concatenate(([True], sorted_s[1:] != sorted_s[:-1])))
    last = np.concatenate((first[1:] - 1, [s.size - 1]))
    avg_rank = 0.5 * (first + last) + 1.0            # 1-based average rank
    group_of = np.repeat(np.arange(first.size), last - first + 1)
    ranks = np.empty(s.size, dtype=float)
    ranks[order] = avg_rank[group_of]

    return float(
        (ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    )


def average_precision(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """
    Area under the precision-recall curve, Σ (Rₖ − Rₖ₋₁)·Pₖ.

    Identical to sklearn.metrics.average_precision_score, and the metric that
    should be quoted first here. ROC-AUC is dominated by the 96% of accounts
    that are obviously legitimate, so it stays high even when the alert queue is
    mostly noise; average precision is computed entirely over the flagged region,
    which is the only region an analyst ever sees. Its trivial baseline is the
    positive rate itself (≈0.04), so the lift is legible without further context.

    Returns NaN when no positives are present.
    """
    y = np.asarray(y_true).astype(int).ravel()
    s = np.asarray(y_score, dtype=float).ravel()
    if y.shape != s.shape:
        raise ValueError(f"shape mismatch: y_true {y.shape}, y_score {s.shape}")
    n_pos = int(y.sum())
    if n_pos == 0:
        return float("nan")

    order = np.argsort(-s, kind="mergesort")
    s_sorted, y_sorted = s[order], y[order]
    # One point per distinct score, exactly as in CostEvaluator.sweep: precision
    # and recall only change where the threshold crosses an observed score.
    cuts = np.append(np.flatnonzero(np.diff(s_sorted) != 0), y.size - 1)
    tp = np.cumsum(y_sorted)[cuts]
    n_flagged = (cuts + 1).astype(float)
    precision = tp / n_flagged
    recall = tp / n_pos
    return float(np.sum(np.diff(np.concatenate(([0.0], recall))) * precision))


# ══════════════════════════════════════════════════════════════════
# Evaluator
# ══════════════════════════════════════════════════════════════════

class CostEvaluator:
    """
    Cost-sensitive evaluation of a binary scorer.

    IMPORTANT — where each method may be pointed:
      • `find_optimal_threshold` selects a threshold and MUST be given
        validation data. Running it on the test split and then reporting the
        result is threshold-selection-on-test: the test precision/recall become
        the best achievable rather than the achieved, which is a leak even
        though the model itself never saw the split. models/train.py selects on
        validation and separately reports the test-side optimum as an explicitly
        labelled ORACLE, to quantify how much was left on the table.
      • `evaluate_at_threshold` is the honest test-set call: a threshold fixed
        beforehand, applied once.
    """

    def __init__(
        self,
        fn_cost: float = DEFAULT_FN_COST,
        fp_cost: float = DEFAULT_FP_COST,
    ):
        self.config = CostConfig(fn_cost=fn_cost, fp_cost=fp_cost)

    # ── convenience passthroughs ──
    @property
    def break_even_probability(self) -> float:
        return self.config.break_even_probability

    def compute_cost(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Total financial cost of a set of hard predictions."""
        y = np.asarray(y_true).astype(int).ravel()
        p = np.asarray(y_pred).astype(int).ravel()
        if y.shape != p.shape:
            raise ValueError(f"shape mismatch: y_true {y.shape}, y_pred {p.shape}")
        fp = int(((p == 1) & (y == 0)).sum())
        fn = int(((p == 0) & (y == 1)).sum())
        return fn * self.config.fn_cost + fp * self.config.fp_cost

    # ── the exact sweep ──
    def sweep(self, y_true: np.ndarray, y_proba: np.ndarray) -> pd.DataFrame:
        """
        Enumerate EVERY distinct achievable confusion matrix, exactly.

        Total cost changes only when the threshold crosses an observed score, so
        the complete set of distinct outcomes is: "flag nothing", plus one row
        per distinct score value (flag everything scoring >= it). Sorting once
        and taking cumulative sums gives all of them in a single pass, so this
        is both exact and cheaper than the old fixed grid.

        Returns one row per achievable operating point, ordered from the
        strictest threshold (flag nothing) to the loosest (flag everything),
        with columns:
            threshold  tp fp tn fn  precision recall f1
            total_cost fp_cost_total fn_cost_total
            alert_rate n_flagged
        """
        y = np.asarray(y_true).astype(int).ravel()
        s = np.asarray(y_proba, dtype=float).ravel()
        if y.shape != s.shape:
            raise ValueError(f"shape mismatch: y_true {y.shape}, y_proba {s.shape}")
        if y.size == 0:
            raise ValueError("cannot sweep thresholds over an empty array")
        if not np.all(np.isfinite(s)):
            raise ValueError(
                "y_proba contains NaN or inf. A threshold sweep over "
                "non-finite scores silently produces nonsense: NaN >= t is "
                "always False, so those rows would be counted as 'not flagged' "
                "rather than reported as broken."
            )

        n = y.size
        n_pos = int(y.sum())
        n_neg = n - n_pos

        # Descending by score; stable so ties keep a deterministic order.
        order = np.argsort(-s, kind="mergesort")
        s_sorted = s[order]
        y_sorted = y[order]

        tp_cum = np.cumsum(y_sorted)
        fp_cum = np.cumsum(1 - y_sorted)

        # A threshold can only cut BETWEEN distinct scores: the last index of
        # each tie group. `score >= s_sorted[k]` flags exactly the top k+1.
        last_of_group = np.flatnonzero(np.diff(s_sorted) != 0)
        cuts = np.append(last_of_group, n - 1)

        # Prepend the "flag nothing" point, reachable by any threshold above
        # the maximum score. The old linspace grid could not express this, so it
        # could never report that a model is not worth deploying.
        thresholds = np.concatenate(([np.nextafter(s_sorted[0], np.inf)],
                                     s_sorted[cuts]))
        tp = np.concatenate(([0], tp_cum[cuts])).astype(np.int64)
        fp = np.concatenate(([0], fp_cum[cuts])).astype(np.int64)
        fn = n_pos - tp
        tn = n_neg - fp
        n_flagged = tp + fp

        with np.errstate(divide="ignore", invalid="ignore"):
            precision = np.where(n_flagged > 0, tp / np.maximum(n_flagged, 1), 0.0)
            recall = (tp / n_pos) if n_pos > 0 else np.zeros_like(tp, dtype=float)
            denom = precision + recall
            f1 = np.where(denom > 0, 2 * precision * recall / np.maximum(denom, 1e-12), 0.0)

        fn_total = fn * self.config.fn_cost
        fp_total = fp * self.config.fp_cost

        return pd.DataFrame({
            "threshold": thresholds,
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "total_cost": fn_total + fp_total,
            "fn_cost_total": fn_total,
            "fp_cost_total": fp_total,
            "n_flagged": n_flagged,
            "alert_rate": n_flagged / n,
        })

    def _row_to_report(
        self,
        row: pd.Series,
        n: int,
    ) -> ThresholdReport:
        """Build a report from a sweep row. Used by callers walking the curve."""
        total = float(row["total_cost"])
        return ThresholdReport(
            threshold=float(row["threshold"]),
            tp=int(row["tp"]), fp=int(row["fp"]),
            tn=int(row["tn"]), fn=int(row["fn"]),
            precision=float(row["precision"]),
            recall=float(row["recall"]),
            f1=float(row["f1"]),
            total_cost=total,
            cost_per_prediction=total / n if n else 0.0,
            alert_rate=float(row["alert_rate"]),
            alerts_per_1000=float(row["alert_rate"]) * 1000.0,
        )

    def evaluate_at_threshold(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        threshold: float,
    ) -> ThresholdReport:
        """
        Evaluate a FIXED threshold. This is the honest test-set call.

        Predicate is `score >= threshold`, matching the sweep and the API.
        """
        y = np.asarray(y_true).astype(int).ravel()
        s = np.asarray(y_proba, dtype=float).ravel()
        if y.shape != s.shape:
            raise ValueError(f"shape mismatch: y_true {y.shape}, y_proba {s.shape}")
        if not np.isfinite(threshold):
            raise ValueError(f"threshold must be finite, got {threshold}")

        pred = s >= threshold
        tp = int((pred & (y == 1)).sum())
        fp = int((pred & (y == 0)).sum())
        fn = int((~pred & (y == 1)).sum())
        tn = int((~pred & (y == 0)).sum())
        precision, recall, f1 = _prf(tp, fp, fn)
        total = fn * self.config.fn_cost + fp * self.config.fp_cost
        n = y.size
        alert_rate = (tp + fp) / n if n else 0.0

        return ThresholdReport(
            threshold=float(threshold),
            tp=tp, fp=fp, tn=tn, fn=fn,
            precision=precision, recall=recall, f1=f1,
            total_cost=float(total),
            cost_per_prediction=float(total / n) if n else 0.0,
            alert_rate=float(alert_rate),
            alerts_per_1000=float(alert_rate) * 1000.0,
        )

    def find_optimal_threshold(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        num_steps: int | None = None,   # accepted and ignored; see below
    ) -> ThresholdReport:
        """
        Minimum-cost threshold, taken from the MIDDLE of the cost plateau.

        Point this at VALIDATION data. See the class docstring.

        Why the middle and not the minimum: total cost is a step function of the
        threshold, so the minimum is typically a contiguous RANGE of thresholds
        rather than a point. v2 returned the first one it saw, i.e. that range's
        left edge, which is the least robust point in the whole optimal region —
        an infinitesimal shift in the score distribution at serve time moves the
        decision set. The midpoint is maximally far from both edges.

        Where several disjoint ranges tie on cost, the WIDEST is chosen, because
        width is exactly the tolerance to distribution shift, and
        `plateau_lo`/`plateau_hi`/`n_equivalent` on the returned report make the
        choice auditable. Note that equal cost does NOT imply an identical
        confusion matrix: with an integer-ish cost ratio, one extra false
        negative can tie exactly against ~13 fewer false positives. So the
        returned counts are obtained by evaluating the CHOSEN threshold, not by
        copying a row out of the plateau — those can differ, and reporting a
        confusion matrix that the stated threshold does not actually produce
        would be a quietly wrong number in metrics.json.

        `num_steps` is accepted for call-site compatibility and ignored: the
        sweep is exact now, so a step count has no meaning.
        """
        curve = self.sweep(y_true, y_proba)
        cost = curve["total_cost"].to_numpy()
        best = float(cost.min())
        at_min = np.flatnonzero(cost == best)

        # Split tied indices into contiguous runs; thresholds descend down the
        # frame, so a run's first index is its high end.
        runs: list[tuple[int, int]] = []
        start = at_min[0]
        for prev, cur in zip(at_min, at_min[1:]):
            if cur != prev + 1:
                runs.append((start, prev))
                start = cur
        runs.append((start, at_min[-1]))

        thr = curve["threshold"].to_numpy()

        def bounds(run: tuple[int, int]) -> tuple[float, float]:
            """Threshold interval (lo, hi] realising this run of rows."""
            hi_i, lo_i = run
            lo = float(thr[lo_i + 1]) if lo_i + 1 < thr.size else 0.0
            return lo, float(thr[hi_i])

        hi_i, lo_i = max(runs, key=lambda r: bounds(r)[1] - bounds(r)[0])
        plateau_lo, plateau_hi = bounds((hi_i, lo_i))

        chosen = 0.5 * (plateau_lo + plateau_hi)
        report = self.evaluate_at_threshold(y_true, y_proba, chosen)

        # The midpoint of (lo, hi] must itself be cost-optimal. It can fail to be
        # only in one degenerate case: when the optimum is the "flag nothing"
        # row, whose interval is (max_score, nextafter(max_score)] and therefore
        # has NO representable interior — the midpoint rounds down to an endpoint
        # that flags one account. Fall back to the interval's top, which always
        # realises the intended row.
        if report.total_cost != best:
            chosen = plateau_hi
            report = self.evaluate_at_threshold(y_true, y_proba, chosen)
        if report.total_cost != best:      # pragma: no cover — defensive
            raise AssertionError(
                f"plateau logic is wrong: threshold {chosen!r} costs "
                f"{report.total_cost} but the sweep minimum is {best}"
            )

        report.plateau_lo = plateau_lo
        report.plateau_hi = plateau_hi
        report.n_equivalent = int(at_min.size)
        return report

    def cost_curve(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        num_steps: int = 200,
    ) -> pd.DataFrame:
        """
        Cost curve for plotting (dashboard/components/cost_slider.py).

        Built from the exact sweep, then thinned to AT MOST `num_steps` rows for
        rendering. Thinning always retains both endpoints and the minimum-cost
        row, so the plotted curve cannot disagree with the reported optimum —
        which a separately-computed grid could, and would be a confusing thing
        for a reviewer to spot on screen mid-demo.
        """
        curve = self.sweep(y_true, y_proba)
        if len(curve) > num_steps:
            required = {0, len(curve) - 1, int(curve["total_cost"].idxmin())}
            budget = max(num_steps - len(required), 2)
            keep = set(np.linspace(0, len(curve) - 1, budget).astype(int))
            keep |= required
            # Trim from the sampled points (never the required ones) if the
            # union still exceeds the budget, so `num_steps` is a real bound.
            while len(keep) > num_steps:
                keep.discard(next(i for i in sorted(keep) if i not in required))
            curve = curve.iloc[sorted(keep)].reset_index(drop=True)
        # Column aliases the dashboard already reads.
        return curve.assign(
            fp_cost=curve["fp_cost_total"],
            fn_cost=curve["fn_cost_total"],
        )

    def threshold_for_alert_budget(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        alerts_per_1000: float,
    ) -> ThresholdReport:
        """
        The strictest threshold that stays inside a given analyst capacity.

        This exists because the cost-optimal threshold is often operationally
        useless. With FN priced at 13.3x FP, minimising total cost drives the
        model to flag ~20% of all accounts — mathematically correct, and nobody
        is staffing a review queue for one account in five. The real constraint
        on a risk team is analyst-hours, so the honest way to present this system
        is BOTH numbers: the cost-optimal point, and the point you can actually
        staff.

        Returns the report for the highest threshold whose alert rate does not
        exceed the budget, i.e. the best recall purchasable with that capacity.
        """
        if alerts_per_1000 < 0:
            raise ValueError(f"alert budget must be >= 0, got {alerts_per_1000}")
        curve = self.sweep(y_true, y_proba)
        affordable = curve[curve["alert_rate"] <= alerts_per_1000 / 1000.0]
        if affordable.empty:            # budget below even one alert
            row = curve.iloc[0]
        else:
            # Thresholds descend down the frame, so the last affordable row is
            # the loosest threshold that still fits — the most recall for the money.
            row = affordable.iloc[-1]
        return self._row_to_report(row, n=int(np.asarray(y_true).size))

    def sensitivity_to_cost_ratio(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        ratios: tuple[float, ...] = (2.0, 5.0, 10.0, 13.33, 25.0, 50.0),
    ) -> pd.DataFrame:
        """
        How the operating point moves as the FN:FP cost ratio changes.

        The ratio is the one assumption in this module that nobody can verify,
        so a result quoted at a single ratio is not a result. This table belongs
        in the README: it shows a reviewer whether the chosen operating point is
        a stable consequence of the model or an artefact of picking 13.3.

        Holds fp_cost fixed and varies fn_cost, since only the ratio affects the
        threshold.
        """
        rows = []
        for r in ratios:
            ev = CostEvaluator(
                fn_cost=self.config.fp_cost * r,
                fp_cost=self.config.fp_cost,
            )
            rep = ev.find_optimal_threshold(y_true, y_proba)
            rows.append({
                "fn_fp_ratio": r,
                "break_even_p": ev.break_even_probability,
                "threshold": rep.threshold,
                "plateau_width": rep.plateau_width,
                "precision": rep.precision,
                "recall": rep.recall,
                "f1": rep.f1,
                "alerts_per_1000": rep.alerts_per_1000,
                "total_cost": rep.total_cost,
            })
        return pd.DataFrame(rows)

    def summary(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        alert_budget_per_1000: float = 20.0,
    ) -> str:
        """
        Human-readable cost analysis.

        Prints three things a reviewer will otherwise have to ask for:
        the empirical optimum, the THEORETICAL break-even cutoff (a large gap
        between them is a calibration problem, not a sweep problem), and the
        operating point that fits a stated analyst capacity — because the
        cost-optimal point usually does not fit one.
        """
        opt = self.find_optimal_threshold(y_true, y_proba)
        budgeted = self.threshold_for_alert_budget(
            y_true, y_proba, alert_budget_per_1000
        )
        auc = roc_auc(y_true, y_proba)
        y = np.asarray(y_true).astype(int).ravel()
        cfg = self.config

        # Cost of the two trivial policies, for scale.
        n_pos, n_neg = int(y.sum()), int(y.size - y.sum())
        cost_flag_none = n_pos * cfg.fn_cost
        cost_flag_all = n_neg * cfg.fp_cost
        best_trivial = min(cost_flag_none, cost_flag_all)
        saved = best_trivial - opt.total_cost

        width = opt.plateau_width
        if width is None:
            thr_line = f"  Chosen threshold:    {opt.threshold:.4f}"
        else:
            note = ("  (wide — robust)" if width > 0.05
                    else "  (NARROW — threshold may be fitted to noise)")
            thr_line = (
                f"  Chosen threshold:    {opt.threshold:.4f}"
                f"   [plateau {opt.plateau_lo:.4f}–{opt.plateau_hi:.4f},"
                f" width {width:.4f}]{note}"
            )

        lines = [
            "━━━ Cost-Sensitive Evaluation ━━━",
            f"  Cost model:          FN ₹{cfg.fn_cost:,.0f} / FP ₹{cfg.fp_cost:,.0f}"
            f"  (ratio {cfg.ratio:.1f}:1)",
            f"  Break-even p*:       {cfg.break_even_probability:.4f}"
            f"   → an alert queue pays for itself above "
            f"{cfg.break_even_probability:.1%} precision",
            f"  ROC-AUC:             {auc:.4f}",
            "",
            "  ── Cost-optimal operating point ──",
            thr_line,
            f"  Precision / Recall:  {opt.precision:.4f} / {opt.recall:.4f}"
            f"   (F1 {opt.f1:.4f})",
            f"  Confusion:           TP {opt.tp}  FP {opt.fp}  "
            f"TN {opt.tn}  FN {opt.fn}",
            f"  Analyst load:        {opt.alerts_per_1000:.1f} alerts "
            f"per 1,000 accounts ({opt.alert_rate:.2%})",
            f"  FN ₹{opt.fn * cfg.fn_cost:,.0f}  +  FP ₹{opt.fp * cfg.fp_cost:,.0f}"
            f"  =  TOTAL ₹{opt.total_cost:,.0f}   (₹{opt.cost_per_prediction:,.2f}"
            f" per account)",
            "",
            f"  ── Capacity-constrained point ({alert_budget_per_1000:.0f}"
            f" alerts per 1,000) ──",
            f"  Threshold:           {budgeted.threshold:.4f}",
            f"  Precision / Recall:  {budgeted.precision:.4f} / "
            f"{budgeted.recall:.4f}   (F1 {budgeted.f1:.4f})",
            f"  Analyst load:        {budgeted.alerts_per_1000:.1f} alerts "
            f"per 1,000 accounts",
            f"  Total cost:          ₹{budgeted.total_cost:,.0f}"
            f"   (₹{budgeted.total_cost - opt.total_cost:+,.0f} vs cost-optimal —"
            f" the price of a staffable queue)",
            "",
            f"  Best trivial policy: ₹{best_trivial:,.0f}"
            f"  ({'flag nothing' if cost_flag_none <= cost_flag_all else 'flag everything'})",
        ]
        if best_trivial > 0:
            lines.append(
                f"  Value of the model:  ₹{saved:,.0f}"
                f"  ({saved / best_trivial:.1%} of the trivial baseline's cost)"
            )
        return "\n".join(lines)
