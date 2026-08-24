"""
dashboard/components/cost_slider.py
────────────────────────────────────
Cost-sensitive threshold analysis, computed from REAL model scores.

─────────────────────────────────────────────────────────────────────────────
WHAT CHANGED FROM v2, AND WHY IT MATTERED
─────────────────────────────────────────────────────────────────────────────
v2 rendered this page from `_simulate_predictions`, which built a score as

    np.clip(raw_scores + noise + y_true * 0.3, 0.01, 0.99)

— i.e. it added the ground-truth label into the score. Every number the page
displayed (precision, recall, F1, cost, the whole confusion matrix, the shape of
the cost curve) was therefore a property of that formula and not of the model.
The function is deleted. Scores now come from `dashboard/scoring.py`, which
loads the trained booster, verifies its feature contract, and calls
`predict_proba`. If that is not possible, the page says so and renders nothing.

Three further honesty fixes:

• The page evaluates the HELD-OUT TEST split by default and labels it. Reading
  cost off the validation split would report the threshold's performance on the
  data the threshold was chosen from.

• The published baseline is drawn on the same axes, and RE-PRICED at whatever
  FN/FP the viewer enters. "Total cost falls to ₹X" is not a result on its own —
  a one-line rule over a single feature also has a cost, and the honest claim is
  the distance between them. metrics.json stores the rule's cost at the default
  prices only, so the rule is replayed over the same accounts here; a fixed
  baseline number beside a moving model number would let the model appear to
  pull ahead purely because the prices changed.

• The threshold slider spans the full range the optimum could occupy. v2 floored
  it at 0.05 while the cost-optimal point at the default costs is ≈0.07, so a
  viewer raising `fn_cost` would push the optimum below the slider's reach and
  never see it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dashboard.scoring import ModelUnavailable, ScoredSplit, score_split  # noqa: E402
from models.cost_matrix import (  # noqa: E402
    DEFAULT_FN_COST,
    DEFAULT_FP_COST,
    CostEvaluator,
)

SLIDER_MIN = 0.01
SLIDER_MAX = 0.99
SLIDER_STEP = 0.005


@st.cache_data(show_spinner="Scoring the held-out split with the real model...")
def _scored(split: str) -> tuple[np.ndarray, np.ndarray, pd.Series, pd.DataFrame]:
    """
    Cached real scores, plus the feature frame they came from.

    The frame is returned because the baseline has to be RE-PRICED at whatever
    FN/FP the viewer types in. metrics.json publishes the baseline's cost at the
    default 200k/15k only, so quoting that fixed number beside a model cost that
    moves with the sliders would let the model appear to pull ahead purely
    because the prices changed. Replaying the rule over the same accounts keeps
    the comparison like for like.
    """
    scored: ScoredSplit = score_split(split)
    return scored.y_true, scored.y_proba, scored.ring_type, scored.features


def _plot_cost_curve(
    cost_df: pd.DataFrame,
    current_threshold: float,
    optimal_threshold: float,
    baseline_cost: float | None,
    baseline_label: str | None,
) -> go.Figure:
    """Cost and precision/recall against threshold, with the baseline drawn in."""
    fig = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=(
            "Total financial cost vs. threshold (held-out split, real model)",
            "Precision / recall / F1 tradeoff",
        ),
        vertical_spacing=0.15,
        row_heights=[0.6, 0.4],
    )

    fig.add_trace(
        go.Scatter(
            x=cost_df["threshold"], y=cost_df["total_cost"],
            mode="lines", name="Total cost",
            line=dict(color="#ff6b6b", width=3),
            fill="tozeroy", fillcolor="rgba(255, 107, 107, 0.1)",
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=cost_df["threshold"], y=cost_df["fn_cost"],
            mode="lines", name="FN cost (missed mules)",
            line=dict(color="#ffa502", width=2, dash="dash"),
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=cost_df["threshold"], y=cost_df["fp_cost"],
            mode="lines", name="FP cost (false alarms)",
            line=dict(color="#70a1ff", width=2, dash="dash"),
        ),
        row=1, col=1,
    )

    # The baseline as a horizontal line: the cost a one-feature rule already
    # achieves. Anything the model does above this line is worse than a rule.
    if baseline_cost is not None:
        fig.add_hline(
            y=baseline_cost,
            line_dash="dashdot",
            line_color="#b2bec3",
            annotation_text=f"baseline: {baseline_label} → ₹{baseline_cost:,.0f}",
            annotation_position="top left",
            row=1, col=1,
        )

    optimal_row = cost_df.iloc[
        (cost_df["threshold"] - optimal_threshold).abs().argsort()[:1]
    ]
    fig.add_trace(
        go.Scatter(
            x=optimal_row["threshold"], y=optimal_row["total_cost"],
            mode="markers+text", name="Cost-optimal threshold",
            marker=dict(color="#2ed573", size=14, symbol="star"),
            text=["optimal"], textposition="top center",
            textfont=dict(color="#2ed573"),
        ),
        row=1, col=1,
    )
    fig.add_vline(
        x=current_threshold, line_dash="dot", line_color="#dfe6e9",
        annotation_text=f"current: {current_threshold:.3f}", row=1, col=1,
    )

    for column, colour, label in (
        ("precision", "#a29bfe", "Precision"),
        ("recall", "#fdcb6e", "Recall"),
        ("f1", "#00b894", "F1"),
    ):
        fig.add_trace(
            go.Scatter(
                x=cost_df["threshold"], y=cost_df[column],
                mode="lines", name=label, line=dict(color=colour, width=2),
            ),
            row=2, col=1,
        )
    fig.add_vline(x=current_threshold, line_dash="dot",
                  line_color="#dfe6e9", row=2, col=1)

    fig.update_layout(
        height=700,
        template="plotly_dark",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.18,
                    xanchor="center", x=0.5),
        margin=dict(t=60, b=90),
    )
    fig.update_xaxes(title_text="Decision threshold", row=2, col=1)
    fig.update_yaxes(title_text="Cost (₹)", row=1, col=1)
    fig.update_yaxes(title_text="Score", row=2, col=1)
    return fig


def _baseline_rule(metrics: dict) -> tuple[str, dict] | None:
    """
    The published single-feature rule to compare against, preferring the
    cost-selected one.

    `models/train.py` selects two rules on the validation split — one minimising
    cost, one maximising F1 — and this page is about cost, so the cost-selected
    rule is the honest opponent. Both are returned by name so the caption can say
    which one is on screen.
    """
    baselines = metrics.get("baselines") or {}
    for name in ("best_single_feature_rule_by_cost", "best_single_feature_rule_by_f1"):
        body = baselines.get(name)
        if isinstance(body, dict) and "feature" in body:
            return name, body
    return None


def _price_baseline(
    body: dict,
    frame: pd.DataFrame,
    y_true: np.ndarray,
    fn_cost: float,
    fp_cost: float,
) -> dict | None:
    """
    Replay a published single-feature rule on this split at the viewer's prices.

    The rule is stored machine-readable in metrics.json as a feature name, a
    direction and a threshold applied to a SIGNED score, exactly as
    `models/train.py` evaluated it:

        score = (+1 if direction == "high" else -1) * x
        flag  = score >= threshold_on_score

    The sign flip is what lets one code path handle "high `cycle_participation`
    is suspicious" and "low `amount_cv` is suspicious" without a second branch,
    and it is why the threshold cannot be compared against the raw feature.

    Returns None when the stored rule is not replayable — an older metrics.json
    without `direction`/`threshold_on_score`, or a feature the extractor no
    longer produces. The caller must then say the baseline cannot be re-priced
    rather than silently showing a cost from a different price ratio.
    """
    feature = body.get("feature")
    direction = body.get("direction")
    threshold_on_score = body.get("threshold_on_score")
    if feature is None or direction is None or threshold_on_score is None:
        return None
    if feature not in frame.columns:
        return None

    sign = 1.0 if str(direction) == "high" else -1.0
    score = sign * frame[feature].to_numpy(dtype=float)
    flagged = score >= float(threshold_on_score)
    positive = y_true == 1

    tp = int((flagged & positive).sum())
    fp = int((flagged & ~positive).sum())
    fn = int((~flagged & positive).sum())
    tn = int((~flagged & ~positive).sum())

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) else 0.0)

    return {
        "rule": body.get("rule") or f"{feature} ({direction})",
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": precision, "recall": recall, "f1": f1,
        "total_cost": fn * fn_cost + fp * fp_cost,
        "alert_rate": (tp + fp) / len(y_true) if len(y_true) else 0.0,
    }


def render_cost_slider(data: dict) -> None:
    """Entry point called from dashboard/app.py."""
    metrics = data.get("metrics") or {}

    st.markdown(
        "Every number on this page is computed from **real `predict_proba` "
        "output** of the trained model on the **held-out test split** — the "
        "split neither training nor threshold selection ever saw. Move the "
        "slider to trade missed mules (FN) against false alarms on legitimate "
        "accounts (FP)."
    )

    split = st.radio(
        "Split to evaluate",
        ["test", "validation", "train"],
        index=0,
        horizontal=True,
        help=(
            "test is the honest one. validation is where the threshold was "
            "chosen, so its numbers are optimistic. train is shown only for "
            "contrast — a model's fit to its own training data is not a result."
        ),
    )
    if split != "test":
        st.warning(
            f"You are looking at the **{split}** split. "
            + ("The operating threshold was selected on this data, so precision "
               "and recall here flatter the model."
               if split == "validation" else
               "The model was fitted on this data. These numbers measure "
               "memorisation, not detection.")
        )

    try:
        y_true, y_proba, ring_type, frame = _scored(split)
    except ModelUnavailable as exc:
        st.error(str(exc))
        return

    st.markdown("### Cost parameters")
    col1, col2 = st.columns(2)
    with col1:
        fn_cost = st.number_input(
            "Cost per missed mule (₹)",
            min_value=1_000, max_value=10_000_000,
            value=int(DEFAULT_FN_COST), step=10_000,
            help="Funds lost plus recovery effort when a mule account is missed.",
        )
    with col2:
        fp_cost = st.number_input(
            "Cost per false alarm (₹)",
            min_value=100, max_value=1_000_000,
            value=int(DEFAULT_FP_COST), step=1_000,
            help=(
                "Analyst review time plus the cost of friction imposed on a "
                "legitimate customer. This is the number the track's bar calls "
                "for, and it is why the threshold is not 0.5."
            ),
        )

    evaluator = CostEvaluator(fn_cost=fn_cost, fp_cost=fp_cost)
    optimal = evaluator.find_optimal_threshold(y_true, y_proba)
    break_even = evaluator.break_even_probability

    st.caption(
        f"At {fn_cost:,.0f} : {fp_cost:,.0f} the break-even probability is "
        f"**{break_even:.4f}** — flag an account once its risk exceeds that, and "
        f"the expected saving beats the expected cost of being wrong. The "
        f"threshold is an economic quantity, not a default of 0.5."
    )

    st.markdown("### Decision threshold")
    default_threshold = float(np.clip(optimal.threshold, SLIDER_MIN, SLIDER_MAX))
    threshold = st.slider(
        "Threshold",
        min_value=SLIDER_MIN, max_value=SLIDER_MAX,
        value=default_threshold, step=SLIDER_STEP,
        help="Accounts with risk score ≥ threshold are flagged for human review.",
    )

    report = evaluator.evaluate_at_threshold(y_true, y_proba, threshold)
    cost_df = evaluator.cost_curve(y_true, y_proba)

    baseline_name, baseline = None, None
    selected = _baseline_rule(metrics)
    if selected is not None:
        baseline_name, body = selected
        baseline = _price_baseline(body, frame, y_true, fn_cost, fp_cost)
    baseline_cost = baseline["total_cost"] if baseline else None
    baseline_label = baseline["rule"] if baseline else None

    st.markdown(f"### Impact at threshold {threshold:.3f}")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric(
        "Total cost", f"₹{report.total_cost:,.0f}",
        delta=f"₹{report.total_cost - optimal.total_cost:+,.0f} vs optimal",
        delta_color="inverse",
    )
    col2.metric("Precision", f"{report.precision:.3f}")
    col3.metric("Recall", f"{report.recall:.3f}")
    col4.metric("F1", f"{report.f1:.3f}")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("True positives", report.tp)
    col2.metric("False positives", report.fp, delta_color="inverse")
    col3.metric("True negatives", report.tn)
    col4.metric("False negatives", report.fn, delta_color="inverse")

    # The operational number that decides whether any of this is deployable.
    st.metric(
        "Alert rate",
        f"{report.alert_rate:.1%} of accounts ({report.alerts_per_1000:.0f} per 1,000)",
        help=(
            "The cost-optimal threshold minimises rupees, not analyst hours. If "
            "this is 20%, the model is telling a review team to work one account "
            "in five, which no team is staffed for. models/cost_matrix.py also "
            "reports a capacity-constrained threshold for that reason."
        ),
    )

    st.markdown("### Against the published baseline")
    if baseline is not None:
        delta = baseline["total_cost"] - report.total_cost
        verdict = "cheaper than" if delta > 0 else "MORE EXPENSIVE than"
        st.markdown(
            f"The best single-feature rule selected on validation — "
            f"`{baseline['rule']}` — costs **₹{baseline['total_cost']:,.0f}** on "
            f"this split at these same prices "
            f"(P {baseline['precision']:.3f} / R {baseline['recall']:.3f} / "
            f"F1 {baseline['f1']:.3f}, {baseline['fp']:,} false alarms, "
            f"{baseline['fn']:,} misses). The model at threshold {threshold:.3f} "
            f"is **₹{abs(delta):,.0f} {verdict}** that one-line rule.\n\n"
            f"That difference is the result. An absolute cost is not a claim: a "
            f"graph model that cannot beat a single comparison on a single "
            f"feature has not earned its complexity, and the rule is re-scored "
            f"here at whatever FN/FP you type above, so the gap you see is a "
            f"like-for-like one."
        )
        st.table(pd.DataFrame({
            "": ["Total cost", "Precision", "Recall", "F1",
                 "False alarms (FP)", "Missed mules (FN)", "Alert rate"],
            f"Baseline rule ({baseline_name})": [
                f"₹{baseline['total_cost']:,.0f}",
                f"{baseline['precision']:.3f}",
                f"{baseline['recall']:.3f}",
                f"{baseline['f1']:.3f}",
                f"{baseline['fp']:,}",
                f"{baseline['fn']:,}",
                f"{baseline['alert_rate']:.1%}",
            ],
            f"Model @ {threshold:.3f}": [
                f"₹{report.total_cost:,.0f}",
                f"{report.precision:.3f}",
                f"{report.recall:.3f}",
                f"{report.f1:.3f}",
                f"{report.fp:,}",
                f"{report.fn:,}",
                f"{report.alert_rate:.1%}",
            ],
        }).set_index(""))
    elif selected is not None:
        st.warning(
            f"metrics.json carries a baseline (`{selected[1].get('rule', '?')}`) "
            f"but not the `direction` and `threshold_on_score` fields needed to "
            f"replay it, so it cannot be re-priced at your FN/FP. Its published "
            f"cost was computed at ₹{DEFAULT_FN_COST:,.0f} / "
            f"₹{DEFAULT_FP_COST:,.0f} and showing that fixed number next to a "
            f"model cost that moves with the sliders would be misleading. "
            f"Re-run `python -m models.train` to write the current schema."
        )
    else:
        st.info(
            "No `baselines` block in metrics.json, so there is nothing to "
            "compare against. Re-run `python -m models.train` — an absolute cost "
            "with no reference point is not a claim."
        )

    if abs(threshold - optimal.threshold) > SLIDER_STEP:
        extra = report.total_cost - optimal.total_cost
        st.info(
            f"The cost-optimal threshold at these prices is "
            f"**{optimal.threshold:.4f}** (total cost ₹{optimal.total_cost:,.0f}). "
            f"Your {threshold:.3f} costs ₹{extra:,.0f} more."
        )

    if optimal.plateau_width is not None:
        st.caption(
            f"The optimum sits on a cost plateau {optimal.plateau_lo:.4f}–"
            f"{optimal.plateau_hi:.4f} wide ({optimal.n_equivalent} thresholds "
            f"tie on cost); the reported value is its midpoint. A single "
            f"minimum-cost point would be an artefact of one account's score."
        )

    st.plotly_chart(
        _plot_cost_curve(cost_df, threshold, optimal.threshold,
                         baseline_cost, baseline_label),
        use_container_width=True,
    )

    st.markdown("### Cost breakdown")
    fn_total = report.fn * fn_cost
    fp_total = report.fp * fp_cost
    total = max(report.total_cost, 1.0)
    st.table(pd.DataFrame({
        "Category": ["Missed mules (FN)", "False alarms (FP)"],
        "Count": [report.fn, report.fp],
        "Unit cost": [f"₹{fn_cost:,.0f}", f"₹{fp_cost:,.0f}"],
        "Total": [f"₹{fn_total:,.0f}", f"₹{fp_total:,.0f}"],
        "Share": [f"{fn_total / total * 100:.1f}%",
                  f"{fp_total / total * 100:.1f}%"],
    }))

    # Recall by ring archetype: the aggregate figure hides which kinds of ring
    # the model actually catches, and "we catch the obvious ones" is the failure
    # mode a single recall number is best at concealing.
    if ring_type is not None and len(ring_type) == len(y_true):
        flagged = y_proba >= threshold
        rows = []
        for archetype in sorted(set(ring_type[y_true == 1])):
            mask = (ring_type == archetype) & (y_true == 1)
            n = int(mask.sum())
            if n:
                rows.append({
                    "Ring archetype": archetype,
                    "Mule accounts": n,
                    "Caught": int((mask & flagged).sum()),
                    "Recall": f"{(mask & flagged).sum() / n:.1%}",
                })
        if rows:
            st.markdown("### Recall by ring archetype")
            st.caption(
                "A single recall figure can hide a model that only finds the "
                "loudest pattern. Stealth rings are designed to be the hard "
                "ones; if their recall is near zero, the headline number is "
                "carried entirely by the easy archetypes."
            )
            st.table(pd.DataFrame(rows))
