"""
dashboard/app.py
────────────────
Streamlit entry point for the UPI Mule-Ring Sentinel dashboard.

    streamlit run dashboard/app.py

─────────────────────────────────────────────────────────────────────────────
WHAT CHANGED FROM v2, AND WHY IT MATTERED
─────────────────────────────────────────────────────────────────────────────
The v2 Overview printed four numbers straight from metrics.json:

    st.metric("ROC-AUC",   metrics.get("roc_auc", 0))     # 0.9999
    st.metric("Precision", metrics.get("precision", 0))   # 1.0000
    st.metric("Recall",    metrics.get("recall", 0))
    st.metric("F1 Score",  metrics.get("f1", 0))

Three things were wrong with that, and all three are the kind of wrong that wins
a hackathon slot and then falls apart in the interview:

1. THE NUMBERS WERE A PROPERTY OF A BROKEN DATASET.  precision 1.0 / AUC 0.9999
   came from the v2 generator, where legitimate accounts never paid anyone
   twice, so `repeat_ratio` alone scored AUC 0.9989. The headline was measuring
   the generator's forgetfulness, not a detector. A perfect score on a fraud
   model is not a triumph, it is a symptom.

2. THEY WERE UNLABELLED AS TO SPLIT.  `roc_auc` / `precision` at top level are
   the TEST numbers AT THE THRESHOLD CHOSEN ON VALIDATION. That provenance is
   the entire difference between an honest metric and an optimistic one, and the
   page hid it.

3. metrics.json CAN BE STALE.  It is written by the last `train.py` run and
   nothing forces it to match the current code. The file on disk right now says
   `model_version: sentinel_v2` and lists `net_flow`, a feature v3 dropped.
   Reading a headline off it would report a different model's performance.

So this page no longer trusts stored scalars for its headline. It scores the
held-out test split with the REAL booster (via dashboard/scoring.py) and
computes precision/recall/F1/cost live, at the threshold metrics.json published,
and says so in those words. metrics.json is used only for what genuinely must
come from training and cannot be recomputed here: the published baselines, the
SHAP importances, and the dataset provenance block. Every one of those degrades
to an explanation if the block is absent, because half a panel with a caption
beats a whole panel with an invented zero.

The Score Demo's timestamp was also outside the serving context window
(2025-03-02 … 2025-04-30), so every demo request silently dropped all context
and scored on a one-edge graph. It now defaults inside the window and says why
that matters.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dashboard.components.cost_slider import render_cost_slider  # noqa: E402
from dashboard.components.graph_viz import render_graph_visualization  # noqa: E402
from dashboard.scoring import (  # noqa: E402
    ModelUnavailable,
    feature_importance,
    load_metrics,
    resolve_threshold,
    score_split,
)
from models.cost_matrix import (  # noqa: E402
    DEFAULT_FN_COST,
    DEFAULT_FP_COST,
    CostEvaluator,
)
from models.features import MODEL_VERSION, TARGET_COL  # noqa: E402

# The serving context window, from data/generator.py. The Score Demo must date
# its transactions inside this range or the API drops the context graph.
CONTEXT_WINDOW = ("2025-03-02", "2025-04-30")
DEMO_TIMESTAMP = "2025-04-15T14:30:00"

SPLITS = (("train", "Train"), ("val", "Validation"), ("test", "Test"))


st.set_page_config(
    page_title="UPI Mule-Ring Sentinel",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .main-header {
        font-size: 2.4rem; font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 0.3rem;
      }
      .subtitle { font-size: 1.05rem; color: #8a94a6; margin-bottom: 1.5rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ══════════════════════════════════════════════════════════════════
# Data loading
# ══════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def load_data() -> dict:
    """
    Load every split's features and edges, plus metrics.json if present.

    v2 loaded train and test only, so the whole validation split — the one the
    threshold is actually chosen on, and the one a viewer most needs to compare
    against test — was invisible. All three are loaded now, and the graph and
    cost components read whichever the viewer selects.
    """
    data: dict = {}
    processed = PROJECT_ROOT / "data" / "processed"
    raw = PROJECT_ROOT / "data" / "raw"

    for split, _ in SPLITS:
        fpath = processed / f"{split}_features.csv"
        if fpath.exists():
            data[f"{split}_features"] = pd.read_csv(fpath)
        epath = raw / f"{split}_edges.csv"
        if epath.exists():
            data[f"{split}_edges"] = pd.read_csv(epath)

    try:
        data["metrics"] = load_metrics()
    except ModelUnavailable as exc:
        data["metrics_error"] = str(exc)

    return data


# ══════════════════════════════════════════════════════════════════
# Sidebar
# ══════════════════════════════════════════════════════════════════

def render_sidebar(data: dict) -> str:
    st.sidebar.markdown("## 🛡️ Sentinel")
    page = st.sidebar.radio(
        "View",
        ["Overview", "Graph explorer", "Cost analysis", "Score demo"],
        index=0,
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Data on disk")
    rows = [(f"{s}_features", f"{label} features") for s, label in SPLITS]
    rows += [(f"{s}_edges", f"{label} edges") for s, label in SPLITS]
    for key, label in rows:
        st.sidebar.markdown(f"{'✅' if key in data else '⬜'} {label}")

    st.sidebar.markdown("---")
    if "metrics" in data:
        version = data["metrics"].get("model_version", "?")
        if version == MODEL_VERSION:
            st.sidebar.markdown(f"✅ metrics.json — `{version}`")
        else:
            st.sidebar.markdown(
                f"⚠️ metrics.json is `{version}`, code is `{MODEL_VERSION}`"
            )
    else:
        st.sidebar.markdown("⬜ metrics.json — not found")

    st.sidebar.caption(
        "Headline metrics on the Overview are computed live from the trained "
        "model on the held-out test split, not read from metrics.json."
    )
    return page


# ══════════════════════════════════════════════════════════════════
# Overview
# ══════════════════════════════════════════════════════════════════

def _headline_from_model(metrics: dict | None) -> None:
    """
    Score the test split with the real model and report metrics live.

    Everything here is a function of `predict_proba` on the held-out split at the
    threshold metrics.json published — not a stored scalar. If the model or the
    threshold is unavailable, the panel says exactly what to run and shows
    nothing else, because a headline metric with a fabricated value is worse than
    an honest blank.
    """
    try:
        scored = score_split("test")
    except ModelUnavailable as exc:
        st.warning(
            "Headline metrics are computed from the trained model, and it is not "
            f"available yet.\n\n{exc}"
        )
        return

    threshold, source = None, "the validation-selected threshold in metrics.json"
    if metrics is not None:
        try:
            threshold = resolve_threshold(metrics)
        except ModelUnavailable:
            threshold = None
    if threshold is None:
        evaluator = CostEvaluator(fn_cost=DEFAULT_FN_COST, fp_cost=DEFAULT_FP_COST)
        threshold = float(evaluator.break_even_probability)
        source = (
            f"the break-even probability at ₹{DEFAULT_FN_COST:,.0f} : "
            f"₹{DEFAULT_FP_COST:,.0f} (metrics.json carried no threshold)"
        )

    evaluator = CostEvaluator(fn_cost=DEFAULT_FN_COST, fp_cost=DEFAULT_FP_COST)
    report = evaluator.evaluate_at_threshold(scored.y_true, scored.y_proba, threshold)

    st.markdown(
        f"**Held-out test split**, scored live by the trained model at threshold "
        f"**{threshold:.4f}** — {source}. "
        f"{scored.n:,} accounts, {scored.n_positive} labelled mules "
        f"({scored.prevalence:.2%} prevalence)."
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Precision", f"{report.precision:.3f}",
                help="Of the accounts flagged, the share that were mules.")
    col2.metric("Recall", f"{report.recall:.3f}",
                help="Of the mules present, the share the model flagged.")
    col3.metric("F1", f"{report.f1:.3f}")
    col4.metric("Alert rate", f"{report.alert_rate:.1%}",
                help=f"{report.alerts_per_1000:.0f} of every 1,000 accounts sent "
                     f"to a human. The number a review team is actually staffed "
                     f"against.")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("True positives", report.tp)
    col2.metric("False positives", report.fp, delta_color="inverse")
    col3.metric("False negatives", report.fn, delta_color="inverse",
                help="Missed mules — the expensive error at ₹200k each.")
    col4.metric("Total cost", f"₹{report.total_cost:,.0f}",
                help="At the default ₹200k FN / ₹15k FP. See the Cost analysis "
                     "tab to move these prices.")

    _baseline_line(metrics, report, threshold, scored)


def _baseline_line(metrics, report, threshold, scored) -> None:
    """The one comparison that turns an absolute score into a claim."""
    baselines = (metrics or {}).get("baselines") or {}
    rule = (baselines.get("best_single_feature_rule_by_f1")
            or baselines.get("best_single_feature_rule_by_cost"))
    if not isinstance(rule, dict) or "feature" not in rule:
        st.caption(
            "No baseline block in metrics.json, so there is nothing yet to "
            "compare against. A precision and recall with no reference point is "
            "not a result — re-run `python -m models.train` to publish the "
            "best single-feature rule, and read the model as lift over it."
        )
        return

    b_f1 = rule.get("test_f1")
    b_rule = rule.get("rule", rule["feature"])
    if b_f1 is not None:
        lift = report.f1 - float(b_f1)
        verdict = "above" if lift >= 0 else "BELOW"
        st.markdown(
            f"**Lift over the baseline.** The best single-feature rule "
            f"(`{b_rule}`) scores F1 **{float(b_f1):.3f}** on this split. The "
            f"model at threshold {threshold:.4f} scores F1 **{report.f1:.3f}** — "
            f"{abs(lift):.3f} {verdict} the rule. That gap, not the absolute "
            f"number, is what a graph model has to earn to justify its "
            f"complexity."
        )


def _feature_importance_panel(metrics: dict | None) -> None:
    """Feature importance, SHAP-preferred and labelled as to which it is."""
    if metrics is None:
        return
    try:
        frame, source = feature_importance(metrics)
    except ModelUnavailable:
        st.caption(
            "metrics.json carries no feature-importance block. Re-run "
            "`python -m models.train`; it writes both XGBoost gain and mean "
            "|SHAP| on the test split."
        )
        return
    st.markdown("#### Feature importance")
    st.caption(
        f"Source: {source}. Gain and SHAP answer different questions — gain is a "
        "training-time split statistic, SHAP is how much each feature moves "
        "predictions on the held-out split — so which one is on screen is stated "
        "rather than left for the reader to guess."
    )
    st.bar_chart(frame)


def _dataset_panel(data: dict) -> None:
    """Prevalence and size for every split that is on disk."""
    present = [(s, label) for s, label in SPLITS if f"{s}_features" in data]
    if not present:
        return
    st.markdown("#### Dataset")
    rows = []
    for split, label in present:
        frame = data[f"{split}_features"]
        pos = int(frame[TARGET_COL].sum()) if TARGET_COL in frame else 0
        rows.append({
            "Split": label,
            "Accounts": f"{len(frame):,}",
            "Mules": pos,
            "Prevalence": f"{pos / max(len(frame), 1):.2%}",
        })
    st.table(pd.DataFrame(rows).set_index("Split"))

    metrics = data.get("metrics") or {}
    dataset = metrics.get("dataset")
    if isinstance(dataset, dict) and dataset.get("leakage_gate"):
        gate = dataset["leakage_gate"]
        st.caption(
            f"Leakage gate: the strongest single feature reaches AUC "
            f"{gate.get('max_single_feature_auc', float('nan')):.3f} on test "
            f"(`{gate.get('feature', '?')}`); the build fails at "
            f"{gate.get('threshold', 0.99)}. The point of the gate is that no "
            f"one feature can carry the label — if it could, the model would be "
            f"learning a generator artefact, which is exactly what v2 did."
        )
    else:
        st.caption(
            "Splits are disjoint 60-day windows by construction "
            "(entity- and ring-disjoint; see tests/). Equal windows matter "
            "because window length multiplies every count and sum feature, so "
            "unequal splits would make the model look like it generalised when "
            "it had only changed scale."
        )


def page_overview(data: dict) -> None:
    st.markdown('<h1 class="main-header">UPI Mule-Ring Sentinel</h1>',
                unsafe_allow_html=True)
    st.markdown(
        '<p class="subtitle">Graph-based detection of mule-account rings in UPI '
        'transaction networks — defence-only, with honest false-positive cost.</p>',
        unsafe_allow_html=True,
    )

    metrics = data.get("metrics")
    if metrics is None and "metrics_error" in data:
        st.info(data["metrics_error"])

    if isinstance(metrics, dict):
        version = metrics.get("model_version")
        if version and version != MODEL_VERSION:
            st.warning(
                f"metrics.json on disk was written for **{version}**, but the "
                f"code is on **{MODEL_VERSION}**. Its stored scalars describe a "
                f"different model and a different feature set, so this page "
                f"ignores them and recomputes the headline from the live model. "
                f"Re-run `python -m models.train` to refresh the file."
            )

    _headline_from_model(metrics)
    st.markdown("---")
    _feature_importance_panel(metrics)
    st.markdown("---")
    _dataset_panel(data)


# ══════════════════════════════════════════════════════════════════
# Other pages
# ══════════════════════════════════════════════════════════════════

def page_graph_explorer(data: dict) -> None:
    st.markdown("## Graph explorer")
    if not any(f"{s}_edges" in data for s, _ in SPLITS):
        st.warning("No edge data on disk. Run `python -m data.generator` first.")
        return
    render_graph_visualization(data)


def page_cost_analysis(data: dict) -> None:
    st.markdown("## Cost-sensitive threshold analysis")
    render_cost_slider(data)


def page_score_demo(data: dict) -> None:
    st.markdown("## Live scoring demo")
    st.info(
        "Start the API in another terminal:\n\n"
        "```bash\nuvicorn api.main:app --port 8000\n```\n\n"
        "The API scores an account against a frozen **serving context graph** "
        f"spanning {CONTEXT_WINDOW[0]} → {CONTEXT_WINDOW[1]}. A submitted "
        "transaction only gets real graph features (PageRank, cycle "
        "participation, community ratio) if its accounts connect to that "
        "context, which is why the timestamp below sits inside the window: a "
        "date outside it makes the API drop the context and score on a "
        "one-edge graph, and the risk score would then be meaningless."
    )

    with st.form("scoring_form"):
        col1, col2 = st.columns(2)
        with col1:
            sender = st.text_input("Sender VPA", value="user123@upi")
            amount = st.number_input("Amount (₹)", min_value=1, value=25_000)
        with col2:
            receiver = st.text_input("Receiver VPA", value="merchant456@upi")
            timestamp = st.text_input(
                "Timestamp (inside the context window)", value=DEMO_TIMESTAMP,
            )
        submitted = st.form_submit_button("Score transaction")

    if not submitted:
        return

    try:
        import requests
    except ImportError:
        st.error("The `requests` package is not installed. `pip install requests`.")
        return

    payload = {"transactions": [{
        "sender": sender, "receiver": receiver,
        "amount": amount, "timestamp": timestamp,
    }]}

    try:
        resp = requests.post("http://localhost:8000/score", json=payload, timeout=10)
    except requests.ConnectionError:
        st.error("Cannot reach the API on port 8000. Is `uvicorn` running?")
        return
    except requests.Timeout:
        st.error("The API did not respond within 10s.")
        return

    if resp.status_code != 200:
        st.error(f"API returned {resp.status_code}: {resp.text}")
        return

    result = resp.json()
    context = result.get("context", {})

    # Surface the context health BEFORE the scores, because a score computed on a
    # dropped context is not wrong-looking, it is just wrong.
    if not context.get("window_comparable", True):
        st.warning(
            "The API reports the observation window is not comparable to the "
            "trained window, so scale-dependent features may be off. "
            f"Submitted transactions: {context.get('n_submitted_transactions', '?')}, "
            f"context transactions used: {context.get('n_context_transactions_used', '?')}."
        )
    for warning in context.get("warnings", []):
        st.warning(warning)

    threshold_used = result.get("threshold_used")
    fingerprint = context.get("partition_fingerprint")
    caption = f"Threshold {threshold_used:.4f}" if threshold_used is not None else ""
    if fingerprint:
        caption += f" · partition `{fingerprint}`"
    if caption:
        st.caption(caption)

    for node in result.get("node_scores", []):
        risk = node.get("risk_level", "?")
        colour = {"LOW": "#2ed573", "MEDIUM": "#ffa502",
                  "HIGH": "#ff6b6b", "CRITICAL": "#ff4757"}.get(risk, "#8a94a6")
        seen = node.get("seen_in_context")
        seen_note = "" if seen else "  ·  ⚠️ not in context graph — score is on a bare edge"
        st.markdown(
            f"**{node.get('node_id')}** — "
            f"<span style='color:{colour}'>{risk}</span> "
            f"(risk {node.get('risk_score', 0):.4f}) → {node.get('action')}"
            f"{seen_note}",
            unsafe_allow_html=True,
        )
        factors = node.get("contributing_factors", [])
        if factors:
            st.table(pd.DataFrame([{
                "Feature": f.get("feature"),
                "Value": f"{f.get('value', float('nan')):.4f}",
                "Effect": f.get("effect"),
                "Why": f.get("description"),
            } for f in factors]))

    with st.expander("Raw API response"):
        st.json(result)


# ══════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════

def main() -> None:
    data = load_data()
    page = render_sidebar(data)

    if page == "Overview":
        page_overview(data)
    elif page == "Graph explorer":
        page_graph_explorer(data)
    elif page == "Cost analysis":
        page_cost_analysis(data)
    elif page == "Score demo":
        page_score_demo(data)


if __name__ == "__main__":
    main()
