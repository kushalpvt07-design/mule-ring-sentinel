"""
api/main.py
───────────
FastAPI server for the UPI Mule-Ring Sentinel.

Endpoints:
  GET  /health          → Service health + model status
  POST /score           → Score a batch of transactions

Usage:
    uvicorn api.main:app --reload --port 8000

─────────────────────────────────────────────────────────────────────────────
WHY THIS FILE WAS REWRITTEN, NOT PATCHED
─────────────────────────────────────────────────────────────────────────────
Every defect below made the service return numbers that looked fine and meant
nothing. There was no error to notice, which is what makes this class of bug
expensive.

1. IT SERVED A DIFFERENT MODEL THAN THE ONE THAT WAS EVALUATED.
   `MODEL_PATH` hard-coded `sentinel_v1.xgb` and `_model_version` hard-coded
   "sentinel_v1", while `models/features.py` declared `sentinel_v3.xgb` and
   training wrote that file. So the API loaded a retired 12-feature model and
   applied the CURRENT model's cost-optimal threshold to its scores — two
   unrelated calibrations bolted together. The path now derives from
   `features.MODEL_NAME` and can't drift.

2. IT DECLARED ITS OWN FEATURE LIST.
   A local 12-name `FEATURE_COLS` sat here, including `net_flow` (dropped in v3)
   and `louvain_community` (an arbitrary integer id, dropped in v2 as a feature).
   `assert_feature_contract` existed in models/features.py to catch exactly this,
   and was never called. It is now called at startup, and a mismatch puts the
   service into a degraded state that refuses to score.

3. IT COMPUTED FEATURES ON A GRAPH OF THE REQUEST ALONE.
   `extract_features_from_batch` built a DiGraph from just the submitted
   transactions. On a 4-edge graph `pagerank` is ~1/n by construction,
   `clustering_coefficient` is 0 because there are no triangles, and
   `cycle_participation` is 0 because a ring's cycle is not inside the batch — so
   the graph features that carry the entire thesis of this project were fed to
   the model as constants. It also hard-coded `louvain_community: 0`,
   `amount_cv: 0.0`, and defined `txn_velocity` as `in_degree + out_degree`,
   which is a degree, not a rate. Features now come from
   `data.extractor.compute_node_features` — the same function that produced the
   training set — over the submitted edges merged into a historical context graph.

4. EVERY ALERT CARRIED THE SAME THREE REASONS.
   `contributing_factors` was filled from `_model.feature_importances_`, a
   global, per-model quantity. Two accounts flagged for completely different
   behaviour got identical explanations. Attribution is now per-account exact
   TreeSHAP via models/explain.py, and the response reports the account's own
   value alongside each signed contribution.

5. `threshold_override=0.0` SILENTLY FELL BACK TO THE DEFAULT.
   `request.threshold_override or _optimal_threshold` treats 0.0 as absent. And
   `metrics.get("optimal_threshold", 0.5)` defaulted a missing key to 0.5 — with
   FN/FP at 200k/15k the real operating point is near 0.07, so that default would
   quietly discard most of the recall the model was tuned for. Both now fail
   loudly instead of guessing.

6. AN ACCOUNT'S SCORE DEPENDED ON WHO ELSE WAS IN THE BATCH.
   Every request repartitioned the merged graph with Louvain, and
   `community_internal_ratio` is a per-community scalar shared by every member,
   so redrawing communities moved the feature for accounts that had nothing to do
   with the request. Measured: submitting two accounts that transact only with
   each other and connect to nothing else moved that feature for 100% of the
   2,954 context accounts and flipped 2.84% of decisions at the operating
   threshold. Louvain is deterministic under a fixed seed, so this never showed
   up as flakiness on a fixed graph — it only appears when the node set changes,
   which is every real request.
   The partition is now computed ONCE at startup from the context graph and
   passed into the extractor, which extends it deterministically for accounts it
   has not seen. Same perturbation after the fix: zero accounts changed on every
   non-PageRank feature, zero decision flips. /health publishes a fingerprint of
   the partition so two replicas that disagree are visible.

─────────────────────────────────────────────────────────────────────────────
KNOWN LIMITATIONS — read before quoting a latency figure
─────────────────────────────────────────────────────────────────────────────
• Graph features are recomputed over the whole merged graph on every request:
  PageRank and bounded cycle enumeration across ~60k edges. That is tens to
  hundreds of milliseconds, not a sub-millisecond feature-store lookup. The
  response reports `feature_computation_ms` so the cost is visible rather than
  claimed away. A production deployment would keep incrementally-maintained node
  features in a store and this endpoint would read them.

• PageRank is a global fixpoint over a normalised rank vector, so adding any
  account shifts every account's value slightly — this is inherent, not a bug.
  Measured magnitude for a disconnected two-account addition: max 1.0e-05 with
  rank correlation > 0.9999, which moved no decision. It is bounded and
  negligible, unlike the community defect above, which was neither.

• Scores are only comparable to the reported metrics when the observation window
  matches the trained one, because `in_amount_sum`, `out_amount_sum` and
  `txn_velocity` all scale with how long an account was watched. The window is
  enforced per request and `context.window_comparable` says whether it holds.

• An account with no history in the context graph is scored on the submitted
  transactions alone, so its graph features are weak by construction. Those
  accounts are marked `seen_in_context: false`; a low score there means "not
  enough evidence", not "safe".
"""

from __future__ import annotations

import json
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from fastapi import FastAPI, HTTPException

from api.responder import build_node_response, validate_response_batch
from api.schemas import (
    ContributingFactor,
    HealthResponse,
    ScoringContext,
    ScoringRequest,
    ScoringResponse,
    TransactionEdge,
)
from data.extractor import (
    build_graph,
    compute_louvain_communities,
    compute_node_features,
    partition_fingerprint,
)
from models.explain import shap_contributions, top_factors
from models.features import (
    FEATURE_COLS,
    MODEL_NAME,
    MODEL_VERSION,
    assert_feature_contract,
)

# ──────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models" / "saved_models"

# Derived from the feature contract. Never write a model filename here again:
# that is precisely how this file ended up serving a retired v1 model.
MODEL_PATH = MODEL_DIR / MODEL_NAME
METRICS_PATH = MODEL_DIR / "metrics.json"

# Historical transactions used as graph context. Scoring an account against only
# the handful of transactions in a request cannot see a ring, because a ring is a
# property of the surrounding graph.
CONTEXT_EDGES_PATH = ROOT / "data" / "raw" / "serving_context_edges.csv"

# Only these four columns are ever read from the context file. It also carries
# is_mule, edge_role, ring_id, ring_type and split — ground truth that serving
# code must never see. Restricting `usecols` makes that structural rather than a
# matter of remembering.
CONTEXT_COLS = ["sender", "receiver", "amount", "timestamp"]
LABEL_COLS_NEVER_READ = ["is_mule", "edge_role", "ring_id", "ring_type", "split"]

# Used only to measure the window length the model was trained on. There is no
# WINDOW_DAYS constant to import: data/generator.py derives each window as a
# third of the timeline and asserts the three come out equal, so the length is a
# property of the generated data and is read back from it here.
TRAIN_EDGES_PATH = ROOT / "data" / "raw" / "train_edges.csv"

# Matches WINDOW_LENGTH_TOLERANCE_DAYS in data/generator.py. Same quantity, same
# slack, so the API and the generator agree on what "equal windows" means.
WINDOW_TOLERANCE_DAYS = 1.5

# Max deviation tolerated between sigmoid(sum of SHAP contributions + bias) and
# the model's own predicted probability. Exact TreeSHAP satisfies this identity
# to floating-point precision; anything larger means the attribution does not
# explain the score, and a wrong explanation on a fraud alert is worse than none.
SHAP_IDENTITY_TOLERANCE = 1e-4

TOP_FACTORS_PER_ACCOUNT = 3


class ServiceState:
    """
    Everything loaded at startup, in one object.

    A module-level `global` per item is how `_model_version` came to say
    "sentinel_v1" while the model file said v3 — nothing tied them together. A
    single state object means the model, its version, its threshold and its
    provenance are set in one place or not at all.
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """
        Return to the not-loaded state.

        Startup calls this first so it is idempotent. Without it, a second
        startup in the same process — `uvicorn --reload`, or a test suite —
        inherits `contract_verified = True` from the previous run and a model
        that FAILS the contract check is served anyway, because nothing ever set
        the flag back to False. A stale success flag is the worst possible
        default for a gate whose whole job is to refuse.
        """
        self.model: xgb.XGBClassifier | None = None
        self.model_version: str = "unknown"
        self.model_file: str | None = None
        self.threshold: float | None = None
        self.threshold_source: str = "none"
        self.contract_verified: bool = False
        self.context_edges: pd.DataFrame | None = None
        self.reference_partition: dict[str, int] | None = None
        self.partition_fingerprint: str | None = None
        self.trained_window_days: float | None = None
        self.degraded_reason: str | None = None

    @property
    def ready(self) -> bool:
        """Scoring is allowed only when every prerequisite actually holds."""
        return (
            self.model is not None
            and self.contract_verified
            and self.threshold is not None
            and self.context_edges is not None
            and self.reference_partition is not None
        )


STATE = ServiceState()


# ──────────────────────────────────────────────────────────────────
# Loading helpers
# ──────────────────────────────────────────────────────────────────

def _to_naive_utc(values) -> pd.Series:
    """
    Normalise any timestamp input to tz-naive UTC.

    One conversion path for both sources. Context timestamps arrive naive from
    CSV; request timestamps arrive tz-aware because api/schemas.py normalises
    them. Comparing the two raises `TypeError`, and the comparison happens inside
    the window trim where that traceback explains nothing. Naive input is read as
    UTC; aware input is converted to UTC. The tz is then stripped so the column
    survives `to_numpy(dtype="datetime64[ns]")` in `build_graph`, which rejects
    tz-aware input on pandas 2.x.
    """
    ts = pd.to_datetime(pd.Series(list(values), dtype="object"), utc=True)
    return ts.dt.tz_convert("UTC").dt.tz_localize(None)


def load_metrics() -> tuple[float | None, str, dict]:
    """
    Read the decision threshold from metrics.json.

    Returns (threshold, source, full metrics dict). The threshold is None if it
    cannot be established, and the service then declines to score.

    There is deliberately no default. The old code did
    `metrics.get("optimal_threshold", 0.5)`, which silently substituted 0.5 for a
    missing key — and with false negatives at ₹200k against false positives at
    ₹15k the cost-optimal threshold is around 0.07. Falling back to 0.5 does not
    degrade gracefully; it throws away most of the recall the model was tuned
    for, while every response continues to look perfectly well-formed.
    """
    if not METRICS_PATH.exists():
        return None, "none", {}

    try:
        metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"  metrics.json could not be read: {exc}")
        return None, "none", {}

    raw = metrics.get("optimal_threshold")
    if raw is None:
        print("  metrics.json has no 'optimal_threshold' key.")
        return None, "none", metrics

    threshold = float(raw)
    if not (0.0 < threshold <= 1.0):
        print(f"  metrics.json 'optimal_threshold' is {threshold}, "
              f"which is not a usable operating point.")
        return None, "none", metrics

    # A metrics.json from a different model version is worse than none: its
    # threshold was chosen on a different score distribution.
    recorded = metrics.get("model_version")
    if recorded is not None and recorded != MODEL_VERSION:
        print(f"  metrics.json was written for {recorded}, but the feature "
              f"contract declares {MODEL_VERSION}. Refusing its threshold — a "
              f"threshold is only meaningful for the model it was tuned on.")
        return None, "none", metrics

    return threshold, "metrics.json", metrics


def load_context_edges() -> pd.DataFrame | None:
    """
    Load the historical graph context, labels excluded by construction.
    """
    if not CONTEXT_EDGES_PATH.exists():
        return None

    edges = pd.read_csv(CONTEXT_EDGES_PATH, usecols=CONTEXT_COLS)
    edges["timestamp"] = _to_naive_utc(edges["timestamp"])
    edges["amount"] = edges["amount"].astype(float)

    leaked = [c for c in LABEL_COLS_NEVER_READ if c in edges.columns]
    if leaked:
        raise RuntimeError(
            f"Ground-truth columns {leaked} reached the serving path. "
            f"CONTEXT_COLS is meant to make this impossible; if it is failing, "
            f"the scores this service returns cannot be trusted."
        )

    # A self-loop would inflate the account's degree and make reciprocity and
    # cycle_participation describe a cycle of one. Rejected on the request path
    # in api/schemas.py; dropped here for the same reason.
    self_loops = int((edges["sender"] == edges["receiver"]).sum())
    if self_loops:
        print(f"  dropped {self_loops} self-loop edges from the context file")
        edges = edges[edges["sender"] != edges["receiver"]].reset_index(drop=True)

    return edges


def measure_trained_window_days() -> float | None:
    """
    Span of the training window, in days, read back from the training edges.

    Needed because magnitude features scale with observation length, so a score
    computed over a different span is on a different scale — silently. Returns
    None if the raw training file isn't present (it isn't needed to serve, only
    to state whether the window matches).
    """
    if not TRAIN_EDGES_PATH.exists():
        return None
    try:
        ts = _to_naive_utc(pd.read_csv(TRAIN_EDGES_PATH, usecols=["timestamp"])
                           ["timestamp"])
    except (OSError, ValueError, KeyError) as exc:
        print(f"  could not measure the trained window: {exc}")
        return None
    if ts.empty:
        return None
    return float((ts.max() - ts.min()).total_seconds() / 86_400.0)


# ──────────────────────────────────────────────────────────────────
# Lifecycle
# ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load the model, verify its feature contract, and build the context graph.

    A failure here leaves the process running but degraded rather than crashing
    it. Both refuse to score, and the difference is diagnosability: /health can
    explain what is wrong, where a dead process gives the dashboard a bare
    connection error.
    """
    problems: list[str] = []

    # Idempotent: a stale success flag from an earlier startup in this process
    # would let a contract-failing model be served.
    STATE.reset()

    print("─" * 62)
    print("UPI Mule-Ring Sentinel — starting up")
    print("─" * 62)

    # ── model ──
    if MODEL_PATH.exists():
        model = xgb.XGBClassifier()
        model.load_model(str(MODEL_PATH))
        STATE.model = model
        STATE.model_version = MODEL_VERSION
        STATE.model_file = MODEL_NAME
        print(f"  model loaded: {MODEL_NAME}")

        # The check that this file previously skipped. A model whose columns
        # don't line up positionally with FEATURE_COLS produces wrong scores
        # without raising, so this has to gate serving, not just log.
        try:
            assert_feature_contract(model.get_booster().feature_names)
            STATE.contract_verified = True
            print(f"  feature contract verified: {len(FEATURE_COLS)} features")
        except RuntimeError as exc:
            problems.append(str(exc))
            print(f"  FEATURE CONTRACT FAILED\n{exc}")
    else:
        problems.append(
            f"Model not found at {MODEL_PATH}. Run `python -m models.train`.")
        print(f"  model MISSING at {MODEL_PATH}")

    # ── threshold ──
    threshold, source, _ = load_metrics()
    STATE.threshold, STATE.threshold_source = threshold, source
    if threshold is None:
        problems.append(
            "No usable decision threshold. metrics.json must carry an "
            "'optimal_threshold' in (0, 1] matching the current model version. "
            "Run `python -m models.train`.")
        print("  threshold UNAVAILABLE")
    else:
        print(f"  threshold: {threshold:.4f} (from {source})")

    # ── context graph ──
    try:
        STATE.context_edges = load_context_edges()
    except RuntimeError as exc:
        problems.append(str(exc))
        STATE.context_edges = None

    if STATE.context_edges is None:
        problems.append(
            f"Context edges not found at {CONTEXT_EDGES_PATH}. Run "
            f"`python -m data.generator`. Without graph context, PageRank, "
            f"clustering and cycle participation cannot be computed and the "
            f"model's strongest features would all be constants.")
        print("  context graph MISSING")
    else:
        ctx = STATE.context_edges
        span = (ctx["timestamp"].max() - ctx["timestamp"].min())
        n_nodes = len(set(ctx["sender"]) | set(ctx["receiver"]))
        print(f"  context graph: {len(ctx):,} transactions, {n_nodes:,} accounts, "
              f"{span.total_seconds() / 86_400:.1f} days "
              f"({ctx['timestamp'].min().date()} to {ctx['timestamp'].max().date()})")

        # ── reference community partition, computed ONCE ──
        # Louvain shuffles node visit order from its seed, so repartitioning the
        # merged graph inside every request made `community_internal_ratio` — and
        # with it the decision — depend on whoever else happened to be in the
        # batch. Measured on this graph: two accounts transacting only with each
        # other, connected to nothing, moved that feature for 100% of accounts
        # and flipped 2.84% of decisions at the cost-optimal threshold.
        #
        # Freezing it here is also the honest deployment shape: community
        # assignment is a batch job that runs on the account graph, not something
        # a scoring call gets to redraw.
        try:
            t0 = time.perf_counter()
            STATE.reference_partition = compute_louvain_communities(
                build_graph(ctx).to_undirected())
            STATE.partition_fingerprint = partition_fingerprint(
                STATE.reference_partition)
            print(f"  reference partition: "
                  f"{len(set(STATE.reference_partition.values())):,} communities "
                  f"over {len(STATE.reference_partition):,} accounts "
                  f"({(time.perf_counter() - t0) * 1000:.0f} ms, computed once)")
            # Published so a restart or a second replica that partitions
            # differently is visible instead of silently rescoring accounts.
            print(f"  partition fingerprint: {STATE.partition_fingerprint}")
        except Exception as exc:  # noqa: BLE001 — degrade, never crash startup
            STATE.reference_partition = None
            STATE.partition_fingerprint = None
            problems.append(
                f"Could not partition the context graph ({type(exc).__name__}: "
                f"{exc}). community_internal_ratio would then be recomputed per "
                f"request, which makes an account's score depend on unrelated "
                f"accounts in the same batch.")
            print(f"  reference partition FAILED: {exc}")

    STATE.trained_window_days = measure_trained_window_days()
    if STATE.trained_window_days is not None:
        print(f"  trained observation window: "
              f"{STATE.trained_window_days:.1f} days")

    STATE.degraded_reason = "\n".join(problems) if problems else None
    print("─" * 62)
    print("  READY" if STATE.ready else "  DEGRADED — /score will return 503")
    print("─" * 62)

    yield

    STATE.model = None
    STATE.context_edges = None
    STATE.reference_partition = None
    STATE.partition_fingerprint = None
    print("Model unloaded.")


# ──────────────────────────────────────────────────────────────────
# App
# ──────────────────────────────────────────────────────────────────

app = FastAPI(
    title="UPI Mule-Ring Sentinel",
    description=(
        "Graph-based mule account detection API. Scores UPI transaction "
        "patterns and returns defense-only risk assessments: the strongest "
        "recommendation it can make is human review."
    ),
    version="3.0.0",
    lifespan=lifespan,
)


# ──────────────────────────────────────────────────────────────────
# Feature computation
# ──────────────────────────────────────────────────────────────────

def submitted_to_frame(transactions: list[TransactionEdge]) -> pd.DataFrame:
    """Request transactions as an edge frame shaped like the context file."""
    return pd.DataFrame({
        "sender": [t.sender for t in transactions],
        "receiver": [t.receiver for t in transactions],
        "amount": [float(t.amount) for t in transactions],
        "timestamp": _to_naive_utc([t.timestamp for t in transactions]),
    })


def merge_with_context(
    submitted: pd.DataFrame,
    context: pd.DataFrame,
    window_days: float | None,
) -> tuple[pd.DataFrame, dict]:
    """
    Merge submitted transactions into the context graph, holding the observation
    window fixed.

    The window matters because `in_amount_sum`, `out_amount_sum` and
    `txn_velocity` are magnitudes: watch an account twice as long and they
    roughly double, with no change in behaviour. The model learned them over a
    window of one fixed length, so scoring against a graph spanning a different
    length compares two different scales and reports the difference as risk.

    Rules, and the reasoning behind each:

      • The window ends at the latest transaction in the merged set, so "now" is
        the most recent thing actually observed rather than wall-clock time,
        which lets a historical batch be scored reproducibly.

      • CONTEXT is trimmed to that window. It is the part that can be dropped
        without discarding the question being asked.

      • SUBMITTED transactions are NEVER trimmed. The caller asked about them; a
        window rule that silently drops the request is worse than a wide window,
        which at least gets reported.

    Returns the merged frame and a diagnostics dict.
    """
    diagnostics: dict = {"warnings": []}

    window_end = max(submitted["timestamp"].max(), context["timestamp"].max())

    if window_days is None:
        window_days = float(
            (context["timestamp"].max() - context["timestamp"].min())
            .total_seconds() / 86_400.0)
        diagnostics["warnings"].append(
            f"Trained window length unknown (data/raw/train_edges.csv is "
            f"absent), so the context file's own span of {window_days:.1f} days "
            f"was used. Comparability with the reported metrics is assumed, not "
            f"verified.")

    window_start = window_end - pd.Timedelta(days=float(window_days))

    kept = context[(context["timestamp"] >= window_start)
                   & (context["timestamp"] <= window_end)]
    merged = (pd.concat([kept, submitted], ignore_index=True)
              .sort_values("timestamp", kind="mergesort")
              .reset_index(drop=True))

    observed_days = float(
        (merged["timestamp"].max() - merged["timestamp"].min())
        .total_seconds() / 86_400.0)

    diagnostics.update({
        "n_context_used": int(len(kept)),
        "n_context_available": int(len(context)),
        "observed_days": observed_days,
        "window_days": float(window_days),
        "window_start": window_start,
        "window_end": window_end,
        # Accounts with history INSIDE the window, judged on the trimmed context
        # rather than the whole file: an account whose only transactions fall
        # outside the window contributed nothing to this graph, so calling it
        # `seen_in_context` would overstate the evidence behind its score.
        "context_accounts": set(kept["sender"]) | set(kept["receiver"]),
    })

    # The failure that matters: the batch is dated outside the context range, so
    # the trim removed everything and we are back to scoring a bare batch graph —
    # the exact defect this function exists to fix. It must be loud.
    if kept.empty:
        diagnostics["warnings"].append(
            f"NO historical context survived the observation window. The "
            f"submitted transactions are dated "
            f"{submitted['timestamp'].min().date()} to "
            f"{submitted['timestamp'].max().date()}, while the context file "
            f"covers {context['timestamp'].min().date()} to "
            f"{context['timestamp'].max().date()}. Graph features are therefore "
            f"computed on the submitted transactions alone: PageRank collapses "
            f"to ~1/n, clustering and cycle participation to 0. These scores are "
            f"not comparable to the reported metrics. Date the transactions "
            f"inside the context window.")
    elif len(kept) < 0.5 * len(context):
        diagnostics["warnings"].append(
            f"Only {len(kept):,} of {len(context):,} context transactions fall "
            f"inside the window, so the graph is sparser than the one the model "
            f"was trained on and graph features are correspondingly weaker.")

    if window_days > 0:
        drift = abs(observed_days - window_days)
        comparable = drift <= WINDOW_TOLERANCE_DAYS
    else:
        comparable = False
    diagnostics["comparable"] = bool(comparable)

    if not comparable:
        diagnostics["warnings"].append(
            f"Effective observation window is {observed_days:.1f} days against a "
            f"trained window of {window_days:.1f} days. in_amount_sum, "
            f"out_amount_sum and txn_velocity scale with window length, so they "
            f"are on a different scale than in training and these scores should "
            f"be treated as indicative only.")

    return merged, diagnostics


def attribute(
    model: xgb.XGBClassifier,
    X: pd.DataFrame,
    probabilities: np.ndarray,
) -> tuple[list[list[ContributingFactor]], list[str]]:
    """
    Per-account SHAP attributions, with the summation identity checked first.

    Exact TreeSHAP guarantees `sigmoid(contributions.sum() + bias)` equals the
    model's predicted probability. Verifying it here is not ceremony: the two
    ways this silently breaks in XGBoost are an `iteration_range` mismatch after
    early stopping (contributions from all trees, prediction from the best
    subset) and mistaking the trailing bias column for a feature, which shifts
    every attribution by one and produces confident, wrong explanations. If the
    identity fails, factors are dropped and the caller is told — a fraud alert
    with no reason attached is recoverable; one with a fabricated reason sends an
    analyst down the wrong path.
    """
    warnings: list[str] = []
    try:
        contribs, bias = shap_contributions(model, X)
    except Exception as exc:
        return [[] for _ in range(len(X))], [
            f"Per-account attribution unavailable ({type(exc).__name__}: {exc}). "
            f"Scores are unaffected; explanations are omitted."]

    reconstructed = 1.0 / (1.0 + np.exp(-np.clip(contribs.sum(axis=1) + bias,
                                                 -60, 60)))
    deviation = float(np.max(np.abs(reconstructed - probabilities))) \
        if len(X) else 0.0
    if deviation > SHAP_IDENTITY_TOLERANCE:
        return [[] for _ in range(len(X))], [
            f"SHAP contributions do not reconstruct the predicted "
            f"probabilities (max deviation {deviation:.2e} > "
            f"{SHAP_IDENTITY_TOLERANCE:.0e}), so they do not explain these "
            f"scores. Explanations withheld rather than served misleadingly."]

    values = X.to_numpy(dtype=float)
    factors = [
        [ContributingFactor(**f) for f in
         top_factors(contribs[i], values[i], k=TOP_FACTORS_PER_ACCOUNT)]
        for i in range(len(X))
    ]
    return factors, warnings


# ──────────────────────────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Check whether the service can actually score, and if not, why."""
    ctx = STATE.context_edges
    return HealthResponse(
        status="healthy" if STATE.ready else "degraded",
        model_loaded=STATE.model is not None,
        model_version=STATE.model_version,
        model_file=STATE.model_file,
        optimal_threshold=STATE.threshold if STATE.threshold is not None else 0.0,
        threshold_source=STATE.threshold_source,
        n_features=len(FEATURE_COLS),
        feature_contract_verified=STATE.contract_verified,
        context_graph_loaded=ctx is not None,
        context_transactions=int(len(ctx)) if ctx is not None else 0,
        partition_fingerprint=STATE.partition_fingerprint,
        detail=STATE.degraded_reason,
    )


@app.post("/score", response_model=ScoringResponse, tags=["Scoring"])
async def score_transactions(request: ScoringRequest):
    """
    Score a batch of UPI transactions for mule-ring risk.

    Returns a risk score, risk level, recommended action and per-account SHAP
    attributions for each unique account in the batch. Features are computed over
    the submitted transactions merged into the historical context graph, using
    the same extractor that built the training set.
    """
    if not STATE.ready:
        raise HTTPException(
            status_code=503,
            detail=STATE.degraded_reason or "Service is not ready to score.",
        )

    model = STATE.model
    threshold = STATE.threshold
    threshold_source = "metrics.json"
    if request.threshold_override is not None:
        # `or` was wrong here: it treats 0.0 as absent. The schema now rejects 0
        # outright, and an explicit None test means any accepted value is used.
        threshold = float(request.threshold_override)
        threshold_source = "request_override"

    submitted = submitted_to_frame(request.transactions)
    merged, diag = merge_with_context(
        submitted, STATE.context_edges, STATE.trained_window_days)

    # ── features, from the same code path that produced the training set ──
    # The frozen reference partition goes in, so community membership is not
    # redrawn per request. Accounts the reference graph has never seen are
    # assigned deterministically inside `extend_partition`.
    started = time.perf_counter()
    graph = build_graph(merged)
    features = compute_node_features(graph, partition=STATE.reference_partition)
    compute_ms = (time.perf_counter() - started) * 1000.0

    missing = [c for c in FEATURE_COLS if c not in features.columns]
    if missing:
        raise HTTPException(
            status_code=500,
            detail=(f"Extractor did not produce {missing}. data/extractor.py and "
                    f"models/features.py have diverged; retrain and redeploy."),
        )

    # ── which accounts to return ──
    batch_accounts = set(submitted["sender"]) | set(submitted["receiver"])
    context_accounts = diag["context_accounts"]
    if request.include_context_accounts:
        wanted = features["node"].isin(batch_accounts | context_accounts)
    else:
        wanted = features["node"].isin(batch_accounts)

    scored = features.loc[wanted].reset_index(drop=True)
    if scored.empty:
        raise HTTPException(
            status_code=500,
            detail="No accounts survived node selection, which should be "
                   "impossible for a non-empty batch.",
        )

    # Selecting by FEATURE_COLS guarantees order matches the contract. XGBoost is
    # positional, so a reordered frame scores every account wrongly in silence.
    X = scored[FEATURE_COLS].astype(float)
    probabilities = model.predict_proba(X)[:, 1]

    factors, attribution_warnings = attribute(model, X, probabilities)

    node_scores = []
    for i, node in enumerate(scored["node"].tolist()):
        node_scores.append(build_node_response(
            node_id=str(node),
            # positional, not .iterrows() label lookup: the old code indexed
            # `probabilities[i]` with an index LABEL, which happened to coincide
            # with position only because the frame was freshly built.
            risk_score=float(probabilities[i]),
            threshold=threshold,
            contributing_factors=factors[i],
            seen_in_context=node in context_accounts,
        ))

    # Final safety validation — re-derives tier and action from the score, so a
    # response cannot reach the caller with a downgraded action.
    node_scores = validate_response_batch(node_scores, threshold)

    risk_counts: dict[str, int] = {}
    for ns in node_scores:
        risk_counts[ns.risk_level.value] = \
            risk_counts.get(ns.risk_level.value, 0) + 1

    n_flagged = sum(1 for ns in node_scores if ns.risk_score >= threshold)
    n_new = sum(1 for ns in node_scores if not ns.seen_in_context)

    return ScoringResponse(
        request_id=str(uuid.uuid4()),
        scored_at=datetime.now(timezone.utc),
        threshold_used=threshold,
        node_scores=node_scores,
        context=ScoringContext(
            model_version=STATE.model_version,
            partition_fingerprint=STATE.partition_fingerprint,
            threshold_source=threshold_source,
            n_submitted_transactions=len(request.transactions),
            n_context_transactions_used=diag["n_context_used"],
            n_nodes_in_graph=int(graph.number_of_nodes()),
            observation_window_days=round(diag["observed_days"], 2),
            trained_window_days=(round(STATE.trained_window_days, 2)
                                 if STATE.trained_window_days is not None
                                 else None),
            window_comparable=diag["comparable"],
            warnings=diag["warnings"] + attribution_warnings,
        ),
        summary={
            "total_nodes_scored": len(node_scores),
            "risk_distribution": risk_counts,
            "high_risk_count": (risk_counts.get("HIGH", 0)
                                + risk_counts.get("CRITICAL", 0)),
            "flagged_at_threshold": n_flagged,
            "accounts_with_no_history": n_new,
            "feature_computation_ms": round(compute_ms, 1),
        },
    )
