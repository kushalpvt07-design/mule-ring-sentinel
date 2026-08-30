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
   "sentinel_v1", while `models/features.py` declared a later version and
   training wrote that file. So the API loaded a retired 12-feature model and
   applied the CURRENT model's cost-optimal threshold to its scores — two
   unrelated calibrations bolted together. The path now derives from
   `features.MODEL_NAME` and can't drift. No version literal appears in this
   file for the same reason.

2. IT DECLARED ITS OWN FEATURE LIST.
   A local 12-name `FEATURE_COLS` sat here, including `net_flow` (dropped in v3)
   and `louvain_community` (an arbitrary integer id, dropped in v2 as a feature).
   `assert_feature_contract` existed in models/features.py to catch exactly this,
   and was never called. It is now called at startup, and a mismatch puts the
   service into a degraded state that refuses to score.

3. IT COMPUTED FEATURES ON A GRAPH OF THE REQUEST ALONE.
   `extract_features_from_batch` built a DiGraph from just the submitted
   transactions. On a 4-edge graph `pagerank` is uniform by construction (the
   feature is now emitted as pagerank × N, so that reads as a flat 1.0 rather
   than a flat 1/n — equally constant either way),
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

• Scores are comparable to the reported metrics on magnitude, but not on
  structure. The three magnitude features (`in_amount_sum`, `out_amount_sum`,
  `repeat_ratio`) are rescaled to a 60-day reference window, so window length no
  longer distorts them. `txn_velocity` needs no rescaling because it is already a
  rate — it divides by the account's own active span — and treating it as a
  window-scaling magnitude would be a warning about a bug that no longer exists.
  What a mismatched window still changes is the graph's STRUCTURE, which no
  per-graph constant can correct: over a different window the counterparty set,
  the reciprocal pairs and the repeated-edge cycle core are built from a different
  amount of evidence than training used, which moves `degree_balance`,
  `reciprocity`, `clustering_coefficient` and `cycle_participation`. Distinct
  counterparties saturate rather than scale, so that residual does not divide out.
  `context.window_comparable` says whether the effective window matched the trained
  one, and it is the flag to check before reading a score against the published
  metrics.

  A batch dated outside the window is SCORED, not refused. The window is anchored
  to the context file's own last timestamp, so nothing the caller sends can move it
  and no other account's history is dropped to accommodate a late request; the
  request's own edges are never trimmed either. What the caller gets instead of a
  refusal is `window_comparable: false` and a warning saying how far outside the
  batch fell. An earlier version refused these with 422, which sounded stricter and
  was worse: the refusal only fired once a batch was late enough to slide the
  window clear of the context entirely, and every batch short of that was quietly
  scored against a graph the request itself had truncated.

  The 422 still exists for the case that is a deployment fault rather than a caller
  fault: a context file that loaded but holds no usable rows, leaving nothing for
  the window to keep. Graph features are constants at that point and a tier derived
  from them is a threshold calibrated on a 60-day graph applied to a handful of
  edges.

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

from console import enable_utf8_stdout, hr
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
    undirected_projection,
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

# What the context file IS, said in the response rather than only in the README.
#
# data/generator.py writes serving_context_edges.csv from the validation split, so
# on the shipped artefacts it is edge-for-edge identical to val_edges.csv. That has
# a consequence worth stating to anyone reading a single response body: an account
# scored against this context is being scored in-sample with respect to the GRAPH.
# Its neighbours, its ring, its community — all of it was visible when the
# threshold was selected. The model file itself never saw validation labels, and
# the split integrity checks in models/train.py are about labels, so nothing here
# invalidates them; but a precision figure computed from this endpoint's output
# would be optimistic and is not a number this project publishes.
#
# The honest out-of-sample figures are the test-split ones in README.md, which come
# from models/report.py reading metrics.json. Regenerating the context from a
# fourth, held-out window would remove the caveat, and is the right fix if this
# ever serves anything real; documenting it is the honest thing to do meanwhile.
CONTEXT_PROVENANCE = (
    "serving_context_edges.csv is the validation split, edge for edge. Accounts "
    "are therefore scored in-sample with respect to the graph: their neighbourhood "
    "was visible when the operating threshold was selected. No labels leak — the "
    "model never trained on validation — but precision measured from this "
    "endpoint's output would be optimistic. The out-of-sample figures are the "
    "test-split ones in README.md."
)

# Used only to measure the window length the model was trained on. There is no
# WINDOW_DAYS constant to import: data/generator.py derives each window as a
# third of the timeline and asserts the three come out equal, so the length is a
# property of the generated data and is read back from it here.
TRAIN_EDGES_PATH = ROOT / "data" / "raw" / "train_edges.csv"

# Matches WINDOW_LENGTH_TOLERANCE_DAYS in data/generator.py. Same quantity, same
# slack, so the API and the generator agree on what "equal windows" means.
WINDOW_TOLERANCE_DAYS = 1.5

# Max deviation tolerated between the SHAP reconstruction of the margin
# (contributions.sum(1) + bias) and the model's own `predict(output_margin=True)`.
# LOG-ODDS, matching MARGIN_TOLERANCE in models/train.py — same quantity, same
# constant, so training and serving hold the attributions to one standard.
#
# It replaced SHAP_IDENTITY_TOLERANCE, which was the same 1e-4 applied in
# PROBABILITY space. Sigmoid saturates, so that number meant something different
# at every score: near p = 0.9999 it tolerated over 1.5 log-odds of attribution
# error, and past a clip at ±60 it tolerated anything at all. XGBoost accumulates
# tree outputs in float32 and the two paths sum in different orders, so exact
# equality is not available; 1e-4 log-odds is ~three orders above that noise floor.
MARGIN_IDENTITY_TOLERANCE = 1e-4

# Max spread tolerated in the TreeSHAP bias column across a batch. For a binary
# booster with a scalar base_score and no per-row base_margin it is one value
# repeated, so this is 0 up to float32 accumulation. It is the ONLY check that can
# catch the bias column being read from the wrong end of `pred_contribs` — the
# summation identity cannot, because the row sum is invariant to the split. See
# `attribute`.
BIAS_CONSTANT_TOLERANCE = 1e-6

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
        self.break_even_probability: float | None = None
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
            and self.break_even_probability is not None
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


def read_break_even_probability(metrics: dict) -> float | None:
    """
    Read the cost model's break-even probability p* out of metrics.json.

    Loaded alongside the threshold because api/responder.py needs both: the
    threshold is where an alert starts, p* is where an account stops being cheap
    enough to ignore, and the step-up band is the space between them. Hard-coding
    0.0698 here would put a second copy of a cost assumption into the serving
    path, which is the same mistake as the local feature list that made this file
    serve a v1 model against v3's threshold.

    If the explicit key is absent it is recomputed as `fp / (fp + fn)` from the
    same block. That is p*'s definition rather than a default, so it cannot
    disagree with the costs metrics.json publishes. Returns None if neither is
    available, and the service then declines to score instead of inventing a tier
    boundary.
    """
    cost = metrics.get("cost_config")
    if not isinstance(cost, dict):
        return None

    raw = cost.get("break_even_probability")
    if raw is None:
        fn_cost, fp_cost = cost.get("fn_cost"), cost.get("fp_cost")
        if fn_cost is None or fp_cost is None:
            print("  metrics.json cost_config carries neither "
                  "'break_even_probability' nor the costs it follows from.")
            return None
        total = float(fn_cost) + float(fp_cost)
        if total <= 0.0:
            return None
        raw = float(fp_cost) / total

    break_even = float(raw)
    if not (0.0 < break_even <= 1.0):
        print(f"  metrics.json gives a break-even probability of {break_even}, "
              f"which is not a probability.")
        return None
    return break_even


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

    enable_utf8_stdout()
    print(hr())
    print("UPI Mule-Ring Sentinel -- starting up")
    print(hr())

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
    threshold, source, metrics = load_metrics()
    STATE.threshold, STATE.threshold_source = threshold, source
    if threshold is None:
        problems.append(
            "No usable decision threshold. metrics.json must carry an "
            "'optimal_threshold' in (0, 1] matching the current model version. "
            "Run `python -m models.train`.")
        print("  threshold UNAVAILABLE")
    else:
        print(f"  threshold: {threshold:.4f} (from {source})")

    # ── break-even probability: the floor of the step-up band ──
    # Required, not optional. Without it api/responder.py would have no economic
    # basis for the ALLOW/step-up line, and the fraction-of-threshold rule it
    # replaced returned ALLOW across a band the cost model prices as reviewable.
    STATE.break_even_probability = read_break_even_probability(metrics)
    if STATE.break_even_probability is None:
        problems.append(
            "No break-even probability. metrics.json must carry a 'cost_config' "
            "block with 'break_even_probability', or the 'fn_cost' and 'fp_cost' "
            "it follows from. It sets the floor of the step-up band, and a "
            "guessed floor means answering ALLOW on accounts the cost model says "
            "are worth a challenge. Run `python -m models.train`.")
        print("  break-even probability UNAVAILABLE")
    else:
        print(f"  break-even p*: {STATE.break_even_probability:.6f} "
              f"(step-up floor; the alert cutoff stays the threshold above)")

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
        #
        # `undirected_projection`, NOT `.to_undirected()`. This line used to read
        # `compute_louvain_communities(build_graph(ctx).to_undirected())` while
        # `compute_node_features` projected with `undirected_projection` — the
        # train/serve skew data/extractor.py names at its own docstring. networkx
        # resolves a reciprocal pair by letting one direction's attribute dict
        # overwrite the other's, and `best_partition` optimises WEIGHTED
        # modularity, so the frozen partition was an optimum of a graph nothing
        # else in the system used. Measured on this context file: 1,092 of 18,426
        # undirected pairs are reciprocal, carrying 11.1% of all transaction
        # weight, so last-wins discarded ~5.7% of the graph's weight. Worse, the
        # partition then went to `extend_partition`, whose heaviest-neighbour
        # tie-break runs on the summed projection — two weight definitions in one
        # path.
        try:
            t0 = time.perf_counter()
            STATE.reference_partition = compute_louvain_communities(
                undirected_projection(build_graph(ctx)))
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
    print(hr())
    print("  READY" if STATE.ready else "  DEGRADED -- /score will return 503")
    print(hr())

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

      • The window ends at the latest CONTEXT transaction, not the latest
        transaction overall. See "WHY THE WINDOW IS ANCHORED TO THE CONTEXT"
        below: anchoring it to the merged maximum let the submitted batch's dates
        decide how much of unrelated accounts' history survived.

      • CONTEXT is trimmed to that window. It is the part that can be dropped
        without discarding the question being asked.

      • SUBMITTED transactions are NEVER trimmed. The caller asked about them; a
        window rule that silently drops the request is worse than a wide window,
        which at least gets reported.

      • Exact duplicates between the two are dropped, keeping the context copy.
        See "WHY THE MERGE DE-DUPLICATES" below.

    ─────────────────────────────────────────────────────────────────────────
    WHY THE WINDOW IS ANCHORED TO THE CONTEXT
    ─────────────────────────────────────────────────────────────────────────
    This was `window_end = max(submitted.max(), context.max())`, with only the
    context trimmed. A batch dated after the context end therefore slid the
    window forward and deleted the OLDEST context edges — history belonging to
    accounts with nothing to do with the request. Measured against the shipped
    context (60,379 edges, 2025-03-02 → 2025-04-30) with a four-transaction batch
    dated n days past the context end: 1 day dropped 1,004 edges, 10 days dropped
    9,694, and 20 days dropped 19,829 — a third of the graph — every one of them
    reported as `window_comparable: true` with no warning.

    The comparability check could not see it by construction. `observed_days` was
    measured on the MERGED frame, and sliding the window forward keeps that
    pinned near the trained length however much context is deleted, so drift
    stayed inside tolerance while the graph emptied. The 50%-of-context floor did
    not trip until roughly 30 days out.

    The window is now anchored to `context["timestamp"].max()`, which is the
    graph the reference partition was frozen on and the graph the reported
    metrics describe. Nothing the caller sends can move it. A batch dated outside
    that window still scores — its own edges are never trimmed — but it is
    scored against the full historical graph rather than a truncated one, and
    `submitted_outside_window` records how far outside it fell.

    The trim is also reported whenever it removes ANYTHING, not only past a 50%
    floor. `window_days` is read from train_edges.csv while the context file is
    its own slightly shorter span, so on the shipped data the trim removes
    nothing at all; if that ever stops being true it should be visible on the
    first request rather than the ten-thousandth.

    ─────────────────────────────────────────────────────────────────────────
    WHY THE MERGE DE-DUPLICATES
    ─────────────────────────────────────────────────────────────────────────
    This was a bare `pd.concat`. `build_graph` then aggregates
    `weight=("amount", "size")` and `total_amount=("amount", "sum")`, so a
    transaction present in both the request and the context file counted twice —
    inflating `in_amount_sum`, `out_amount_sum`, `repeat_ratio`, `txn_velocity`
    and `burst_ratio` for both endpoints of that edge.

    That is not an exotic input. The context file ships in the repo and copying rows
    out of it is the obvious way to build a request that scores against real
    history, so the natural first attempt at using this API was silently mis-scored.

    Exact matches on (sender, receiver, amount, timestamp) are dropped, keeping
    the context copy, and the count is reported. Two genuinely distinct
    transactions that agree on all four fields to the microsecond are
    indistinguishable from a replay in this data and are the far rarer case; the
    shipped context file contains no such pair.

    Returns the merged frame and a diagnostics dict.
    """
    diagnostics: dict = {"warnings": []}

    window_end = context["timestamp"].max()

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

    # Drop request rows that replay a context transaction exactly. `keep="first"`
    # on a frame with the context first means the context copy survives, so the
    # de-duplication cannot silently discard the caller's row in favour of a
    # different one — they are identical in all four fields by construction.
    key = ["sender", "receiver", "amount", "timestamp"]
    stacked = pd.concat([kept, submitted], ignore_index=True)
    duplicated = stacked.duplicated(subset=key, keep="first")
    n_duplicates = int(duplicated.sum())
    merged = (stacked.loc[~duplicated]
              .sort_values("timestamp", kind="mergesort")
              .reset_index(drop=True))

    observed_days = float(
        (merged["timestamp"].max() - merged["timestamp"].min())
        .total_seconds() / 86_400.0)

    # How far the request falls outside the reference window, in days, signed:
    # positive is later than the context ends, negative is earlier than it starts.
    # Reported rather than acted on — submitted edges are never trimmed — because
    # it is the cause of which `observed_days` drift is only the symptom.
    outside = submitted[(submitted["timestamp"] < window_start)
                        | (submitted["timestamp"] > window_end)]
    late_days = float(
        (submitted["timestamp"].max() - window_end).total_seconds() / 86_400.0)
    early_days = float(
        (window_start - submitted["timestamp"].min()).total_seconds() / 86_400.0)

    diagnostics.update({
        "n_context_used": int(len(kept)),
        "n_context_available": int(len(context)),
        "n_duplicates_dropped": n_duplicates,
        "n_submitted_outside_window": int(len(outside)),
        "submitted_days_after_window_end": round(max(late_days, 0.0), 3),
        "submitted_days_before_window_start": round(max(early_days, 0.0), 3),
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

    if n_duplicates:
        diagnostics["warnings"].append(
            f"{n_duplicates:,} of {len(submitted):,} submitted transactions "
            f"exactly replay a transaction already in the context file (same "
            f"sender, receiver, amount and timestamp) and were counted once, not "
            f"twice. Without this, each replayed edge would have doubled its own "
            f"weight and amount, inflating in_amount_sum, out_amount_sum, "
            f"repeat_ratio, txn_velocity and burst_ratio for both endpoints. The "
            f"scores are correct; the request is redundant.")

    # The failure that matters: nothing at all survived the trim, so we are back to
    # scoring a bare batch graph — the exact defect this function exists to fix. It
    # must be loud.
    #
    # Now that the window is anchored to `context["timestamp"].max()`, that instant
    # is always inside the window, so a non-empty context with parseable timestamps
    # cannot reach this. Reaching it means the context frame is empty or its
    # timestamps are NaT. The message therefore names THAT, and deliberately no
    # longer ends with "date the transactions inside the context window": under
    # anchoring, re-dating the request changes nothing here, and an instruction that
    # cannot fix the problem sends the caller to debug their own input while a
    # broken deployment serves constants.
    if kept.empty:
        diagnostics["warnings"].append(
            f"NO historical context survived the observation window, and the "
            f"submitted dates are not the reason — the window is anchored to the "
            f"context file, so the request cannot move it. The context frame holds "
            f"{len(context):,} row(s) and its timestamps read as "
            f"{context['timestamp'].min()} to {context['timestamp'].max()}, so "
            f"either data/raw/serving_context_edges.csv is empty or its timestamp "
            f"column did not parse. Graph features are therefore computed on the "
            f"submitted transactions alone: PageRank collapses to ~1/n, clustering "
            f"and cycle participation to 0. These scores are not comparable to the "
            f"reported metrics. Check the context file, not the request.")
    elif len(kept) < len(context):
        # ANY trim is reported, not only a trim past some floor. The old 50% floor
        # meant a third of the graph could vanish silently; and now that the window
        # is anchored to the context rather than to the request, a trim is a
        # property of the deployment (a context file spanning longer than the
        # trained window) rather than of one caller, so it is worth saying once
        # per request until someone shortens the file.
        dropped = len(context) - len(kept)
        severity = ("Only " if len(kept) < 0.5 * len(context) else "")
        diagnostics["warnings"].append(
            f"{severity}{len(kept):,} of {len(context):,} context transactions "
            f"fall inside the {window_days:.1f}-day observation window "
            f"({dropped:,} older edges trimmed, {100 * dropped / len(context):.1f}"
            f"%), so the graph is sparser than the one the model was trained on "
            f"and graph features are correspondingly weaker. The context file "
            f"spans longer than the trained window; trimming it to match would "
            f"remove this warning without changing any score.")

    if len(outside):
        diagnostics["warnings"].append(
            f"{len(outside):,} of {len(submitted):,} submitted transactions fall "
            f"outside the reference window "
            f"({window_start.date()} to {window_end.date()}): up to "
            f"{max(late_days, 0.0):.1f} days after it ends and "
            f"{max(early_days, 0.0):.1f} days before it starts. They are scored "
            f"anyway — the request is never trimmed — but they widen the effective "
            f"window without adding history, so the magnitude features below are "
            f"measured over a longer span than training used. The window itself is "
            f"anchored to the context file and does not move, so no other "
            f"account's history was dropped to accommodate them.")

    if window_days > 0:
        drift = abs(observed_days - window_days)
        comparable = drift <= WINDOW_TOLERANCE_DAYS
    else:
        comparable = False
    diagnostics["comparable"] = bool(comparable)

    if not comparable:
        # WHAT THIS WARNING IS ABOUT IN v4, WHICH IS NOT WHAT IT WAS ABOUT IN v3.
        #
        # v3's version named the three features MEASURED to scale with window
        # length — in_amount_sum 2.00x, out_amount_sum 1.99x, repeat_ratio 1.88x
        # when the window doubles — because nothing corrected them and a
        # mismatched window put them on a different scale than training. (An
        # earlier draft named `txn_velocity` instead of `repeat_ratio`, which was
        # backwards: txn_velocity divides by the account's own active span so it is
        # already a rate.)
        #
        # `data/extractor.py` now rescales all three to a 60-day reference window,
        # so magnitude is no longer the problem and claiming it is would be a
        # warning about a fixed bug. What a mismatched window still changes is the
        # graph's STRUCTURE, which no per-graph constant can correct: distinct
        # counterparties saturate rather than scale (1.00 median / 1.10 by totals
        # over a doubled window), so `degree_balance`, `reciprocity`,
        # `clustering_coefficient` and the repeated-edge cycle core are all
        # measured over a different amount of evidence than training used. That is
        # the residual, it is real, and it is smaller and differently shaped than
        # what this text used to describe.
        diagnostics["warnings"].append(
            f"Effective observation window is {observed_days:.1f} days against a "
            f"trained window of {window_days:.1f} days. The three magnitude "
            f"features (in_amount_sum, out_amount_sum, repeat_ratio) are rescaled "
            f"to a 60-day reference window and so remain comparable, but graph "
            f"STRUCTURE is not rescalable: over a different window the "
            f"counterparty set, the reciprocal pairs and the repeated-edge cycle "
            f"core are all built from a different amount of evidence than training "
            f"used, which moves degree_balance, reciprocity, "
            f"clustering_coefficient and cycle_participation. Treat these scores "
            f"as indicative only.")

    return merged, diagnostics


def attribute(
    model: xgb.XGBClassifier,
    X: pd.DataFrame,
    probabilities: np.ndarray,
) -> tuple[list[list[ContributingFactor]], list[str]]:
    """
    Per-account SHAP attributions, with the summation identity checked first.

    Exact TreeSHAP guarantees `contributions.sum() + bias` equals the model's raw
    margin. Verifying it here is not ceremony: the two ways this silently breaks in
    XGBoost are an `iteration_range` mismatch after early stopping (contributions
    from all trees, prediction from the best subset) and mistaking the trailing
    bias column for a feature, which shifts every attribution by one and produces
    confident, wrong explanations. If a check fails, factors are dropped and the
    caller is told — a fraud alert with no reason attached is recoverable; one with
    a fabricated reason sends an analyst down the wrong path.

    ─────────────────────────────────────────────────────────────────────────
    WHY THE IDENTITY IS CHECKED ON THE MARGIN AND NOT ON THE PROBABILITY
    ─────────────────────────────────────────────────────────────────────────
    This used to compare `sigmoid(clip(contribs.sum(1) + bias, -60, 60))` against
    `probabilities` at 1e-4 in PROBABILITY space, while `models/train.py` had
    already been corrected to compare margins at 1e-4 in LOG-ODDS. Same constant,
    two different meanings, and the serving copy was the loose one.

    Sigmoid saturates, so a fixed probability tolerance is a wildly varying margin
    tolerance. At a true margin of 9 (p = 0.99988) a reconstruction error of about
    +1.66 log-odds slips through; past ±40 the clip makes the check vacuous
    outright. For scale, the whole HIGH→CRITICAL band is well under half a log-odd
    wide at the shipped threshold — so the undetectable error was several times a
    tier band, on precisely the CRITICAL accounts an analyst opens first. Comparing
    margins directly makes the tolerance mean one thing everywhere.

    The probability agreement is still reported when it fails, because that is the
    number a reader intuits and the number the API serves, but it is a CONSEQUENCE
    of the margin check rather than the gate: |dp| <= |dm| / 4 for any margin.

    ─────────────────────────────────────────────────────────────────────────
    WHY THE BIAS COLUMN IS CHECKED SEPARATELY
    ─────────────────────────────────────────────────────────────────────────
    The docstring here used to claim the summation identity catches "mistaking the
    trailing bias column for a feature". It cannot, and this is worth being precise
    about because the claim was load-bearing. `pred_contribs` returns
    `n_features + 1` columns whose row sum is the margin; `contribs.sum(1) + bias`
    therefore equals that same row sum for ANY split of the columns. So
    `raw[:, :-1] / raw[:, -1]` (correct, and what models/explain.py ships) and
    `raw[:, 1:] / raw[:, 0]` (the off-by-one) produce bit-identical totals and both
    pass. Under the misindex every score stays right and every factor NAME shifts
    by one position — the worst available failure for an analyst-facing
    explanation, and invisible to the check that advertised catching it.

    What does distinguish the two ends: for a binary booster with a scalar
    `base_score` and no per-row `base_margin`, the bias column is the same value in
    every row. A real feature's contributions are not — that is what makes it a
    feature. So `np.ptp(bias) == 0` holds for the correct split and fails
    immediately under the misindex. It is checked with a tolerance rather than
    exactly, since the value arrives through float32 accumulation, and it is
    skipped for a single row, where every column is trivially constant.
    """
    warnings: list[str] = []
    try:
        contribs, bias = shap_contributions(model, X)
    except Exception as exc:
        return [[] for _ in range(len(X))], [
            f"Per-account attribution unavailable ({type(exc).__name__}: {exc}). "
            f"Scores are unaffected; explanations are omitted."]

    if not len(X):
        return [], warnings

    # THE IDENTITY, in the space TreeSHAP is additive in. `output_margin=True`
    # applies the same early-stopping iteration range `predict_proba` does, which
    # is the whole point — a hand-rolled log(p / (1 - p)) would not, and would be
    # catastrophically imprecise at the saturated ends where it matters most.
    margin_from_shap = contribs.sum(axis=1) + bias
    margin_direct = np.asarray(
        model.predict(X, output_margin=True), dtype=float).ravel()
    drift = float(np.abs(margin_from_shap - margin_direct).max())

    if drift > MARGIN_IDENTITY_TOLERANCE:
        prob_drift = float(np.abs(
            1.0 / (1.0 + np.exp(-np.clip(margin_from_shap, -60, 60)))
            - probabilities).max())
        return [[] for _ in range(len(X))], [
            f"SHAP contributions do not reconstruct the model's own margins (max "
            f"deviation {drift:.2e} log-odds > {MARGIN_IDENTITY_TOLERANCE:.0e}; "
            f"{prob_drift:.2e} in probability), so they do not explain these "
            f"scores. Most likely the early-stopping iteration range in "
            f"models/explain.py disagrees with the one used for scoring. "
            f"Explanations withheld rather than served misleadingly."]

    # THE BIAS COLUMN, checked on the one property that separates it from a
    # feature: it is constant down the batch. Needs at least two rows to say
    # anything — with one row every column is constant.
    if len(X) > 1:
        bias_spread = float(np.ptp(np.asarray(bias, dtype=float)))
        if bias_spread > BIAS_CONSTANT_TOLERANCE:
            return [[] for _ in range(len(X))], [
                f"The TreeSHAP bias column varies across the batch (spread "
                f"{bias_spread:.2e} > {BIAS_CONSTANT_TOLERANCE:.0e}), but this "
                f"booster has a scalar base_score and no per-row base_margin, so "
                f"it cannot. The most likely cause is an off-by-one in the "
                f"column split in models/explain.py, which leaves every score "
                f"correct and shifts every feature NAME by one position. "
                f"Explanations withheld: a confidently mislabelled reason is "
                f"worse than no reason."]

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
        # None, not 0.0. Publishing 0.0 for "unavailable" named the one value
        # `classify_risk` refuses outright — a threshold of 0 flags every account
        # — while the schema's own default for the field was 0.5, so the service
        # had two contradictory sentinels for the same missing number and neither
        # said "missing". `optimal_threshold: float | None` says it once.
        optimal_threshold=STATE.threshold,
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
    break_even = STATE.break_even_probability
    threshold_source = "metrics.json"
    if request.threshold_override is not None:
        # `or` was wrong here: it treats 0.0 as absent. The schema now rejects 0
        # outright, and an explicit None test means any accepted value is used.
        threshold = float(request.threshold_override)
        threshold_source = "request_override"

    submitted = submitted_to_frame(request.transactions)
    merged, diag = merge_with_context(
        submitted, STATE.context_edges, STATE.trained_window_days)

    # The trim dropped every historical edge, so features would be computed on the
    # submitted transactions alone: PageRank collapses to ~1/n, clustering and
    # cycle participation to 0. A tier and an action derived from that are a
    # threshold calibrated on a 60-day graph applied to a 4-edge one — well-formed
    # and meaningless.
    #
    # WHAT REACHES THIS, WHICH IS NOT WHAT THIS COMMENT USED TO CLAIM. It said "any
    # present-day timestamp lands here, because the context file ends 2025-04-30, so
    # this is the common case rather than the exotic one." That was true while the
    # window slid forward to the latest submitted transaction. It is not true now:
    # `merge_with_context` anchors the window to `context["timestamp"].max()`, which
    # is always inside the window, so a non-empty context always keeps at least the
    # edges at that instant and a late batch is scored against the whole graph. A
    # present-day timestamp now lands in the `submitted_outside_window` warning and
    # `window_comparable: false`, not here.
    #
    # What still reaches this is a deployment fault: a context file that loaded and
    # holds no rows, or holds timestamps that did not parse, so every comparison
    # against the window is False. Rare, and worth a distinct refusal rather than a
    # score — which is why it is kept and why
    # tests/test_api.py::test_a_request_with_no_usable_context_is_refused_with_422
    # exercises it directly. An unreachable guard nothing tests is a guard that will
    # not work the day it becomes reachable.
    #
    # Refused with 422 rather than returned as a 200 with `risk_level` and
    # `action` suppressed: both are required fields on NodeRiskScore, and a
    # response the caller must inspect for absent fields degrades exactly as
    # quietly as the warning string that used to be the only signal. Refusing is
    # also what this endpoint already does in the other two states it cannot score
    # honestly in — 503 when a prerequisite is missing, 500 when the extractor has
    # diverged from the contract.
    if diag["n_context_used"] == 0:
        raise HTTPException(status_code=422, detail="\n".join(diag["warnings"]))

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
            break_even_probability=break_even,
            contributing_factors=factors[i],
            seen_in_context=node in context_accounts,
        ))

    # Final safety validation — re-derives tier and action from the score, so a
    # response cannot reach the caller with a downgraded action.
    node_scores = validate_response_batch(node_scores, threshold, break_even)

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
            # The second number the tiering rests on. Without it the response
            # named the alert cutoff but not the ALLOW/step-up boundary, so a
            # caller could not tell why an account tiered MEDIUM.
            break_even_probability=break_even,
            context_provenance=CONTEXT_PROVENANCE,
            n_submitted_transactions=len(request.transactions),
            n_context_transactions_used=diag["n_context_used"],
            n_duplicates_dropped=diag["n_duplicates_dropped"],
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
