"""
models/train.py
───────────────
Trains the mule-account detector and produces the numbers that go in the README.

Pipeline:
  1. Load processed features for all three splits from data/processed/
  2. Guard the split boundary: no account and no ring on both sides
  3. Train XGBoost, early stopping on VALIDATION (never on test)
  4. Ring-grouped cross-validation on TRAIN, at the final model's tree count
  5. Choose two operating points on VALIDATION: cost-optimal, and one that fits
     a stated analyst capacity
  6. Evaluate once on TEST at those frozen thresholds
  7. Compute published baselines the model has to beat, on the same test split
  8. Save the model, the metrics, and the attribution summary

Usage:
    python -m models.train
    python -m models.train --no-cv                    # skip cross-validation
    python -m models.train --fn-cost 250000 --fp-cost 10000
    python -m models.train --alert-budget 10          # 10 alerts per 1,000
    python -m models.train --max-depth 6              # override regularisation

─────────────────────────────────────────────────────────────────────────────
THE POINT OF THIS FILE
─────────────────────────────────────────────────────────────────────────────
An accuracy number with nothing to compare it against is not evidence. Track 2
asks for "honest metrics including false-positive cost", so every number this
script prints is accompanied by the thing that makes it interpretable:

  • a threshold chosen before the test split was touched, plus the cost of not
    having peeked (the ORACLE diagnostic);
  • three baselines on the same held-out split — the best possible single-feature
    rule, a logistic regression on the identical features, and the same XGBoost
    with every graph feature removed. If the model cannot beat a one-line `if`
    statement, that is the result, and it belongs in the report;
  • recall broken out per ring archetype and per ring, because 70% overall
    recall means something very different if it is 100% of the loud rings and
    30% of the quiet ones;
  • the alert load in accounts per thousand, since a queue nobody can staff is
    not a deployable operating point;
  • a calibration check, because the cost model implies a break-even cutoff of
    p* ≈ 0.07 and that is only meaningful if the scores are probabilities.

─────────────────────────────────────────────────────────────────────────────
v2 → v3 CHANGES
─────────────────────────────────────────────────────────────────────────────
1. CROSS-VALIDATION GROUPS ON RINGS, NOT LOUVAIN COMMUNITIES.
   v2 grouped folds on `louvain_community`, which sounds right and is not: on v2's
   data 52% of train accounts — and zero positives — landed in one giant organic
   community, so folds were lopsided and some were skipped for being
   single-class. (v3's generator produces a less degenerate community structure —
   the largest is ~17% of train — but the flaw is structural, not a property of
   one dataset: Louvain community size is unbounded and uncorrelated with the
   label.) The grouping that matters is the one that defines the leak: a ring's
   members must not straddle a fold boundary, so the group key is `ring_id`, with
   every legitimate account its own singleton group. See `ring_grouped_folds` for
   why that splitter is hand-rolled rather than `StratifiedGroupKFold`.

2. CROSS-VALIDATION MEASURES THE MODEL THAT WAS ACTUALLY TRAINED.
   v2's fold models used a hard-coded 300 rounds while the final model used
   early stopping, so the CV interval described a different model. Folds now use
   the final fit's `best_iteration + 1`.

3. BASELINES ARE PUBLISHED, NOT IMPLIED.
   Nothing in v2 established that the model beat a threshold on one feature.
   Three baselines are now computed and written to metrics.json, each with its
   threshold selected on validation and applied blind to test, exactly as the
   model's is.

4. PER-ARCHETYPE AND PER-RING RECALL.
   A single recall figure hides which rings are being caught. The generator
   emits three archetypes of deliberately unequal difficulty; recall is reported
   per archetype and per ring (a ring counts as detected if any member is
   flagged, which is what actually starts an investigation).

5. CALIBRATION IS CHECKED, AND MEAN |SHAP| IS RECORDED.
   `scale_pos_weight` deliberately distorts the output scale, so the empirical
   threshold and the theoretical break-even p* need not agree — the Brier score
   and reliability table make the size of that gap visible instead of implicit.
   Global importance is reported as mean |SHAP| alongside gain, and the run
   asserts that the per-account attributions sum to the margin — checked in
   MARGIN space, where TreeSHAP's additivity is actually stated, rather than
   through a sigmoid that compresses the disagreement away on the ~96% of
   accounts scoring near zero. That is what makes api/main.py's explanations
   trustworthy.

6. THE COST CONSTANTS HAVE ONE HOME.
   v2 redefined `DEFAULT_FN_COST` / `DEFAULT_FP_COST` here as well as in
   models/cost_matrix.py. They are imported now.

Carried over from the v1 → v2 pass, and still true: early stopping runs on
validation rather than test, the decision threshold is selected on validation,
test AUC is reported with a bootstrap interval, and FEATURE_COLS is imported
rather than re-declared.

Compatibility note: metrics.json keeps every top-level key the dashboard and API
already read (`roc_auc`, `optimal_threshold`, `precision`, `recall`, `f1`,
`total_cost`, `feature_importance`). New information is added in nested blocks,
so nothing that consumes the old file breaks.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from console import bar, banner, enable_utf8_stdout, sym
from models.cost_matrix import (
    DEFAULT_FN_COST,
    DEFAULT_FP_COST,
    CostEvaluator,
    average_precision,
    brier_score,
    expected_calibration_error,
    fit_platt_scaling,
    reliability_table,
    roc_auc,
    # Underscored, and borrowed deliberately. Precision/recall/F1 are None when
    # not computable (cost_matrix._prf), so every place this module rounds one for
    # metrics.json or formats one for the console needs the same None convention.
    # Re-implementing "how do we print a missing metric" here is how two files
    # drift into disagreeing about what 0.0 means.
    _fmt,
    _none_if_nan,
    _round_or_none,
)
from models.explain import mean_abs_shap, shap_contributions
from models.features import (
    FEATURE_COLS,
    FEATURE_DECISION_SPLIT,
    LABEL_META_COLS,
    LEAKAGE_AUC_CEILING,
    MODEL_NAME,
    MODEL_VERSION,
    TARGET_COL,
    assert_feature_contract,
)

# ──────────────────────────────────────────────────────────────────
# Paths / constants
# ──────────────────────────────────────────────────────────────────

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
MODEL_DIR = Path(__file__).resolve().parent / "saved_models"

SPLITS = ("train", "val", "test")
SPLIT_PAIRS = (("train", "val"), ("train", "test"), ("val", "test"))

RANDOM_SEED = 42
BOOTSTRAP_ROUNDS = 1_000

# What a review team can actually work, per thousand accounts scored. The
# cost-optimal threshold ignores this entirely and will happily flag one account
# in five; both points are reported so the trade-off is explicit.
DEFAULT_ALERT_BUDGET_PER_1000 = 20.0

# Rings seat some of their members by recruiting EXISTING ordinary accounts
# (data/generator.py, `hijack_prob`), which is how money mules actually work:
# a real person with a real account and a real history is paid to move money.
# So an account being an ordinary customer in January and a ring member in May is
# not a labelling contradiction, it is the recruitment timeline, and it runs one
# way only.
#
# The guard worth having here is therefore a FLOOR, not a ceiling. If almost no
# mule had a prior history, every positive would be a freshly-minted account
# whose only counterparties are its co-conspirators — which makes
# `community_internal_ratio` a restatement of the label and the whole benchmark
# too easy. Empirically ~45% of each window's mules are recruited, so a low
# reading means the generator's camouflage stopped working.
MIN_RECRUITED_MULE_RATE = 0.15

# NOTE: LEAKAGE_AUC_CEILING is no longer defined here. If one feature alone
# separates the classes this well, the generator planted the answer and every
# downstream metric is theatre — but that ceiling is a statement about the FEATURE
# CONTRACT, so it now lives in models/features.py beside design rule 4 that
# argues for it, and is imported above. Same name, so tests/test_baselines.py's
# `from models.train import LEAKAGE_AUC_CEILING` agreement check still resolves.
#
# What changed with it: the gate is screened on FEATURE_DECISION_SPLIT
# (validation), not on test. A number that can veto a feature or condemn a
# dataset is a decision, and decisions do not get to read the split reserved for
# the final evaluation. Here it stays a warning either way — train.py should not
# be the thing that decides a dataset is invalid.

# The five features that require the transaction graph. Everything else — degrees,
# sums, ratios, timing, amount dispersion — is computable per account from a flat
# ledger with a GROUP BY. Ablating exactly these answers the question the project
# rests on: does the graph earn its complexity?
GRAPH_FEATURES = [
    "pagerank",
    "clustering_coefficient",
    "cycle_participation",
    "reciprocity",
    "community_internal_ratio",
]
NON_GRAPH_FEATURES = [c for c in FEATURE_COLS if c not in GRAPH_FEATURES]

# Columns log-scaled before the logistic-regression baseline: handing a linear
# model a raw rupee sum builds a straw man, and a baseline worth reporting is one
# that was given a fair chance. A too-weak baseline is not a neutral error — it
# inflates the headline lift, which is the number this project is judged on.
#
# THE SELECTION CRITERION, WRITTEN DOWN SO IT CAN BE APPLIED AND CHECKED
# ──────────────────────────────────────────────────────────────────────
# A feature is log1p-scaled iff, measured on the TRAIN split only:
#
#     (a) it is non-negative            — log1p needs x > -1, and `design()`
#                                         re-checks per column at runtime; and
#     (b) its skewness is >= 1.5        — a heavy right tail is exactly the shape
#                                         a linear-in-x model cannot fit.
#
# Train only, because this is preprocessing: fitting it on validation or test
# would leak, for the same reason the z-score below uses train mean/std.
#
# This list was previously curated by hand and had no stated rule, which is how it
# came to omit `degree_ratio` — skew 9.39, the MOST skewed feature in the contract,
# spanning 0 to 98. That omission was not cosmetic: it left the linear model to fit
# a raw count ratio, and is the likely source of the anomalous LR coefficient
# -5.0882 on degree_ratio against +0.0733 on the bounded degree_balance beside it.
# A baseline crippled on one feature makes the model look better than it is.
#
# Applying the criterion to all 18 features (train skew in brackets) admits:
#     out_degree [10.51]  degree_ratio [9.39]  out_amount_sum [7.81]
#     in_degree [6.34]    burst_ratio [5.82]   community_internal_ratio [5.45]
#     in_amount_sum [5.42] clustering_coefficient [5.22] txn_velocity [4.98]
#     pagerank [3.78]     repeat_ratio [1.95]  cycle_participation [1.82]
# and rejects the six with skew < 1.5 (reciprocity 1.01, degree_balance 0.75,
# counterparty_amount_cv 0.34, amount_cv 0.09, flow_passthrough -0.12,
# fan_in_concentration -0.94).
#
# Worth knowing which of the new entries actually matter: for a column bounded in
# [0,1], log1p is within a hair of affine — Pearson r(x, log1p x) >= 0.9939 for
# every bounded feature here — so after standardisation it cannot change what the
# linear model expresses, and the four bounded additions are no-ops kept only so
# the list follows its own rule. The unbounded magnitudes are where the transform
# bites: r is 0.36 (out_amount_sum), 0.54 (in_amount_sum), 0.68 (out_degree) and
# 0.68 (degree_ratio). degree_ratio sits in that group, which is why its absence
# was a real defect and not a tidiness complaint.
LOG1P_COLS = [
    "in_degree", "out_degree", "degree_ratio", "in_amount_sum",
    "out_amount_sum", "txn_velocity", "burst_ratio", "repeat_ratio",
    "pagerank", "clustering_coefficient", "cycle_participation",
    "community_internal_ratio",
]

# Hyperparameters. Depth is 4 rather than v2's 6 because a depth-6 tree has 64
# leaves and the training split holds ~196 positives — enough for leaves that
# isolate one or two accounts, which is memorisation dressed as a model.
# min_child_weight and reg_lambda push in the same direction. Early stopping on
# validation still has the final say, and --max-depth reopens the choice.
HPARAMS: dict = {
    "max_depth": 4,
    "min_child_weight": 5,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 2.0,
    "n_estimators": 500,
    "eval_metric": "aucpr",      # tracks the minority class; logloss does not
    "importance_type": "gain",   # explicit: the property's default has moved
    "tree_method": "hist",
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
    "verbosity": 0,
}
EARLY_STOPPING_ROUNDS = 30


# ══════════════════════════════════════════════════════════════════
# Data loading
# ══════════════════════════════════════════════════════════════════

def load_data() -> dict[str, pd.DataFrame]:
    """
    Load the train / val / test feature frames, and refuse anything stale.

    A missing column here almost always means data/processed/ was written by an
    older extractor than the current feature contract, which is the failure the
    v1 → v2 refactor produced twice. Better to stop with the column name than to
    train on whatever happens to be present.
    """
    required = set(FEATURE_COLS) | {TARGET_COL, "node", *LABEL_META_COLS}
    frames: dict[str, pd.DataFrame] = {}

    for split in SPLITS:
        path = DATA_DIR / f"{split}_features.csv"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Run:\n"
                "  python -m data.generator\n"
                "  python -m data.extractor"
            )
        # reset_index so positional and label indexing coincide; the fold
        # builder below relies on it.
        #
        # The two attribution columns are read as strings rather than inferred.
        # They hold pipe-separated ring-id sets like "64|12", but when no account
        # in a split bridges two rings, every cell is a bare integer, pandas
        # infers float64, and `_attribution_slack`'s `astype(str)` then yields
        # "64.0" — which `int()` refuses. So the inferred dtype depends on the
        # data: the column parses cleanly as soon as one account bridges two
        # rings and breaks when none does, which is exactly the case the function
        # exists to report as zero slack. Pinning the dtype removes that
        # dependence. dtype keys naming absent columns are ignored by pandas, so
        # the missing-column check below still reports in its own words instead
        # of surfacing a parser error.
        df = pd.read_csv(
            path,
            dtype={"rings_attributed": str, "ring_types_attributed": str},
        ).reset_index(drop=True)

        missing = sorted(required - set(df.columns))
        if missing:
            raise RuntimeError(
                f"{path.name} is missing {len(missing)} required column(s): "
                f"{missing}\n"
                "  This file was written by a different feature contract than "
                "models/features.py declares.\n"
                "  Re-run `python -m data.extractor`."
            )

        for col in FEATURE_COLS:
            if not pd.api.types.is_numeric_dtype(df[col]):
                raise RuntimeError(
                    f"{path.name}: feature '{col}' is {df[col].dtype}, not "
                    "numeric. XGBoost would coerce or fail unpredictably."
                )
            if not np.isfinite(df[col].to_numpy()).all():
                n_bad = int((~np.isfinite(df[col].to_numpy())).sum())
                raise RuntimeError(
                    f"{path.name}: feature '{col}' has {n_bad} NaN/inf value(s).\n"
                    "  The threshold sweep cannot be trusted over non-finite "
                    "scores, and a NaN feature silently becomes a 'missing' "
                    "branch at serve time. Fix data/extractor.py."
                )

        frames[split] = df

    for split, df in frames.items():
        pos = int(df[TARGET_COL].sum())
        rings = int(df.loc[df[TARGET_COL] == 1, "ring_id"].nunique())
        print(f"  {split:<5s}: {str(df.shape):<12s} | "
              f"mule rate {df[TARGET_COL].mean():>6.2%} "
              f"({pos} positives across {rings} rings)")

    return frames


def describe_dataset(frames: dict[str, pd.DataFrame]) -> dict:
    """Per-split shape, prevalence and archetype mix, for metrics.json."""
    out = {}
    for split, df in frames.items():
        pos = df[TARGET_COL] == 1
        archetypes = (df.loc[pos, "ring_type"].value_counts().sort_index()
                      .astype(int).to_dict())
        out[split] = {
            "n_accounts": int(len(df)),
            "n_positives": int(pos.sum()),
            "prevalence": round(float(pos.mean()), 6),
            "n_rings": int(df.loc[pos, "ring_id"].nunique()),
            "positives_by_archetype": archetypes,
        }
    return out


# ══════════════════════════════════════════════════════════════════
# Split integrity
# ══════════════════════════════════════════════════════════════════

def assert_split_integrity(frames: dict[str, pd.DataFrame]) -> dict:
    """
    Guard the split boundary, and record what was checked.

    The leakage that invalidates a test metric is a *ring* appearing on both
    sides: the v1 generator scattered each ring across the whole six-month
    window, so 25 of 25 test rings had members in train and the model was scored
    on rings it had partly memorised.

    Two things are hard failures, and the distinction matters:

      HARD FAIL  a ring_id appears in two splits. The most direct statement of
                 the leak, and cheap to check now that the extractor carries
                 ring_id through.

      HARD FAIL  a labelled mule in one split is also a labelled mule in
                 another. The same leak seen from the account side; kept because
                 it would also catch a generator that reused ring_ids.

    Everything else about accounts recurring across windows is reported rather
    than forbidden, because forbidding it would demand a dataset less realistic
    than production:

      • The same *legitimate* account in two windows is how the problem actually
        looks — you train on January-February customers and score those same
        customers in May. Account IDs are not features and features are
        recomputed from each window's own edges, so a recurring account has
        entirely different feature values in each window and there is nothing to
        memorise.

      • An account that is a ring member in a later window and an ordinary
        customer in an earlier one is the RECRUITMENT TIMELINE, not a
        contradiction: both labels are correct for their own window. Around 45%
        of mules here are recruited this way, and the direction is
        one-way — this function reports the reverse direction separately, and it
        is empirically zero because a ringed account is frozen rather than
        returned to service.

    An earlier draft of this function bounded the recruited share at 5% and
    treated the excess as label noise. That was the wrong model of the data: it
    would have rejected the realistic generator and accepted one where every mule
    is a fresh account with no history — which is the easier, less honest
    dataset. The check now runs in the useful direction, as a floor.
    """
    node_sets = {s: set(df["node"]) for s, df in frames.items()}
    pos_sets = {s: set(df.loc[df[TARGET_COL] == 1, "node"])
                for s, df in frames.items()}
    ring_sets = {s: set(df.loc[df["ring_id"] >= 0, "ring_id"].unique())
                 for s, df in frames.items()}

    # ── hard invariant: rings never span a split ──
    for a, b in SPLIT_PAIRS:
        shared = ring_sets[a] & ring_sets[b]
        if shared:
            raise RuntimeError(
                f"RING LEAKAGE: ring_id(s) {sorted(shared)[:5]} appear in both "
                f"'{a}' and '{b}'.\n"
                "  Test rings would be partly memorised, so held-out precision "
                "and recall mean nothing.\n"
                "  Regenerate with `python -m data.generator`."
            )
    print(f"  {sym('ok')} no ring spans a split boundary "
          f"({len(ring_sets['train'])} / {len(ring_sets['val'])} / "
          f"{len(ring_sets['test'])} rings, disjoint)")

    # ── hard invariant: positives never recur ──
    for a, b in SPLIT_PAIRS:
        shared = pos_sets[a] & pos_sets[b]
        if shared:
            raise RuntimeError(
                f"ENTITY LEAKAGE: {len(shared):,} accounts are labelled mules "
                f"in BOTH '{a}' and '{b}' (e.g. {sorted(shared)[:5]}).\n"
                "  Regenerate with `python -m data.generator`."
            )
    print(f"  {sym('ok')} no account is a labelled mule in two splits "
          f"({len(pos_sets['train'])} / {len(pos_sets['val'])} / "
          f"{len(pos_sets['test'])} positives, disjoint)")

    # ── informational: recurring legitimate customers ──
    recurrence = {}
    for a, b in SPLIT_PAIRS:
        recur = len(node_sets[a] & node_sets[b])
        recurrence[f"{a}|{b}"] = recur
        print(f"  {sym('bullet')} {a}/{b}: {recur:,} accounts recur "
              f"(expected — same customers across time windows)")

    # ── reported both ways: recruitment vs. an ex-mule returning ──
    # SPLITS is in chronological order, so `later` is always the newer window.
    recruited: dict[str, dict] = {}
    returned: dict[str, dict] = {}
    for earlier, later in SPLIT_PAIRS:
        n_recruited = len(pos_sets[later] & (node_sets[earlier]
                                            - pos_sets[earlier]))
        share = n_recruited / max(len(pos_sets[later]), 1)
        recruited[f"{later}_from_{earlier}"] = {
            "n_accounts": n_recruited,
            "share_of_later_positives": round(share, 4),
        }
        print(f"  {sym('bullet')} {n_recruited} of {len(pos_sets[later])} "
              f"'{later}' mules were ordinary '{earlier}' customers "
              f"({share:.0%}) — recruited, not contradictory")
        if share < MIN_RECRUITED_MULE_RATE:
            print(f"      {sym('warn')} below the "
                  f"{MIN_RECRUITED_MULE_RATE:.0%} floor: mules are mostly "
                  "purpose-built accounts with no civilian history, which "
                  "makes community_internal_ratio a proxy for the label. "
                  "Check `hijack_prob` and `_emit_camouflage` in "
                  "data/generator.py.")

        n_returned = len(pos_sets[earlier] & node_sets[later])
        returned[f"{earlier}_into_{later}"] = {"n_accounts": n_returned}
        if n_returned:
            print(f"  {sym('bullet')} {n_returned} '{earlier}' mules reappear "
                  f"in '{later}' — an ex-mule returned to service. Not "
                  "leakage (labels are per-window and disjoint), but worth "
                  "knowing.")

    return {
        "rings_disjoint": True,
        "positives_disjoint": True,
        "recurring_legitimate_accounts": recurrence,
        "recruited_from_earlier_window": recruited,
        "ex_mules_returning_to_service": returned,
        "min_recruited_mule_rate": MIN_RECRUITED_MULE_RATE,
    }


# ══════════════════════════════════════════════════════════════════
# Cross-validation
# ══════════════════════════════════════════════════════════════════

def build_cv_groups(df: pd.DataFrame) -> pd.Series:
    """
    The grouping key for cross-validation: one group per ring, one per legit
    account.

    A ring's members share almost all of their structure — the same cycle, the
    same counterparties, the same burst — so putting two of them on opposite
    sides of a fold boundary is the entity leakage the temporal split exists to
    prevent, reintroduced inside the training data. Grouping them together is
    the fix.

    Legitimate accounts get singleton groups rather than one shared "not a ring"
    group, which would otherwise be an unsplittable 94% of the data.
    """
    is_pos = (df[TARGET_COL] == 1).to_numpy()
    if is_pos.any() and not (df.loc[is_pos, "ring_id"] >= 0).all():
        raise RuntimeError(
            "some positives carry ring_id < 0, so they would all collapse into "
            "one CV group. Re-run `python -m data.extractor`."
        )
    ring = df["ring_id"].to_numpy()
    node = df["node"].astype(str).to_numpy()
    return pd.Series(
        [f"ring:{r}" if p else f"acct:{n}"
         for p, r, n in zip(is_pos, ring, node)],
        index=df.index, name="cv_group",
    )


def ring_grouped_folds(
    df: pd.DataFrame,
    n_folds: int,
    seed: int = RANDOM_SEED,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Folds that are ring-grouped and exactly stratified, by construction.

    Two constraints have to hold simultaneously: no ring may straddle a fold
    boundary, and every fold needs enough positives for AUC to mean anything.
    `StratifiedGroupKFold` satisfies the first and only approximates the second,
    which with 40 rings of 2-8 members produced folds thin enough to skip. The
    structure here is simple enough to do better directly: sort rings by size
    descending and deal them round-robin (longest-processing-time first, the
    standard greedy balance), then deal the shuffled legitimate accounts the same
    way. Ring integrity is exact, positive counts land within one ring of each
    other, and the whole thing is reproducible from `seed`.

    Returns positional (train_idx, val_idx) pairs.
    """
    if not isinstance(df.index, pd.RangeIndex) or df.index.start != 0:
        raise RuntimeError(
            "ring_grouped_folds needs a 0-based RangeIndex so labels and "
            "positions coincide; call .reset_index(drop=True) first."
        )

    rng = np.random.default_rng(seed)
    fold_of = np.full(len(df), -1, dtype=int)

    pos_mask = (df[TARGET_COL] == 1).to_numpy()
    # Positions in the FULL frame, grouped by ring. Deliberately not
    # `df.loc[pos_mask].groupby("ring_id").indices`: that returns positions
    # relative to the filtered sub-frame, so the assignment below would land on
    # unrelated rows — silently, since the indices are all in range. Rings would
    # straddle folds and some positives would never be held out at all.
    pos_positions = np.flatnonzero(pos_mask)
    rings_at_pos = df["ring_id"].to_numpy()[pos_positions]
    ring_members = {int(r): pos_positions[rings_at_pos == r]
                    for r in np.unique(rings_at_pos)}

    ring_ids = list(ring_members)
    rng.shuffle(ring_ids)                     # seed-dependent tie-breaking
    ring_ids.sort(key=lambda r: -len(ring_members[r]))          # stable, LPT
    for i, ring_id in enumerate(ring_ids):
        fold_of[ring_members[ring_id]] = i % n_folds

    neg_positions = rng.permutation(np.flatnonzero(~pos_mask))
    fold_of[neg_positions] = np.arange(neg_positions.size) % n_folds

    if (fold_of < 0).any():                              # pragma: no cover
        raise RuntimeError(
            f"{int((fold_of < 0).sum())} rows were not assigned to any fold"
        )

    all_idx = np.arange(len(df))
    return [(all_idx[fold_of != k], all_idx[fold_of == k])
            for k in range(n_folds)]


def cross_validate(
    frames: dict[str, pd.DataFrame],
    n_rounds: int,
    n_folds: int = 5,
) -> dict:
    """
    Ring-grouped cross-validation on the TRAIN split only.

    This is a stability check on the configuration, not a second performance
    estimate — the headline numbers all come from the untouched test split. It
    answers one question: is the test result a property of the model, or of which
    accounts happened to land in test? A wide fold-to-fold spread means the
    latter, and no amount of held-out purity fixes it.

    `n_rounds` comes from the final model's early-stopped `best_iteration`, so the
    folds measure the same model complexity that was actually shipped.
    """
    df = frames["train"]
    print(banner(f"{n_folds}-Fold Ring-Grouped Cross-Validation (train only)"))

    groups = build_cv_groups(df)
    n_rings = int(df.loc[df[TARGET_COL] == 1, "ring_id"].nunique())
    n_folds = min(n_folds, max(n_rings, 1))
    if n_folds < 2:
        print(f"  {sym('warn')} only {n_rings} rings in train; skipping CV.")
        return {}

    X, y = df[FEATURE_COLS], df[TARGET_COL]
    aucs: list[float] = []
    aps: list[float] = []

    for fold, (tr_idx, va_idx) in enumerate(
        ring_grouped_folds(df, n_folds), start=1
    ):
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]
        if y_va.nunique() < 2 or y_tr.nunique() < 2:      # pragma: no cover
            print(f"  Fold {fold}: skipped (single-class fold)")
            continue

        fold_model = xgb.XGBClassifier(
            **{**HPARAMS, "n_estimators": n_rounds},
            scale_pos_weight=compute_scale_pos_weight(y_tr),
        )
        # No early stopping inside a fold: there is no third split to stop
        # against, and stopping on the fold's own held-out rows is precisely the
        # bug removed from the final fit.
        fold_model.fit(X.iloc[tr_idx], y_tr, verbose=False)

        p_va = fold_model.predict_proba(X.iloc[va_idx])[:, 1]
        fold_auc = roc_auc(y_va.to_numpy(), p_va)
        fold_ap = average_precision(y_va.to_numpy(), p_va)
        aucs.append(float(fold_auc))
        aps.append(float(fold_ap))
        print(f"  Fold {fold}: AUC {fold_auc:.4f}  AP {fold_ap:.4f}   "
              f"({len(va_idx):,} accounts, {int(y_va.sum())} positives, "
              f"{groups.iloc[va_idx].nunique():,} groups)")

    if not aucs:                                          # pragma: no cover
        return {}

    mean_auc, std_auc = float(np.mean(aucs)), float(np.std(aucs))
    print(f"  Mean AUC: {mean_auc:.4f} {sym('plusminus')} {std_auc:.4f}   "
          f"Mean AP: {np.mean(aps):.4f} {sym('plusminus')} {np.std(aps):.4f}")
    print("  (Ring-grouped folds score lower than shuffled ones. That gap was "
          "the leakage.)")

    return {
        "grouping": "ring_id (positives) + per-account singletons (negatives)",
        "n_folds": len(aucs),
        "n_rounds_per_fold": int(n_rounds),
        "fold_auc": aucs,
        "fold_average_precision": aps,
        "mean_auc": mean_auc,
        "std_auc": std_auc,
        "mean_average_precision": float(np.mean(aps)),
        "std_average_precision": float(np.std(aps)),
    }


# ══════════════════════════════════════════════════════════════════
# Training
# ══════════════════════════════════════════════════════════════════

def compute_scale_pos_weight(y: pd.Series) -> float:
    """
    Class-imbalance weight, neg/pos.

    Worth noting where this lands: at ~6% prevalence it comes out near 15, which
    is close to the cost model's 13.3:1 FN:FP ratio, so the training loss already
    approximates the cost function by coincidence. It is deliberately *not* set
    from the cost ratio — the threshold sweep already prices the trade-off, and
    doing it twice would double-count.
    """
    neg = int((y == 0).sum())
    pos = int((y == 1).sum())
    return neg / max(pos, 1)


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    label: str = "model",
) -> xgb.XGBClassifier:
    """
    Fit XGBoost with early stopping on the VALIDATION split.

    Validation does double duty here — it stops the boosting and it fixes the
    decision threshold — so validation-side numbers are mildly optimistic and are
    never reported as performance. Test is touched once, at a threshold chosen
    before it was read.
    """
    spw = compute_scale_pos_weight(y_train)
    model = xgb.XGBClassifier(
        **HPARAMS,
        scale_pos_weight=spw,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    print(f"  {label}: {len(X_train.columns)} features, "
          f"scale_pos_weight {spw:.2f}, best iteration "
          f"{model.best_iteration} of {HPARAMS['n_estimators']} "
          f"(early stopping on validation, {len(X_val):,} accounts)")
    return model


def score(model, X: pd.DataFrame) -> np.ndarray:
    """P(mule) for each row. One call site for the positive-class column index."""
    return np.asarray(model.predict_proba(X), dtype=float)[:, 1]


def margin(model, X: pd.DataFrame) -> np.ndarray:
    """
    Raw log-odds margin for each row — the space TreeSHAP is actually additive in.

    `predict_proba` is `sigmoid(margin)`, so this is the same model one step
    earlier. It exists because the SHAP identity holds here and only here:
    contributions sum to the MARGIN, and checking them against a probability puts
    the comparison through a saturating function that hides disagreement (see
    `importance_report`). Kept beside `score()` so both raw-output conventions
    live in one place, and because `predict(..., output_margin=True)` applies the
    same early-stopping iteration range `predict_proba` does — a hand-rolled
    `log(p / (1 - p))` would not, and would also be catastrophically imprecise at
    the saturated ends.
    """
    return np.asarray(
        model.predict(X, output_margin=True), dtype=float
    ).ravel()


# ══════════════════════════════════════════════════════════════════
# Reporting helpers
# ══════════════════════════════════════════════════════════════════

def bootstrap_clusters(df: pd.DataFrame) -> np.ndarray:
    """
    The resampling unit for the AUC interval: one cluster per ring, one per
    non-member account.

    Same principle as `build_cv_groups`, and for the same reason — a ring is one
    observation, not six — but keyed off `ring_id >= 0` rather than the label, so
    that any future ring-adjacent negative would still cluster with its ring
    instead of quietly becoming an independent draw.
    """
    ring = df["ring_id"].to_numpy()
    node = df["node"].astype(str).to_numpy()
    return np.array(
        [f"ring:{r}" if r >= 0 else f"acct:{n}" for r, n in zip(ring, node)],
        dtype=object,
    )


def bootstrap_auc_ci(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    clusters: np.ndarray,
    rounds: int = BOOTSTRAP_ROUNDS,
    seed: int = RANDOM_SEED,
) -> tuple[float, float]:
    """
    Percentile bootstrap 95% CI for ROC-AUC, resampling CLUSTERS, not accounts.

    WHY THE CLUSTER IS THE RING AND NOT THE ACCOUNT
    ───────────────────────────────────────────────
    An i.i.d. account bootstrap — which is what this function used to do — assumes
    every row is an independent draw from the population. Here that assumption is
    false in exactly the place it matters. A ring's members are generated from one
    template: the same cycle, the same counterparties, the same burst hour, the
    same amounts. Their features are near-perfectly correlated and so are their
    scores, so six accounts from one ring carry barely more information about
    "would this model catch an unseen ring?" than one of them does.

    The test split holds on the order of a hundred positives drawn from a couple
    of dozen rings. Resampling accounts therefore inflates the effective sample
    size by roughly the mean ring size and reports an interval tighter than the
    evidence supports. The honest interval resamples the RINGS with replacement
    (each non-member account is its own singleton cluster, so the negative class
    is still resampled at account level, which is correct — nothing ties two
    random customers together).

    No counts are quoted here on purpose. An earlier version of this paragraph
    said "119 positives but only 24 rings" and "about 2.14x too narrow", measured
    once by hand; the positive count was already wrong by 5 and a reader had no
    way to know which of the three numbers had aged. The live figures are in
    metrics.json — `n_rings` in the recall breakdown, and both intervals side by
    side, one clustered and one naive — so the size of the understatement is a
    published measurement rather than a remembered one.

    That is not a cosmetic difference in a hackathon judged on honest metrics: a
    narrow interval is the claim "this AUC would hold on your data", and the
    number of independent fraud rings behind it is two dozen.

    Returns the percentile interval. `main` also prints the naive account-level
    interval beside it, obtained by passing singleton clusters, so the size of the
    understatement is on the record rather than silently corrected.
    """
    y = np.asarray(y_true).astype(int).ravel()
    s = np.asarray(y_proba, dtype=float).ravel()
    g = np.asarray(clusters).ravel()
    if not (y.size == s.size == g.size):
        raise ValueError(
            f"shape mismatch: y_true {y.size}, y_proba {s.size}, "
            f"clusters {g.size}"
        )
    if y.size == 0:
        raise ValueError("cannot bootstrap an empty array")

    # Group positions by cluster ONCE, as a ragged structure: `order` holds row
    # positions sorted by cluster, `starts`/`sizes` delimit each cluster inside
    # it. Drawing a cluster then means copying a contiguous slice of `order`,
    # which keeps the whole resample vectorised — a Python loop over 2,800
    # clusters x 1,000 rounds is minutes, and this is seconds.
    order = np.argsort(g.astype(str), kind="mergesort")
    g_sorted = g.astype(str)[order]
    starts = np.flatnonzero(
        np.concatenate(([True], g_sorted[1:] != g_sorted[:-1]))
    )
    sizes = np.diff(np.append(starts, g.size))
    n_clusters = starts.size

    rng = np.random.default_rng(seed)
    samples: list[float] = []

    for _ in range(rounds):
        pick = rng.integers(0, n_clusters, n_clusters)
        counts = sizes[pick]
        total = int(counts.sum())
        # Expand each drawn cluster to its member positions: base offset repeated
        # per member, plus 0..size-1 within the cluster.
        base = np.repeat(starts[pick], counts)
        within = np.arange(total) - np.repeat(np.cumsum(counts) - counts, counts)
        idx = order[base + within]
        if len(np.unique(y[idx])) < 2:      # pragma: no cover — degenerate draw
            continue
        samples.append(roc_auc(y[idx], s[idx]))

    if not samples:                         # pragma: no cover
        return (float("nan"), float("nan"))
    return (float(np.percentile(samples, 2.5)),
            float(np.percentile(samples, 97.5)))


def print_confusion_table(tp: int, fp: int, tn: int, fn: int) -> None:
    """
    Both classes' precision/recall/F1, replacing sklearn's classification_report.

    Written out rather than imported because every other metric in this project
    now comes from models/cost_matrix.py, and two sources for the same four
    numbers is how they drift apart.

    `_prf` returns None for a metric with an empty denominator, which this table
    prints as "n/a". Both rows can hit it: a model that flags everything leaves
    tn + fn = 0, so the legitimate class has no precision to report. "n/a" is the
    honest cell there — 0.0000 would read as a model that gets every legitimate
    account wrong, when in fact it never called one legitimate.
    """
    from models.cost_matrix import _prf

    rows = [
        ("Legitimate", *_prf(tn, fn, fp), tn + fp),   # roles swapped
        ("Mule", *_prf(tp, fp, fn), tp + fn),
    ]
    print(f"  {'':<12s}{'precision':>11s}{'recall':>9s}{'f1':>9s}{'support':>9s}")
    for name, precision, recall, f1, support in rows:
        print(f"  {name:<12s}{_fmt(precision):>11s}{_fmt(recall):>9s}"
              f"{_fmt(f1):>9s}{support:>9d}")


def calibration_report(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n_bins: int = 5,
) -> dict:
    """
    Brier score, expected calibration error, and a reliability table.

    The cost model derives a break-even cutoff of p* = fp/(fp+fn) ≈ 0.07, which
    is only actionable if the model's outputs are probabilities. They are not,
    quite: `scale_pos_weight` inflates positive-class scores by design, so the
    empirical optimum sits well above p*. That is a defensible choice, but only
    if it is stated — so the size of the distortion is measured here rather than
    left for a reviewer to discover.

    The three quantities are now DELEGATED to models/cost_matrix.py rather than
    computed inline. They were open-coded here, and then `calibration_diagnostic`
    below needed the same binning to compare raw against calibrated scores — at
    which point this file would have held two independent ECE implementations free
    to disagree about bin edges, empty-bin handling and weighting. That is the
    exact failure `classification_table` above names in its own docstring: "two
    sources for the same four numbers is how they drift apart."

    The returned shape is unchanged — same keys, same rounding, populated bins
    only — because models/report.py renders `reliability` by these key names.
    """
    y = np.asarray(y_true).astype(int).ravel()
    p = np.asarray(y_proba, dtype=float).ravel()

    frame = reliability_table(y, p, n_bins)
    populated = frame[frame["n"] > 0]
    table = [
        {
            "bin": f"[{row.bin_lower:.1f}, {row.bin_upper:.1f})",
            "n": int(row.n),
            "mean_predicted": round(float(row.mean_predicted), 4),
            "observed_rate": round(float(row.observed_frequency), 4),
        }
        for row in populated.itertuples()
    ]

    return {
        "brier_score": round(brier_score(y, p), 6),
        # `_round_or_none`, not round(): ECE is None on an empty split, and 0.0
        # there would read as perfect calibration of nothing.
        "expected_calibration_error": _round_or_none(
            expected_calibration_error(y, p, n_bins), 4),
        "mean_predicted_probability": round(float(p.mean()), 4),
        "actual_prevalence": round(float(y.mean()), 4),
        "reliability": table,
        "note": (
            "scale_pos_weight inflates scores on purpose; the operating "
            "threshold is chosen empirically on validation, not from p*."
        ),
    }


def archetype_breakdown(
    df: pd.DataFrame,
    y_proba: np.ndarray,
    threshold: float,
) -> dict:
    """
    Recall per ring archetype, at two levels, with each level's denominator named.

    One overall recall figure hides the only thing a risk lead will ask: which
    rings are we missing? The generator emits three archetypes of deliberately
    unequal difficulty, so reporting them separately turns a single number into
    an honest difficulty gradient.

    TWO RECALLS, TWO DENOMINATORS, AND THEY ARE NOT INTERCHANGEABLE
    ──────────────────────────────────────────────────────────────
      account_recall  flagged mule accounts / labelled mule accounts
      ring_recall     rings with >= 1 flagged member / rings in the split

    Ring-level recall counts a ring as detected if ANY member is flagged, which
    is closer to how this is used — one thread is enough to start pulling, and a
    ring where 2 of 6 accounts alert is a caught ring, not a 33% failure. The
    corollary is that ring recall SATURATES: it rises with ring size for a fixed
    per-account detection rate, so a high figure is partly a property of the
    generator's ring sizes and must never be quoted as if it were the account
    number.

    The two levels were previously reported as `detected/n_accounts` beside
    `rings_with_an_alert/n_rings` under a single "recall by archetype" heading,
    with the complements left to be derived by subtraction — so "we missed N"
    could be read off either column and the two were easy to mix, especially at
    the archetype level where a per-archetype ring shortfall and a global escaped
    count look identical on the page. Every count now ships with its denominator
    in its key name and its complement alongside it (`accounts_missed`,
    `rings_with_no_alert`), per archetype AND overall, so no consumer has to do
    arithmetic across the two levels to state a failure. models/report.py's
    "produced no alert at all" paragraph is exactly that arithmetic
    (`n_rings - rings_with_an_alert`, printed beside a list of archetype names);
    the per-archetype field is published so it need not be inferred.

    WHAT THIS DOES ABOUT ACCOUNTS THAT BELONG TO MORE THAN ONE RING
    ──────────────────────────────────────────────────────────────
    `ring_id` and `ring_type` are an ATTRIBUTION, not a membership list.
    data/extractor.py's `label_nodes` walks the ring edges and keeps the FIRST
    ring each account appears in, so an account bridging two rings is counted
    under one of them and is invisible to the other. Its own comment calls the
    situation "pathological"; on the shipped data it is simply uncommon.

    Two consequences, both real:

      • An archetype's ACCOUNT denominator is short by the members it lent to
        another archetype, and the borrowing archetype's is long by the same
        accounts.

      • A ring's membership as seen here can be smaller than the ring the
        generator built. A ring whose only alerting account was attributed to
        another ring therefore reads as escaped when an analyst would in fact
        have been looking at it, so ring recall carries slack in that direction
        — worth stating next to a figure like 23/24, which is otherwise easy to
        read as exact.

    HOW BIG THAT SLACK IS, IS MEASURED AND NOT REMEMBERED
    ────────────────────────────────────────────────────
    `ring_attribution.measured` in the returned dict carries the counts for the
    split in hand: positives in more than one ring, positives bridging two
    archetypes, and the exact number of rings short a member. See
    `_attribution_slack`.

    This paragraph used to quote those counts as literals, and so did the string
    this function wrote into metrics.json. They had been measured once by hand on
    an earlier dataset and were wrong by the time anyone read them — they said
    119 test positives against an actual 124 — while sitting inside the artefact
    whose entire claim is that its numbers are derived. Nothing here quotes a
    count any more; `label_nodes` records full membership and this function
    counts it.

    No ring disappears entirely under this attribution on the shipped data — every
    ring keeps at least one member, so the ring DENOMINATOR is complete even where
    membership is short, and `ring_id` determines `ring_type` in all three splits.
    Both properties are checked below rather than trusted, because neither is
    guaranteed by anything.
    """
    flagged = np.asarray(y_proba) >= threshold
    positives = (df[TARGET_COL] == 1).to_numpy()
    n_positives = int(positives.sum())

    # Both label columns must be present and real on every positive, checked here
    # by name. Otherwise the failures are silent or misleading: pandas' groupby
    # drops null keys by default, so a missing `ring_id` would shrink the ring
    # population without shrinking anything that reads like a denominator; a
    # missing `ring_type` would place an account in no archetype at all and then
    # surface as a TypeError from sorted() comparing str to nan; and the `-1`
    # non-member sentinel on a positive would be grouped as if it were one extra
    # ring shared by every such account.
    labels = df.loc[positives, ["ring_id", "ring_type"]]
    missing = labels.isna().any(axis=1)
    sentinel = labels["ring_id"].notna() & (labels["ring_id"] < 0)
    if bool(missing.any()) or bool(sentinel.any()):
        raise RuntimeError(
            "labelled mules with unusable ring labels — cannot report recall by "
            "archetype:\n"
            f"  {int(missing.sum())} positive(s) missing ring_id or ring_type\n"
            f"  {int(sentinel.sum())} positive(s) carrying the ring_id < 0 "
            "non-member sentinel\n"
            "  Every account with is_mule == 1 is a member of exactly one "
            "attributed ring; check data/extractor.py's label_nodes."
        )

    per_archetype = {}
    for archetype in sorted(df.loc[positives, "ring_type"].unique()):
        mask = positives & (df["ring_type"] == archetype).to_numpy()
        n = int(mask.sum())
        detected = int(flagged[mask].sum())
        per_archetype[str(archetype)] = {
            # The original three keys, unchanged: models/report.py reads them.
            "n_accounts": n,
            "detected": detected,
            "account_recall": round(detected / n, 4) if n else None,
            # The same figures under names that carry their denominator, plus the
            # complement, so neither has to be inferred from the ring block.
            "n_mule_accounts": n,
            "accounts_flagged": detected,
            "accounts_missed": n - detected,
            "denominator": "labelled mule accounts attributed to this archetype",
            "mean_score": round(float(np.asarray(y_proba)[mask].mean()), 4),
            "median_score": round(float(np.median(np.asarray(y_proba)[mask])), 4),
        }

    rings = df.loc[positives, ["ring_id", "ring_type"]].assign(
        _flagged=flagged[positives]
    )
    # Grouped by ring_id ALONE. This used to group by ["ring_type", "ring_id"],
    # which counts a ring once per archetype label attached to it — so a single
    # inconsistent ring_type would inflate the ring denominator and deflate ring
    # recall without anything looking wrong. One ring is one group by definition;
    # the archetype is a property OF the ring, checked here rather than assumed.
    types_per_ring = rings.groupby("ring_id")["ring_type"].nunique()
    if (types_per_ring > 1).any():
        offenders = types_per_ring[types_per_ring > 1].to_dict()
        raise RuntimeError(
            "ring_id does not determine ring_type — one ring is carrying two "
            f"archetype labels: {offenders}.\n"
            "  Ring-level recall would count those rings once per label, so its "
            "denominator would not be 'the rings in this split'.\n"
            "  Fix data/extractor.py's label_nodes or the generator's ring_type "
            "assignment before trusting any archetype figure."
        )

    by_ring = rings.groupby("ring_id")
    ring_hits = by_ring["_flagged"].any()
    ring_frac = by_ring["_flagged"].mean()
    ring_archetype = by_ring["ring_type"].first()
    n_rings = int(ring_hits.size)

    per_archetype_rings = {}
    for archetype in sorted(ring_archetype.unique()):
        sel = (ring_archetype == archetype).to_numpy()
        hits = ring_hits[sel]
        per_archetype_rings[str(archetype)] = {
            "n_rings": int(hits.size),
            "rings_with_an_alert": int(hits.sum()),
            "rings_with_no_alert": int(hits.size - hits.sum()),
            "ring_recall": round(float(hits.mean()), 4) if hits.size else None,
            "mean_share_of_ring_flagged": round(
                float(ring_frac[sel].mean()), 4),
            "denominator": (
                "rings of this archetype in the split; a ring counts as "
                "detected if at least one attributed member is flagged"
            ),
        }

    # Each level must account for its own population exactly — a backstop behind
    # the label precondition above, for a future change that filters archetypes or
    # rings on the way into either loop. If attribution ever drops an account or
    # splits a ring, these are the two sums that catch it, and a breakdown that
    # does not add up is worse than no breakdown, because it looks like a finding.
    accounts_counted = sum(v["n_mule_accounts"] for v in per_archetype.values())
    rings_counted = sum(v["n_rings"] for v in per_archetype_rings.values())
    if accounts_counted != n_positives or rings_counted != n_rings:
        raise RuntimeError(
            "archetype breakdown does not reconcile with its own inputs:\n"
            f"  accounts: archetypes sum to {accounts_counted}, split has "
            f"{n_positives} labelled mules\n"
            f"  rings:    archetypes sum to {rings_counted}, split has "
            f"{n_rings} distinct ring_ids\n"
            "  Every positive carries exactly one ring_id and one ring_type, so "
            "these must be equal; a gap means label_nodes left an account "
            "unattributed."
        )

    accounts_flagged = int(flagged[positives].sum())
    return {
        "by_archetype_accounts": per_archetype,
        "by_archetype_rings": per_archetype_rings,
        "overall_ring_recall": round(float(ring_hits.mean()), 4) if n_rings else None,
        "n_rings": n_rings,
        "rings_with_an_alert": int(ring_hits.sum()),
        # Complements and the account-level totals, so the two levels can be
        # quoted side by side without either being derived from the other.
        "rings_with_no_alert": n_rings - int(ring_hits.sum()),
        "n_mule_accounts": n_positives,
        "accounts_flagged": accounts_flagged,
        "accounts_missed": n_positives - accounts_flagged,
        "overall_account_recall": (round(accounts_flagged / n_positives, 4)
                                   if n_positives else None),
        "denominators": {
            "account_recall": (
                "flagged mule accounts / labelled mule accounts in the split"
            ),
            "ring_recall": (
                "rings with >= 1 flagged member / distinct ring_ids in the split"
            ),
            "warning": (
                "different denominators — never subtract or average across the "
                "two levels. Ring recall saturates with ring size, so it is the "
                "operationally useful figure and the flattering one at once."
            ),
        },
        "ring_attribution": {
            "rule": (
                "data.extractor.label_nodes keeps the FIRST ring each account "
                "appears in, so ring_id/ring_type is an attribution, not full "
                "membership"
            ),
            "consequence": (
                "an account in two rings is counted under one of them; a ring "
                "detected only through such an account can read as escaped, and "
                "archetype account counts can be short or long by those accounts"
            ),
            # MEASURED on the split in hand, from the full membership sets
            # data.extractor.label_nodes now carries. This used to be a sentence
            # of hand-typed counts ("4 of 119 positives…") republished verbatim
            # every run, inside the artefact whose whole point is that its
            # numbers are derived. The test split has 124 positives, not 119.
            "measured": _attribution_slack(df, positives),
        },
    }


def _attribution_slack(df: pd.DataFrame, positives: np.ndarray) -> dict | None:
    """
    How much ring recall's denominator loses to first-ring attribution.

    Returns None — absence, not zero — when the feature table predates the
    `rings_attributed` / `ring_types_attributed` columns, because "no accounts
    bridge two rings" and "we cannot tell" are different findings and only one of
    them should be publishable as a reassurance.

    `rings_short_a_member` is exact rather than a bound: a ring loses exactly
    those accounts that are endpoints of its edges while attributed elsewhere,
    and both facts are in the two columns.
    """
    needed = ("rings_attributed", "ring_types_attributed")
    if any(col not in df.columns for col in needed):
        return None

    rows = df.loc[positives, ["ring_id", *needed]]
    # Read back from CSV, an empty cell is NaN; positives always have a ring, so
    # this only guards the type.
    ring_sets = rows["rings_attributed"].fillna("").astype(str)
    type_sets = rows["ring_types_attributed"].fillna("").astype(str)

    per_account_rings = [
        {int(r) for r in cell.split("|") if r} for cell in ring_sets
    ]
    n_types = [len([t for t in cell.split("|") if t]) for cell in type_sets]

    # Rings that own an edge endpoint they were not credited with.
    shorted: set[int] = set()
    for attributed, rings in zip(rows["ring_id"].astype(int), per_account_rings):
        shorted |= (rings - {attributed})

    multi = sum(1 for rings in per_account_rings if len(rings) > 1)
    return {
        "n_positives": int(len(rows)),
        "positives_in_more_than_one_ring": int(multi),
        "positives_bridging_two_archetypes": int(sum(1 for k in n_types if k > 1)),
        "rings_short_a_member": int(len(shorted)),
        "note": (
            "rings_short_a_member is the count of distinct rings that own an "
            "edge endpoint credited to a different ring; ring recall carries up "
            "to that many rings of slack in the escaped-looking direction"
        ),
    }


# ══════════════════════════════════════════════════════════════════
# Operating points
# ══════════════════════════════════════════════════════════════════

def select_operating_points(
    model,
    frames: dict[str, pd.DataFrame],
    evaluator: CostEvaluator,
    alert_budget: float,
) -> dict:
    """
    Pick both thresholds using ONLY validation data, and freeze them.

    Two are needed because they answer different questions. The cost-optimal
    threshold minimises FN·₹200,000 + FP·₹15,000 and is the right answer to
    "what does the cost model imply?" — but at a 13.3:1 ratio it typically flags
    around one account in five, which no review team can staff. The
    capacity-constrained threshold answers "what is the most recall we can buy
    with the analysts we have?". Reporting only the first would be arithmetically
    correct and operationally dishonest.
    """
    print(banner("Threshold Selection (validation split)"))

    X_val = frames["val"][FEATURE_COLS]
    y_val = frames["val"][TARGET_COL].to_numpy()
    p_val = score(model, X_val)

    val_auc = roc_auc(y_val, p_val)
    val_ap = average_precision(y_val, p_val)
    optimal = evaluator.find_optimal_threshold(y_val, p_val)
    budgeted = evaluator.threshold_for_alert_budget(y_val, p_val, alert_budget)

    print(f"  Validation ROC-AUC {val_auc:.4f} | average precision {val_ap:.4f} "
          f"(trivial AP = {y_val.mean():.4f})")
    print(f"  Break-even p* from the cost model: "
          f"{evaluator.break_even_probability:.4f}")
    # _fmt everywhere a precision/recall/F1 is printed: these are None at a
    # degenerate operating point, and a training run must not die inside a print
    # statement after the expensive part succeeded.
    print(f"  {sym('bullet')} cost-optimal threshold   {optimal.threshold:.4f}  "
          f"P {_fmt(optimal.precision)} R {_fmt(optimal.recall)} | "
          f"{optimal.alerts_per_1000:.1f} alerts/1,000")
    width = optimal.plateau_width
    if width is not None:
        verdict = ("wide — robust" if width > 0.05
                   else "NARROW — may be fitted to noise")
        print(f"      plateau [{optimal.plateau_lo:.4f}, "
              f"{optimal.plateau_hi:.4f}] width {width:.4f} ({verdict})")
    print(f"  {sym('bullet')} capacity threshold       {budgeted.threshold:.4f}  "
          f"P {_fmt(budgeted.precision)} R {_fmt(budgeted.recall)} | "
          f"{budgeted.alerts_per_1000:.1f} alerts/1,000 "
          f"(budget {alert_budget:.0f})")
    print(f"  {sym('arrow')} both thresholds are now frozen and applied blind "
          "to test.")

    return {
        "auc": float(val_auc),
        "average_precision": float(val_ap),
        "cost_optimal": optimal,
        "capacity_constrained": budgeted,
        "_scores": p_val,          # underscore key: never serialised
    }


def evaluate_on_test(
    model,
    frames: dict[str, pd.DataFrame],
    evaluator: CostEvaluator,
    thresholds: dict,
    alert_budget: float,
) -> dict:
    """
    Evaluate once on test, at thresholds fixed on validation.

    Also reports what the test-optimal threshold *would* have been. That number
    is a diagnostic — the size of the selection bias v2 was reporting as
    performance — and must never be quoted or written to `optimal_threshold`.
    """
    print(banner("Held-out Test Evaluation"))

    df = frames["test"]
    X_test, y_test = df[FEATURE_COLS], df[TARGET_COL].to_numpy()
    p_test = score(model, X_test)

    threshold = float(thresholds["cost_optimal"].threshold)
    capacity_threshold = float(thresholds["capacity_constrained"].threshold)

    auc = roc_auc(y_test, p_test)
    ap = average_precision(y_test, p_test)
    # Clustered on rings: the published interval. The account-level interval is
    # computed too, but only so the printout can show how much narrower an
    # unclustered bootstrap would have claimed — passing singleton clusters is
    # exactly the old i.i.d. resample.
    n_rings_test = int(df.loc[df["ring_id"] >= 0, "ring_id"].nunique())
    ci_lo, ci_hi = bootstrap_auc_ci(y_test, p_test, bootstrap_clusters(df))
    naive_lo, naive_hi = bootstrap_auc_ci(
        y_test, p_test, np.arange(y_test.size).astype(str)
    )

    at_threshold = evaluator.evaluate_at_threshold(y_test, p_test, threshold)
    at_capacity = evaluator.evaluate_at_threshold(y_test, p_test,
                                                  capacity_threshold)
    oracle = evaluator.find_optimal_threshold(y_test, p_test)

    print(f"  ROC-AUC {auc:.4f} (95% CI {ci_lo:.4f}-{ci_hi:.4f}, "
          f"{BOOTSTRAP_ROUNDS} bootstrap resamples over "
          f"{n_rings_test} rings + singleton non-members)")
    naive_width = naive_hi - naive_lo
    width = ci_hi - ci_lo
    print(f"  [honesty] an i.i.d. ACCOUNT bootstrap would report "
          f"{naive_lo:.4f}-{naive_hi:.4f} — {width / max(naive_width, 1e-12):.2f}x "
          f"narrower than the evidence supports, because it treats the "
          f"{int((df[TARGET_COL] == 1).sum())} members of "
          f"{n_rings_test} rings as that many independent observations. The "
          "published interval is the clustered one.")
    print(f"  Average precision {ap:.4f} — lift of "
          f"{ap / max(y_test.mean(), 1e-12):.1f}x over the trivial "
          f"{y_test.mean():.4f}")
    print()
    print(f"  ── at the validation-selected cost-optimal threshold "
          f"({threshold:.4f}) ──")
    print_confusion_table(at_threshold.tp, at_threshold.fp,
                          at_threshold.tn, at_threshold.fn)
    print(f"    FN cost {sym('rupee')}"
          f"{at_threshold.fn * evaluator.config.fn_cost:,.0f} + FP cost "
          f"{sym('rupee')}{at_threshold.fp * evaluator.config.fp_cost:,.0f} = "
          f"{sym('rupee')}{at_threshold.total_cost:,.0f} "
          f"({sym('rupee')}{at_threshold.cost_per_prediction:,.2f}/account)")
    print(f"    Analyst load {at_threshold.alerts_per_1000:.1f} alerts per "
          f"1,000 accounts")
    print()
    print(f"  ── at the capacity threshold ({capacity_threshold:.4f}, budget "
          f"{alert_budget:.0f}/1,000) ──")
    print(f"    P {_fmt(at_capacity.precision)} R {_fmt(at_capacity.recall)} "
          f"F1 {_fmt(at_capacity.f1)} | {at_capacity.alerts_per_1000:.1f} "
          f"alerts/1,000 | {sym('rupee')}{at_capacity.total_cost:,.0f}")

    gap = at_threshold.total_cost - oracle.total_cost
    print()
    print(f"  [diagnostic] test-optimal threshold would have been "
          f"{oracle.threshold:.4f} (F1 {_fmt(oracle.f1)}, "
          f"{sym('rupee')}{oracle.total_cost:,.0f}).")
    print(f"  [diagnostic] cost of not having peeked at test: "
          f"{sym('rupee')}{gap:,.0f}. Reported metrics use the honest "
          "threshold.")

    breakdown = archetype_breakdown(df, p_test, threshold)
    print()
    # Both denominators spelled out in the heading. The two columns are recalls
    # over different populations (mule accounts vs rings) and the ring one
    # saturates with ring size, so a bare "recall by archetype" invites reading
    # one as a restatement of the other.
    print("  ── recall by ring archetype ──")
    print("     accounts = mule accounts flagged / mule accounts attributed to "
          "the archetype")
    print("     rings    = rings with >= 1 flagged member / rings of that "
          "archetype")
    for archetype, stats in breakdown["by_archetype_accounts"].items():
        rings = breakdown["by_archetype_rings"][archetype]
        print(f"    {archetype:<16s} accounts {stats['accounts_flagged']:>3d}/"
              f"{stats['n_mule_accounts']:<3d} = "
              f"{_fmt(stats['account_recall'], '.0%')}"
              f"   rings {rings['rings_with_an_alert']:>2d}/"
              f"{rings['n_rings']:<2d} = {_fmt(rings['ring_recall'], '.0%')}"
              f"   silent rings {rings['rings_with_no_alert']:>2d}"
              f"   mean score {stats['mean_score']:.3f}")
    print(f"    {'ALL':<16s} accounts {breakdown['accounts_flagged']:>3d}/"
          f"{breakdown['n_mule_accounts']:<3d} = "
          f"{_fmt(breakdown['overall_account_recall'], '.0%')}"
          f"   rings {breakdown['rings_with_an_alert']:>2d}/"
          f"{breakdown['n_rings']:<2d} = "
          f"{_fmt(breakdown['overall_ring_recall'], '.0%')}"
          f"   silent rings {breakdown['rings_with_no_alert']:>2d}")
    print("     [caveat] ring membership here is first-ring-wins attribution "
          "(data.extractor.label_nodes),")
    print("              so a ring caught only through an account credited to "
          "another ring reads as silent.")

    calibration = calibration_report(y_test, p_test)
    print()
    print(f"  Calibration: Brier {calibration['brier_score']:.4f}, ECE "
          f"{calibration['expected_calibration_error']:.4f}, mean predicted "
          f"{calibration['mean_predicted_probability']:.4f} vs actual "
          f"prevalence {calibration['actual_prevalence']:.4f}")

    return {
        "auc": float(auc),
        # PUBLISHED interval: clusters = rings. See bootstrap_auc_ci for why an
        # account-level resample overstates the precision of this number.
        "auc_ci_95": [ci_lo, ci_hi],
        "auc_ci_95_method": (
            f"cluster bootstrap, {BOOTSTRAP_ROUNDS} rounds, resampling "
            f"{n_rings_test} rings and each non-member account as a singleton"
        ),
        "auc_ci_95_iid_accounts_diagnostic": {
            "ci": [naive_lo, naive_hi],
            "warning": (
                "DIAGNOSTIC ONLY — assumes ring members are independent "
                "observations, which they are not. Recorded to show the size of "
                "the understatement; never quote this interval."
            ),
        },
        "average_precision": float(ap),
        "average_precision_trivial": round(float(y_test.mean()), 6),
        "at_selected_threshold": at_threshold.as_dict(),
        "at_capacity_threshold": at_capacity.as_dict(),
        "oracle_threshold_diagnostic": at_oracle_dict(oracle, gap),
        "recall_breakdown": breakdown,
        "calibration": calibration,
        "_scores": p_test,          # popped before serialisation
    }


def at_oracle_dict(oracle, gap: float) -> dict:
    """The oracle block, labelled so it cannot be mistaken for a result."""
    payload = oracle.as_dict()
    payload["cost_of_not_peeking_at_test"] = round(float(gap), 2)
    payload["warning"] = (
        "DIAGNOSTIC ONLY — this threshold was chosen using test labels. "
        "Never quote these numbers as performance."
    )
    return payload


# ══════════════════════════════════════════════════════════════════
# Baselines
# ══════════════════════════════════════════════════════════════════

def _rule_text(feature: str, direction: str, threshold: float) -> str:
    """The baseline rule as something a reader can check by hand."""
    if direction == "high":
        return f"{feature} >= {threshold:.4f}"
    # score = -x, so score >= t is the rule x <= -t
    return f"{feature} <= {-threshold:.4f}"


def single_feature_rule_baseline(
    frames: dict[str, pd.DataFrame],
    evaluator: CostEvaluator,
    criterion: str = "cost",
    screen_split: str = FEATURE_DECISION_SPLIT,
) -> dict:
    """
    The best possible one-line `if` statement, selected on validation.

    This is the baseline that matters. A graph pipeline, a gradient-boosted
    ensemble and a cost matrix have to beat `if cycle_participation >= 0.45:
    review` before any of the complexity is justified, and until someone computes
    that rule nobody knows whether they do.

    Every feature is tried in both directions — some signals are inverted, and
    `amount_cv` is one of them — with the threshold chosen on validation and
    applied blind to test, exactly as the model's is. Selecting by `cost` matches
    the model's own objective; `f1` is also offered because it is the more
    familiar quantity and the two do not always agree.

    TWO DIFFERENT USES OF SINGLE-FEATURE AUC, KEPT APART
    ────────────────────────────────────────────────────
    `screen_split` (default validation, per models/features.py rule 4) is the
    split the LEAKAGE GATE reads. That gate is a decision — it can condemn the
    dataset and send someone back to data/generator.py — so it must not be
    measured on test; doing so spends the final evaluation on feature selection.

    The per-feature `test_auc` column and the `strongest_single_feature_test_auc`
    block are a different thing: a post-hoc DESCRIPTION of the shipped test data,
    published so tests/test_baselines.py can reconcile the claim in metrics.json
    against the CSVs in the repo. They are reported, never acted on. Both figures
    carry their split in the name so neither can be quoted as the other.
    """
    if criterion not in ("cost", "f1"):
        raise ValueError(f"criterion must be 'cost' or 'f1', got {criterion!r}")
    if screen_split not in frames:
        raise ValueError(
            f"screen_split must be one of {sorted(frames)}, got {screen_split!r}"
        )
    if screen_split == "test":
        raise ValueError(
            "refusing to screen the leakage ceiling on test: that gate decides "
            "whether the dataset is usable, and a decision read off the final "
            "split makes the final evaluation a measurement of its own selection."
        )

    y_val = frames["val"][TARGET_COL].to_numpy()
    y_test = frames["test"][TARGET_COL].to_numpy()
    y_screen = frames[screen_split][TARGET_COL].to_numpy()
    rows = []

    for feature in FEATURE_COLS:
        x_val = frames["val"][feature].to_numpy(dtype=float)
        x_test = frames["test"][feature].to_numpy(dtype=float)
        x_screen = frames[screen_split][feature].to_numpy(dtype=float)
        for direction, sign in (("high", 1.0), ("low", -1.0)):
            s_val, s_test = sign * x_val, sign * x_test

            if criterion == "cost":
                chosen = evaluator.find_optimal_threshold(y_val, s_val)
                threshold = float(chosen.threshold)
            else:
                curve = evaluator.sweep(y_val, s_val)
                threshold = float(curve.loc[curve["f1"].idxmax(), "threshold"])
                chosen = evaluator.evaluate_at_threshold(y_val, s_val, threshold)

            test_report = evaluator.evaluate_at_threshold(y_test, s_test,
                                                          threshold)
            rows.append({
                "feature": feature,
                "direction": direction,
                "rule": _rule_text(feature, direction, threshold),
                "threshold_on_score": round(threshold, 6),
                "val_cost": float(chosen.total_cost),
                # Left unrounded (it is a sort key when criterion == "f1"), but
                # passed through as-is because it may be None. pandas stores that
                # as NaN, which sort_values puts last — the right place for a rule
                # whose F1 could not be computed.
                "val_f1": chosen.f1,
                "screen_auc": float(roc_auc(y_screen, sign * x_screen)),
                "test_auc": float(roc_auc(y_test, s_test)),
                # _round_or_none, not round(float(...)): a rule whose threshold
                # flags nothing has no precision, and float(None) is a TypeError
                # that would kill the run 36 rules deep.
                "test_precision": _round_or_none(test_report.precision, 4),
                "test_recall": _round_or_none(test_report.recall, 4),
                "test_f1": _round_or_none(test_report.f1, 4),
                "test_total_cost": float(test_report.total_cost),
                "test_alerts_per_1000": round(
                    float(test_report.alerts_per_1000), 1),
            })

    table = pd.DataFrame(rows)
    # Rank on the SELECTION split only. Sorting this table by test_f1 would be
    # the same selection-on-test mistake the model's threshold no longer makes.
    key = "val_cost" if criterion == "cost" else "val_f1"
    table = table.sort_values(key, ascending=(criterion == "cost"))
    best = table.iloc[0]

    # The single-feature AUC ceiling. A feature that separates the classes this
    # cleanly was planted by the generator, not learned.
    #
    # Direction-corrected on purpose: a feature with AUC 0.13 is not weak, it is
    # strongly inverted, and ranking on the raw value would report 0.13 against a
    # 0.99 ceiling and call a near-perfect predictor harmless. `max(auc, 1 - auc)`
    # is the discriminative power available to a model that can flip a sign — which
    # every model here can.
    #
    # Computed twice, on purpose, and labelled: `screen` is the gate (validation),
    # `test` is the published description reconciled by tests/test_baselines.py.
    def strongest_on(auc_col: str) -> pd.Series:
        discriminative = np.maximum(table[auc_col], 1.0 - table[auc_col])
        return (table.assign(discriminative_auc=discriminative)
                .sort_values("discriminative_auc", ascending=False).iloc[0])

    screened = strongest_on("screen_auc")
    strongest = strongest_on("test_auc")

    return {
        "selected_on": "validation",
        "criterion": criterion,
        "rule": str(best["rule"]),
        "feature": str(best["feature"]),
        # The rule's machine-readable form. Present so a consumer can REPLAY the
        # baseline instead of quoting a cost that was priced at one FN/FP ratio:
        # `sign * x >= threshold_on_score`, sign = +1 for "high", -1 for "low".
        # dashboard/components/cost_slider.py uses these to re-price the baseline
        # whenever a viewer changes the cost inputs, so the comparison stays like
        # for like — a fixed baseline cost beside a moving model cost would
        # silently flatter whichever side the prices happened to favour.
        "direction": str(best["direction"]),
        "threshold_on_score": float(best["threshold_on_score"]),
        # _none_if_nan on the way out: the row above may have stored None, and
        # pandas keeps None as NaN in a float column. Round-tripping through the
        # frame must not turn "not computable" into a number.
        "test_precision": _none_if_nan(best["test_precision"]),
        "test_recall": _none_if_nan(best["test_recall"]),
        "test_f1": _none_if_nan(best["test_f1"]),
        "test_total_cost": float(best["test_total_cost"]),
        "test_alerts_per_1000": float(best["test_alerts_per_1000"]),
        "top_10_rules": table.head(10).to_dict(orient="records"),
        # THE GATE. Screened on validation (models/features.py rule 4); this is
        # the block whose AUC is allowed to condemn the dataset.
        "leakage_screen": {
            "split": screen_split,
            "feature": str(screened["feature"]),
            "auc": round(float(screened["discriminative_auc"]), 4),
            "raw_auc": round(float(screened["screen_auc"]), 4),
            "inverted": bool(screened["screen_auc"] < 0.5),
            "ceiling": LEAKAGE_AUC_CEILING,
            "note": (
                "Decision gate, measured on the split named above so that "
                "acting on it does not spend the test split. Compare with "
                "strongest_single_feature_test_auc, which is descriptive only."
            ),
        },
        # DESCRIPTIVE. Same statistic on test, published so
        # tests/test_baselines.py can reconcile metrics.json against the shipped
        # CSVs. Never an input to a decision — see the docstring.
        "strongest_single_feature_test_auc": {
            "feature": str(strongest["feature"]),
            "auc": round(float(strongest["discriminative_auc"]), 4),
            "raw_auc": round(float(strongest["test_auc"]), 4),
            "inverted": bool(strongest["test_auc"] < 0.5),
            "ceiling": LEAKAGE_AUC_CEILING,
        },
    }


def logistic_regression_baseline(
    frames: dict[str, pd.DataFrame],
    evaluator: CostEvaluator,
) -> dict:
    """
    A linear model on identical features: does the model need to be non-linear?

    Deliberately given a fair chance rather than set up to fail — heavy-tailed
    magnitudes are log-scaled and everything is standardised on train statistics,
    because a logistic regression fed raw rupee sums is a straw man and beating a
    straw man proves nothing.

    Imported lazily so a missing scikit-learn costs one baseline rather than the
    whole training run.
    """
    try:
        from sklearn.linear_model import LogisticRegression
    except ImportError:                                   # pragma: no cover
        print(f"  {sym('warn')} scikit-learn not installed; skipping the "
              "logistic-regression baseline.")
        return {"skipped": "scikit-learn not installed"}

    def design(df: pd.DataFrame) -> np.ndarray:
        out = df[FEATURE_COLS].to_numpy(dtype=float).copy()
        for j, col in enumerate(FEATURE_COLS):
            if col in LOG1P_COLS and out[:, j].min() >= 0:
                out[:, j] = np.log1p(out[:, j])
        return out

    # The LOG1P_COLS criterion, re-derived from the data rather than trusted. This
    # is a warning and not a failure — a shifted skew on regenerated data is not
    # grounds for refusing to train — but it means the next omission announces
    # itself instead of quietly weakening the baseline, which is how degree_ratio
    # (skew 9.39) stayed out of the list.
    train_skew = frames["train"][FEATURE_COLS].skew()
    implied = {c for c in FEATURE_COLS
               if frames["train"][c].min() >= 0 and train_skew[c] >= 1.5}
    if implied != set(LOG1P_COLS):
        missing = sorted(implied - set(LOG1P_COLS))
        extra = sorted(set(LOG1P_COLS) - implied)
        print(f"  {sym('warn')} LOG1P_COLS no longer matches its stated "
              f"criterion (non-negative, train skew >= 1.5): "
              f"missing {missing or 'none'}, unjustified {extra or 'none'}. "
              "The logistic-regression baseline is the thing the headline lift "
              "is measured against, so a hand-curated list that drifts from its "
              "rule flatters the model.")

    Z = {s: design(frames[s]) for s in SPLITS}
    mean = Z["train"].mean(axis=0)
    std = Z["train"].std(axis=0)
    std[std == 0] = 1.0                    # a constant column contributes nothing
    Z = {s: (z - mean) / std for s, z in Z.items()}
    y = {s: frames[s][TARGET_COL].to_numpy() for s in SPLITS}

    lr = LogisticRegression(
        max_iter=5_000,
        class_weight="balanced",           # the LR analogue of scale_pos_weight
        random_state=RANDOM_SEED,
    )
    lr.fit(Z["train"], y["train"])

    p_val = lr.predict_proba(Z["val"])[:, 1]
    p_test = lr.predict_proba(Z["test"])[:, 1]
    chosen = evaluator.find_optimal_threshold(y["val"], p_val)
    report = evaluator.evaluate_at_threshold(y["test"], p_test,
                                             chosen.threshold)

    coefficients = dict(sorted(
        zip(FEATURE_COLS, (float(c) for c in lr.coef_[0])),
        key=lambda kv: -abs(kv[1]),
    ))
    return {
        "selected_on": "validation",
        "threshold": round(float(chosen.threshold), 6),
        "test_auc": round(float(roc_auc(y["test"], p_test)), 4),
        "test_average_precision": round(
            float(average_precision(y["test"], p_test)), 4),
        "test_precision": _round_or_none(report.precision, 4),
        "test_recall": _round_or_none(report.recall, 4),
        "test_f1": _round_or_none(report.f1, 4),
        "test_total_cost": float(report.total_cost),
        "test_alerts_per_1000": round(float(report.alerts_per_1000), 1),
        "preprocessing": ("log1p on non-negative columns with train skew "
                          ">= 1.5, then train-fit z-score"),
        "log1p_columns": sorted(LOG1P_COLS),
        "coefficients_standardised": {k: round(v, 4)
                                      for k, v in coefficients.items()},
        # Underscore-prefixed: handed to capacity_fair_comparison so this
        # baseline can be re-thresholded under the analyst budget, and stripped
        # by build_metrics_payload before serialisation. Two float vectors per
        # split do not belong in metrics.json.
        "_val_scores": p_val,
        "_test_scores": p_test,
    }


def graph_ablation_baseline(
    frames: dict[str, pd.DataFrame],
    evaluator: CostEvaluator,
) -> dict:
    """
    The same model with every graph feature removed.

    This is the ablation the project stands or falls on. Everything left in
    NON_GRAPH_FEATURES is computable per account from a flat ledger with a GROUP
    BY — no NetworkX, no Louvain, no cycle search. If the gap is small, the
    graph is decoration and the honest thing to do is say so; if it is large,
    that gap is the contribution, quantified in rupees.
    """
    y = {s: frames[s][TARGET_COL] for s in SPLITS}
    X = {s: frames[s][NON_GRAPH_FEATURES] for s in SPLITS}

    model = train_model(X["train"], y["train"], X["val"], y["val"],
                        label="ablation (no graph features)")
    p_val = score(model, X["val"])
    p_test = score(model, X["test"])
    chosen = evaluator.find_optimal_threshold(y["val"].to_numpy(), p_val)
    report = evaluator.evaluate_at_threshold(y["test"].to_numpy(), p_test,
                                            chosen.threshold)

    return {
        "selected_on": "validation",
        "features_removed": list(GRAPH_FEATURES),
        "n_features": len(NON_GRAPH_FEATURES),
        "threshold": round(float(chosen.threshold), 6),
        "test_auc": round(float(roc_auc(y["test"].to_numpy(), p_test)), 4),
        "test_average_precision": round(
            float(average_precision(y["test"].to_numpy(), p_test)), 4),
        "test_precision": _round_or_none(report.precision, 4),
        "test_recall": _round_or_none(report.recall, 4),
        "test_f1": _round_or_none(report.f1, 4),
        "test_total_cost": float(report.total_cost),
        "test_alerts_per_1000": round(float(report.alerts_per_1000), 1),
        # See the note in logistic_regression_baseline: underscore keys are
        # stripped before metrics.json is written.
        "_val_scores": p_val,
        "_test_scores": p_test,
    }


def trivial_baselines(y_test: np.ndarray, evaluator: CostEvaluator) -> dict:
    """
    Flag nobody / flag everybody — the floor any result must clear.

    The metrics are taken from `cost_matrix._prf` rather than written out here,
    because these two policies are exactly where the degenerate cases live and
    hand-arithmetic got them wrong: flag-nothing published precision 0.0 and F1
    0.0 for an alert queue it never opened, and flag-everything hard-coded recall
    1.0, which is a false claim on a split with no positives. Routing both through
    the one helper means the null-versus-zero rule is decided in one place.

    `flag_nothing` is the case that made the convention matter. It is published as
    the comparison floor, and tests/test_baselines.py prices the model against it
    — so a fabricated 0.0 precision on the baseline flattered the model in the one
    table that exists to keep it honest.
    """
    from models.cost_matrix import _prf

    n_pos = int(np.asarray(y_test).sum())
    n_neg = int(np.asarray(y_test).size - n_pos)
    if n_pos + n_neg == 0:
        raise ValueError(
            "cannot price trivial baselines on an empty test split: every "
            "figure here would be a zero standing in for a missing measurement."
        )
    flag_none = n_pos * evaluator.config.fn_cost
    flag_all = n_neg * evaluator.config.fp_cost

    # tp, fp, fn for each policy. Flag nothing catches nothing (fn = every
    # positive); flag everything catches everything and takes every negative as
    # a false positive.
    none_p, none_r, none_f1 = _prf(0, 0, n_pos)
    all_p, all_r, all_f1 = _prf(n_pos, n_neg, 0)

    return {
        "flag_nothing": {
            # precision and F1 are null: no alerts were raised, so there is no
            # queue whose purity could be measured. Recall is 0.0 and belongs
            # here — the split has positives and all of them were missed, which
            # is a measurement, not a gap.
            "test_precision": _round_or_none(none_p, 4),
            "test_recall": _round_or_none(none_r, 4),
            "test_f1": _round_or_none(none_f1, 4),
            "test_total_cost": float(flag_none), "test_alerts_per_1000": 0.0,
        },
        "flag_everything": {
            "test_precision": _round_or_none(all_p, 4),
            "test_recall": _round_or_none(all_r, 4),
            "test_f1": _round_or_none(all_f1, 4),
            "test_total_cost": float(flag_all),
            "test_alerts_per_1000": 1000.0,
        },
        "cheapest_trivial_policy_cost": float(min(flag_none, flag_all)),
    }


def compute_baselines(
    frames: dict[str, pd.DataFrame],
    evaluator: CostEvaluator,
    model_report: dict,
) -> dict:
    """Run every baseline and print the comparison table that goes in the README."""
    print(banner("Baselines (thresholds selected on validation)"))

    y_test = frames["test"][TARGET_COL].to_numpy()
    baselines = {
        "trivial": trivial_baselines(y_test, evaluator),
        "best_single_feature_rule_by_cost": single_feature_rule_baseline(
            frames, evaluator, criterion="cost"),
        "best_single_feature_rule_by_f1": single_feature_rule_baseline(
            frames, evaluator, criterion="f1"),
        "logistic_regression": logistic_regression_baseline(frames, evaluator),
        "xgboost_without_graph_features": graph_ablation_baseline(
            frames, evaluator),
    }

    rule_cost = baselines["best_single_feature_rule_by_cost"]
    rule_f1 = baselines["best_single_feature_rule_by_f1"]
    lr = baselines["logistic_regression"]
    ablation = baselines["xgboost_without_graph_features"]

    def line(name: str, block: dict) -> str:
        if "test_f1" not in block:
            return f"  {name:<34s} {'(skipped)':>10s}"
        # Right-aligned strings, not floats: flag-nothing's precision and F1 are
        # null, and "n/a" is what the comparison table should show there. Column
        # widths match the header below.
        return (f"  {name:<34s} {_fmt(block['test_precision']):>9s}"
                f"{_fmt(block['test_recall']):>8s}{_fmt(block['test_f1']):>8s}"
                f"{block['test_alerts_per_1000']:>9.1f}"
                f"  {sym('rupee')}{block['test_total_cost']:>13,.0f}")

    print(f"  {'':<34s}{'precision':>9s}{'recall':>8s}{'f1':>8s}"
          f"{'alerts/1k':>9s}{'total cost':>16s}")
    print(line("flag nothing", baselines["trivial"]["flag_nothing"]))
    print(line("flag everything", baselines["trivial"]["flag_everything"]))
    print(line(f"rule: {rule_cost['rule']}", rule_cost))
    if rule_f1["rule"] != rule_cost["rule"]:
        print(line(f"rule: {rule_f1['rule']}", rule_f1))
    print(line("logistic regression (same features)", lr))
    print(line("XGBoost, no graph features", ablation))
    print(line("XGBoost, full model  <-- THIS MODEL", model_report))

    screen = rule_cost["leakage_screen"]
    strongest = rule_cost["strongest_single_feature_test_auc"]
    print()
    # The gate reads the screening split (validation). The test figure is printed
    # beside it as a description, clearly labelled, so nobody mistakes the
    # published number for the one that decides anything.
    print(f"  Leakage gate ({screen['split']} split): strongest single feature "
          f"is {screen['feature']} AUC {screen['auc']:.4f}"
          + (f" (inverted; raw {screen['raw_auc']:.4f})"
             if screen["inverted"] else "")
          + f" — ceiling {screen['ceiling']}")
    print(f"  [descriptive] on test it is {strongest['feature']} "
          f"AUC {strongest['auc']:.4f} — reported for "
          "tests/test_baselines.py to reconcile, not acted on.")
    if screen["auc"] >= LEAKAGE_AUC_CEILING:
        print(f"  {sym('fail')} that feature alone essentially solves the task "
              "— the generator is planting the label. Fix data/generator.py; "
              "every metric above is meaningless until you do.")

    if "test_f1" in ablation:
        delta_cost = ablation["test_total_cost"] - model_report["test_total_cost"]
        delta_ap = (model_report.get("test_average_precision", float("nan"))
                    - ablation["test_average_precision"])
        print(f"  Graph features are worth {sym('rupee')}{delta_cost:,.0f} in "
              f"avoided cost and {delta_ap:+.4f} average precision on test.")
        baselines["graph_feature_value"] = {
            "cost_avoided_vs_no_graph": float(delta_cost),
            "average_precision_gain": round(float(delta_ap), 4),
        }

    return baselines


# ══════════════════════════════════════════════════════════════════
# One analyst budget, applied to every policy
# ══════════════════════════════════════════════════════════════════

def capacity_fair_comparison(
    frames: dict[str, pd.DataFrame],
    evaluator: CostEvaluator,
    baselines: dict,
    *,
    model_val_scores: np.ndarray,
    model_test_scores: np.ndarray,
    alert_budget: float,
) -> pd.DataFrame:
    """
    Every policy priced under ONE analyst budget, each threshold chosen on val.

    WHY THIS TABLE EXISTS: THE CAPPED COMPARISON WAS NOT LIKE FOR LIKE
    ─────────────────────────────────────────────────────────────────
    The capacity discussion published a capped MODEL against UNCAPPED baselines,
    and then reported that the model loses on cost to the one-line rule. Both
    halves of that were true and the comparison behind them was not: with a miss
    priced at ~13x a false alert, total cost is dominated by misses, so any policy
    free to flood the review queue wins on cost by construction. The rule bought
    its recall with roughly 12x the alerts the cap allows. Comparing it against a
    model held to the cap measures the cap, not the model.

    So this table applies the SAME budget to every policy. Note what that is and
    is not: it does not weaken the baseline to flatter the model — the rule's
    uncapped cost stays published in the main baseline table, right beside this
    one, and a reader can compare either way. It removes an allowance the
    baseline was never entitled to under the stated operating constraint.

    SELECTION IS STILL VALIDATION-ONLY
    ──────────────────────────────────
    Each policy's budget threshold comes from `threshold_for_alert_budget` on
    VALIDATION and is then evaluated once on test at that frozen value — the same
    discipline as the headline and the cost-ratio table. A consequence worth
    reading rather than hiding: the frozen threshold can spill over the budget on
    test, because the score distribution moved and the threshold was fixed before
    test was seen. `test_within_budget` reports that per row instead of quietly
    re-fitting the threshold until the constraint holds, which would be selection
    on test wearing a capacity argument as a disguise.

    The two trivial policies are priced closed-form and kept in the table on
    purpose. Flag-nothing is the cost floor any real policy must beat;
    flag-everything is marked infeasible, because 1,000 alerts per 1,000 accounts
    is not a queue any budget below that permits, and a reader should see the
    excluded policy rather than wonder whether it was quietly dropped.

    WHAT `feasible_under_budget` MEANS, AND WHY IT IS COMPUTED
    ─────────────────────────────────────────────────────────
    True exactly when the policy forms a NON-EMPTY review queue that fits inside
    the budget. Both halves matter, and the column used to be the literal `True`
    for every scored policy, which made it decoration.

    It is not decoration, because a coarse rule can fail the first half. Scores
    tie: on the shipped validation split 126 of 2,947 accounts share the maximum
    `cycle_participation`, so the strictest non-empty cut this rule can form
    flags 42.8 per 1,000 — over twice the 20-per-1,000 cap. There is no threshold
    at which the rule is both non-empty and affordable, so
    `threshold_for_alert_budget` returns the flag-nothing operating point, and
    with the column hardcoded True the table published that as a policy which
    caught no mules at a cost of every miss. The model then "beat" it. That is
    not a win over a rival; it is a win over an abstention, and reporting it as
    the former inflates the model's standing with a comparison it never made.

    `val_strictest_nonempty_alerts_per_1000` carries the reason, so a reader can
    see by how much the cap was missed rather than only that a row was excluded.
    """
    from models.cost_matrix import _prf

    y_val = frames["val"][TARGET_COL].to_numpy()
    y_test = frames["test"][TARGET_COL].to_numpy()

    # (label, validation scores, test scores) for every policy that HAS a score
    # to threshold. Order mirrors compute_baselines' printed table, model last.
    scorers: list[tuple[str, np.ndarray, np.ndarray]] = []

    # The one-line rule is REPLAYED from its published machine-readable form
    # rather than having its score vectors threaded out of the baseline: those are
    # loop-local to the rule search, and `sign * x >= threshold_on_score` is the
    # replay contract that function already documents and the dashboard already
    # uses. Re-deriving here keeps one definition of what the rule means.
    rule = baselines.get("best_single_feature_rule_by_cost") or {}
    if rule.get("feature"):
        sign = 1.0 if rule.get("direction") == "high" else -1.0
        feature = str(rule["feature"])
        scorers.append((
            f"one-line rule: {rule.get('rule', feature)}",
            sign * frames["val"][feature].to_numpy(dtype=float),
            sign * frames["test"][feature].to_numpy(dtype=float),
        ))

    for key, label in (
        ("logistic_regression", "logistic regression, same features"),
        ("xgboost_without_graph_features", "XGBoost, no graph features"),
    ):
        block = baselines.get(key) or {}
        # Absent when the baseline was skipped (no scikit-learn, --no-baselines).
        # A missing row is better than a fabricated one.
        if block.get("_val_scores") is not None:
            scorers.append((label,
                            np.asarray(block["_val_scores"], dtype=float),
                            np.asarray(block["_test_scores"], dtype=float)))

    scorers.append(("XGBoost, full model",
                    np.asarray(model_val_scores, dtype=float),
                    np.asarray(model_test_scores, dtype=float)))

    rows = []
    for label, s_val, s_test in scorers:
        budgeted = evaluator.threshold_for_alert_budget(y_val, s_val,
                                                        alert_budget)
        realised = evaluator.evaluate_at_threshold(y_test, s_test,
                                                   budgeted.threshold)
        rows.append({
            "policy": label,
            "is_model": label == "XGBoost, full model",
            # Computed, not asserted. `budget_feasible` is False when no
            # non-empty cut fits the cap; see the docstring.
            "feasible_under_budget": bool(budgeted.budget_feasible),
            "val_threshold": float(budgeted.threshold),
            "val_alerts_per_1000": round(float(budgeted.alerts_per_1000), 1),
            "val_strictest_nonempty_alerts_per_1000":
                _none_if_nan_or_none(
                    budgeted.strictest_nonempty_alerts_per_1000),
            "test_precision": _none_if_nan_or_none(realised.precision),
            "test_recall": _none_if_nan_or_none(realised.recall),
            "test_f1": _none_if_nan_or_none(realised.f1),
            "test_alerts_per_1000": round(float(realised.alerts_per_1000), 1),
            "test_within_budget": bool(
                realised.alerts_per_1000 <= alert_budget + 1e-9),
            "test_total_cost": float(realised.total_cost),
        })

    # ── the two trivial policies, closed form ──
    n_pos = int(y_test.sum())
    n_neg = int(y_test.size - n_pos)
    none_p, none_r, none_f1 = _prf(tp=0, fp=0, fn=n_pos)
    all_p, all_r, all_f1 = _prf(tp=n_pos, fp=n_neg, fn=0)

    rows.append({
        "policy": "flag nothing", "is_model": False,
        # An empty queue is affordable at any budget and is not a policy, so it
        # is infeasible under the definition above. This is the row whose cost
        # every feasible policy must beat, not one of the policies being ranked.
        "feasible_under_budget": False,
        "val_threshold": None, "val_alerts_per_1000": 0.0,
        "val_strictest_nonempty_alerts_per_1000": None,
        "test_precision": none_p, "test_recall": none_r, "test_f1": none_f1,
        "test_alerts_per_1000": 0.0, "test_within_budget": True,
        "test_total_cost": float(n_pos * evaluator.config.fn_cost),
    })
    rows.append({
        "policy": "flag everything", "is_model": False,
        # The point of the row: cheap on paper, impossible to staff.
        "feasible_under_budget": bool(alert_budget >= 1000.0),
        "val_threshold": None, "val_alerts_per_1000": 1000.0,
        "val_strictest_nonempty_alerts_per_1000": 1000.0,
        "test_precision": all_p, "test_recall": all_r, "test_f1": all_f1,
        "test_alerts_per_1000": 1000.0,
        "test_within_budget": bool(alert_budget >= 1000.0),
        "test_total_cost": float(n_neg * evaluator.config.fp_cost),
    })

    return pd.DataFrame(rows)


def _none_if_nan_or_none(value):
    """
    Pass None through, map NaN to None, otherwise return a float.

    `ThresholdReport` already emits None for a not-computable metric, but these
    values pass through a DataFrame on the way to metrics.json and pandas stores
    None in a float column as NaN. Without this, "no queue to measure" would
    round-trip into the published table as a number.
    """
    if value is None:
        return None
    v = float(value)
    return None if np.isnan(v) else v


# ══════════════════════════════════════════════════════════════════
# Do the probabilities mean what they say?
# ══════════════════════════════════════════════════════════════════

def calibration_diagnostic(
    y_val: np.ndarray,
    p_val: np.ndarray,
    y_test: np.ndarray,
    p_test: np.ndarray,
    *,
    n_bins: int = 10,
) -> dict:
    """
    Measure how far the scores are from being probabilities, and by how much a
    two-parameter calibrator would close the gap.

    Distinct from `calibration_report` above, which DESCRIBES the raw scores on a
    single split. This one fits a calibrator on validation, applies it to test,
    and reports the before/after — so it answers "how much of the miscalibration
    is fixable" rather than only "how much is there".

    WHY THIS IS REPORTED AT ALL
    ───────────────────────────
    Every headline number this project publishes is a RANKING number — AUC,
    average precision, precision and recall at a threshold. None of them constrain
    the scores to mean anything as probabilities, and this model has a specific
    reason to be badly calibrated: `scale_pos_weight` re-weights the positive
    class during training, which deliberately inflates predicted probabilities. So
    the scores are a good ordering and a bad probability, while the dashboard
    prints them to two decimals beside the word "probability". Better to measure
    that and publish the size of it.

    FIT ON VALIDATION, MEASURED ON TEST
    ───────────────────────────────────
    The calibrator is fitted on the VALIDATION split and every figure below is
    reported on TEST. Fitting and reporting on the same split would show the
    improvement a calibrator can memorise rather than the one it generalises, and
    the generalising one is the whole claim.

    WHAT THIS DOES NOT DO
    ─────────────────────
    It does not change the model, the shipped threshold, or any published
    operating point. `ranking_invariance` in the returned payload is the check
    that keeps that claim honest: the calibrated AUC is recomputed and compared
    against the raw one, so if the map ever stopped being monotone the run says so
    instead of quietly re-ranking the alert queue.
    """
    y_val = np.asarray(y_val).astype(int).ravel()
    y_test = np.asarray(y_test).astype(int).ravel()
    p_val = np.asarray(p_val, dtype=float).ravel()
    p_test = np.asarray(p_test, dtype=float).ravel()

    scaler = fit_platt_scaling(y_val, p_val, fit_on="validation")
    calibrated = scaler.transform(p_test)

    raw_brier = brier_score(y_test, p_test)
    cal_brier = brier_score(y_test, calibrated)
    raw_ece = expected_calibration_error(y_test, p_test, n_bins)
    cal_ece = expected_calibration_error(y_test, calibrated, n_bins)
    raw_auc = roc_auc(y_test, p_test)
    cal_auc = roc_auc(y_test, calibrated)

    print(banner("Probability Calibration (diagnostic only)"))
    print(f"  Platt map fitted on validation ({scaler.n_positive_fit} positives "
          f"of {scaler.n_fit}): slope {scaler.slope:.4f}, "
          f"intercept {scaler.intercept:+.4f}"
          + ("" if scaler.converged else "  [DID NOT CONVERGE]"))
    print(f"  {'':<22s}{'raw':>10s}{'calibrated':>12s}{'change':>10s}")
    print(f"  {'Brier score':<22s}{raw_brier:>10.5f}{cal_brier:>12.5f}"
          f"{cal_brier - raw_brier:>+10.5f}")
    print(f"  {'expected cal. error':<22s}{_fmt(raw_ece, '.5f'):>10s}"
          f"{_fmt(cal_ece, '.5f'):>12s}"
          f"{_fmt(None if (cal_ece is None or raw_ece is None) else cal_ece - raw_ece, '+.5f'):>10s}")
    print(f"  {'ROC-AUC':<22s}{raw_auc:>10.5f}{cal_auc:>12.5f}"
          f"{cal_auc - raw_auc:>+10.5f}   <- must be ~0: the map is monotone")
    # A slope below 1 is the signature of over-confidence, which is what
    # scale_pos_weight is expected to produce. Naming the direction stops the
    # number being reported without being read.
    if scaler.slope < 1.0:
        print("  Slope < 1: the raw scores are OVER-confident — they are spread "
              "wider than the evidence supports, which is the expected effect of "
              "scale_pos_weight.")
    elif scaler.slope > 1.0:
        print("  Slope > 1: the raw scores are UNDER-confident — the calibrator "
              "sharpens them.")
    print("  Not applied at serving time: the shipped threshold was selected on "
          "the raw scale, so rescaling scores underneath it would move the "
          "operating point without saying so.")

    return {
        "purpose": ("diagnostic only — the serving path and every published "
                    "operating point use the RAW scores"),
        "fitted_on": "validation",
        "measured_on": "test",
        "n_bins": int(n_bins),
        "scaler": scaler.as_dict(),
        "test_raw": {
            "brier_score": round(float(raw_brier), 6),
            "expected_calibration_error": _round_or_none(raw_ece, 6),
        },
        "test_calibrated": {
            "brier_score": round(float(cal_brier), 6),
            "expected_calibration_error": _round_or_none(cal_ece, 6),
        },
        "ranking_invariance": {
            "test_auc_raw": round(float(raw_auc), 6),
            "test_auc_calibrated": round(float(cal_auc), 6),
            # Tolerance is float-summation slack, not a claim about how much
            # re-ranking is acceptable. Any real re-ranking exceeds it by orders
            # of magnitude; tests/test_baselines.py asserts against this key.
            "abs_delta": float(abs(cal_auc - raw_auc)),
            "tolerance": 1e-6,
        },
        "test_reliability_raw": reliability_table(
            y_test, p_test, n_bins).to_dict(orient="records"),
        "test_reliability_calibrated": reliability_table(
            y_test, calibrated, n_bins).to_dict(orient="records"),
    }


# ══════════════════════════════════════════════════════════════════
# Importance / attribution
# ══════════════════════════════════════════════════════════════════

def importance_report(model, X_test: pd.DataFrame) -> tuple[dict, dict]:
    """
    Gain-based importance, mean |SHAP|, and the check that ties them to the score.

    The assertion at the end is the load-bearing part. api/main.py returns
    per-account SHAP attributions as `contributing_factors`; if those did not sum
    to the margin behind the score, the API would be explaining a different model
    than the one it served — silently, and forever. Verifying it here means every
    saved model has been checked once, on real data, before it can be served.

    THE CHECK IS DONE IN MARGIN SPACE, NOT PROBABILITY SPACE
    ────────────────────────────────────────────────────────
    TreeSHAP's additivity guarantee is `contribs.sum(1) + bias == raw margin`.
    That is the identity, and margin is the only space it is stated in. This
    function used to verify it by pushing the reconstructed margin through a
    sigmoid and comparing probabilities to a 1e-5 tolerance — testing a corollary
    instead of the theorem, and testing it through a function that destroys
    exactly the information the test is for.

    Sigmoid compresses: dp = p(1 - p)·dm. So a probability tolerance of 1e-5
    permits a margin error of 1e-5 / p(1 - p), which is 0.01 log-odds at p = 0.001
    and 0.1 log-odds at p = 0.0001. On this dataset that is the common case, not
    the corner: ~96% of accounts are legitimate and score near zero, so the old
    check was at its loosest precisely where most of the data lives, and a
    0.1-log-odds attribution error is more than enough to reorder the top-3
    `contributing_factors` an analyst is shown. Meanwhile at p ≈ 0.5 the same
    tolerance demanded 4e-5 — the check was up to 2,500x stricter on the rows it
    mattered least for. Comparing margins directly makes the tolerance mean one
    thing everywhere.

    The probability-space agreement is still printed, because it is what a reader
    intuits and what the API serves, but it is a CONSEQUENCE now, not the gate:
    |dp| <= |dm| / 4 for any margin, so passing here bounds it automatically.
    """
    print(banner("Feature Importance"))

    gain = {feature: float(value) for feature, value
            in zip(X_test.columns, model.feature_importances_)}

    contribs, bias = shap_contributions(model, X_test)
    shap = mean_abs_shap(contribs, list(X_test.columns))

    # The identity as TreeSHAP states it, checked as TreeSHAP states it.
    margin_from_shap = contribs.sum(axis=1) + bias
    margin_direct = margin(model, X_test)
    drift = float(np.abs(margin_from_shap - margin_direct).max())

    # 1e-4 log-odds. XGBoost accumulates tree outputs in float32 and the two
    # paths sum in different orders, so exact equality is not available; 1e-4 is
    # ~three orders of magnitude above that noise floor and ~three below the
    # 0.1 the previous check tolerated at the saturated end.
    MARGIN_TOLERANCE = 1e-4
    if drift > MARGIN_TOLERANCE:
        raise RuntimeError(
            "TreeSHAP attributions do not reconstruct the model's own margins "
            f"(max deviation {drift:.2e} log-odds, tolerance "
            f"{MARGIN_TOLERANCE:.0e}).\n"
            "  Something is inconsistent between predict(output_margin=True) "
            "and pred_contribs — most likely the early-stopping iteration range "
            "in models/explain.py.\n"
            "  api/main.py's explanations would be describing a different "
            "model than the one it scores with, so this run is refusing to "
            "save."
        )
    # Reported, not enforced: the same disagreement measured where a reader will
    # picture it. Bounded by drift/4, so it cannot fail on its own.
    prob_drift = float(np.abs(1.0 / (1.0 + np.exp(-margin_from_shap))
                              - score(model, X_test)).max())
    print(f"  {sym('ok')} SHAP contributions reconstruct every test margin "
          f"(max deviation {drift:.1e} log-odds; {prob_drift:.1e} in "
          f"probability) — the API's explanations are exact")

    print(f"  {'feature':<26s}{'mean|SHAP|':>11s}  {'gain':>7s}")
    for feature, value in sorted(shap.items(), key=lambda kv: -kv[1]):
        print(f"  {feature:<26s}{value:>11.4f}  {gain[feature]:>7.4f} "
              f"{bar(value, width=34)}")

    return gain, shap


# ══════════════════════════════════════════════════════════════════
# Persistence
# ══════════════════════════════════════════════════════════════════

def _public_only(baselines: dict) -> dict:
    """
    Strip underscore-prefixed keys from every baseline block, one level deep.

    THE DEFECT THIS PREVENTS: `build_metrics_payload` filters underscore keys out
    of the `test` block but passed `baselines` through whole. The moment a
    baseline started carrying `_val_scores` for the capacity-fair table, two float
    vectors per baseline — thousands of numbers nobody reads — would have been
    serialised into metrics.json, and `report.py --check` compares that file
    byte-for-byte against a regenerated one. So this is not only bloat: score
    vectors are the least stable thing in the payload, and they would have turned
    every rerun into a spurious staleness failure.

    One level deep is deliberate and sufficient: underscore keys are only ever
    added to the baseline blocks themselves. Recursing would silently launder a
    private key out of a nested structure where its presence is a bug worth
    seeing.
    """
    return {
        name: ({k: v for k, v in block.items() if not k.startswith("_")}
               if isinstance(block, dict) else block)
        for name, block in baselines.items()
    }


def build_metrics_payload(
    *,
    evaluator: CostEvaluator,
    dataset: dict,
    integrity: dict,
    thresholds: dict,
    test_results: dict,
    baselines: dict,
    gain: dict,
    shap: dict,
    cv_results: dict,
    sensitivity: pd.DataFrame,
    capacity_fair: pd.DataFrame | None,
    prevalence_projection: pd.DataFrame | None,
    calibration: dict | None,
    alert_budget: float,
    hparams: dict,
    n_train: int,
) -> dict:
    """
    Assemble metrics.json.

    The top-level keys are exactly the ones the dashboard and API already read,
    holding TEST numbers at the VALIDATION-selected threshold — same names as
    before, honest values. Everything new lives in nested blocks so existing
    consumers are unaffected.
    """
    at_threshold = test_results["at_selected_threshold"]
    optimal = thresholds["cost_optimal"]
    budgeted = thresholds["capacity_constrained"]

    return {
        # ── keys consumed by dashboard/ and api/ — do not rename ──
        "roc_auc": test_results["auc"],
        "optimal_threshold": float(optimal.threshold),
        "precision": at_threshold["precision"],
        "recall": at_threshold["recall"],
        "f1": at_threshold["f1"],
        "total_cost": at_threshold["total_cost"],
        "feature_importance": gain,

        # ── provenance, so a stale metrics.json is obvious ──
        "model_version": MODEL_VERSION,
        "model_file": MODEL_NAME,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_features": len(FEATURE_COLS),
        "feature_cols": list(FEATURE_COLS),
        "n_train_accounts": int(n_train),
        "hyperparameters": hparams,
        "threshold_selected_on": "validation",
        "roc_auc_ci_95": test_results["auc_ci_95"],
        "average_precision": test_results["average_precision"],
        "cost_config": {
            "fn_cost": float(evaluator.config.fn_cost),
            "fp_cost": float(evaluator.config.fp_cost),
            "fn_fp_ratio": round(float(evaluator.config.ratio), 4),
            "break_even_probability": round(
                float(evaluator.break_even_probability), 6),
            "alert_budget_per_1000": float(alert_budget),
        },

        # ── full detail ──
        "dataset": dataset,
        "split_integrity": integrity,
        "feature_importance_mean_abs_shap": shap,
        "validation": {
            "auc": thresholds["auc"],
            "average_precision": thresholds["average_precision"],
            "cost_optimal": optimal.as_dict(),
            "capacity_constrained": budgeted.as_dict(),
        },
        "test": {k: v for k, v in test_results.items()
                 if not k.startswith("_")},
        "baselines": _public_only(baselines),
        "cost_ratio_sensitivity": sensitivity.to_dict(orient="records"),
        # Each of the three is None rather than absent when it could not be built
        # (--no-baselines, a missing dependency). An explicit null tells report.py
        # "this run did not produce it"; a missing key is indistinguishable from a
        # metrics.json written by an older version of this script.
        "capacity_fair_comparison": (
            None if capacity_fair is None
            else capacity_fair.to_dict(orient="records")),
        "prevalence_projection": (
            None if prevalence_projection is None
            else prevalence_projection.to_dict(orient="records")),
        "probability_calibration": calibration,
        "cross_validation": cv_results or None,
    }


def _json_safe(value):
    """
    Map every non-finite float to `null` on the way into metrics.json.

    NaN, inf and -inf are not JSON. `json.dump` writes them as the bare tokens
    `NaN` / `Infinity`, so the published file is not valid JSON: a strict parser
    rejects it outright, and a lenient one hands the caller back a float that
    poisons any arithmetic it enters.

    Everything non-finite that legitimately reaches here means NOT COMPUTABLE —
    `roc_auc` on a single-class archetype slice, and the precision/recall/F1 NaNs
    that `cost_matrix.sweep` uses because a float column cannot hold None (see
    `cost_matrix._prf`). `null` states that in JSON, and unlike 0.0 it cannot be
    averaged, plotted or quoted as a result by accident. Anything non-finite for
    a different reason is a bug that should have raised further upstream;
    publishing it as null at least stops it being published as a number.

    Note this is the LAST line of defence, not the mechanism: the metrics dict is
    built from `ThresholdReport.as_dict()`, which already emits None directly.
    """
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    # np.floating and Python float both: np.float64 subclasses float but
    # np.float32 does not, and pandas hands us both.
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    return value


def save_artifacts(model, metrics: dict) -> None:
    """Write the model binary and metrics.json."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODEL_DIR / MODEL_NAME
    model.save_model(str(model_path))
    print(f"  Model saved   {sym('arrow')} {model_path}")

    metrics_path = MODEL_DIR / "metrics.json"
    # newline="\n" so the committed artifact is byte-identical whoever wrote it.
    # Python's text mode translates "\n" to the platform terminator, so the same
    # metrics.json written on Windows and on Linux differ in every single line —
    # which turns a retrain that changed one number into a whole-file diff and
    # buries the change nobody can now see.
    with open(metrics_path, "w", encoding="utf-8", newline="\n") as f:
        # allow_nan=False turns a `_json_safe` miss into an exception here instead
        # of a metrics.json containing the bare token `NaN`. That file is read by
        # the dashboard, the API and `report.py --check`; a strict parser rejects
        # it and a lenient one silently yields a float that poisons whatever it
        # touches. Failing at the write is the cheaper of the two outcomes, and
        # `_json_safe` runs first precisely so this never fires.
        json.dump(_json_safe(metrics), f, indent=2, default=float,
                  allow_nan=False)
    print(f"  Metrics saved {sym('arrow')} {metrics_path}")


# ══════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the Sentinel mule detector.")
    p.add_argument("--no-cv", action="store_true",
                   help="skip cross-validation (faster)")
    p.add_argument("--cv-folds", type=int, default=5,
                   help="number of ring-grouped CV folds (default: 5)")
    p.add_argument("--fn-cost", type=float, default=DEFAULT_FN_COST,
                   help=f"cost of a missed mule (default: {DEFAULT_FN_COST:,.0f})")
    p.add_argument("--fp-cost", type=float, default=DEFAULT_FP_COST,
                   help=f"cost of a false flag (default: {DEFAULT_FP_COST:,.0f})")
    p.add_argument("--alert-budget", type=float,
                   default=DEFAULT_ALERT_BUDGET_PER_1000,
                   help="alerts per 1,000 accounts a review team can work "
                        f"(default: {DEFAULT_ALERT_BUDGET_PER_1000:.0f})")
    p.add_argument("--max-depth", type=int, default=HPARAMS["max_depth"],
                   help=f"XGBoost tree depth (default: {HPARAMS['max_depth']})")
    p.add_argument("--min-child-weight", type=float,
                   default=HPARAMS["min_child_weight"],
                   help="minimum leaf weight; raise it to regularise further "
                        f"(default: {HPARAMS['min_child_weight']})")
    p.add_argument("--no-baselines", action="store_true",
                   help="skip the baseline comparison (not recommended — the "
                        "baselines are what make the headline number mean "
                        "something)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    enable_utf8_stdout()

    HPARAMS["max_depth"] = int(args.max_depth)
    HPARAMS["min_child_weight"] = float(args.min_child_weight)

    print(banner("UPI Mule-Ring Sentinel: Model Training"))
    evaluator = CostEvaluator(fn_cost=args.fn_cost, fp_cost=args.fp_cost)
    print(f"  Cost model: FN {sym('rupee')}{args.fn_cost:,.0f} / "
          f"FP {sym('rupee')}{args.fp_cost:,.0f} "
          f"(ratio {evaluator.config.ratio:.1f}:1, break-even p* "
          f"{evaluator.break_even_probability:.4f})")

    print("\n[1/7] Loading features...")
    frames = load_data()
    dataset = describe_dataset(frames)

    print("\n[2/7] Checking split integrity...")
    integrity = assert_split_integrity(frames)

    print("\n[3/7] Training final model (early stopping on validation)...")
    model = train_model(
        frames["train"][FEATURE_COLS], frames["train"][TARGET_COL],
        frames["val"][FEATURE_COLS], frames["val"][TARGET_COL],
        label="full model",
    )
    # Verify the contract now, not at serve time: a model whose columns do not
    # match models/features.py must never reach models/saved_models/.
    assert_feature_contract(model.get_booster().feature_names)
    print(f"  {sym('ok')} feature contract verified against "
          f"models/features.py ({len(FEATURE_COLS)} features, in order)")

    print("\n[4/7] Cross-validating on train split...")
    if args.no_cv:
        print("  skipped (--no-cv)")
        cv_results: dict = {}
    else:
        cv_results = cross_validate(
            frames, n_rounds=int(model.best_iteration) + 1,
            n_folds=args.cv_folds,
        )

    print("\n[5/7] Selecting operating points on validation...")
    thresholds = select_operating_points(model, frames, evaluator,
                                        args.alert_budget)

    print("\n[6/7] Evaluating on held-out test...")
    test_results = evaluate_on_test(model, frames, evaluator, thresholds,
                                    args.alert_budget)

    model_report = {
        "test_precision": test_results["at_selected_threshold"]["precision"],
        "test_recall": test_results["at_selected_threshold"]["recall"],
        "test_f1": test_results["at_selected_threshold"]["f1"],
        "test_total_cost": test_results["at_selected_threshold"]["total_cost"],
        "test_alerts_per_1000":
            test_results["at_selected_threshold"]["alerts_per_1000_accounts"],
        "test_average_precision": test_results["average_precision"],
    }
    if args.no_baselines:
        print(f"\n  {sym('warn')} baselines skipped (--no-baselines)")
        baselines: dict = {}
    else:
        baselines = compute_baselines(frames, evaluator, model_report)

    sensitivity = evaluator.sensitivity_to_cost_ratio(
        frames["val"][TARGET_COL].to_numpy(), thresholds["_scores"],
        frames["test"][TARGET_COL].to_numpy(), test_results["_scores"],
    )
    print(banner("Sensitivity to the FN:FP Cost Assumption"))
    print(sensitivity.to_string(index=False,
                                float_format=lambda v: f"{v:,.4f}"))
    print("  (The ratio is the one number in the cost model nobody can verify, "
          "so the operating point is reported across a range of it.")
    print("   Each row's threshold is picked on VALIDATION and then evaluated "
          "on TEST at that frozen value — the same")
    print("   discipline as the headline number. Selecting these per-ratio "
          "thresholds on test instead would understate")
    print("   cost at the shipped 13.33 ratio by the full "
          "not-peeking gap, which is exactly the trap this table exists to "
          "disprove.)")

    gain, shap = importance_report(model, frames["test"][FEATURE_COLS])

    # ── The three blocks that answer the stated limitations ──
    y_val = frames["val"][TARGET_COL].to_numpy()
    y_test = frames["test"][TARGET_COL].to_numpy()

    # 1. What the published precision and rupee cost become at a production base
    #    rate. The operating point is re-derived here rather than read out of
    #    `test_results`, because the projection needs the raw confusion counts and
    #    `at_selected_threshold` has already been flattened for JSON.
    selected = evaluator.evaluate_at_threshold(
        y_test, test_results["_scores"],
        float(thresholds["cost_optimal"].threshold))
    prevalence_projection = evaluator.project_to_prevalence(selected)
    print(banner("Precision and Cost at Lower Base Rates"))
    print(prevalence_projection.to_string(index=False,
                                          float_format=lambda v: f"{v:,.4f}"))
    pi_star = evaluator.break_even_prevalence(selected)
    print("  (Recall is CONSTANT down this table on purpose: TPR and FPR are "
          "within-class rates, so a change of")
    print("   base rate cannot touch them. Precision and total cost are not "
          "within-class, and they move a great deal.")
    if pi_star is None:
        print("   This operating point has no break-even prevalence — it never "
              "clears p*, at any base rate.)")
    else:
        print(f"   Below a prevalence of {pi_star:.4%} the queue stops paying "
              f"for itself at this threshold: projected precision falls under "
              f"the break-even")
        print(f"   p* of {evaluator.break_even_probability:.4%}, and the false "
              "alerts then cost more than the misses they prevent.)")

    # 2. Every policy under one analyst budget. Needs the baselines, so it is
    #    skipped rather than faked when they were skipped.
    if baselines:
        capacity_fair = capacity_fair_comparison(
            frames, evaluator, baselines,
            model_val_scores=thresholds["_scores"],
            model_test_scores=test_results["_scores"],
            alert_budget=args.alert_budget,
        )
        print(banner(f"Every Policy at the Same {args.alert_budget:.0f} "
                     "Alerts/1,000 Budget"))
        print(capacity_fair.to_string(index=False,
                                      float_format=lambda v: f"{v:,.4f}"))
        print("  (The baseline table above lets each policy raise as many alerts "
              "as it likes. At 13.3:1 that is an")
        print("   advantage and not a fair fight, because misses dominate cost — "
              "so the one-line rule bought its")
        print("   recall with roughly 12x the queue the cap allows. Here every "
              "policy is held to the SAME queue size,")
        print("   with each threshold still selected on validation. "
              "test_within_budget=False means a val-selected")
        print("   threshold overflowed the budget on test; that is reported "
              "rather than re-fitted, because re-fitting")
        print("   it until the constraint held would be selection on test in a "
              "capacity argument's clothing.)")
    else:
        capacity_fair = None

    # 3. Whether the scores mean anything as probabilities.
    calibration = calibration_diagnostic(
        y_val, thresholds["_scores"], y_test, test_results["_scores"])

    print("\n[7/7] Saving artifacts...")
    metrics = build_metrics_payload(
        evaluator=evaluator,
        dataset=dataset,
        integrity=integrity,
        thresholds=thresholds,
        test_results=test_results,
        baselines=baselines,
        gain=gain,
        shap=shap,
        cv_results=cv_results,
        sensitivity=sensitivity,
        capacity_fair=capacity_fair,
        prevalence_projection=prevalence_projection,
        calibration=calibration,
        alert_budget=args.alert_budget,
        hparams={**HPARAMS, "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
                 "best_iteration": int(model.best_iteration)},
        n_train=len(frames["train"]),
    )
    save_artifacts(model, metrics)

    at_threshold = test_results["at_selected_threshold"]
    print(f"\n{sym('ok')} Training complete.")
    print(f"  Test ROC-AUC {metrics['roc_auc']:.4f} "
          f"(95% CI {metrics['roc_auc_ci_95'][0]:.4f}-"
          f"{metrics['roc_auc_ci_95'][1]:.4f}), "
          f"average precision {metrics['average_precision']:.4f}")
    print(f"  At the validation-selected threshold "
          f"{metrics['optimal_threshold']:.4f}: "
          f"P {_fmt(at_threshold['precision'])} "
          f"R {_fmt(at_threshold['recall'])} F1 {_fmt(at_threshold['f1'])}, "
          f"{at_threshold['alerts_per_1000_accounts']:.1f} alerts/1,000")
    if baselines:
        rule = baselines["best_single_feature_rule_by_cost"]
        print(f"  Best single-feature rule was `{rule['rule']}` at F1 "
              f"{_fmt(rule['test_f1'])} — quote this model as lift over that, "
              "not in isolation.")


if __name__ == "__main__":
    main()
