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
   asserts that the per-account attributions sum to the margin, which is what
   makes api/main.py's explanations trustworthy.

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
    roc_auc,
)
from models.explain import mean_abs_shap, shap_contributions
from models.features import (
    FEATURE_COLS,
    LABEL_META_COLS,
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

# If one feature alone separates the classes this well, the generator planted the
# answer and every downstream metric is theatre. tests/test_baselines.py fails
# the build on this; here it is only a warning, because train.py should not be
# the thing that decides a dataset is invalid.
LEAKAGE_AUC_CEILING = 0.99

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

# Heavy-tailed, non-negative columns. Log-scaled before the logistic-regression
# baseline: handing a linear model a raw rupee sum builds a straw man, and a
# baseline worth reporting is one that was given a fair chance.
LOG1P_COLS = [
    "in_degree", "out_degree", "in_amount_sum", "out_amount_sum",
    "txn_velocity", "repeat_ratio", "pagerank",
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
        df = pd.read_csv(path).reset_index(drop=True)

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


# ══════════════════════════════════════════════════════════════════
# Reporting helpers
# ══════════════════════════════════════════════════════════════════

def bootstrap_auc_ci(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    rounds: int = BOOTSTRAP_ROUNDS,
    seed: int = RANDOM_SEED,
) -> tuple[float, float]:
    """
    Percentile bootstrap 95% CI for ROC-AUC.

    With ~119 positives in test, the difference between 0.94 and 0.96 is inside
    the noise. Reporting the interval keeps the sample size visible instead of
    hiding it behind four decimal places.
    """
    rng = np.random.default_rng(seed)
    y = np.asarray(y_true)
    n = y.size
    samples: list[float] = []

    for _ in range(rounds):
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) < 2:      # pragma: no cover — degenerate draw
            continue
        samples.append(roc_auc(y[idx], np.asarray(y_proba)[idx]))

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
    """
    from models.cost_matrix import _prf

    rows = [
        ("Legitimate", *_prf(tn, fn, fp), tn + fp),   # roles swapped
        ("Mule", *_prf(tp, fp, fn), tp + fn),
    ]
    print(f"  {'':<12s}{'precision':>11s}{'recall':>9s}{'f1':>9s}{'support':>9s}")
    for name, precision, recall, f1, support in rows:
        print(f"  {name:<12s}{precision:>11.4f}{recall:>9.4f}{f1:>9.4f}"
              f"{support:>9d}")


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
    """
    y = np.asarray(y_true).astype(float).ravel()
    p = np.asarray(y_proba, dtype=float).ravel()
    brier = float(np.mean((p - y) ** 2))

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = np.clip(np.digitize(p, edges[1:-1], right=False), 0, n_bins - 1)
    table, ece = [], 0.0
    for b in range(n_bins):
        mask = bins == b
        if not mask.any():
            continue
        mean_p = float(p[mask].mean())
        observed = float(y[mask].mean())
        weight = float(mask.mean())
        ece += weight * abs(mean_p - observed)
        table.append({
            "bin": f"[{edges[b]:.1f}, {edges[b + 1]:.1f})",
            "n": int(mask.sum()),
            "mean_predicted": round(mean_p, 4),
            "observed_rate": round(observed, 4),
        })

    return {
        "brier_score": round(brier, 6),
        "expected_calibration_error": round(float(ece), 4),
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
    Recall per ring archetype, and per ring.

    One overall recall figure hides the only thing a risk lead will ask: which
    rings are we missing? The generator emits three archetypes of deliberately
    unequal difficulty, so reporting them separately turns a single number into
    an honest difficulty gradient.

    Ring-level recall counts a ring as detected if ANY member is flagged, which
    is closer to how this is used — one thread is enough to start pulling, and a
    ring where 2 of 6 accounts alert is a caught ring, not a 33% failure.
    """
    flagged = np.asarray(y_proba) >= threshold
    positives = (df[TARGET_COL] == 1).to_numpy()

    per_archetype = {}
    for archetype in sorted(df.loc[positives, "ring_type"].unique()):
        mask = positives & (df["ring_type"] == archetype).to_numpy()
        n = int(mask.sum())
        detected = int(flagged[mask].sum())
        per_archetype[str(archetype)] = {
            "n_accounts": n,
            "detected": detected,
            "account_recall": round(detected / n, 4) if n else None,
            "mean_score": round(float(np.asarray(y_proba)[mask].mean()), 4),
            "median_score": round(float(np.median(np.asarray(y_proba)[mask])), 4),
        }

    rings = df.loc[positives].assign(_flagged=flagged[positives])
    by_ring = rings.groupby(["ring_type", "ring_id"])["_flagged"]
    ring_hits = by_ring.any()
    ring_frac = by_ring.mean()

    per_archetype_rings = {}
    for archetype in sorted(ring_hits.index.get_level_values(0).unique()):
        hits = ring_hits.loc[archetype]
        per_archetype_rings[str(archetype)] = {
            "n_rings": int(hits.size),
            "rings_with_an_alert": int(hits.sum()),
            "ring_recall": round(float(hits.mean()), 4),
            "mean_share_of_ring_flagged": round(
                float(ring_frac.loc[archetype].mean()), 4),
        }

    return {
        "by_archetype_accounts": per_archetype,
        "by_archetype_rings": per_archetype_rings,
        "overall_ring_recall": round(float(ring_hits.mean()), 4),
        "n_rings": int(ring_hits.size),
        "rings_with_an_alert": int(ring_hits.sum()),
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
    print(f"  {sym('bullet')} cost-optimal threshold   {optimal.threshold:.4f}  "
          f"P {optimal.precision:.4f} R {optimal.recall:.4f} | "
          f"{optimal.alerts_per_1000:.1f} alerts/1,000")
    width = optimal.plateau_width
    if width is not None:
        verdict = ("wide — robust" if width > 0.05
                   else "NARROW — may be fitted to noise")
        print(f"      plateau [{optimal.plateau_lo:.4f}, "
              f"{optimal.plateau_hi:.4f}] width {width:.4f} ({verdict})")
    print(f"  {sym('bullet')} capacity threshold       {budgeted.threshold:.4f}  "
          f"P {budgeted.precision:.4f} R {budgeted.recall:.4f} | "
          f"{budgeted.alerts_per_1000:.1f} alerts/1,000 "
          f"(budget {alert_budget:.0f})")
    print(f"  {sym('arrow')} both thresholds are now frozen and applied blind "
          "to test.")

    return {
        "auc": float(val_auc),
        "average_precision": float(val_ap),
        "cost_optimal": optimal,
        "capacity_constrained": budgeted,
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
    ci_lo, ci_hi = bootstrap_auc_ci(y_test, p_test)

    at_threshold = evaluator.evaluate_at_threshold(y_test, p_test, threshold)
    at_capacity = evaluator.evaluate_at_threshold(y_test, p_test,
                                                  capacity_threshold)
    oracle = evaluator.find_optimal_threshold(y_test, p_test)

    print(f"  ROC-AUC {auc:.4f} (95% CI {ci_lo:.4f}-{ci_hi:.4f}, "
          f"{BOOTSTRAP_ROUNDS} bootstrap resamples)")
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
    print(f"    P {at_capacity.precision:.4f} R {at_capacity.recall:.4f} "
          f"F1 {at_capacity.f1:.4f} | {at_capacity.alerts_per_1000:.1f} "
          f"alerts/1,000 | {sym('rupee')}{at_capacity.total_cost:,.0f}")

    gap = at_threshold.total_cost - oracle.total_cost
    print()
    print(f"  [diagnostic] test-optimal threshold would have been "
          f"{oracle.threshold:.4f} (F1 {oracle.f1:.4f}, "
          f"{sym('rupee')}{oracle.total_cost:,.0f}).")
    print(f"  [diagnostic] cost of not having peeked at test: "
          f"{sym('rupee')}{gap:,.0f}. Reported metrics use the honest "
          "threshold.")

    breakdown = archetype_breakdown(df, p_test, threshold)
    print()
    print("  ── recall by ring archetype (accounts / rings) ──")
    for archetype, stats in breakdown["by_archetype_accounts"].items():
        rings = breakdown["by_archetype_rings"][archetype]
        print(f"    {archetype:<16s} accounts {stats['detected']:>3d}/"
              f"{stats['n_accounts']:<3d} = {stats['account_recall']:.0%}"
              f"   rings {rings['rings_with_an_alert']:>2d}/"
              f"{rings['n_rings']:<2d} = {rings['ring_recall']:.0%}"
              f"   mean score {stats['mean_score']:.3f}")
    print(f"    {'ALL RINGS':<16s} "
          f"{'':>21s}rings {breakdown['rings_with_an_alert']:>2d}/"
          f"{breakdown['n_rings']:<2d} = "
          f"{breakdown['overall_ring_recall']:.0%}")

    calibration = calibration_report(y_test, p_test)
    print()
    print(f"  Calibration: Brier {calibration['brier_score']:.4f}, ECE "
          f"{calibration['expected_calibration_error']:.4f}, mean predicted "
          f"{calibration['mean_predicted_probability']:.4f} vs actual "
          f"prevalence {calibration['actual_prevalence']:.4f}")

    return {
        "auc": float(auc),
        "auc_ci_95": [ci_lo, ci_hi],
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
    """
    if criterion not in ("cost", "f1"):
        raise ValueError(f"criterion must be 'cost' or 'f1', got {criterion!r}")

    y_val = frames["val"][TARGET_COL].to_numpy()
    y_test = frames["test"][TARGET_COL].to_numpy()
    rows = []

    for feature in FEATURE_COLS:
        x_val = frames["val"][feature].to_numpy(dtype=float)
        x_test = frames["test"][feature].to_numpy(dtype=float)
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
                "val_f1": float(chosen.f1),
                "test_auc": float(roc_auc(y_test, s_test)),
                "test_precision": round(float(test_report.precision), 4),
                "test_recall": round(float(test_report.recall), 4),
                "test_f1": round(float(test_report.f1), 4),
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
    discriminative = np.maximum(table["test_auc"], 1.0 - table["test_auc"])
    ranked = table.assign(discriminative_auc=discriminative).sort_values(
        "discriminative_auc", ascending=False)
    strongest = ranked.iloc[0]

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
        "test_precision": float(best["test_precision"]),
        "test_recall": float(best["test_recall"]),
        "test_f1": float(best["test_f1"]),
        "test_total_cost": float(best["test_total_cost"]),
        "test_alerts_per_1000": float(best["test_alerts_per_1000"]),
        "top_10_rules": table.head(10).to_dict(orient="records"),
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
        "test_precision": round(float(report.precision), 4),
        "test_recall": round(float(report.recall), 4),
        "test_f1": round(float(report.f1), 4),
        "test_total_cost": float(report.total_cost),
        "test_alerts_per_1000": round(float(report.alerts_per_1000), 1),
        "preprocessing": "log1p on heavy-tailed columns, then train-fit z-score",
        "coefficients_standardised": {k: round(v, 4)
                                      for k, v in coefficients.items()},
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
        "test_precision": round(float(report.precision), 4),
        "test_recall": round(float(report.recall), 4),
        "test_f1": round(float(report.f1), 4),
        "test_total_cost": float(report.total_cost),
        "test_alerts_per_1000": round(float(report.alerts_per_1000), 1),
    }


def trivial_baselines(y_test: np.ndarray, evaluator: CostEvaluator) -> dict:
    """Flag nobody / flag everybody — the floor any result must clear."""
    n_pos = int(np.asarray(y_test).sum())
    n_neg = int(np.asarray(y_test).size - n_pos)
    flag_none = n_pos * evaluator.config.fn_cost
    flag_all = n_neg * evaluator.config.fp_cost
    return {
        "flag_nothing": {
            "test_precision": 0.0, "test_recall": 0.0, "test_f1": 0.0,
            "test_total_cost": float(flag_none), "test_alerts_per_1000": 0.0,
        },
        "flag_everything": {
            "test_precision": round(n_pos / (n_pos + n_neg), 4),
            "test_recall": 1.0,
            "test_f1": round(2 * n_pos / (2 * n_pos + n_neg), 4),
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
        return (f"  {name:<34s} {block['test_precision']:>9.4f}"
                f"{block['test_recall']:>8.4f}{block['test_f1']:>8.4f}"
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

    strongest = rule_cost["strongest_single_feature_test_auc"]
    print()
    print(f"  Strongest single feature on test: {strongest['feature']} "
          f"AUC {strongest['auc']:.4f}"
          + (f" (inverted; raw {strongest['raw_auc']:.4f})"
             if strongest["inverted"] else "")
          + f" — leakage ceiling {strongest['ceiling']}")
    if strongest["auc"] >= LEAKAGE_AUC_CEILING:
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
    """
    print(banner("Feature Importance"))

    gain = {feature: float(value) for feature, value
            in zip(X_test.columns, model.feature_importances_)}

    contribs, bias = shap_contributions(model, X_test)
    shap = mean_abs_shap(contribs, list(X_test.columns))

    margin = contribs.sum(axis=1) + bias
    from_shap = 1.0 / (1.0 + np.exp(-margin))
    direct = score(model, X_test)
    drift = float(np.abs(from_shap - direct).max())
    if drift > 1e-5:
        raise RuntimeError(
            "TreeSHAP attributions do not reconstruct the model's own scores "
            f"(max deviation {drift:.2e}).\n"
            "  Something is inconsistent between predict_proba and "
            "pred_contribs — most likely the early-stopping iteration range in "
            "models/explain.py.\n"
            "  api/main.py's explanations would be describing a different "
            "model than the one it scores with, so this run is refusing to "
            "save."
        )
    print(f"  {sym('ok')} SHAP contributions reconstruct every test score "
          f"(max deviation {drift:.1e}) — the API's explanations are exact")

    print(f"  {'feature':<26s}{'mean|SHAP|':>11s}  {'gain':>7s}")
    for feature, value in sorted(shap.items(), key=lambda kv: -kv[1]):
        print(f"  {feature:<26s}{value:>11.4f}  {gain[feature]:>7.4f} "
              f"{bar(value, width=34)}")

    return gain, shap


# ══════════════════════════════════════════════════════════════════
# Persistence
# ══════════════════════════════════════════════════════════════════

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
        "baselines": baselines,
        "cost_ratio_sensitivity": sensitivity.to_dict(orient="records"),
        "cross_validation": cv_results or None,
    }


def save_artifacts(model, metrics: dict) -> None:
    """Write the model binary and metrics.json."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODEL_DIR / MODEL_NAME
    model.save_model(str(model_path))
    print(f"  Model saved   {sym('arrow')} {model_path}")

    metrics_path = MODEL_DIR / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, default=float)
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
        frames["test"][TARGET_COL].to_numpy(), test_results["_scores"]
    )
    print(banner("Sensitivity to the FN:FP Cost Assumption (test)"))
    print(sensitivity.to_string(index=False,
                                float_format=lambda v: f"{v:,.4f}"))
    print("  (The ratio is the one number in the cost model nobody can verify, "
          "so the operating point is reported across a range of it.)")

    gain, shap = importance_report(model, frames["test"][FEATURE_COLS])

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
          f"{metrics['optimal_threshold']:.4f}: P {at_threshold['precision']:.4f} "
          f"R {at_threshold['recall']:.4f} F1 {at_threshold['f1']:.4f}, "
          f"{at_threshold['alerts_per_1000_accounts']:.1f} alerts/1,000")
    if baselines:
        rule = baselines["best_single_feature_rule_by_cost"]
        print(f"  Best single-feature rule was `{rule['rule']}` at F1 "
              f"{rule['test_f1']:.4f} — quote this model as lift over that, "
              "not in isolation.")


if __name__ == "__main__":
    main()
