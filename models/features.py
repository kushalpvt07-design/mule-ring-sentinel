"""
models/features.py
──────────────────
Single source of truth for the model's feature contract.

Everything that computes, trains on, serves, or displays features imports from
this module. Three copies of a feature list is three chances for training and
serving to drift apart, and drift in an ML service is silent: nothing errors,
every score is quietly wrong.

─────────────────────────────────────────────────────────────────────────────
DESIGN RULES for anything added to FEATURE_COLS
─────────────────────────────────────────────────────────────────────────────
1. STRUCTURAL, NOT NOMINAL.
   A feature must describe the shape of behaviour, not carry an arbitrary
   identifier. `louvain_community` was dropped in v2 for this reason: XGBoost
   was splitting on "community id < 17.5", which is meaningless, and the
   numbering had no correspondence between graphs.

2. IDENTICALLY COMPUTABLE AT TRAIN AND SERVE TIME.
   Every feature here comes out of `data.extractor.compute_node_features`, and
   both `models/train.py` and `api/main.py` call exactly that function. A
   feature that can only be computed in batch does not belong in this list.

3. SCALE-FREE WHERE POSSIBLE, AND WINDOW-STABLE ALWAYS.
   Prefer bounded ratios over raw magnitudes. `community_size` was dropped in
   v3: it is a raw count that grows with the graph, and empirically it scored
   AUC 0.10 — i.e. it had become an inverted proxy for "am I in the giant
   organic blob", which is a property of the sample, not of fraud.

   WHICH SPLIT SAID SO — AND THE ADMISSION THAT GOES WITH IT.
   That 0.10 was originally measured on TEST, and this file used to cite it as
   "test AUC 0.10". Dropping a feature on the strength of a test-split number is
   selection on test: it makes the final evaluation partly a measurement of
   choices that were themselves tuned against it, which is the exact discipline
   the rest of this repo refuses to break (see models/cost_matrix.py on
   threshold selection, and `sensitivity_to_cost_ratio`'s v3 note). The decision
   stands, because the argument for it is a priori — a count that grows with the
   graph is not comparable across windows regardless of what it scores — but the
   number is no longer offered as the reason, and cannot now be re-measured on
   validation because the column is not emitted any more. Feature-level
   decisions from here on are made on VALIDATION (`FEATURE_DECISION_SPLIT`
   below); the historical provenance is recorded rather than quietly deleted,
   because a repo whose selling point is honest metrics does not get to launder
   its own history.

   `in_amount_sum` and `out_amount_sum` are the two deliberate exceptions. They
   stay because the cost model in models/cost_matrix.py is denominated in rupees
   and an analyst reviewing an alert needs the absolute exposure, not a ratio.

   The exception carries an obligation. A sum, a count and a rate all scale with
   how long you watched the account, so these features are only comparable
   across train, validation, test and serving if every observation window is the
   SAME LENGTH. That is not a style preference, it is a correctness requirement,
   and it was violated: v2 split the timeline 60/18/22, giving windows of
   108/32/39 days, so train features were roughly 3x test features purely by
   duration — and the API's context file spanned train+val, a third scale again,
   which is silent train/serve skew.

   The constant itself lives in data/generator.py (a models/ module must not
   depend on data/), enforced there by `assert_equal_window_lengths()` and
   covered by tests/test_leakage.py. Anything that builds a graph to score
   against — including `serving_context_edges.csv` — must span one window.

4. NO FEATURE MAY ENCODE THE LABEL.
   The generator must not be able to plant a feature that separates classes by
   construction. The ceiling is `LEAKAGE_AUC_CEILING` (0.99) on the
   direction-corrected single-feature AUC, and it is SCREENED ON VALIDATION —
   `FEATURE_DECISION_SPLIT`. models/train.py's `single_feature_rule_baseline`
   takes `screen_split` for exactly this and defaults it to validation, so the
   number that can veto a feature never comes from test.

   Why validation and not test: this ceiling is a decision gate. Reading it on
   test and then acting on it — dropping a column, regenerating the data —
   consumes the one evaluation the split exists for, and does so repeatedly,
   which is how a "held-out" number becomes a fitted one. Validation is already
   spent on early stopping and threshold selection, so spending it here costs
   nothing that has not already been spent.

   ADMISSION: through v3 this rule read "AUC >= 0.99 on the test split", and the
   guard in tests/test_baselines.py still measures the shipped TEST table,
   because its job is different — it is a post-hoc audit of published data
   ("does the dataset in this repo match the claim in metrics.json?"), not an
   input to any choice, and metrics.json's `strongest_single_feature_test_auc`
   is the figure it reconciles against. Both numbers are now reported with the
   split in their name. On the shipped v3 data the strongest single feature is
   `cycle_participation` at 0.8485 on validation and 0.8708 on test — the gate
   fires on the first of those.

5. STABLE UNDER GRAPH CHANGES THAT DO NOT TOUCH THE ACCOUNT.
   Adding an unrelated account, or a transaction between two strangers, must not
   move an account's features. Determinism is not enough — the same graph scored
   twice matching is the easy half. The requirement is that the feature depends
   only on the account's own neighbourhood, or degrades gracefully when it can't.

   Two features need care:

   `community_internal_ratio` failed this outright. Louvain shuffles node visit
   order from its seed, so a fixed seed is reproducible on a fixed node set but
   the partition moves the moment the node set changes — and since the ratio is a
   per-COMMUNITY scalar broadcast to every member, one repartition moves the
   feature for accounts nowhere near the change. Measured on the 2,954-account
   validation graph: two accounts transacting only with each other, connected to
   nothing else, moved this feature for 100% of accounts (median |Δ| 0.0151, max
   0.2906) and flipped 84 of 2,954 decisions — 2.84% — at the cost-optimal
   threshold. Pinning that one feature to its original value dropped the flips to
   zero. The fix is not to drop the feature but to stop redrawing communities per
   request: serving computes the partition ONCE from the reference graph and
   passes it into `compute_node_features(..., partition=...)`, which extends it
   deterministically for unseen accounts. See `data.extractor.extend_partition`.

   `pagerank` is a global fixpoint over a normalised rank vector, so strictly
   every node shifts when any node is added. That is inherent to the definition,
   not a bug, and the magnitude is what matters: the same perturbation moved
   pagerank by at most 1.0e-05 with rank correlation > 0.9999. Bounded and
   negligible is acceptable; 0.29 is not.

   tests/test_features.py asserts this: perturb the graph with disconnected
   accounts, require bit-identical values for every feature except pagerank, and
   require pagerank within 1e-4.

─────────────────────────────────────────────────────────────────────────────
v2 → v3 CHANGES
─────────────────────────────────────────────────────────────────────────────
DROPPED
  community_size    Raw count, scale-dependent (rule 3). AUC 0.10 — an
                    inverted proxy for organic-blob membership. (That figure was
                    read on test; see rule 3. The scale argument, not the
                    figure, is what carries the decision.)
  net_flow          Redundant. `in_amount_sum - out_amount_sum` is a linear
                    combination of two features already in the list, and
                    `flow_passthrough` already expresses the pass-through
                    signature it was there to capture — scale-free, and without
                    a third rupee magnitude for a tree to split on arbitrarily.

ADDED
  cycle_participation  Fraction of the node's repeated-edge neighbours that lie
                       on a short directed cycle through it. This is the actual
                       thesis of the project ("rings are circular") expressed as
                       a feature, rather than being left implicit in PageRank.
  reciprocity          Share of counterparties that both pay and get paid.
                       Distinguishes a layering cycle from ordinary two-way
                       social payment, which is why the generator now emits
                       reciprocal organic traffic.
  burst_ratio          Busiest hour's share of the node's transactions.
                       Separates "many transactions" from "many transactions
                       crammed into a window", which is the laundering pattern.
  counterparty_amount_cv  CV of a node's *per-counterparty mean* amounts.
                       A ring pays every hop near-identically; a real user pays
                       rent, a kirana store and a friend very differently. This
                       survives the generator fix that killed raw `amount_cv`
                       as a giveaway.

KEPT BUT NO LONGER LOAD-BEARING
  repeat_ratio      Still informative, but the v2 generator made it a
                    giveaway (AUC 0.9989) by drawing organic
                    counterparties i.i.d., so legitimate accounts essentially
                    never paid anyone twice. Real users repeat constantly. The
                    v3 generator gives organic accounts recurring
                    relationships, which is what moves this feature from
                    "artifact" to "signal": on v3 data it scores 0.6315 on
                    validation (0.6632 on test), comfortably below the rule-4
                    ceiling.

                    Provenance, stated rather than tidied away: the 0.9989 was
                    a TEST-split reading, and it is the number that triggered
                    the v2 → v3 generator rewrite — a dataset-level decision
                    made on test. It cannot be re-measured on validation now
                    because the v2 data no longer exists; the two v3 figures
                    above are the ones that can be, and the validation one is
                    what rule 4 screens.
"""

from __future__ import annotations

# ──────────────────────────────────────────────────────────────────
# The contract
# ──────────────────────────────────────────────────────────────────

# Ordered exactly as the model expects them. Order matters: XGBoost stores
# feature names on the booster, and assert_feature_contract() below compares
# this list against them positionally.
FEATURE_COLS: list[str] = [
    # ── Degree structure ──
    "in_degree",
    "out_degree",
    "degree_ratio",
    "degree_balance",
    # ── Value flow ──
    "in_amount_sum",
    "out_amount_sum",
    "flow_passthrough",
    # ── Centrality / local topology ──
    "pagerank",
    "clustering_coefficient",
    "cycle_participation",
    "reciprocity",
    # ── Behavioural ──
    "fan_in_concentration",
    "txn_velocity",
    "burst_ratio",
    "amount_cv",
    "counterparty_amount_cv",
    "repeat_ratio",
    # ── Community structure (scale-free only) ──
    "community_internal_ratio",
]

TARGET_COL = "is_mule"

# ──────────────────────────────────────────────────────────────────
# Which split is allowed to decide things (design rules 3 and 4)
# ──────────────────────────────────────────────────────────────────

# The split any FEATURE-LEVEL decision must be measured on: dropping a column,
# demoting one, or firing the leakage gate below. Test is reserved for the single
# final evaluation, and a feature set chosen against test turns that evaluation
# into a partial measurement of its own selection — the same failure as tuning a
# threshold on test, just one level up and harder to see.
#
# It lives here, next to the contract, because the contract IS the accumulated
# result of those decisions; a constant in models/train.py would be invisible to
# anyone reading the list of features to find out how it was chosen.
FEATURE_DECISION_SPLIT = "val"

# Rule 4's ceiling on direction-corrected single-feature AUC. Above this, one
# column essentially solves the task and the generator planted the label rather
# than the model learning it, so every downstream metric is theatre.
#
# Declared here rather than in models/train.py so the number the contract cites
# and the number the code enforces cannot drift: train.py imports it (and keeps
# re-exporting the name `LEAKAGE_AUC_CEILING`, which tests/test_baselines.py
# imports to assert agreement), and `single_feature_rule_baseline(screen_split=)`
# defaults to FEATURE_DECISION_SPLIT above.
LEAKAGE_AUC_CEILING = 0.99

# Analyst-facing descriptions. api/main.py returns these alongside per-node
# SHAP attributions so a reviewer sees "money in ≈ money out" rather than
# "flow_passthrough=0.98". A risk product that cannot explain itself does not
# get used, and the field name `contributing_factors` promises this.
FEATURE_DESCRIPTIONS: dict[str, str] = {
    "in_degree": "number of distinct accounts paying in",
    "out_degree": "number of distinct accounts paid out to",
    "degree_ratio": "ratio of payees to payers",
    "degree_balance": "how evenly split between paying in and out",
    "in_amount_sum": "total value received",
    "out_amount_sum": "total value sent",
    "flow_passthrough": "money in ≈ money out (pass-through account)",
    "pagerank": "centrality in the payment network",
    "clustering_coefficient": "how tightly its counterparties inter-transact",
    "cycle_participation": "sits on a repeating circular payment path",
    "reciprocity": "share of counterparties it both pays and is paid by",
    "fan_in_concentration": "inbound value concentrated in few sources",
    "txn_velocity": "transactions per hour while active",
    "burst_ratio": "transactions crammed into a single hour",
    "amount_cv": "variation across individual transaction amounts",
    "counterparty_amount_cv": "pays every counterparty near-identical amounts",
    "repeat_ratio": "transactions per distinct counterparty",
    "community_internal_ratio": "how closed its transaction community is",
}

# ──────────────────────────────────────────────────────────────────
# Model identity
# ──────────────────────────────────────────────────────────────────

# Bumped from v2 because the feature contract changed. api/main.py derives the
# path it loads from MODEL_NAME — it must never hard-code a filename, which is
# how it ended up serving a stale v1 model against v2's threshold.
MODEL_NAME = "sentinel_v3.xgb"
MODEL_VERSION = "sentinel_v3"

# Columns the extractor emits that are NOT fed to the model. Retained in the
# CSVs for analysis, joins, dashboard display and CV grouping.
METADATA_COLS: list[str] = [
    "node",
    "louvain_community",  # inspection/plots only — never a feature (rule 1)
    "split",
]

# Ground-truth columns the extractor attaches from the edge file. `ring_id` is
# the correct grouping key for cross-validation: grouping on Louvain community is
# unsound because community size is unbounded and unrelated to the label — on v2's
# data it put 52% of train nodes, and zero positives, in a single group, so folds
# were silently skipped. `ring_type` enables per-archetype recall reporting,
# which is where an honest evaluation shows the stealthy rings are harder.
LABEL_META_COLS: list[str] = [
    "ring_id",
    "ring_type",
]


def assert_feature_contract(booster_feature_names: list[str] | None) -> None:
    """
    Fail loudly if a loaded model was trained on a different feature set.

    Silently serving a model whose columns don't line up is the single most
    expensive failure mode in an ML service: nothing raises, every score is
    wrong, and the bug is invisible until someone audits outcomes.

    This is not hypothetical here. Before v3, api/main.py hard-coded a 12-name
    list matching the retired `sentinel_v1.xgb`, so it loaded that stale model
    without error and applied the *current* model's threshold to it. This
    function existed at the time and was never called. It is now called from
    the API's startup path, and covered by tests/test_contract.py.
    """
    if booster_feature_names is None:
        # Model was trained from a raw numpy array, so XGBoost kept no names.
        # models/train.py always fits from a DataFrame, so this means the model
        # was produced by something other than our pipeline.
        raise RuntimeError(
            "FEATURE CONTRACT UNVERIFIABLE — refusing to serve this model.\n"
            "  The booster carries no feature names, which means it was not\n"
            "  trained from a DataFrame by models/train.py.\n"
            "  Retrain with `python -m models.train`."
        )

    if list(booster_feature_names) != FEATURE_COLS:
        model_set, contract_set = set(booster_feature_names), set(FEATURE_COLS)
        missing = [c for c in FEATURE_COLS if c not in model_set]
        extra = [c for c in booster_feature_names if c not in contract_set]
        reordered = (
            not missing
            and not extra
            and list(booster_feature_names) != FEATURE_COLS
        )
        raise RuntimeError(
            "FEATURE CONTRACT MISMATCH — refusing to serve this model.\n"
            f"  Model was trained on {len(booster_feature_names)} features; "
            f"FEATURE_COLS declares {len(FEATURE_COLS)}.\n"
            f"  Missing from model:  {missing or 'none'}\n"
            f"  Unexpected in model: {extra or 'none'}\n"
            + ("  Same features, DIFFERENT ORDER — XGBoost is positional, so "
               "every score would be wrong.\n" if reordered else "")
            + "  Retrain with `python -m models.train`."
        )
