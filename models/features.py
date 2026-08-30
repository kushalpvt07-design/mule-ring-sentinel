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

   `in_amount_sum` and `out_amount_sum` are the two deliberate exceptions to the
   "prefer bounded ratios" half of this rule. They stay unbounded because the cost
   model in models/cost_matrix.py is denominated in rupees and an analyst
   reviewing an alert needs the absolute exposure, not a ratio.

   They are NOT exceptions to the "window-stable always" half, and in v3 they
   were. So was `repeat_ratio`, which is the more interesting case because it
   looks scale-free: `n_txns / n_counterparties` is a ratio, but only the
   numerator grows with the window — a longer watch adds transactions to
   counterparties already seen far faster than it adds new counterparties.

   MEASURED, and this time with the estimator named. On
   `serving_context_edges.csv`, the full 60 days against the last 30, over the
   2,946 accounts present in both windows, as the median of per-account ratios
   (ratio of totals in brackets):

     feature          v3, raw           v4, referenced to 60 days
     in_amount_sum    2.0000 (1.9748)   1.0002 (0.9876)
     out_amount_sum   1.9893 (1.9949)   0.9949 (0.9977)
     repeat_ratio     1.8750 (1.8525)   0.9377 (0.9265)
     txn_velocity     0.9332            unchanged — already a rate
     distinct cps     1.0000 (1.1041)   unchanged — saturates, see below

   NAMING THE ESTIMATOR IS PART OF THE FIX. This rule previously cited "2.00x,
   2.00x, 1.84x" and "in_degree and out_degree 1.11x" with no estimator attached,
   and re-measuring showed those four figures came from three DIFFERENT ones: 2.00
   is a median, 1.11 is a mean (its median is 1.0000), and 1.84 is nearest the
   ratio of totals. Mixing estimators inside one list makes a spread of 1.0–1.11
   read as a single measured constant. The mean of per-account ratios is the worst
   choice of the three here and is the reason to state it: a ratio-of-ratios is
   heavy-tailed, so accounts with a near-zero half-window value dominate it, and
   `in_amount_sum` comes out at 3.17 under it. Median and totals are both quoted
   above precisely because they do not always agree.

   WHAT v4 DOES ABOUT IT. `data/extractor.py` multiplies all three by one
   per-graph constant, `REFERENCE_WINDOW_DAYS / observed_days`, so the emitted
   value is what the account would have shown over a 60-day watch. Two things
   follow, and both matter:

     * It cannot reorder accounts. One constant per graph means every account is
       multiplied by the same number, so within a split the rank order — and
       therefore every threshold a tree can choose — is bit-identical. This fix
       buys nothing in-split; it buys comparability BETWEEN graphs.
     * On the shipped splits the constant is 1.000067 / 1.000264 / 1.000112, so
       no published rupee figure moves. The dependency is what goes away.

   `repeat_ratio`'s correction is approximate and is documented as such in
   `window_scale`: it grows sublinearly (2x window → 1.875x), so dividing by the
   full factor overshoots and leaves ~6% the other way against 88% uncorrected. A
   fitted exponent would close that on one pair of windows and is not offered.

   THE OBLIGATION THIS USED TO CARRY, AND WHY IT IS NOT DISCHARGED. v3 kept these
   features comparable by REQUIRING every observation window to be the same
   length — "not a style preference, a correctness requirement" — and it was
   violated: v2 split the timeline 60/18/22, giving windows of 108/32/39 days, so
   train features were roughly 3x test features purely by duration, and the API's
   context file spanned train+val, a third scale again. v4 divides that
   requirement out of the three features that depended on it, but equal windows
   are still required for a different reason the rescale cannot touch: window
   length changes the graph's STRUCTURE, not just its magnitudes. Distinct
   counterparties saturate (1.00 median, 1.10 by totals — sublinear, not flat),
   which moves `degree_balance`, `reciprocity` and the cycle core. So
   `assert_equal_window_lengths()` in data/generator.py stays, tests/test_leakage.py
   keeps covering it, and anything that builds a graph to score against —
   including `serving_context_edges.csv` — must still span one window.

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

   v4 NOTE, and an admission about that tolerance — twice over, because the first
   attempt at this note was also wrong. `pagerank` is now emitted as pagerank × N
   (rule 3), which changes what a tolerance on it means. First, the honest reading
   of the v3 guard: raw pagerank has mean 1/N ≈ 3.4e-04 on these graphs, and the
   guard allowed an absolute 1e-4 — i.e. 29% of a typical value, and 10x the drift
   ever measured. "Within 1e-4" sounded tight and was nearly vacuous.

   Second, the direction of the fix, which this note previously understated as
   "cancels that term to first order". The cancellation is EXACT for the
   perturbation the test applies. Two new accounts transacting only with each other
   form a closed component: no edge enters it from the main graph and none leaves
   it. PageRank's teleport is uniform, so the only quantity that changes for a
   pre-existing account is the teleport magnitude, (1-d)/N → (1-d)/N'; the main
   component's linear system is otherwise untouched, so every raw value scales by
   exactly N/N' and emitting `raw × N'` cancels it in full. The expected drift is
   zero, not "materially smaller".

   That matters because it decides what the guard should be. The absolute 1e-3 that
   replaced v3's 1e-4 was the wrong SHAPE, not the wrong number: an absolute budget
   on a column defined as a multiple of the graph's own baseline means a different
   thing in every graph, and the retrain duly measured 6.9e-3 against it — solver
   residual from `nx.pagerank`'s power iteration, amplified by N, not a centrality
   shift. tests/test_features.py now asserts the invariant three ways instead:
   every other feature bit-identical, pagerank's rank order preserved
   (PAGERANK_PERTURBATION_RANK_CORRELATION, the claim this file already made in
   prose), pagerank's magnitude within a RELATIVE budget
   (PAGERANK_PERTURBATION_RELATIVE_TOLERANCE), and the ×N scaling itself pinned by
   `mean == 1.0`, which is an identity rather than a property of this data and is
   what keeps the relative budget from passing vacuously.

─────────────────────────────────────────────────────────────────────────────
v3 → v4 CHANGES
─────────────────────────────────────────────────────────────────────────────
NOTHING WAS ADDED OR DROPPED. FEATURE_COLS is the same 18 names in the same
order. Four of them mean something different, which is a more dangerous kind of
change than adding a column and is why the version is bumped:
`assert_feature_contract` below compares NAMES against the booster, so a
`sentinel_v3.xgb` left on disk would have loaded against v4 features and scored
every account confidently wrong without raising anything. Renaming the artefact
is what stops that — the API derives its path from MODEL_NAME, so the old file is
simply not found and the service refuses to start until `python -m models.train`
has run.

REDEFINED
  pagerank          Emitted as pagerank × N, a multiple of the uniform baseline.
                    Raw PageRank sums to 1, so its mean is exactly 1/N and the
                    column tracked node count: means of 0.000322997 / 0.000339328
                    / 0.000348918 are 1/3096, 1/2947, 1/2866 to nine decimals —
                    an 8% shift in the mean of the #2 feature by test AUC (0.786)
                    driven by nothing but how many accounts were in the window.
                    Rule 3 retired `community_size` for precisely this.
  clustering_coefficient
                    Computed on the DIRECTED graph. On the undirected projection a
                    layering loop (A→B→C→A) and reciprocal social payment
                    (A↔B↔C↔A) are the same triangle — while the v2/v3 docstrings
                    claimed this feature was what told them apart. It was the
                    weakest of the 18 instead: test AUC 0.524, 65% of test nodes
                    at exactly 0.0. Left unweighted, because networkx's weighted
                    variant normalises by the largest edge weight in the graph and
                    would trade a direction bug for a scale dependence.
  in_amount_sum     All three rescaled to a 60-day reference window (rule 3).
  out_amount_sum    Unit unchanged — still rupees, read as "per 60 days
  repeat_ratio      observed". On the shipped splits the factor is 1.0001-1.0003,
                    so no published figure moves; what goes away is a dependence
                    on window length that v3 held off by hand.

WHAT DID NOT CHANGE, and is worth stating because it bounds the retrain
None of the four can reorder accounts within one graph. Three are a multiplication
by a single per-graph constant, and the fourth is a per-node recomputation on the
same neighbourhood. So the trees see the same in-split orderings they always did,
and the retrain is not expected to move the headline numbers much — the exception
is `clustering_coefficient`, which is a genuinely different measurement and was
the weakest feature in the set, so it has room to improve and little to lose.

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
    "in_amount_sum": "total value received (per 60 days observed)",
    "out_amount_sum": "total value sent (per 60 days observed)",
    "flow_passthrough": "money in ≈ money out (pass-through account)",
    "pagerank": "centrality, as a multiple of the average account's",
    "clustering_coefficient": "how tightly its counterparties inter-transact",
    "cycle_participation": "sits on a repeating circular payment path",
    "reciprocity": "share of counterparties it both pays and is paid by",
    "fan_in_concentration": "inbound value concentrated in few sources",
    "txn_velocity": "transactions per hour while active",
    "burst_ratio": "transactions crammed into a single hour",
    "amount_cv": "variation across individual transaction amounts",
    "counterparty_amount_cv": "pays every counterparty near-identical amounts",
    "repeat_ratio": "transactions per counterparty (per 60 days observed)",
    "community_internal_ratio": "how closed its transaction community is",
}

# ──────────────────────────────────────────────────────────────────
# Model identity
# ──────────────────────────────────────────────────────────────────

# Bumped from v3 because four feature DEFINITIONS changed while all eighteen
# names stayed the same — see the v3 → v4 section above. That is the case the
# version number is actually load-bearing for: `assert_feature_contract` below
# compares names, so it cannot tell a v3 booster from a v4 one, and a stale
# `sentinel_v3.xgb` would have been served against redefined features in silence.
# Renaming the artefact makes the failure loud and immediate instead.
#
# api/main.py derives the path it loads from MODEL_NAME — it must never hard-code
# a filename, which is how it ended up serving a stale v1 model against v2's
# threshold. The same applies to `models/saved_models/metrics.json`, whose
# `model_version` tests/test_baselines.py compares against this constant: until
# `python -m models.train` has run, that file describes v3 and the suite fails on
# purpose. A metrics file describing a model that no longer exists is a published
# claim about nothing.
MODEL_NAME = "sentinel_v4.xgb"
MODEL_VERSION = "sentinel_v4"

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
    # First-ring attribution is lossy: `ring_id` records one ring per account, so
    # an account bridging two rings is credited to whichever edge was visited
    # first, and ring recall's denominator silently shrinks. These two columns
    # carry the FULL attribution set, pipe-separated, which is what lets
    # train.py::_attribution_slack report that loss exactly rather than assume it
    # is zero.
    #
    # Both are derived from the label and must never become features. Nothing
    # here enforces that by exclusion — it holds because every feature matrix in
    # models/train.py is built positively as `df[FEATURE_COLS]`, never by
    # dropping columns from the frame. Keep it that way: a `drop`-based X would
    # feed the answer straight into the model.
    #
    # They were emitted by the extractor and read by train.py from the start but
    # never declared here, so `process_split`'s column-set assertion rejected the
    # extractor's own output and the pipeline could not run at all. Declared
    # columns are the contract; emitting one without declaring it is the bug.
    "rings_attributed",
    "ring_types_attributed",
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
