"""
tests/test_leakage.py
─────────────────────
Split hygiene: the tests that decide whether the reported precision and recall
mean anything at all.

─────────────────────────────────────────────────────────────────────────────
WHAT THE PREVIOUS VERSION OF THIS FILE CHECKED, AND WHY IT WASN'T ENOUGH
─────────────────────────────────────────────────────────────────────────────
It checked one thing: that the earliest test timestamp is not before the latest
train timestamp. That is necessary and nowhere near sufficient, and it passed
throughout the period when the evaluation was, in fact, meaningless.

Three leaks it could not see:

  1. ENTITY LEAKAGE. v1 scattered each ring's transactions across the whole
     six-month timeline, so a ring had members in January and in June. Every
     timestamp was in the right split; 25 of 25 test rings still had members the
     model had partly memorised from train. Timestamps say nothing about which
     ACCOUNTS crossed the boundary — this file now checks the accounts and the
     ring ids directly, across all three splits, at both the edge level and the
     node level the model is actually trained on.

  2. UNEQUAL OBSERVATION WINDOWS. v2 split the timeline 60/18/22, producing
     windows of 108 / 32 / 39 days. `in_amount_sum`, `out_amount_sum` and
     `txn_velocity` are magnitudes: watch an account three times as long and they
     roughly triple with no change in behaviour. Train features were on a
     different scale from test features, which the strictly-ordered timestamps
     were perfectly happy about. The window contract is now asserted on day
     spans, and — more usefully — on the feature values themselves.

  3. SERVING SKEW. `serving_context_edges.csv` spanned train+val, a third scale
     again, so the API computed features the model had never seen the like of.
     Checked here too, because a leak at serving time costs real money rather
     than a wrong number in a README.

─────────────────────────────────────────────────────────────────────────────
WHAT IS DELIBERATELY *NOT* FORBIDDEN
─────────────────────────────────────────────────────────────────────────────
Ordinary accounts recur across windows. That is not leakage, it is the problem:
you train on January's customers and score those same customers in May. Account
ids are not features, and features are recomputed from each window's own edges,
so a recurring account has entirely different values in each split and there is
nothing to memorise. `test_legitimate_accounts_do_recur` asserts the overlap is
LARGE, because a dataset where it were zero would be easier than production and
would flatter the model.

Likewise, an account that is clean in an earlier window and a ring member in a
later one is the recruitment timeline, and both labels are correct for their own
window. The reverse — a known mule returned to service as a clean negative —
would be a labelling contradiction, and is asserted to be impossible.

Usage:
    pytest tests/test_leakage.py -v
"""

from __future__ import annotations

import pandas as pd
import pytest

from data.generator import (
    RING_ARCHETYPES,
    WINDOW_LENGTH_TOLERANCE_DAYS,
    ring_member_nodes,
)
from models.features import TARGET_COL

SPLITS = ("train", "val", "test")
ADJACENT = (("train", "val"), ("val", "test"))
ALL_PAIRS = (("train", "val"), ("train", "test"), ("val", "test"))

# Ratio between the largest and smallest per-split mean of a magnitude feature,
# measured over NEGATIVES only so a difference in ring prevalence cannot explain
# it. Measured on the shipped data: 1.04-1.06 for every feature below. v2's
# unequal windows produced roughly 3x, so 1.25 fails loudly on the defect while
# leaving ample room for ordinary sampling noise.
MAGNITUDE_RATIO_CEILING = 1.25

MAGNITUDE_FEATURES = (
    "in_amount_sum",
    "out_amount_sum",
    "txn_velocity",
    "in_degree",
    "out_degree",
)


def _span_days(df: pd.DataFrame) -> float:
    return float(
        (df["timestamp"].max() - df["timestamp"].min()).total_seconds() / 86_400.0)


# ══════════════════════════════════════════════════════════════════
# 1. Temporal order
# ══════════════════════════════════════════════════════════════════

class TestTemporalOrder:
    """The necessary-but-insufficient part: no split contains the future."""

    @pytest.mark.parametrize("earlier,later", ADJACENT)
    def test_split_boundaries_do_not_overlap(self, raw_edges, earlier, later):
        e_max = raw_edges[earlier]["timestamp"].max()
        l_min = raw_edges[later]["timestamp"].min()
        assert l_min >= e_max, (
            f"TEMPORAL LEAKAGE: '{later}' starts at {l_min}, before '{earlier}' "
            f"ends at {e_max} (overlap {e_max - l_min}). The model would be "
            f"trained on transactions it is later scored against."
        )

    def test_train_val_test_are_strictly_ordered(self, raw_edges):
        """Transitivity too, so a mis-ordered val cannot hide behind two pairs."""
        maxima = [raw_edges[s]["timestamp"].max() for s in SPLITS]
        minima = [raw_edges[s]["timestamp"].min() for s in SPLITS]
        assert minima == sorted(minima) and maxima == sorted(maxima), (
            f"splits are not in chronological order: "
            f"starts {minima}, ends {maxima}"
        )

    def test_no_identical_transaction_in_two_splits(self, raw_edges):
        """
        A duplicated row would be the same event scored twice, once as training
        signal and once as held-out evidence.
        """
        cols = ["sender", "receiver", "timestamp", "amount"]
        keys = {
            s: set(map(tuple, raw_edges[s][cols].itertuples(index=False, name=None)))
            for s in SPLITS
        }
        for a, b in ALL_PAIRS:
            shared = keys[a] & keys[b]
            assert not shared, (
                f"{len(shared)} identical transaction(s) appear in both '{a}' and "
                f"'{b}'. Examples: {sorted(shared)[:3]}"
            )


# ══════════════════════════════════════════════════════════════════
# 2. Entity and ring disjointness — the leak timestamps cannot see
# ══════════════════════════════════════════════════════════════════

class TestEntityDisjointness:
    """
    No ring, and no ring member, may appear in two splits.

    This is the check that would have caught v1, where every timestamp was in
    the correct split and 25 of 25 test rings were still partly memorised.
    """

    @pytest.mark.parametrize("a,b", ALL_PAIRS)
    def test_ring_members_are_disjoint_at_edge_level(self, raw_edges, a, b):
        """Uses the shipped definition of a ring member, not a copy of it."""
        shared = ring_member_nodes(raw_edges[a]) & ring_member_nodes(raw_edges[b])
        assert not shared, (
            f"ENTITY LEAKAGE: {len(shared)} account(s) sit on a ring edge in both "
            f"'{a}' and '{b}'. Examples: {sorted(shared)[:5]}\n"
            f"The model would be scored on rings it had partly seen."
        )

    @pytest.mark.parametrize("a,b", ALL_PAIRS)
    def test_ring_ids_are_disjoint_at_edge_level(self, raw_edges, a, b):
        def rings(df):
            return set(df.loc[df["ring_id"] >= 0, "ring_id"].unique())

        shared = rings(raw_edges[a]) & rings(raw_edges[b])
        assert not shared, (
            f"RING LEAKAGE: ring_id(s) {sorted(shared)[:5]} appear in both "
            f"'{a}' and '{b}'."
        )

    @pytest.mark.parametrize("a,b", ALL_PAIRS)
    def test_positive_nodes_are_disjoint_in_the_feature_tables(
        self, node_features, a, b
    ):
        """
        The same guarantee at the level the model is actually fitted on.

        The edge-level checks above could both pass while the extractor's
        node labelling still put one account in the positive class of two splits
        — for instance if `label_nodes` derived mule status from edge incidence
        rather than from ring membership. This closes that gap.
        """
        def positives(df):
            return set(df.loc[df[TARGET_COL] == 1, "node"])

        shared = positives(node_features[a]) & positives(node_features[b])
        assert not shared, (
            f"POSITIVE-CLASS LEAKAGE: {len(shared)} account(s) are labelled mules "
            f"in both '{a}' and '{b}'. Examples: {sorted(shared)[:5]}"
        )

    @pytest.mark.parametrize("a,b", ALL_PAIRS)
    def test_ring_ids_are_disjoint_in_the_feature_tables(self, node_features, a, b):
        """`ring_id` is the CV grouping key, so a shared id also breaks the folds."""
        def rings(df):
            return set(df.loc[df["ring_id"] >= 0, "ring_id"].unique())

        shared = rings(node_features[a]) & rings(node_features[b])
        assert not shared, (
            f"RING LEAKAGE in the feature tables: ring_id(s) {sorted(shared)[:5]} "
            f"appear in both '{a}' and '{b}'. Cross-validation folds grouped on "
            f"ring_id would not be independent."
        )

    @pytest.mark.parametrize("a,b", ALL_PAIRS)
    def test_no_mule_is_returned_to_service_as_a_negative(
        self, node_features, a, b
    ):
        """
        Recruitment runs one way only.

        Clean-then-mule is the recruitment timeline and is expected (measured: 52
        to 54 accounts per pair). Mule-then-clean would be a labelling
        contradiction — the same account asserted to be both, with the negative
        label training the model to ignore a known ring member.
        """
        fa = node_features[a].set_index("node")[TARGET_COL]
        fb = node_features[b].set_index("node")[TARGET_COL]
        common = fa.index.intersection(fb.index)
        contradictory = common[(fa.loc[common] == 1) & (fb.loc[common] == 0)]
        assert len(contradictory) == 0, (
            f"{len(contradictory)} account(s) are labelled mules in '{a}' and "
            f"clean in '{b}'. Examples: {sorted(contradictory)[:5]}\n"
            f"A ringed account should be frozen, not returned to service as a "
            f"negative training example."
        )

    def test_legitimate_accounts_do_recur(self, node_features):
        """
        The overlap that SHOULD exist.

        Asserted as a floor, not a ceiling. If a future change made the splits
        share no accounts at all, the dataset would have stopped resembling
        production — where you score the same customers month after month — and
        every reported number would be flattering for the wrong reason. Measured:
        2,841-2,894 shared accounts per pair.
        """
        for a, b in ALL_PAIRS:
            shared = set(node_features[a]["node"]) & set(node_features[b]["node"])
            smaller = min(len(node_features[a]), len(node_features[b]))
            assert len(shared) > 0.5 * smaller, (
                f"only {len(shared)} of {smaller} accounts recur between '{a}' and "
                f"'{b}'. Splits that share no population are easier than "
                f"production; check the generator's account pool."
            )


# ══════════════════════════════════════════════════════════════════
# 3. The equal-window contract (models/features.py design rule 3)
# ══════════════════════════════════════════════════════════════════

class TestEqualObservationWindows:
    """
    Every window must be the same length, because six features are magnitudes.

    Asserted on DAY SPANS and on FEATURE MEANS. Not on medians: an integer-valued
    median (`in_degree`, say) is quantised so coarsely that a 3x window change can
    leave it untouched, which is how this defect survived a "distributions look
    similar" eyeball check.
    """

    def test_split_windows_are_equal_length(self, raw_edges):
        spans = {s: _span_days(df) for s, df in raw_edges.items()}
        spread = max(spans.values()) - min(spans.values())
        assert spread <= WINDOW_LENGTH_TOLERANCE_DAYS, (
            "UNEQUAL OBSERVATION WINDOWS: "
            + ", ".join(f"{s}={d:.2f}d" for s, d in spans.items())
            + f" (spread {spread:.2f}d > {WINDOW_LENGTH_TOLERANCE_DAYS}d).\n"
            "Every count, sum and rate feature scales with window length, so the "
            "splits are not comparable. Fix SPLIT_FRACTIONS in data/generator.py."
        )

    def test_serving_context_window_matches_training(
        self, raw_edges, serving_context
    ):
        """
        The API's context graph must span one window too.

        v2's context file covered train+val — roughly double — so every magnitude
        feature the API computed was on a scale the model had never been fitted
        against. Nothing raised; the scores were simply wrong.
        """
        train_span = _span_days(raw_edges["train"])
        context_span = _span_days(serving_context)
        drift = abs(context_span - train_span)
        assert drift <= WINDOW_LENGTH_TOLERANCE_DAYS, (
            f"SERVING SKEW: serving_context_edges.csv spans {context_span:.2f}d "
            f"against a trained window of {train_span:.2f}d (drift {drift:.2f}d).\n"
            f"in_amount_sum, out_amount_sum and txn_velocity would be on a "
            f"different scale at serving time than in training."
        )

    @pytest.mark.parametrize("feature", MAGNITUDE_FEATURES)
    def test_magnitude_features_are_comparable_across_splits(
        self, node_features, feature
    ):
        """
        The window contract, verified where it actually bites: the values.

        Day spans are the cause; these means are the effect, and the effect is
        what the model sees. Restricted to negatives so that differing ring
        prevalence between splits cannot masquerade as window drift.
        """
        means = {
            s: float(df.loc[df[TARGET_COL] == 0, feature].mean())
            for s, df in node_features.items()
        }
        lo, hi = min(means.values()), max(means.values())
        assert lo > 0, f"{feature} has a zero mean on the negatives of some split"
        ratio = hi / lo
        assert ratio <= MAGNITUDE_RATIO_CEILING, (
            f"{feature} differs by {ratio:.2f}x across splits "
            + "(" + ", ".join(f"{s}={m:,.4f}" for s, m in means.items()) + ").\n"
            f"A magnitude feature on a different scale per split is a "
            f"distribution shift the model cannot see or correct for. The usual "
            f"cause is unequal observation windows."
        )


# ══════════════════════════════════════════════════════════════════
# 4. The serving context must not be the held-out set
# ══════════════════════════════════════════════════════════════════

class TestServingContextIsolation:
    """
    The demo graph must not be the data the headline metrics were measured on.

    Pointing the API's context file at `test_edges.csv` would make the live demo
    look better and quietly turn the held-out split into training input for the
    graph features. Cheap to do by accident when regenerating; caught here.
    """

    def test_context_does_not_overlap_the_test_window(
        self, raw_edges, serving_context
    ):
        test_start = raw_edges["test"]["timestamp"].min()
        context_end = serving_context["timestamp"].max()
        assert context_end < test_start, (
            f"serving_context_edges.csv extends to {context_end}, into the test "
            f"window that opens at {test_start}. The held-out split would be "
            f"contributing graph structure to live scores, and the reported "
            f"metrics would no longer describe unseen data."
        )

    def test_context_transactions_are_not_test_transactions(
        self, raw_edges, serving_context
    ):
        cols = ["sender", "receiver", "timestamp", "amount"]
        ctx = set(map(tuple, serving_context[cols].itertuples(index=False, name=None)))
        tst = set(map(tuple, raw_edges["test"][cols].itertuples(index=False, name=None)))
        shared = ctx & tst
        assert not shared, (
            f"{len(shared)} transaction(s) appear in both the serving context and "
            f"the test split."
        )


# ══════════════════════════════════════════════════════════════════
# 5. Structural sanity — cheap guards on numbers everything else rests on
# ══════════════════════════════════════════════════════════════════

class TestStructuralSanity:
    """Defects here would silently corrupt every metric downstream."""

    @pytest.mark.parametrize("split", SPLITS)
    def test_split_has_both_classes(self, node_features, split):
        counts = node_features[split][TARGET_COL].value_counts()
        assert set(counts.index) == {0, 1}, (
            f"'{split}' does not contain both classes (found {dict(counts)}). "
            f"Precision and recall are undefined."
        )

    @pytest.mark.parametrize("split", SPLITS)
    def test_prevalence_is_plausible(self, node_features, split):
        """
        Between 0.5% and 25%.

        Not a style check. Below the floor a single flipped label moves recall by
        percentage points and the confidence intervals swamp the result; above the
        ceiling the "rare event" framing the cost matrix assumes breaks down.
        Measured: 6.34% / 3.72% / 4.09%.
        """
        prevalence = float(node_features[split][TARGET_COL].mean())
        assert 0.005 <= prevalence <= 0.25, (
            f"'{split}' prevalence is {prevalence:.2%}, outside [0.5%, 25%]."
        )

    @pytest.mark.parametrize("split", SPLITS)
    def test_every_ring_archetype_appears(self, node_features, split):
        """
        Otherwise the per-archetype recall table is not comparable across splits,
        and an aggregate recall can hide "we only catch the loud ones".
        """
        present = set(
            node_features[split].loc[
                node_features[split][TARGET_COL] == 1, "ring_type"].unique())
        expected = {a.name for a in RING_ARCHETYPES}
        assert expected <= present, (
            f"'{split}' is missing archetype(s) {sorted(expected - present)}; "
            f"per-archetype recall would not be comparable across splits."
        )

    @pytest.mark.parametrize("split", SPLITS)
    def test_no_self_transfers(self, raw_edges, split):
        """A self-loop inflates both degrees and fakes a one-account cycle."""
        loops = raw_edges[split]["sender"] == raw_edges[split]["receiver"]
        assert not loops.any(), (
            f"{int(loops.sum())} self-transfer(s) in '{split}'."
        )

    @pytest.mark.parametrize("split", SPLITS)
    def test_amounts_are_positive(self, raw_edges, split):
        bad = raw_edges[split]["amount"] <= 0
        assert not bad.any(), (
            f"{int(bad.sum())} non-positive amount(s) in '{split}'."
        )

    @pytest.mark.parametrize("split", SPLITS)
    def test_negatives_have_no_ring_metadata(self, node_features, split):
        """
        A negative carrying a real ring_id would corrupt the CV grouping and the
        per-archetype breakdown.
        """
        neg = node_features[split][node_features[split][TARGET_COL] == 0]
        assert (neg["ring_id"] == -1).all(), (
            f"{int((neg['ring_id'] != -1).sum())} negative(s) in '{split}' carry a "
            f"ring_id."
        )
        assert (neg["ring_type"] == "organic").all(), (
            f"negative(s) in '{split}' carry a non-organic ring_type: "
            f"{sorted(set(neg.loc[neg['ring_type'] != 'organic', 'ring_type']))}"
        )

    @pytest.mark.parametrize("split", SPLITS)
    def test_positives_all_carry_ring_metadata(self, node_features, split):
        """The mirror image: a mule with no ring is ungroupable and unexplainable."""
        pos = node_features[split][node_features[split][TARGET_COL] == 1]
        assert (pos["ring_id"] >= 0).all(), (
            f"{int((pos['ring_id'] < 0).sum())} mule(s) in '{split}' have no "
            f"ring_id, so they cannot be grouped for cross-validation."
        )
        assert (pos["ring_type"] != "organic").all(), (
            f"{int((pos['ring_type'] == 'organic').sum())} mule(s) in '{split}' "
            f"have ring_type 'organic'."
        )
