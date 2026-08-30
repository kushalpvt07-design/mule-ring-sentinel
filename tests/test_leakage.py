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
     windows of 108 / 32 / 39 days. `in_amount_sum` and `out_amount_sum` are
     magnitudes: watch an account three times as long and they roughly triple with
     no change in behaviour (`repeat_ratio` drifts the same way, sub-linearly).
     Train features were on a different scale from test features, which the
     strictly-ordered timestamps were perfectly happy about. v4 rescales those
     three to a 60-day reference window; the window contract is still asserted on
     day spans and — more usefully — on the feature values themselves, because
     window length also changes graph STRUCTURE, which no rescale can correct.

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
# taken over NEGATIVES only so a difference in ring prevalence between splits
# cannot masquerade as scale drift.
#
# THE CEILING IS A BOUND, NOT A MEASUREMENT. The shipped splits sit at 1.09-1.11
# for every feature below. An earlier version of this comment claimed 1.04-1.06;
# that was wrong, and a guard whose stated margin is fictitious is worse than
# none — it invites someone to "tighten it to the measured range" and trip a
# false failure. 1.18 is chosen to clear the observed 1.11 with headroom for
# regeneration noise, not measured: the true post-regen ratio cannot be known
# until the pipeline is re-run on this machine. If the regen shows it well below
# 1.18, tighten this and record the figure here.
#
# This is a SECONDARY guard. `test_split_windows_are_equal_length` bounds the
# window spread to WINDOW_LENGTH_TOLERANCE_DAYS directly, which caps the
# window-driven component of any magnitude ratio at ~2.5% on a ~60-day window;
# v2's unequal windows produced roughly 3x and would fail THAT test first. What
# this ceiling adds is a check on the effect the model actually sees, catching a
# rescale regression that skews magnitudes without touching the day spans — e.g.
# the v4 reference-window rescale of the amount sums being dropped.
MAGNITUDE_RATIO_CEILING = 1.18

# Per-split scale-stability probes. in_amount_sum and out_amount_sum are the
# rescaled magnitudes this primarily guards — a dropped v4 rescale spikes their
# ratio on unequal windows. txn_velocity is a rate and in_/out_degree are counts;
# they are included because a large cross-split drift in ANY of them betrays
# unequal windows regardless of feature kind. Named for the dominant case, not a
# claim that every member is a magnitude.
MAGNITUDE_FEATURES = (
    "in_amount_sum",
    "out_amount_sum",
    "txn_velocity",
    "in_degree",
    "out_degree",
)


def _span_days(df: pd.DataFrame) -> float:
    """
    Elapsed days between a split's first and last transaction.

    `total_seconds() / 86_400` measures absolute elapsed time rather than
    wall-clock days, and that is the right measure here rather than an accident of
    convenience. Every timestamp that reaches this arithmetic is UTC: the
    generator writes naive UTC, and the API converts each incoming edge to UTC in
    `api/schemas.py` before `api/main.py` normalises the column with
    `pd.to_datetime(..., utc=True)`. UTC has no daylight-saving discontinuity, so
    a day is always 86,400 seconds and the ratio is exact.

    Were a local-time series ever passed in, a DST transition would move a single
    span by one hour — 0.042d against `WINDOW_LENGTH_TOLERANCE_DAYS` of 1.5d, so
    even then it could not flip these assertions. Recording that here because the
    division looks like the naive-day bug it is often mistaken for.
    """
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
        # Strict. `>=` permitted `l_min == e_max`, which is one transaction — the
        # boundary one — sitting in both windows at once. The generator assigns it
        # to the later split by construction (`searchsorted(side="right")` on the
        # interior boundaries), so equality here is not a tie to be tolerated, it
        # means the splits were cut somewhere other than where that rule says.
        assert l_min > e_max, (
            f"TEMPORAL LEAKAGE: '{later}' starts at {l_min}, at or before "
            f"'{earlier}' ends at {e_max} (overlap {e_max - l_min}). A boundary "
            f"timestamp belongs to the later split, so this is at least one "
            f"transaction the model is both trained on and scored against."
        )

    def test_train_val_test_are_strictly_ordered(self, raw_edges):
        """
        Transitivity too, so a mis-ordered val cannot hide behind two pairs.

        Strictly increasing, not merely sorted. The previous form compared each
        list against `sorted(...)`, which is satisfied by ties: three splits that
        all opened at the same instant would have passed. Nothing in the repo
        makes that reachable today — the adjacent-boundary test above forces
        `train_max < val_min < val_max < test_min`, which orders these lists
        strictly as a side effect — but an assertion that only holds because a
        sibling test holds is not the guard its docstring claims to be, and it is
        the sibling that would be deleted first if the boundary rule ever changed.
        """
        maxima = [raw_edges[s]["timestamp"].max() for s in SPLITS]
        minima = [raw_edges[s]["timestamp"].min() for s in SPLITS]
        for label, values in (("starts", minima), ("ends", maxima)):
            assert all(a < b for a, b in zip(values, values[1:])), (
                f"splits are not in strictly increasing order by {label}: "
                + ", ".join(f"{s}={v}" for s, v in zip(SPLITS, values))
                + ".\nEqual boundaries mean two windows open or close at the same "
                  "instant, which is not a chronological ordering."
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
        members_a = ring_member_nodes(raw_edges[a])
        members_b = ring_member_nodes(raw_edges[b])
        # Guard against a vacuous pass BEFORE trusting the disjointness below.
        # `ring_member_nodes` keys off the edge-role column; rename or drop that
        # column upstream and it returns the empty set for every split, at which
        # point `empty & empty` is empty and this test goes green while asserting
        # nothing about leakage. An empty result is itself a defect — the
        # generator plants rings in every split (measured 209/136/124 members for
        # train/val/test) — so require both operands non-empty first.
        assert members_a, (
            f"ring_member_nodes('{a}') returned no accounts. The generator plants "
            f"rings in every split (measured 209/136/124 for train/val/test), so an "
            f"empty set means the ring-member definition has stopped matching the "
            f"data — and the disjointness assertion below would pass vacuously."
        )
        assert members_b, (
            f"ring_member_nodes('{b}') returned no accounts — see the '{a}' "
            f"assertion above; an empty operand makes the disjointness check vacuous."
        )
        shared = members_a & members_b
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
            f"The amount features are rescaled to a reference window, but graph "
            f"structure — degrees, reciprocity, clustering, cycle participation — "
            f"is not, and would be measured over a different window at serving time "
            f"than in training."
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

    def test_window_scale_maps_window_length_to_the_reference(self):
        """
        The Tier-3a rescale, unit-tested directly — because the ratio ceiling above
        cannot see it. The shipped splits are equal by construction
        (`test_split_windows_are_equal_length`), so `window_scale` returns ~1.0 on
        every one of them and dropping it would move no cross-split ratio far enough
        to trip the 1.18 bound. The function the whole scale-free story rests on
        therefore had no test that goes red when it breaks; this is that test.

        `window_scale(w)` is what makes in_amount_sum, out_amount_sum and
        repeat_ratio comparable between graphs of different length: a magnitude
        watched over `w` days is multiplied by REFERENCE_WINDOW_DAYS / w, so a graph
        watched twice as long is scaled back down by half. Two properties are
        load-bearing:

          • the reference window is a no-op, and the correction runs the right way
            (longer window -> smaller multiplier). An inverted ratio would scale a
            long-window graph UP and double the very skew the rescale exists to
            remove, while still passing on the near-equal shipped splits.

          • below MIN_WINDOW_DAYS_FOR_RESCALE the rescale switches OFF and returns
            exactly 1.0. Without the floor a near-empty serving window divides by a
            tiny number and multiplies every magnitude into nonsense, and a zero
            window is a ZeroDivisionError rather than a score.
        """
        pytest.importorskip("networkx", reason="data.extractor imports networkx")
        from data.extractor import (
            MIN_WINDOW_DAYS_FOR_RESCALE,
            REFERENCE_WINDOW_DAYS,
            window_scale,
        )

        assert window_scale(REFERENCE_WINDOW_DAYS) == 1.0, (
            "the reference window must be a no-op; a feature already measured over "
            "REFERENCE_WINDOW_DAYS should not be touched."
        )
        assert window_scale(2 * REFERENCE_WINDOW_DAYS) == pytest.approx(0.5), (
            "a graph watched twice as long must be scaled DOWN by half. A multiplier "
            "of 2.0 here means the ratio is inverted and the rescale doubles the "
            "skew instead of removing it."
        )
        assert window_scale(REFERENCE_WINDOW_DAYS / 2) == pytest.approx(2.0)
        # Direction as an ordering, so a subtler sign error is still caught.
        assert (window_scale(REFERENCE_WINDOW_DAYS / 2)
                > window_scale(REFERENCE_WINDOW_DAYS)
                > window_scale(2 * REFERENCE_WINDOW_DAYS)), (
            "longer windows must map to smaller multipliers"
        )

        # The floor: strictly below the minimum, and at a zero window, the
        # multiplier is exactly 1.0 rather than a blow-up or a divide-by-zero.
        assert window_scale(MIN_WINDOW_DAYS_FOR_RESCALE / 2) == 1.0
        assert window_scale(0.0) == 1.0


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
