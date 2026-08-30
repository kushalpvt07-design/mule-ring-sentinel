"""
tests/test_features.py
──────────────────────
The extractor's invariants, and in particular DESIGN RULE 5 from
models/features.py, which cites this file by name:

    5. STABLE UNDER GRAPH CHANGES THAT DO NOT TOUCH THE ACCOUNT.
       Adding an unrelated account, or a transaction between two strangers, must
       not move an account's features.

       tests/test_features.py asserts this: perturb the graph with disconnected
       accounts, require bit-identical values for every feature except pagerank,
       and require pagerank to hold both its scale and its rank order.

─────────────────────────────────────────────────────────────────────────────
WHY THIS IS THE MOST IMPORTANT TEST IN THE FILE
─────────────────────────────────────────────────────────────────────────────
Determinism — the same graph scored twice giving the same answer — is the easy
half, and it was the only half anyone was checking. The half that costs money is
stability: `community_internal_ratio` is a per-COMMUNITY scalar broadcast to
every member, and Louvain repartitions the moment the node set changes. Measured
on the 2,954-account val graph, adding two accounts that transact only with each
other moved that feature for 100% of accounts (median |Δ| 0.021, max 0.317) and
flipped 84 decisions at the cost-optimal threshold — 2.84% of the population.

Two strangers paying each other changed the risk verdict on 84 unrelated
customers. Nothing raised, and no metric would have shown it.

The fix is not to drop the feature but to stop redrawing communities per request:
serving computes the partition once and passes it into `compute_node_features`,
which extends it deterministically. This file asserts the fixed path holds the
line — every feature except pagerank bit-identical, and pagerank holding both its
scale and its rank order — so that deleting the `partition=` plumbing fails the
build instead of silently reintroducing the defect.

PageRank is exempt from BIT-identity because it is a global fixpoint over a
normalised rank vector, and the solver that computes it stops at a tolerance
rather than at a fixed point. It is not exempt from the invariant, and the
invariant here is stronger than "small": for a disconnected addition the emitted
`pagerank × N` column is EXACTLY unchanged in exact arithmetic, because the ×N
cancels the teleport rescale in full. See
PAGERANK_PERTURBATION_RELATIVE_TOLERANCE below for that derivation and for why
the v3 figure of 9.5e-06 — measured on the differently scaled raw column — could
not be carried over, and why the absolute 1e-3 that replaced it was the wrong
shape of bound rather than the wrong number.

Usage:
    pytest tests/test_features.py -v
    pytest tests/test_features.py -v -m "not slow"      # skip the graph rebuilds
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from models.features import FEATURE_COLS, LABEL_META_COLS, METADATA_COLS, TARGET_COL

SPLITS = ("train", "val", "test")

# Design rule 5's budget for pagerank. models/features.py names this constant and
# defers its value to here.
#
# WHY IT IS RELATIVE AND NOT ABSOLUTE
# ───────────────────────────────────
# v3 allowed an absolute 1e-4 on the raw column, whose mean is 1/N ≈ 3.4e-4. When
# v4 began emitting `pagerank × N`, that budget was moved to 1e-3 — one order of
# magnitude, eyeballed — on a column whose scale had grown by a factor of
# N ≈ 3,000. The retrain measured 6.9e-3 and the test failed. Widening it again
# would be the third guess at a number that should never have been absolute:
# `pagerank` is defined as a multiple of the graph's OWN uniform baseline, so a
# fixed budget on it means a different thing in every graph it is applied to.
#
# WHAT THE CORRECT BOUND IS, DERIVED RATHER THAN FITTED
# ────────────────────────────────────────────────────
# For the perturbation this file applies — two new accounts transacting only with
# each other — the emitted column is EXACTLY invariant, not approximately. The
# probe pair is a closed component: no edge enters it from the main graph and none
# leaves it. PageRank's teleport is uniform, so the only quantity the addition
# changes for a main-component node is the teleport magnitude, (1-d)/N → (1-d)/N'.
# The main component's linear system is otherwise untouched, so every pre-existing
# RAW value scales by exactly N/N' — and emitting `raw × N'` cancels that
# precisely: raw_i · (N/N') · N' = raw_i · N. The expected drift is ZERO.
#
# So the 6.9e-3 that was measured is neither a centrality shift nor a regression
# in the scaling. It is `nx.pagerank`'s power-iteration residual (it stops when
# the L1 change falls below N · tol, with tol=1e-6) read on a column N times
# larger than the vector the solver actually converged. That residual is a
# property of the solver, the platform and the networkx version, so pinning it
# tightly would buy flakiness rather than safety.
#
# The budget below is therefore relative to the column's own largest value, with
# deliberate headroom: measured 4.1e-4 of the maximum on val (6.9e-3 against a max
# of 16.9), bounded at 1e-2. It is the backstop, not the real guard. The two sharp
# assertions sit beside it, and neither can be satisfied by noise:
# PAGERANK_PERTURBATION_RANK_CORRELATION pins the ordering the trees actually
# split on, and TestEmittedValues::test_pagerank_is_a_multiple_of_the_uniform_
# baseline pins the ×N scaling itself. That last one is load-bearing for this
# constant: without it a relative budget would pass VACUOUSLY if someone deleted
# the scaling, because a raw 1/N column drifts by ~1e-5 in absolute terms and
# clears any relative bound trivially.
PAGERANK_PERTURBATION_RELATIVE_TOLERANCE = 1e-2

# Spearman correlation required between the pre- and post-perturbation pagerank
# columns. This is the assertion that matches the claim data/extractor.py and
# api/main.py both make in prose — "rank correlation > 0.9999, which moved no
# decision" — and it is the one that matters for this model: gradient-boosted
# trees split on order, so order is the thing that has to hold.
PAGERANK_PERTURBATION_RANK_CORRELATION = 0.9999

# Features that are mathematically confined to [0, 1]. Asserted on the emitted
# data because a value outside the range means the arithmetic is wrong, not that
# the data is unusual.
#
# `pagerank` is deliberately NOT in this list. It was, and it belonged here while
# the raw pagerank vector was emitted: that sums to 1, so every entry is in [0, 1]
# by construction. v4 emits `pagerank × N`, a multiple of the uniform baseline
# with mean exactly 1.0, and a value above 1 is the normal case rather than an
# arithmetic error — the shipped splits reach 13.6 / 16.9 / 21.4. Leaving it here
# asserted the OLD definition, so it failed on correct output. Its replacement
# invariant is test_pagerank_is_a_multiple_of_the_uniform_baseline, which is
# strictly stronger than a range check: a bound of [0, ∞) would have been
# satisfied by any non-negative garbage, while "mean is exactly 1.0" can only be
# satisfied by the scaling actually being applied. Removing it from this tuple
# also adds it to NON_NEGATIVE_FEATURES below, which is derived as the complement,
# so the lower bound is not lost.
BOUNDED_UNIT_FEATURES = (
    "degree_balance",
    "flow_passthrough",
    "clustering_coefficient",
    "cycle_participation",
    "reciprocity",
    "fan_in_concentration",
    "burst_ratio",
    "community_internal_ratio",
)

NON_NEGATIVE_FEATURES = tuple(
    c for c in FEATURE_COLS if c not in BOUNDED_UNIT_FEATURES)


@pytest.fixture(scope="module")
def perturbation_edges() -> pd.DataFrame:
    """
    Three transactions between two brand-new accounts and nobody else.

    Deliberately a closed pair: it shares no counterparty with any existing
    account, so under any defensible definition of a local feature it must be
    invisible to the other 2,954. That is what makes it a clean probe.
    """
    return pd.DataFrame({
        "sender": ["PERTURB_A", "PERTURB_B", "PERTURB_A"],
        "receiver": ["PERTURB_B", "PERTURB_A", "PERTURB_B"],
        "amount": [1234.0, 999.0, 4321.0],
        "timestamp": pd.to_datetime([
            "2025-03-15 10:00:00",
            "2025-03-15 12:00:00",
            "2025-03-15 15:00:00",
        ]),
    })


@pytest.fixture(scope="module")
def base_features(val_graph, frozen_partition):
    """Features on the unperturbed val graph, frozen partition — the reference."""
    from data.extractor import compute_node_features
    return (compute_node_features(val_graph, partition=frozen_partition)
            .set_index("node").sort_index())


@pytest.fixture(scope="module")
def perturbed_features(raw_edges, perturbation_edges, frozen_partition):
    """
    Features after the perturbation, computed the way SERVING computes them:
    the partition frozen from the reference graph, extended to cover newcomers.
    """
    from data.extractor import build_graph, compute_node_features
    cols = ["sender", "receiver", "amount", "timestamp"]
    merged = (
        pd.concat([raw_edges["val"][cols], perturbation_edges], ignore_index=True)
        .sort_values("timestamp", kind="mergesort")
        .reset_index(drop=True)
    )
    graph = build_graph(merged)
    return (compute_node_features(graph, partition=frozen_partition)
            .set_index("node").sort_index())


# ══════════════════════════════════════════════════════════════════
# 1. Design rule 5 — stability under an unrelated graph change
# ══════════════════════════════════════════════════════════════════

@pytest.mark.slow
class TestPerturbationStability:
    """models/features.py design rule 5, asserted."""

    def test_perturbation_actually_grew_the_graph(
        self, base_features, perturbed_features
    ):
        """Guard against a vacuous pass: the probe must have done something."""
        newcomers = set(perturbed_features.index) - set(base_features.index)
        assert newcomers == {"PERTURB_A", "PERTURB_B"}, (
            f"expected exactly the two probe accounts to be new, got {newcomers}. "
            f"If the perturbation did not reach the graph, the stability "
            f"assertions below prove nothing."
        )

    @pytest.mark.parametrize(
        "feature", [c for c in FEATURE_COLS if c != "pagerank"])
    def test_feature_is_bit_identical_after_perturbation(
        self, base_features, perturbed_features, feature
    ):
        """
        Every feature except pagerank must be UNCHANGED — not close, identical.

        A tolerance here would be the wrong shape of assertion. These features are
        defined on the account's own neighbourhood, which the probe does not
        touch, so any movement at all means something global leaked in. The one
        that used to move, `community_internal_ratio`, moved by up to 0.317.
        """
        common = base_features.index
        before = base_features[feature].to_numpy(dtype=float)
        after = perturbed_features.loc[common, feature].to_numpy(dtype=float)
        moved = int((before != after).sum())
        assert moved == 0, (
            f"{feature} moved for {moved} of {len(common)} pre-existing accounts "
            f"after adding two accounts that transact only with each other "
            f"(max |Δ| {np.abs(after - before).max():.3e}).\n"
            f"This feature is supposed to depend only on the account's own "
            f"neighbourhood. If this is community_internal_ratio, the frozen "
            f"partition is no longer being threaded through "
            f"compute_node_features(partition=...)."
        )

    def test_pagerank_moves_only_negligibly(
        self, base_features, perturbed_features
    ):
        """
        PageRank is exempt from bit-identity but not from a bound.

        The exemption is narrower than it looks. For THIS perturbation the emitted
        column is exactly invariant in exact arithmetic — the probe pair is a closed
        component, so the only thing that changes for a pre-existing account is the
        uniform teleport magnitude, every raw value scales by exactly N/N', and the
        ×N' emission cancels it. See PAGERANK_PERTURBATION_RELATIVE_TOLERANCE for
        the derivation. So the expected drift is zero and the residual measured here
        is `nx.pagerank`'s power-iteration tolerance amplified by N, which is a
        property of the solver rather than of the feature.

        That is why the magnitude check is relative and generous while the rank
        check is tight. A regression in the ×N scaling shows up as a rank change or
        as a mean that is no longer 1.0, not as a slightly larger residual.
        """
        common = base_features.index
        before = base_features["pagerank"].to_numpy(dtype=float)
        after = perturbed_features.loc[common, "pagerank"].to_numpy(dtype=float)

        worst = float(np.abs(after - before).max())
        scale = float(np.abs(before).max())
        budget = PAGERANK_PERTURBATION_RELATIVE_TOLERANCE * scale
        assert worst <= budget, (
            f"pagerank moved by up to {worst:.3e} on a column whose largest value "
            f"is {scale:.4f} — {worst / scale:.2e} of full scale, above the "
            f"{PAGERANK_PERTURBATION_RELATIVE_TOLERANCE:.0e} relative budget in "
            f"design rule 5.\n"
            f"Adding two accounts that transact only with each other should move "
            f"this column by EXACTLY nothing: the probe pair is a closed component, "
            f"so ×N cancels the teleport rescale in full. A residual this large is "
            f"not solver noise. Check that compute_pagerank_vs_uniform is still "
            f"multiplying by G.number_of_nodes() of the SAME graph it ranked, and "
            f"that the probe accounts really are disconnected."
        )

        # The assertion that matches what the model consumes. Trees split on order,
        # so an ordering that survives the perturbation is the operative claim —
        # and it is the claim data/extractor.py and api/main.py both make in prose.
        # Spearman via ranks + Pearson, so this needs no scipy.
        ranked_before = pd.Series(before).rank().to_numpy()
        ranked_after = pd.Series(after).rank().to_numpy()
        rho = float(np.corrcoef(ranked_before, ranked_after)[0, 1])
        assert rho >= PAGERANK_PERTURBATION_RANK_CORRELATION, (
            f"pagerank's rank order changed after the perturbation (Spearman "
            f"{rho:.6f} < {PAGERANK_PERTURBATION_RANK_CORRELATION}). The magnitude "
            f"budget above can be met by a column that has been quietly reordered; "
            f"this cannot. Reordering means the trees see different splits for "
            f"accounts the perturbation never touched, which is design rule 5's "
            f"actual failure mode."
        )

    def test_no_pre_existing_account_changes_community(
        self, base_features, perturbed_features
    ):
        """
        The mechanism behind the stability above, checked directly.

        If `extend_partition` ever reassigns an existing account, every member of
        both communities gets a new `community_internal_ratio` and the feature
        starts describing the sample rather than the account.
        """
        common = base_features.index
        changed = int((
            perturbed_features.loc[common, "louvain_community"].to_numpy()
            != base_features["louvain_community"].to_numpy()
        ).sum())
        assert changed == 0, (
            f"{changed} pre-existing account(s) were moved to a different "
            f"community by extend_partition."
        )

    def test_newcomers_receive_a_community(self, perturbed_features):
        """Unseen accounts must be placed, not left with a sentinel id."""
        assigned = perturbed_features.loc[
            ["PERTURB_A", "PERTURB_B"], "louvain_community"]
        assert (assigned >= 0).all(), (
            f"newcomer(s) left unassigned: {assigned.to_dict()}. A negative "
            f"community id would collapse community_internal_ratio to the "
            f"fallback value for every unseen account."
        )


# ══════════════════════════════════════════════════════════════════
# 2. Determinism — the easy half, still worth pinning
# ══════════════════════════════════════════════════════════════════

@pytest.mark.slow
class TestDeterminism:
    """The same edges must produce the same numbers, run to run."""

    def test_recomputation_is_bit_identical(
        self, raw_edges, frozen_partition, base_features
    ):
        from data.extractor import build_graph, compute_node_features

        again = (compute_node_features(build_graph(raw_edges["val"]),
                                       partition=frozen_partition)
                 .set_index("node").sort_index())
        assert list(again.index) == list(base_features.index), (
            "node order changed between two runs on identical input"
        )
        drifted = [
            c for c in FEATURE_COLS
            if not np.array_equal(again[c].to_numpy(dtype=float),
                                  base_features[c].to_numpy(dtype=float))
        ]
        assert not drifted, (
            f"features differ between two runs on identical input: {drifted}. "
            f"An unseeded random source or a dict-ordering dependency has crept in."
        )

    def test_partition_fingerprint_is_reproducible(self, val_graph):
        """
        The fingerprint /health publishes must be stable across recomputations.

        Two replicas partitioning differently would return different scores for
        the same account, and only this string would reveal it.
        """
        from data.extractor import compute_louvain_communities, partition_fingerprint

        undirected = val_graph.to_undirected()
        first = partition_fingerprint(compute_louvain_communities(undirected))
        second = partition_fingerprint(compute_louvain_communities(undirected))
        assert first == second, (
            f"Louvain partition is not reproducible on identical input: "
            f"{first} vs {second}. LOUVAIN_SEED is not being honoured."
        )


# ══════════════════════════════════════════════════════════════════
# 3. Independent recomputation — is the arithmetic right?
# ══════════════════════════════════════════════════════════════════

class TestArithmeticAgainstAnIndependentImplementation:
    """
    Recompute the countable features straight from the edge file with plain
    pandas, and require the extractor to agree.

    This is the test that catches a real class of bug the graph code is prone to:
    `DiGraph.add_edge` in a per-transaction loop keeps only the LAST edge's
    attributes, so a naive implementation silently drops every repeat transaction
    between the same pair — 86% of pairs here — and every sum, count and rate
    comes out wrong while still looking plausible.

    ON THE WINDOW RESCALE, AND WHY THIS STAYS AN INDEPENDENT IMPLEMENTATION
    ──────────────────────────────────────────────────────────────────────
    Three of the five features below — `in_amount_sum`, `out_amount_sum` and
    `repeat_ratio` — are emitted on a 60-day reference window rather than raw (see
    `data/extractor.py::window_scale`). This fixture knew nothing about that and so
    disagreed with correct output for ~97% of accounts, by exactly the rescale
    factor: 0.034% on the shipped test split, whose window is 59.979 days.

    The recomputation now applies the same normalisation, and the way it does so is
    the point. It imports the two DECLARED CONSTANTS, `REFERENCE_WINDOW_DAYS` and
    `MIN_WINDOW_DAYS_FOR_RESCALE`, and derives the factor itself from the edge
    file's own timestamps. It deliberately does NOT call `window_scale()` or
    `observation_window_days()`. Calling either would make the comparison circular
    — the extractor's own helper on both sides of an equals sign proves only that
    the helper is deterministic — whereas re-deriving the span from the raw CSV
    still fails if the extractor measures the wrong window, applies the factor to
    the wrong features, applies it twice, or applies it in the wrong order relative
    to its 2-decimal rounding. Knowing the declared UNIT of a column is not the
    same as borrowing its implementation.

    One consequence worth naming: if a future regeneration produces a split
    spanning exactly REFERENCE_WINDOW_DAYS, the factor is 1.0 and these three
    assertions quietly degenerate into the raw comparison they used to be. That is
    still a valid check of the summation, just a weaker check of the rescale.
    `test_the_window_rescale_is_actually_exercised` fails loudly in that case
    rather than letting the coverage evaporate silently.
    """

    @classmethod
    @pytest.fixture(scope="class")
    def window_rescale(cls, raw_edges) -> float:
        """
        The reference window factor for the test split, derived from the raw edge
        file without touching the extractor's own window helpers.

        Integer nanoseconds and the same day divisor the extractor uses, so this is
        the identical arithmetic reached by an independent route — not a
        floating-point approximation of it that would leave every comparison
        hostage to a rounding boundary.

        PIN THE UNIT BEFORE CASTING TO int64. `to_numpy(dtype="datetime64[ns]")`
        is not decoration; it is the whole correctness of this fixture, and it is
        the same idiom `data/extractor.py:267` uses for the same reason. An
        unpinned `pd.to_datetime(...).astype("int64")` returns epoch counts in
        WHATEVER resolution pandas happened to infer while parsing — seconds for
        ISO strings on newer pandas, nanoseconds on older — so the divisor below
        silently becomes wrong by a factor of a billion. That is not hypothetical:
        this fixture shipped that way for one run. The span came out at 6e-8 days,
        fell through the `< MIN_WINDOW_DAYS_FOR_RESCALE` branch, returned exactly
        1.0, and took three recomputation assertions down with it while passing
        cleanly on the pandas version it was written against. Deriving a fact
        independently is worth nothing if the derivation is environment-dependent.
        """
        from data.extractor import (MIN_WINDOW_DAYS_FOR_RESCALE,
                                    REFERENCE_WINDOW_DAYS)

        stamps = (raw_edges["test"]["timestamp"]
                  .to_numpy(dtype="datetime64[ns]")
                  .astype("int64"))
        span_days = float((stamps.max() - stamps.min()) / 86_400_000_000_000)
        if span_days < MIN_WINDOW_DAYS_FOR_RESCALE:
            return 1.0
        return REFERENCE_WINDOW_DAYS / span_days

    @classmethod
    @pytest.fixture(scope="class")
    def reference(cls, raw_edges, window_rescale) -> pd.DataFrame:
        edges = raw_edges["test"]
        pairs = (edges.groupby(["sender", "receiver"], sort=False)
                 .agg(total_amount=("amount", "sum")).reset_index())

        nodes = sorted(set(edges["sender"]) | set(edges["receiver"]))
        out = pd.DataFrame(index=pd.Index(nodes, name="node"))
        out["in_degree"] = pairs.groupby("receiver").size().reindex(
            nodes).fillna(0).astype(int)
        out["out_degree"] = pairs.groupby("sender").size().reindex(
            nodes).fillna(0).astype(int)

        # Scale THEN round, in that order: the extractor rounds the rescaled rupee
        # figure, so rounding first here would disagree in the last paisa for a
        # value sitting near a boundary and the failure would read as an arithmetic
        # bug rather than an ordering one.
        raw_in = pairs.groupby("receiver")["total_amount"].sum().reindex(
            nodes).fillna(0.0)
        raw_out = pairs.groupby("sender")["total_amount"].sum().reindex(
            nodes).fillna(0.0)
        out["in_amount_sum"] = (raw_in * window_rescale).round(2)
        out["out_amount_sum"] = (raw_out * window_rescale).round(2)

        # Distinct UNDIRECTED counterparties, then transactions per counterparty.
        both_ways = pd.concat([
            pairs[["sender", "receiver"]].rename(
                columns={"sender": "self", "receiver": "other"}),
            pairs[["receiver", "sender"]].rename(
                columns={"receiver": "self", "sender": "other"}),
        ])
        n_counterparties = both_ways.groupby("self")["other"].nunique().reindex(
            nodes).fillna(0)
        n_txns = pd.concat([
            edges.groupby("sender").size(),
            edges.groupby("receiver").size(),
        ]).groupby(level=0).sum().reindex(nodes).fillna(0)
        # Not rounded, because the extractor does not round this one.
        out["repeat_ratio"] = (
            (n_txns / n_counterparties.replace(0, np.nan)) * window_rescale
        ).fillna(0.0)
        return out

    def test_the_window_rescale_is_actually_exercised(self, raw_edges,
                                                      window_rescale):
        """
        Guard against a vacuous pass in the three rescaled features above.

        If the test split ever spans exactly the reference window the factor is
        1.0, the recomputation stops distinguishing rescaled output from raw, and
        three assertions silently become weaker without anyone editing them. The
        shipped span is 59.979 days, giving 1.000343.

        The span is re-derived here rather than read off the fixture, because the
        two ways of arriving at a factor of exactly 1.0 need telling apart and only
        the span separates them: a split that genuinely spans REFERENCE_WINDOW_DAYS
        is a legitimate loss of coverage, whereas a span of 6e-8 days means the
        timestamps were read in the wrong RESOLUTION and fell through the
        `< MIN_WINDOW_DAYS_FOR_RESCALE` branch. This test has already caught the
        second case once, and reported it as the first — hence the split assertion.
        """
        from data.extractor import MIN_WINDOW_DAYS_FOR_RESCALE

        # Same pinned-unit idiom as the fixture and as data/extractor.py:267.
        stamps = (raw_edges["test"]["timestamp"]
                  .to_numpy(dtype="datetime64[ns]")
                  .astype("int64"))
        span_days = float((stamps.max() - stamps.min()) / 86_400_000_000_000)

        assert span_days >= MIN_WINDOW_DAYS_FOR_RESCALE, (
            f"the test split's span reads as {span_days!r} days, under "
            f"MIN_WINDOW_DAYS_FOR_RESCALE ({MIN_WINDOW_DAYS_FOR_RESCALE}), so "
            f"window_rescale returns 1.0 and the three rescaled comparisons below "
            f"degenerate to raw. A file of ~60 days of transactions cannot really "
            f"span under a day: the likely cause is an epoch count read in the "
            f"wrong resolution — seconds or microseconds rather than nanoseconds — "
            f"which is off by a factor of a billion and looks exactly like this. "
            f"Pin the unit with to_numpy(dtype='datetime64[ns]') before casting to "
            f"int64; do not trust whatever resolution pandas inferred at parse."
        )
        assert window_rescale != 1.0, (
            f"the window rescale factor for the test split is exactly 1.0 off a "
            f"span of {span_days} days, so test_matches_a_plain_pandas_recomputation "
            f"no longer distinguishes a correctly rescaled in_amount_sum / "
            f"out_amount_sum / repeat_ratio from an unrescaled one. The span is "
            f"sane, so this is the legitimate case: either the split now spans "
            f"exactly REFERENCE_WINDOW_DAYS, or the rescale has been switched off. "
            f"Neither is wrong on its own; losing the coverage without noticing is."
        )

    @pytest.mark.parametrize("feature", [
        "in_degree", "out_degree", "in_amount_sum", "out_amount_sum", "repeat_ratio",
    ])
    def test_matches_a_plain_pandas_recomputation(
        self, node_features, reference, feature
    ):
        got = node_features["test"].set_index("node").reindex(reference.index)
        assert not got[feature].isna().any(), (
            f"the feature table is missing accounts present in the edge file"
        )
        delta = np.abs(got[feature].to_numpy(dtype=float)
                       - reference[feature].to_numpy(dtype=float))
        worst = int(delta.argmax())
        assert delta.max() <= 1e-6, (
            f"{feature} disagrees with an independent recomputation for "
            f"{int((delta > 1e-6).sum())} of {len(delta)} accounts "
            f"(worst: {reference.index[worst]}, extractor "
            f"{got[feature].iloc[worst]}, recomputed "
            f"{reference[feature].iloc[worst]})."
        )

    def test_every_account_in_the_edge_file_has_features(
        self, raw_edges, node_features
    ):
        """An account dropped from the feature table is an account never scored."""
        for split in SPLITS:
            in_edges = set(raw_edges[split]["sender"]) | set(
                raw_edges[split]["receiver"])
            in_features = set(node_features[split]["node"])
            assert in_edges == in_features, (
                f"'{split}': {len(in_edges - in_features)} account(s) appear in "
                f"the edge file but not the feature table, and "
                f"{len(in_features - in_edges)} the other way round."
            )


# ══════════════════════════════════════════════════════════════════
# 4. Emitted values are usable
# ══════════════════════════════════════════════════════════════════

class TestEmittedValues:
    """
    Range and finiteness checks on what actually lands in the CSVs.

    XGBoost tolerates NaN by design, which is precisely why an accidental NaN is
    dangerous: it becomes a learned branch rather than an error, and the same
    column at serving time may be NaN for an entirely different reason.
    """

    @pytest.mark.parametrize("split", SPLITS)
    def test_no_missing_or_infinite_values(self, node_features, split):
        frame = node_features[split][list(FEATURE_COLS)]
        nan_counts = frame.isna().sum()
        offenders = {c: int(n) for c, n in nan_counts.items() if n}
        assert not offenders, f"'{split}' has NaN feature values: {offenders}"

        numeric = frame.to_numpy(dtype=float)
        assert np.isfinite(numeric).all(), (
            f"'{split}' has non-finite feature values in "
            f"{[c for c in FEATURE_COLS if not np.isfinite(frame[c].to_numpy(dtype=float)).all()]}"
        )

    @pytest.mark.parametrize("feature", BOUNDED_UNIT_FEATURES)
    def test_bounded_features_stay_in_the_unit_interval(
        self, node_features, feature
    ):
        """
        Note on `reciprocity`: it is mathematically bounded by 1 but empirically
        peaks at exactly 0.50 on this data, with 47% of accounts at 0. That looks
        broken and is not — every account carries one-directional relationships
        (salary in, rent and subscriptions out), so mutual pairs can only ever be
        a minority. The assertion is the real bound, 1.0, so that a generator
        change producing genuinely reciprocal accounts does not fail a test that
        had quietly hard-coded an accident.
        """
        for split in SPLITS:
            values = node_features[split][feature]
            assert values.min() >= 0.0 and values.max() <= 1.0, (
                f"'{split}' {feature} ranges [{values.min()}, {values.max()}], "
                f"outside [0, 1]."
            )

    @pytest.mark.parametrize("split", SPLITS)
    def test_pagerank_is_a_multiple_of_the_uniform_baseline(
        self, node_features, split
    ):
        """
        `pagerank` replaced a [0, 1] range check with this, and it is the stronger
        assertion of the two.

        The column is `raw_pagerank × N` — a multiple of the graph's uniform 1/N
        baseline, so "2.0" reads as "twice as central as the average account"
        whether the graph holds 3,000 accounts or 300,000. Raw PageRank sums to 1,
        so the emitted column sums to exactly N and its MEAN is exactly 1.0. That
        is an identity, not a distributional fact about this data, which is why it
        can be asserted this tightly: the shipped splits sit within 2.3e-16 of it.

        Why this beats the range check it replaced. `[0, 1]` was true of the raw
        column and false of this one, so it failed on correct output; the obvious
        repair, widening to `[0, ∞)`, would have been satisfied by any non-negative
        garbage — including the raw column with the ×N scaling deleted, which is
        precisely the regression worth catching. "Mean is exactly 1.0" can only be
        satisfied by the scaling being applied to the right node count. It is also
        what makes design rule 5's relative perturbation budget non-vacuous; see
        PAGERANK_PERTURBATION_RELATIVE_TOLERANCE.

        The lower bound is not lost: removing `pagerank` from BOUNDED_UNIT_FEATURES
        adds it to NON_NEGATIVE_FEATURES, which is derived as the complement and is
        asserted by test_unbounded_features_are_non_negative directly below.
        """
        values = node_features[split]["pagerank"]
        n = len(values)
        mean = float(values.mean())
        assert abs(mean - 1.0) < 1e-12, (
            f"'{split}' pagerank has mean {mean:.12f}, not 1.0. The column is "
            f"defined as raw pagerank × N and raw pagerank sums to 1, so the mean "
            f"is 1.0 by identity for any N. A mean near 1/{n} means "
            f"compute_pagerank_vs_uniform is emitting the raw vector; any other "
            f"value means it multiplied by a node count that is not this graph's."
        )
        assert values.max() > 1.0, (
            f"'{split}' pagerank never exceeds 1.0 (max {values.max():.6f}) even "
            f"though its mean is 1.0, which means the column is constant. A "
            f"centrality that does not vary carries no signal and would have been "
            f"silently accepted by the [0, 1] range check this test replaced."
        )

    @pytest.mark.parametrize("feature", NON_NEGATIVE_FEATURES)
    def test_unbounded_features_are_non_negative(self, node_features, feature):
        """
        The lower half of the bound for every feature that is NOT confined to
        [0, 1]: counts, sums, rates, dispersions and `pagerank` are all unbounded
        above and all floored at zero.

        This test is the reason `pagerank` could be removed from
        BOUNDED_UNIT_FEATURES without losing anything. NON_NEGATIVE_FEATURES is
        derived as the complement of that tuple, so deleting a name from there
        moves it here automatically and the floor follows it across. Keep the two
        tuples complementary: if a feature ends up in neither, it is asserted by
        nothing at all, and `test_columns_match_the_contract_exactly` will not
        notice because the column still exists.
        """
        for split in SPLITS:
            worst = node_features[split][feature].min()
            assert worst >= 0.0, (
                f"'{split}' {feature} has a negative value ({worst})."
            )

    @pytest.mark.parametrize("split", SPLITS)
    def test_repeat_ratio_is_at_least_one_for_active_accounts(
        self, node_features, split
    ):
        """
        `repeat_ratio` is transactions per distinct counterparty, so it cannot be
        below 1 for an account with any counterparty at all. Below 1 would mean
        more counterparties than transactions, i.e. the pair aggregation and the
        transaction count disagree.
        """
        frame = node_features[split]
        active = frame[(frame["in_degree"] + frame["out_degree"]) > 0]
        assert (active["repeat_ratio"] >= 1.0).all(), (
            f"'{split}': {int((active['repeat_ratio'] < 1.0).sum())} active "
            f"account(s) have repeat_ratio < 1."
        )

    @pytest.mark.parametrize("split", SPLITS)
    def test_no_feature_is_constant(self, node_features, split):
        """
        A constant column carries no information and is usually a bug — a feature
        that silently returns its fallback for every account, for instance.
        """
        constant = [
            c for c in FEATURE_COLS
            if node_features[split][c].nunique(dropna=False) <= 1
        ]
        assert not constant, (
            f"'{split}' has constant feature(s) {constant}. Either the "
            f"computation is failing into a fallback, or the feature should be "
            f"dropped from FEATURE_COLS."
        )


# ══════════════════════════════════════════════════════════════════
# 5. The features mean what their descriptions claim
# ══════════════════════════════════════════════════════════════════

class TestFeaturesCarrySignal:
    """
    Weak, directional sanity checks on the two features the project's thesis
    rests on.

    Deliberately loose. A tight bound here would be a leakage test in disguise
    (tests/test_baselines.py owns that, with a hard AUC ceiling); the point is
    only that `cycle_participation` is not silently returning noise, which a
    refactor of the cycle enumeration could easily cause without failing anything
    else in this file.
    """

    def test_cycle_participation_is_higher_for_every_ring_archetype(
        self, node_features
    ):
        frame = node_features["test"]
        organic = frame.loc[frame[TARGET_COL] == 0, "cycle_participation"].mean()
        for archetype, group in frame[frame[TARGET_COL] == 1].groupby("ring_type"):
            ring_mean = group["cycle_participation"].mean()
            assert ring_mean > organic, (
                f"archetype '{archetype}' has mean cycle_participation "
                f"{ring_mean:.4f}, at or below the organic mean {organic:.4f}. "
                f"The feature that expresses this project's whole thesis is not "
                f"separating that ring shape at all."
            )

    def test_mules_are_more_central_than_organic_accounts(self, node_features):
        frame = node_features["test"]
        mule_pr = frame.loc[frame[TARGET_COL] == 1, "pagerank"].mean()
        organic_pr = frame.loc[frame[TARGET_COL] == 0, "pagerank"].mean()
        assert mule_pr > organic_pr, (
            f"mean pagerank is {mule_pr:.6f} for mules against {organic_pr:.6f} "
            f"for organic accounts. Count-weighted centrality should be higher "
            f"for accounts on a repeated laundering route."
        )


# ══════════════════════════════════════════════════════════════════
# 6. Column layout of the emitted tables
# ══════════════════════════════════════════════════════════════════

class TestTableLayout:
    """
    The CSVs must carry exactly the declared columns, in the declared order.

    XGBoost is positional: a reordering that keeps the same names produces a
    model where every score is wrong and nothing warns you. `process_split`
    builds the order from the contract, so this test is what keeps that promise
    honest against a hand-edited CSV or a partially regenerated pipeline.
    """

    EXPECTED = (
        ["node"]
        + list(FEATURE_COLS)
        + [c for c in METADATA_COLS if c != "node"]
        + list(LABEL_META_COLS)
        + [TARGET_COL]
    )

    @pytest.mark.parametrize("split", SPLITS)
    def test_columns_match_the_contract_exactly(self, node_features, split):
        got = list(node_features[split].columns)
        assert got == self.EXPECTED, (
            f"'{split}' column layout drifted from models/features.py.\n"
            f"  expected: {self.EXPECTED}\n"
            f"  got:      {got}\n"
            f"  missing:  {[c for c in self.EXPECTED if c not in got]}\n"
            f"  extra:    {[c for c in got if c not in self.EXPECTED]}"
        )

    @pytest.mark.parametrize("split", SPLITS)
    def test_split_column_says_what_split_it_is(self, node_features, split):
        values = set(node_features[split]["split"].unique())
        assert values == {split}, (
            f"{split}_features.csv carries split labels {values}."
        )
