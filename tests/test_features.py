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
       and require pagerank within 1e-4.

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
line — every feature except pagerank bit-identical, pagerank within 1e-4 — so
that deleting the `partition=` plumbing fails the build instead of silently
reintroducing the defect.

PageRank is exempt because it is a global fixpoint over a normalised rank vector:
strictly every node shifts when any node is added. That is the definition, not a
bug. What matters is the magnitude — measured at 9.5e-06, four orders of
magnitude below the 0.317 that made the community feature unusable.

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

# Design rule 5's tolerance for pagerank, quoted from models/features.py.
# Measured on the shipped val graph: 9.5e-06.
PAGERANK_TOLERANCE = 1e-4

# Features that are mathematically confined to [0, 1]. Asserted on the emitted
# data because a value outside the range means the arithmetic is wrong, not that
# the data is unusual.
BOUNDED_UNIT_FEATURES = (
    "degree_balance",
    "flow_passthrough",
    "pagerank",
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

        It is a global fixpoint, so every node shifts by construction; the
        question is whether the shift is small enough to be irrelevant to a
        decision. Measured: 9.5e-06 against a 1e-4 budget.
        """
        common = base_features.index
        before = base_features["pagerank"].to_numpy(dtype=float)
        after = perturbed_features.loc[common, "pagerank"].to_numpy(dtype=float)
        worst = float(np.abs(after - before).max())
        assert worst <= PAGERANK_TOLERANCE, (
            f"pagerank moved by up to {worst:.3e}, above the {PAGERANK_TOLERANCE:.0e} "
            f"budget in design rule 5. A global centrality is allowed to drift when "
            f"the graph grows, but not enough to change a verdict."
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
    """

    @classmethod
    @pytest.fixture(scope="class")
    def reference(cls, raw_edges) -> pd.DataFrame:
        edges = raw_edges["test"]
        pairs = (edges.groupby(["sender", "receiver"], sort=False)
                 .agg(total_amount=("amount", "sum")).reset_index())

        nodes = sorted(set(edges["sender"]) | set(edges["receiver"]))
        out = pd.DataFrame(index=pd.Index(nodes, name="node"))
        out["in_degree"] = pairs.groupby("receiver").size().reindex(
            nodes).fillna(0).astype(int)
        out["out_degree"] = pairs.groupby("sender").size().reindex(
            nodes).fillna(0).astype(int)
        out["in_amount_sum"] = pairs.groupby("receiver")["total_amount"].sum().reindex(
            nodes).fillna(0.0).round(2)
        out["out_amount_sum"] = pairs.groupby("sender")["total_amount"].sum().reindex(
            nodes).fillna(0.0).round(2)

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
        out["repeat_ratio"] = (
            n_txns / n_counterparties.replace(0, np.nan)).fillna(0.0)
        return out

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

    @pytest.mark.parametrize("feature", NON_NEGATIVE_FEATURES)
    def test_unbounded_features_are_non_negative(self, node_features, feature):
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
