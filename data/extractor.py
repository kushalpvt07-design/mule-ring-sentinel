"""
data/extractor.py
─────────────────
Turns a raw UPI edge list into the per-node feature table the model consumes.

This module is the SINGLE implementation of feature computation. `api/main.py`
imports `build_graph` and `compute_node_features` from here rather than keeping
its own copy, so serving-time features are produced by the same code path as
training-time features by construction. A duplicate implementation is how the
API previously ended up sending `louvain_community = 0` and `amount_cv = 0.0`
for every account while the model treated one of them as a top-three input.

The authoritative feature list lives in `models/features.py`. This file must
produce exactly those columns; the check at the end of `compute_node_features`
fails loudly if the two drift apart.

Usage:
    python -m data.extractor

─────────────────────────────────────────────────────────────────────────────
WHAT EACH FEATURE IS FOR
─────────────────────────────────────────────────────────────────────────────
Degree structure
  in_degree, out_degree      distinct counterparties in each direction
  degree_ratio               out/in, unbounded
  degree_balance             min/max, bounded [0,1] — a pass-through account
                             is balanced; a sink or a source is not
Value flow
  in_amount_sum,             absolute value at risk. The only unbounded
  out_amount_sum             magnitudes kept, because the cost model in
                             models/cost_matrix.py is denominated in rupees and
                             an analyst needs the number. Comparable across
                             splits only because all observation windows are the
                             same length — see data/generator.py.
  flow_passthrough           min/max of inbound vs outbound value, bounded [0,1].
                             ≈1 means "money in ≈ money out": the pass-through
                             signature of a mule — and of a gig worker, which is
                             why the generator emits gig accounts.
Centrality / local topology
  pagerank                   centrality in the payment network
  clustering_coefficient     how tightly the node's counterparties inter-transact
  cycle_participation        the project's thesis as a feature — see below
  reciprocity                share of counterparties that both pay and are paid
Behavioural
  fan_in_concentration       Herfindahl index on inbound value
  txn_velocity               transactions per hour while active
  burst_ratio                busiest single hour's share of the node's traffic
  amount_cv                  CV of INDIVIDUAL transaction amounts
  counterparty_amount_cv     CV across PER-COUNTERPARTY mean amounts
  repeat_ratio               transactions per distinct counterparty
Community structure
  community_internal_ratio   how closed the node's Louvain community is

─────────────────────────────────────────────────────────────────────────────
v2 → v3 CHANGES
─────────────────────────────────────────────────────────────────────────────
1. `cycle_participation` added, and it is computed exactly, not heuristically.
   See compute_cycle_participation() for the algorithm and why it does not use
   nx.simple_cycles().

2. `reciprocity`, `burst_ratio` and `counterparty_amount_cv` added. Each exists
   because a v2 feature was doing two jobs at once: `clustering_coefficient` was
   the only thing distinguishing a layering loop from ordinary two-way social
   payment, and `txn_velocity` conflated "many transactions" with "many
   transactions crammed into one hour".

3. `community_size` and `net_flow` dropped. `community_size` is a raw count that
   grows with the graph and scored test AUC 0.10 — it had become an inverted
   proxy for "am I in the giant organic blob", a property of the sample rather
   than of fraud. `net_flow` was redundant with `flow_passthrough`.

4. `repeat_ratio` and `reciprocity` now divide by DISTINCT UNDIRECTED
   counterparties. v2 used `in_degree + out_degree`, which double-counts every
   mutual pair, so an account with reciprocal relationships had its repeat_ratio
   silently halved. That systematically understated exactly the organic
   accounts that most resemble rings.

5. Per-transaction timestamps are stored as int64 epoch nanoseconds rather than
   pd.Timestamp objects. `burst_ratio` needs an hour bucket per transaction and
   doing that through pandas once per node dominated the whole extraction.

v2 KEPT AND STILL TRUE
  • `louvain_community` is not a model feature. It carried 30.5% of model
    importance while being an arbitrary integer: XGBoost split on "community
    id < 17.5", train had 41 communities and test 54 with no correspondence
    between the numbering, and the API sent a constant 0. It survives as a
    metadata column for plots.
  • `amount_cv` uses individual transaction amounts, not per-edge totals.
    Over totals, an edge carrying 20 transactions of ₹50k looked like one ₹1M
    transaction, which inverted the feature's meaning.
  • Node labelling uses `edge_role`, not `is_mule`. A node is a mule if it sits
    inside a ring cycle. Accounts that merely paid into a ring
    (edge_role="fan_in") are unwitting sources and stay label 0.
  • `fan_in_concentration` is documented in the correct direction: a wide
    fan-in hub has LOW concentration.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict, deque
from pathlib import Path

import community as community_louvain  # python-louvain
import networkx as nx
import numpy as np
import pandas as pd

from console import banner, enable_utf8_stdout, hr, sym
from models.features import (
    FEATURE_COLS,
    LABEL_META_COLS,
    METADATA_COLS,
    TARGET_COL,
)

RAW_DIR = Path(__file__).resolve().parent / "raw"
PROCESSED_DIR = Path(__file__).resolve().parent / "processed"

SPLITS = ("train", "val", "test")

LOUVAIN_SEED = 42

# ── Cycle-detection parameters ────────────────────────────────────
# A pair must have transacted at least this many times to count as a standing
# relationship. One-off transfers create spurious cycles: with a single payment
# per pair, any three accounts that happened to pay each other once look like a
# ring. Requiring repetition is also what makes the search tractable.
MIN_REPEATS = 2

# Longest directed cycle considered. Ring sizes in the generator top out at 8,
# so 8 is what it takes to see a full ring loop. Chosen empirically: on the test
# split, single-feature AUC for cycle_participation rises 0.55 → 0.58 → 0.73 →
# 0.80 → 0.87 as the bound goes 3 → 4 → 5 → 6 → 8, while the share of NEGATIVE
# nodes with a nonzero score also rises (2.8% → 23%). Both moving together is
# the point: the feature is finding real structure, not the label.
MAX_CYCLE_LEN = 8

# Cycles of length 2 are mutual pairs, which `reciprocity` already measures.
MIN_CYCLE_LEN = 3


# ══════════════════════════════════════════════════════════════════
# Graph construction
# ══════════════════════════════════════════════════════════════════

def build_graph(edges_df: pd.DataFrame) -> nx.DiGraph:
    """
    Build a directed transaction graph from an edge DataFrame.

    Parallel transactions between the same ordered pair collapse into one edge
    that retains the full per-transaction detail:

        weight         number of transactions
        total_amount   sum of amounts
        amounts        list of individual amounts            (amount_cv)
        timestamps_ns  list of int64 epoch nanoseconds       (velocity, burst)

    Vectorised with groupby; the original per-row `iterrows()` loop was the
    slowest step in the pipeline.
    """
    df = edges_df
    if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])

    # int64 nanoseconds once, here, rather than converting per node later.
    work = pd.DataFrame({
        "sender": df["sender"].to_numpy(),
        "receiver": df["receiver"].to_numpy(),
        "amount": df["amount"].to_numpy(dtype=float),
        "ts_ns": df["timestamp"].to_numpy(dtype="datetime64[ns]").astype("int64"),
    })

    grouped = (
        work.groupby(["sender", "receiver"], sort=False)
        .agg(
            weight=("amount", "size"),
            total_amount=("amount", "sum"),
            amounts=("amount", list),
            timestamps_ns=("ts_ns", list),
        )
        .reset_index()
    )

    G = nx.DiGraph()
    # sorted(), not set(): `partition_fingerprint` and `extend_partition` both
    # rest on the claim that "build_graph produces node insertion order
    # deterministically", and iterating a set of str does not. CPython randomises
    # str hashing per process unless PYTHONHASHSEED is pinned, so two replicas of
    # the service — different processes — enumerated the same accounts in
    # different orders, Louvain consumed its seeded RNG stream differently, and
    # the frozen reference partition differed between them. Verified: three
    # interpreters, three different orderings of the same 3,000 ids. The existing
    # determinism test compares two runs inside ONE process, so it could never
    # see this. Sorting costs one O(n log n) pass on ~3k ids.
    G.add_nodes_from(sorted(set(work["sender"]) | set(work["receiver"])))
    G.add_edges_from(
        (
            row.sender,
            row.receiver,
            {
                "weight": int(row.weight),
                "total_amount": float(row.total_amount),
                "amounts": row.amounts,
                "timestamps_ns": row.timestamps_ns,
            },
        )
        for row in grouped.itertuples(index=False)
    )
    return G


# ══════════════════════════════════════════════════════════════════
# Graph-level features
# ══════════════════════════════════════════════════════════════════

def compute_pagerank(G: nx.DiGraph) -> dict[str, float]:
    """
    PageRank on the transaction graph, with a convergence fallback.

    NOTE: networkx defaults to `weight="weight"`, and our `weight` attribute is
    the TRANSACTION COUNT for the pair, not the rupee value. So this is
    count-weighted PageRank: a relationship used 20 times passes more rank than
    one used once. That is the behaviour we want — repetition is what
    distinguishes a standing laundering route from a one-off payment — but it is
    silent, so it is stated here. Switching to `weight="total_amount"` would make
    it value-weighted and would need re-tuning, not just a keyword change.
    """
    try:
        return nx.pagerank(G, alpha=0.85, max_iter=200)
    except nx.PowerIterationFailedConvergence:
        return nx.pagerank(G, alpha=0.85, max_iter=1000, tol=1e-4)


def undirected_projection(G: nx.DiGraph) -> nx.Graph:
    """
    Collapse the directed graph to an undirected one, SUMMING reciprocal edges.

    THE DEFECT THIS REPLACES
    ────────────────────────
    `G.to_undirected()` was used here, and networkx resolves a reciprocal pair by
    copying one direction's attribute dict over the other's — last edge iterated
    wins. So for every pair that transacts both ways, the undirected `weight`
    became ONE direction's transaction count instead of the pair's total, and the
    other direction's traffic vanished.

    Measured on the shipped data: 5.9% of undirected pairs are reciprocal, and
    they lose 55.9% (train) / 55.7% (test) of their own transaction weight —
    6.28% and 6.40% of ALL transaction weight in the graph. 24.3% of those pairs
    on train touch a ring account, because reciprocal traffic is precisely the
    camouflage a layering ring wears.

    That matters to two consumers that read `weight` off this graph:
      * `compute_louvain_communities` — python-louvain optimises WEIGHTED
        modularity, so half the evidence for keeping a mutual pair together was
        being discarded;
      * `extend_partition` — assigns an unseen account to its heaviest known
        neighbour, so the tie-break was decided on half a relationship.
    (`nx.clustering` and `compute_community_internal_ratio` count edges rather
    than weights and are unaffected either way.)

    `total_amount` is summed for the same reason. The per-transaction `amounts`
    and `timestamps_ns` lists are deliberately NOT carried across: no consumer
    reads them off the undirected graph, and concatenating them here would
    double the peak memory of the largest attributes in the graph for nothing.

    ⚠ TRAIN/SERVE SKEW, NOT YET CLOSED: `api/main.py` builds its own reference
    projection with `build_graph(ctx).to_undirected()`, and `tests/conftest.py`
    does the same for its Louvain fixture. Until both call this function, the
    frozen serving partition is computed on last-wins weights while training
    uses summed ones — which is exactly the class of skew `test_contract.py`'s
    TRAIN/SERVE check exists to catch.
    """
    UG = nx.Graph()
    UG.add_nodes_from(G.nodes())
    for u, v, data in G.edges(data=True):
        weight = int(data.get("weight", 1))
        amount = float(data.get("total_amount", 0.0))
        if UG.has_edge(u, v):
            edge = UG[u][v]
            edge["weight"] += weight
            edge["total_amount"] += amount
        else:
            UG.add_edge(u, v, weight=weight, total_amount=amount)
    return UG


def compute_louvain_communities(UG: nx.Graph) -> dict[str, int]:
    """
    Louvain communities on the undirected projection.

    The integer ids are NOT a model feature — they feed the structural community
    feature below and survive as a metadata column for plots.
    """
    return community_louvain.best_partition(UG, random_state=LOUVAIN_SEED)


# Backwards-compatible alias for the original public name.
compute_louvain_partition = compute_louvain_communities


def extend_partition(UG: nx.Graph, partition: dict[str, int]) -> dict[str, int]:
    """
    Extend a FROZEN partition to cover nodes it has never seen, deterministically.

    ─────────────────────────────────────────────────────────────────────────
    WHY A FROZEN PARTITION EXISTS AT ALL
    ─────────────────────────────────────────────────────────────────────────
    Louvain shuffles node visit order from its `random_state`. A fixed seed makes
    it reproducible on a fixed node set, but change the node set and the RNG
    stream is consumed differently, so the partition moves — and because
    `community_internal_ratio` is a per-COMMUNITY scalar broadcast to every
    member, a partition change moves the feature for accounts nowhere near the
    change.

    Measured on the 2,954-account validation graph: adding two accounts that
    transact only with each other, connected to nothing, shifted
    `community_internal_ratio` for 100% of accounts (median |Δ| 0.0151, 95th pct
    0.1460, max 0.2906) and flipped 84 of 2,954 decisions — 2.84% — at the
    cost-optimal threshold. Pinning that one feature to its original value
    dropped the flips to zero, so it was the sole cause. The same graph scored
    twice is bit-identical, so this is not nondeterminism; it is sensitivity to
    an irrelevant change.

    A risk service that can flip an account from ALLOW to HOLD_FOR_REVIEW because
    an unrelated customer signed up is not defensible, so serving computes the
    partition ONCE on the reference graph and holds it. That is also how this
    would actually deploy: community assignment is a batch job, not something
    recomputed inside a scoring call.

    Training is unaffected and unchanged — each split still partitions its own
    graph once, so the offline metrics keep the same meaning.

    ─────────────────────────────────────────────────────────────────────────
    ASSIGNMENT RULE for a node the frozen partition doesn't cover
    ─────────────────────────────────────────────────────────────────────────
    Join the community of the neighbour it transacts with most by weight, with
    ties broken on node id so the result cannot depend on dict ordering. Repeat
    until nothing more can be placed, so a chain of new accounts attaches to the
    graph rather than to whichever one happened to come first.

    Whatever is left touches no known account. Those nodes are grouped by
    CONNECTED COMPONENT, one fresh community per component — not one per node.
    The difference is not cosmetic: a set of brand-new accounts that transact
    only among themselves scores `community_internal_ratio` 1.0 as a component
    and 0.0 as singletons, and 1.0 is both the structurally correct reading and
    the one training saw, since a split's own Louvain pass would have grouped
    them too. Singletons would hand the clearest ring shape in the batch — a
    closed loop of unknown accounts — the most innocent value the feature has.
    """
    extended = dict(partition)
    unseen = [n for n in UG.nodes() if n not in extended]
    if not unseen:
        return extended

    next_id = (max(extended.values()) + 1) if extended else 0

    remaining = sorted(unseen)
    while True:
        placed = []
        for node in remaining:
            best_comm, best_weight, best_nbr = None, -1.0, None
            for nbr, data in UG[node].items():
                if nbr not in extended:
                    continue
                w = float(data.get("weight", 1.0))
                if w > best_weight or (w == best_weight and
                                       (best_nbr is None or nbr < best_nbr)):
                    best_comm, best_weight, best_nbr = extended[nbr], w, nbr
            if best_comm is not None:
                extended[node] = best_comm
                placed.append(node)
        if not placed:
            break
        remaining = [n for n in remaining if n not in extended]

    # Connected components among the nodes that attached to nothing known.
    # Explicit BFS over adjacency rather than nx.connected_components on a
    # subgraph view, so the traversal order — and therefore the community ids —
    # is fixed by the sort, not by the graph library's internal iteration.
    leftover = set(remaining)
    for start in remaining:
        if start in extended:
            continue
        component, queue = [], deque([start])
        extended[start] = next_id
        while queue:
            node = queue.popleft()
            component.append(node)
            for nbr in sorted(UG[node]):
                if nbr in leftover and nbr not in extended:
                    extended[nbr] = next_id
                    queue.append(nbr)
        next_id += 1

    return extended


def compute_community_internal_ratio(
    UG: nx.Graph,
    partition: dict[str, int],
) -> dict[int, float]:
    """
    Reduce each Louvain community to one transferable scalar:

        internal_ratio = internal_edges / (internal_edges + boundary_edges)

      → 1.0  a closed group that only transacts with itself, which is what a
             layering ring looks like
      → ~0   a community whose members mostly transact outside it

    This is the structural content the raw community *id* never carried.
    `community_size` used to be returned alongside and is gone: it is a raw
    count that grows with the graph, and it scored test AUC 0.10.
    """
    internal: dict[int, int] = defaultdict(int)
    boundary: dict[int, int] = defaultdict(int)

    for u, v in UG.edges():
        cu, cv = partition.get(u, -1), partition.get(v, -1)
        if cu == cv:
            internal[cu] += 1
        else:
            boundary[cu] += 1
            boundary[cv] += 1

    ratios: dict[int, float] = {}
    for comm in set(partition.values()):
        i, b = internal.get(comm, 0), boundary.get(comm, 0)
        ratios[comm] = float(i / (i + b)) if (i + b) > 0 else 0.0
    return ratios


# ══════════════════════════════════════════════════════════════════
# Cycle participation
# ══════════════════════════════════════════════════════════════════

def repeated_edge_cycle_core(
    G: nx.DiGraph,
    min_repeats: int = MIN_REPEATS,
) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    """
    Reduce the graph to the part that can actually contain a cycle.

    Two exact reductions, in order:

    1. Keep only edges with weight >= min_repeats. A cycle built from one-off
       transfers is not a laundering ring, it is coincidence.

    2. Iteratively delete every node with no inbound or no outbound remaining
       edge. Such a node provably cannot lie on any directed cycle, and deleting
       it can strip its neighbours of their last inbound/outbound edge, so this
       repeats to a fixed point.

    Step 2 is where the tractability comes from, and it is lossless. Measured on
    the test split: 16,158 repeated edges over 2,905 nodes reduce to 4,970 edges
    over 1,724 nodes — a 3.3x shrink — while retaining 116 of 119 ring members.
    The three dropped are members of open-chain rings that genuinely never close
    a loop, so a cycle feature *should* score them zero.

    Returns (successors, predecessors) as plain dict-of-set adjacency.
    """
    succ: dict[str, set[str]] = defaultdict(set)
    pred: dict[str, set[str]] = defaultdict(set)
    for u, v, data in G.edges(data=True):
        if data.get("weight", 1) >= min_repeats:
            succ[u].add(v)
            pred[v].add(u)

    # Prune to the fixed point.
    queue = deque(set(succ) | set(pred))
    while queue:
        n = queue.popleft()
        if succ.get(n) and pred.get(n):
            continue
        # n cannot be on a cycle: detach it and re-examine its neighbours.
        for m in succ.pop(n, set()):
            pred[m].discard(n)
            queue.append(m)
        for m in pred.pop(n, set()):
            succ[m].discard(n)
            queue.append(m)

    return (
        {n: s for n, s in succ.items() if s},
        {n: p for n, p in pred.items() if p},
    )


def _bounded_forward_distances(
    succ: dict[str, set[str]],
    start: str,
    depth: int,
    skip_edge: tuple[str, str] | None = None,
) -> dict[str, int]:
    """
    BFS hop-distance from `start` to every node within `depth` hops.

    `skip_edge`
        One directed edge (u, v) to pretend is absent for this traversal only.
        Needed because BFS reports the SHORTEST distance, and a shortest path of
        length 1 hides every longer path to the same node — see
        `compute_cycle_participation` for why that mattered.
    """
    dist = {start: 0}
    queue = deque(((start, 0),))
    while queue:
        node, d = queue.popleft()
        if d == depth:
            continue
        for nxt in succ.get(node, ()):
            if skip_edge is not None and node == skip_edge[0] and nxt == skip_edge[1]:
                continue
            if nxt not in dist:
                dist[nxt] = d + 1
                queue.append((nxt, d + 1))
    del dist[start]
    return dist


def compute_cycle_participation(
    G: nx.DiGraph,
    min_repeats: int = MIN_REPEATS,
    max_len: int = MAX_CYCLE_LEN,
) -> dict[str, float]:
    """
    Fraction of a node's standing counterparties that sit with it on a short
    directed cycle.

        cycle_participation(n) =
            |{m : m is a repeated-edge neighbour of n, and some directed cycle
                  of length in [MIN_CYCLE_LEN, max_len] passes through both}|
            ────────────────────────────────────────────────────────────────
            |repeated-edge neighbours of n|

    Bounded in [0,1] and independent of graph size, so it transfers between the
    training graph and the serving graph.

    WHY NOT nx.simple_cycles(..., length_bound=8)
    ─────────────────────────────────────────────
    Enumerating cycles is the wrong shape of work for this question and it is
    unbounded in output size: the number of cycles up to length 8 in a graph
    with mean out-degree 2.7 can run to millions, all of which would be
    generated and thrown away. It also silently requires networkx >= 3.1, since
    `length_bound` does not exist before that.

    The question actually being asked is per-EDGE, not per-cycle: for the edge
    n → m, is there a return path m ⇝ n of length <= max_len - 1? That is one
    bounded BFS per node, giving O(V · E) worst case with a small constant, and
    it yields the exact same membership answer with no enumeration at all.

    Cycles of length 2 (mutual pairs) are excluded — `reciprocity` measures
    those, and counting them here would make every reciprocal social payment
    look like laundering.

    THE MUTUAL-PAIR DEFECT THIS FUNCTION USED TO HAVE
    ────────────────────────────────────────────────
    "Exclude length-2 cycles" was implemented as a length test on the ONE
    distance BFS returns, and BFS returns the SHORTEST distance. So for a mutual
    pair n ⇄ m, `reach[m][n]` was always 1 — the direct edge m → n — giving a
    candidate cycle length of 2, which failed `MIN_CYCLE_LEN <= 2` and dropped m
    from the numerator. The shortest path masked every longer one: if n and m
    ALSO sat together on a genuine 3-to-8-hop ring, that cycle was invisible.

    The rule being enforced was meant to be "the length-2 cycle does not count".
    What was enforced was "a mutual counterparty never counts" — which discards
    exactly the accounts a layering ring dresses in reciprocal traffic to look
    social. Measured on the shipped data, correcting it raises the score for
    443 / 3,090 training nodes (14.3%) and 241 / 2,907 test nodes (8.3%), and
    recovers 666 / 360 neighbour slots respectively; mean cycle_participation
    among ring accounts goes 0.3125 → 0.3382 on train and 0.2988 → 0.3139 on
    test. Direction-corrected single-feature AUC moves by under 0.005, so this
    is a definition correction rather than a scoreboard change — which is the
    point: the old number did not measure what the docstring claimed.

    The correction is to re-run the BFS for mutual pairs only, with the single
    reciprocal edge masked out (`skip_edge`). Masking exactly one edge is what
    keeps the answer honest:

      * the returned distance d' is then >= 2 by construction, so
        `MIN_CYCLE_LEN` keeps its meaning and pure 2-cycles are still excluded;
      * a BFS shortest path never revisits its own start, so n → m ⇝ n is a
        genuine SIMPLE cycle, not a closed walk that merely passes twice
        through the pair. (This is why the cheaper trick — asking whether some
        successor of m can reach n — is wrong: that path may route back through
        m, and n → m → x ⇝ m → n contains a cycle through m alone.)

    Cost is one extra BFS per mutual pair inside the core, not per node: 3,124
    extra traversals over 885 mutual pairs on train, 2,907 over 782 on test,
    which is roughly a doubling of this function's work and still under 5s.

    MEASURED on the training split: 2.5s for 3,090 nodes / 20,741 pairs, of
    which the cycle core reduces 16,158 repeated edges to 4,970. Serving cost
    matters because api/main.py rebuilds this graph per scoring batch, so if the
    serving context ever grows past one observation window this is the function
    that will notice first.
    """
    succ, pred = repeated_edge_cycle_core(G, min_repeats)
    core_nodes = set(succ) | set(pred)

    # Undirected standing-relationship neighbourhood — the denominator. Taken
    # from the full repeated-edge graph, not the pruned core, so that pruning a
    # node cannot inflate its score by shrinking its own denominator.
    nbrs: dict[str, set[str]] = defaultdict(set)
    for u, v, data in G.edges(data=True):
        if data.get("weight", 1) >= min_repeats:
            nbrs[u].add(v)
            nbrs[v].add(u)

    # One bounded BFS per core node. max_len - 1 because the closing edge of the
    # cycle supplies the final hop.
    reach = {
        n: _bounded_forward_distances(succ, n, max_len - 1)
        for n in core_nodes
    }

    def closes(d: int | None) -> bool:
        """Does a return path of `d` hops make a cycle of allowed length?"""
        return d is not None and MIN_CYCLE_LEN <= d + 1 <= max_len

    out: dict[str, float] = {}
    for node in G.nodes():
        neighbours = nbrs.get(node)
        if not neighbours or node not in core_nodes:
            out[node] = 0.0
            continue

        on_cycle: set[str] = set()

        # Edge node → m closes a cycle if m can reach node within max_len - 1.
        for m in succ.get(node, ()):
            d = reach.get(m, {}).get(node)
            if closes(d):
                on_cycle.add(m)
            elif d == 1:
                # d == 1 means the edge m → node exists: a mutual pair, whose
                # length-2 cycle is correctly rejected above. But BFS reported
                # only that shortest hop, so any LONGER return path m ⇝ node was
                # never seen. Ask again with the reciprocal edge masked; the
                # answer is then the shortest path of length >= 2, which is the
                # shortest cycle through this pair that is not the mutual pair.
                d2 = _bounded_forward_distances(
                    succ, m, max_len - 1, skip_edge=(m, node)).get(node)
                if closes(d2):
                    on_cycle.add(m)

        # Edge m → node closes a cycle if node can reach m within max_len - 1.
        node_reach = reach.get(node, {})
        for m in pred.get(node, ()):
            if m in on_cycle:
                continue
            d = node_reach.get(m)
            if closes(d):
                on_cycle.add(m)
            elif d == 1:
                # Mirror of the case above, for the other direction round the
                # pair. Both are checked because they are different cycles: one
                # uses the edge node → m, the other the edge m → node, and
                # either one qualifies m as sharing a cycle with node.
                d2 = _bounded_forward_distances(
                    succ, node, max_len - 1, skip_edge=(node, m)).get(m)
                if closes(d2):
                    on_cycle.add(m)

        out[node] = len(on_cycle & neighbours) / len(neighbours)

    return out


# ══════════════════════════════════════════════════════════════════
# Per-node features
# ══════════════════════════════════════════════════════════════════

_NS_PER_HOUR = 3_600_000_000_000

# ══════════════════════════════════════════════════════════════════
# THE UNDEFINED CASE, AND WHY IT IS NOT 0.0
# ══════════════════════════════════════════════════════════════════
# Three features below are genuinely undefined for some accounts: HHI needs at
# least one inbound counterparty, a coefficient of variation needs at least two
# observations. All three used to return 0.0 there — and for all three, 0.0 is
# ALSO the most fraud-like value the feature can take (low HHI = a wide fan-in
# collector; low CV = a ring paying every hop the same sum). "No data" and
# "maximally suspicious" were the same number, so the model could not tell them
# apart and the undefined accounts were pooled with the strongest fraud signal
# the column carries.
#
# WHAT WOULD BE CLEANEST, AND WHY IT IS UNAVAILABLE
# ────────────────────────────────────────────────
# NaN. XGBoost learns a missing-branch natively and it is unambiguous. It is
# blocked twice over by code this module must not fight:
#   * models/train.py raises on any non-finite feature value ("the threshold
#     sweep cannot be trusted over non-finite scores"), naming this file in the
#     error;
#   * tests/test_features.py asserts every FEATURE_COLS value is finite and
#     non-NaN on all three splits.
# An out-of-range sentinel (-1.0, or 2.0 for the HHI) is blocked by the same
# test file: `fan_in_concentration` is asserted inside [0, 1] and the two CVs are
# asserted non-negative. Adding a companion `is_defined` column is blocked by
# `compute_node_features`, which refuses any column not declared in
# models/features.py — a file this pass does not own.
#
# WHAT IS DONE INSTEAD
# ────────────────────
# Each undefined case returns the least fraud-like value in the feature's legal
# range, so the collision moves off the signal instead of sitting on top of it.
# The remaining ambiguity is with genuine values at that end, and it is
# resolvable: `in_degree` distinguishes "no inbound counterparties" from "one",
# and `repeat_ratio` distinguishes "one transaction" from "many".
#
# This is rare on the shipped data — 23/3,090 train, 32/2,954 val, 33/2,907 test
# accounts for the HHI, and 1-3 and 4-9 accounts respectively for the two CVs —
# which is why it is a correctness fix rather than a scoreboard one. Rare also
# means a tree can never isolate these accounts into their own leaf, so the only
# thing that matters is WHICH neighbourhood they get pooled into.
UNDEFINED_HHI = 1.0     # no inbound value at all → not a fan-in hub
UNDEFINED_CV = 1.0      # CV of an exponential: the no-structure reference point


def compute_fan_in_concentration(in_amounts: list[float]) -> float:
    """
    Herfindahl-Hirschman Index over inbound value: HHI = Σ(shareᵢ²).

      → near 1.0  a single counterparty supplies nearly all inbound value
                  (a dedicated collector account)
      → near 0.0  inbound value is spread thinly across many counterparties,
                  which is what a wide fan-in hub looks like

    The original docstring claimed "high concentration = suspicious", which had
    the direction backwards: a fan-in hub with many feeders has *low* HHI. Both
    extremes are informative and the model may use either; what matters is that
    the documentation matches the arithmetic. Wide fan-in shows up as high
    `in_degree` combined with low HHI.

    UNDEFINED (no inbound edges, or inbound value summing to zero) returns
    `UNDEFINED_HHI` = 1.0, not 0.0. With k inbound counterparties HHI is bounded
    below by 1/k, so 0.0 was unreachable-when-defined — it read as "the widest
    fan-in in the graph", which is the exact opposite of the truth for an account
    with no inbound traffic whatsoever. 1.0 says "inbound value is not spread
    across anyone", which is the honest reading of an empty fan-in and sits at
    the benign end. See the block above for why not NaN.
    """
    total = float(sum(in_amounts))
    if not in_amounts or total <= 0:
        return UNDEFINED_HHI
    shares = np.asarray(in_amounts, dtype=float) / total
    return float(np.sum(shares ** 2))


def compute_txn_velocity(timestamps_ns: list[int]) -> float:
    """
    Transactions per hour across the node's active span.

    Takes epoch nanoseconds so no datetime parsing happens per node.
    """
    n = len(timestamps_ns)
    if n < 2:
        return 0.0
    span_h = (max(timestamps_ns) - min(timestamps_ns)) / _NS_PER_HOUR
    if span_h <= 0:
        return float(n)  # everything landed inside the same instant
    return float(n / span_h)


def compute_burst_ratio(timestamps_ns: list[int]) -> float:
    """
    Share of the node's transactions falling in its single busiest clock hour.

    Bounded [0,1]. Separates "many transactions" from "many transactions crammed
    into a window", which is the laundering pattern — and also what payday looks
    like, which is why the generator emits post-salary bursts.
    """
    n = len(timestamps_ns)
    if n == 0:
        return 0.0
    buckets: dict[int, int] = defaultdict(int)
    for ts in timestamps_ns:
        buckets[ts // _NS_PER_HOUR] += 1
    return float(max(buckets.values()) / n)


def compute_amount_cv(amounts: list[float]) -> float:
    """
    Coefficient of variation over INDIVIDUAL transaction amounts.

    Rings repeat near-identical values as they layer funds, so a low CV across
    many transactions is the signal. Computing this over per-edge totals — the
    v1 behaviour — destroyed exactly that signal.

    UNDEFINED (fewer than two transactions, or a zero mean) returns
    `UNDEFINED_CV` = 1.0, not 0.0. On the shipped test split the DEFINED
    distribution runs p05 0.449 / median 1.749 / p95 2.904, with ring accounts at
    median 1.333 against 1.803 for the rest — so 0.0 sat below the entire defined
    range, at the far end of the direction that means "ring". One transaction is
    no evidence of uniformity at all; 1.0 is the CV of an exponential, the
    maximum-entropy spread for a positive quantity, and lands inside the bulk of
    the defined distribution where "no evidence either way" belongs.
    """
    if len(amounts) < 2:
        return UNDEFINED_CV
    arr = np.asarray(amounts, dtype=float)
    mean = arr.mean()
    if mean == 0:
        return UNDEFINED_CV
    return float(arr.std() / mean)


def compute_counterparty_amount_cv(per_counterparty_means: list[float]) -> float:
    """
    CV across the node's per-counterparty MEAN amounts.

    A ring pays every hop roughly the same sum, so this is low. A real user pays
    rent, a kirana store and a friend amounts that differ by orders of
    magnitude, so it is high — even when their individual-transaction CV is low
    because each relationship is itself consistent.

    That distinction is the reason this feature exists: `amount_cv` alone cannot
    separate "consistent within each relationship" from "consistent across all
    relationships", and the second is the laundering signature.

    UNDEFINED (fewer than two counterparties, or a zero mean) returns
    `UNDEFINED_CV` for the same reason as `compute_amount_cv`: an account with a
    single relationship gives no evidence that it pays everyone alike, and 0.0
    asserted exactly that. `repeat_ratio` and the degree columns are what
    separate a one-counterparty account from a consistent multi-hop one.
    """
    if len(per_counterparty_means) < 2:
        return UNDEFINED_CV
    arr = np.asarray(per_counterparty_means, dtype=float)
    mean = arr.mean()
    if mean == 0:
        return UNDEFINED_CV
    return float(arr.std() / mean)


def partition_fingerprint(partition: dict[str, int]) -> str:
    """
    A short, label-permutation-invariant fingerprint of a partition.

    The reference partition used at serving time is recomputed at every startup.
    python-louvain with a fixed `random_state` is reproducible given the same node
    insertion order, and `build_graph` produces that order deterministically — but
    "should be reproducible" is an assumption about a third-party library, a numpy
    version and a dict ordering, and it is silently load-bearing: two replicas of
    the service that partition differently would return different scores for the
    same account, and nothing would say so.

    So the assumption gets a check instead of a comment. /health publishes this
    string; if two replicas or two restarts disagree, it is immediately visible.

    Community ids are arbitrary, so they are canonicalised to the rank of each
    community's smallest member before hashing. Two partitions with identical
    membership and different labels therefore fingerprint the same, and any real
    membership change shows up.
    """
    members: dict[int, list[str]] = defaultdict(list)
    for node, comm in partition.items():
        members[comm].append(node)

    order = sorted(members, key=lambda c: min(members[c]))
    canonical = {c: rank for rank, c in enumerate(order)}

    payload = ";".join(
        f"{node}:{canonical[partition[node]]}" for node in sorted(partition)
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]
    return f"{digest}/{len(order)}c/{len(partition)}n"


def compute_node_features(
    G: nx.DiGraph,
    verbose: bool = False,
    partition: dict[str, int] | None = None,
) -> pd.DataFrame:
    """
    Compute the full feature table for every node in `G`.

    THIS is the function api/main.py calls, so serving-time and training-time
    features come from one implementation by construction.

    `partition`
        None (the default, and what training uses) → partition this graph with
        Louvain here, exactly as before.

        A frozen partition → reuse it, extending it to cover nodes it has not
        seen via `extend_partition`. Serving passes the partition computed once
        at startup from the reference graph, because repartitioning per request
        made `community_internal_ratio` — and therefore the decision — depend on
        accounts unrelated to the one being scored. See `extend_partition` for
        the measurement.

    Returns a DataFrame with a `node` column, every column in FEATURE_COLS, and
    the `louvain_community` metadata column. Ground-truth columns are attached
    separately by `label_nodes` — this function never sees labels, which is what
    makes it safe to call at serving time.
    """
    if verbose:
        print("  Undirected projection...")
    # Summing projection, not G.to_undirected(): the latter lets one direction of
    # a reciprocal pair overwrite the other's weight, losing 6.3% of all
    # transaction weight. See `undirected_projection`.
    UG = undirected_projection(G)  # computed once; three consumers below

    if verbose:
        print("  PageRank...")
    pagerank = compute_pagerank(G)

    if partition is None:
        if verbose:
            print("  Louvain communities...")
        partition = compute_louvain_communities(UG)
    else:
        if verbose:
            print("  Louvain communities (frozen partition supplied)...")
        partition = extend_partition(UG, partition)

    # Ratios are always recounted on THIS graph. That is deliberate: with the
    # partition held fixed, the counts only move for communities whose edges
    # actually changed, so an unrelated account's feature cannot move — while a
    # community that genuinely closes in on itself still shows it.
    comm_ratios = compute_community_internal_ratio(UG, partition)

    if verbose:
        print("  Clustering coefficients...")
    clustering = nx.clustering(UG)

    if verbose:
        print(f"  Cycle participation (repeated edges, length <= {MAX_CYCLE_LEN})...")
    cycle_part = compute_cycle_participation(G)

    if verbose:
        print("  Per-node features...")
    records: list[dict] = []

    for node in G.nodes():
        # ── one pass over each side of the adjacency ──
        in_amounts_by_cp: list[float] = []
        amounts: list[float] = []
        timestamps: list[int] = []
        cp_total: dict[str, float] = defaultdict(float)
        cp_count: dict[str, int] = defaultdict(int)

        in_amount = 0.0
        for src, _, data in G.in_edges(node, data=True):
            total = float(data.get("total_amount", 0.0))
            in_amount += total
            in_amounts_by_cp.append(total)
            amounts.extend(data.get("amounts", ()))
            timestamps.extend(data.get("timestamps_ns", ()))
            cp_total[src] += total
            cp_count[src] += int(data.get("weight", 0))

        out_amount = 0.0
        for _, dst, data in G.out_edges(node, data=True):
            total = float(data.get("total_amount", 0.0))
            out_amount += total
            amounts.extend(data.get("amounts", ()))
            timestamps.extend(data.get("timestamps_ns", ()))
            cp_total[dst] += total
            cp_count[dst] += int(data.get("weight", 0))

        in_deg = G.in_degree(node)
        out_deg = G.out_degree(node)

        # Distinct UNDIRECTED counterparties. v2 used in_deg + out_deg, which
        # double-counts every mutual pair and so halved repeat_ratio for exactly
        # the reciprocal organic accounts that most resemble rings.
        n_counterparties = len(cp_total)
        n_txns = len(amounts)

        # Counterparties this node both pays and is paid by → `reciprocity`.
        # Empirically this never exceeds 0.5 on our data and 47% of nodes score
        # exactly 0, which looks broken but is correct: verified against an
        # independent recomputation to float precision. Every account carries
        # strictly one-directional relationships (salary in, rent/EMI/
        # subscription/utility out), so mutual pairs can only ever be a minority
        # of any real account's counterparties. The feature therefore has its
        # resolution in a narrow band near the bottom — worth knowing before
        # anyone "fixes" it by rescaling.
        mutual = sum(
            1 for cp in cp_total
            if G.has_edge(node, cp) and G.has_edge(cp, node)
        )

        cp_means = [
            cp_total[cp] / cp_count[cp]
            for cp in cp_total if cp_count[cp] > 0
        ]

        comm = partition.get(node, -1)
        hi_amt, lo_amt = max(in_amount, out_amount), min(in_amount, out_amount)
        hi_deg, lo_deg = max(in_deg, out_deg), min(in_deg, out_deg)

        records.append({
            "node": node,
            # ── degree structure ──
            "in_degree": int(in_deg),
            "out_degree": int(out_deg),
            "degree_ratio": out_deg / max(in_deg, 1),
            "degree_balance": (lo_deg / hi_deg) if hi_deg > 0 else 0.0,
            # ── value flow ──
            "in_amount_sum": round(in_amount, 2),
            "out_amount_sum": round(out_amount, 2),
            "flow_passthrough": (lo_amt / hi_amt) if hi_amt > 0 else 0.0,
            # ── centrality / topology ──
            "pagerank": float(pagerank.get(node, 0.0)),
            "clustering_coefficient": float(clustering.get(node, 0.0)),
            "cycle_participation": float(cycle_part.get(node, 0.0)),
            "reciprocity": (mutual / n_counterparties) if n_counterparties else 0.0,
            # ── behavioural ──
            "fan_in_concentration": compute_fan_in_concentration(in_amounts_by_cp),
            "txn_velocity": compute_txn_velocity(timestamps),
            "burst_ratio": compute_burst_ratio(timestamps),
            "amount_cv": compute_amount_cv(amounts),
            "counterparty_amount_cv": compute_counterparty_amount_cv(cp_means),
            "repeat_ratio": (n_txns / n_counterparties) if n_counterparties else 0.0,
            # ── community structure ──
            "community_internal_ratio": float(comm_ratios.get(comm, 0.0)),
            # ── metadata only, never fed to the model ──
            "louvain_community": int(comm),
        })

    df = pd.DataFrame(records)

    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise RuntimeError(
            f"Extractor did not produce declared features: {missing}.\n"
            "models/features.py and data/extractor.py are out of sync."
        )
    extra = [
        c for c in df.columns
        if c not in FEATURE_COLS and c not in METADATA_COLS
    ]
    if extra:
        raise RuntimeError(
            f"Extractor produced undeclared columns: {extra}.\n"
            "Add them to FEATURE_COLS or METADATA_COLS in models/features.py, "
            "or stop emitting them. Silent extra columns are how a feature ends "
            "up in the CSV but not in the model."
        )
    return df


# Backwards-compatible alias for the original public name.
extract_node_features = compute_node_features


# ══════════════════════════════════════════════════════════════════
# Labelling
# ══════════════════════════════════════════════════════════════════

def label_nodes(features_df: pd.DataFrame, edges_df: pd.DataFrame) -> pd.DataFrame:
    """
    Attach ground truth: the target plus the label metadata used for evaluation.

    A node is a mule iff it is an endpoint of an `edge_role == "ring"` edge.
    Accounts that only appear on `fan_in` edges are unwitting sources and stay 0
    — labelling them positive put hundreds of accounts with entirely organic
    feature profiles into the positive class and was a large part of why v1
    precision sat at 0.499.

    Also attaches, for positives:
      ring_id    the correct grouping key for cross-validation. Grouping on
                 Louvain community put 52% of train nodes — and zero positives —
                 into a single group, so folds were silently skipped.
      ring_type  the archetype, so models/train.py can report recall per ring
                 shape. That breakdown is where an honest evaluation admits the
                 stealthy rings are much harder than the loud ones.

    Negatives get ring_id = -1 and ring_type = "organic".
    """
    if "edge_role" not in edges_df.columns:
        raise RuntimeError(
            "Edge file has no `edge_role` column, so ring members cannot be "
            "distinguished from the accounts that merely paid into a ring.\n"
            "Regenerate with `python -m data.generator`."
        )

    ring_edges = edges_df[edges_df["edge_role"] == "ring"]
    mule_nodes = set(ring_edges["sender"]) | set(ring_edges["receiver"])

    # First ring a node appears in. Rings are window-confined and entity-
    # disjoint across splits (asserted in the generator), so within one split a
    # node has exactly one ring in all but pathological configurations.
    ring_of: dict[str, int] = {}
    type_of: dict[str, str] = {}
    for row in ring_edges[["sender", "receiver", "ring_id", "ring_type"]].itertuples(
        index=False
    ):
        for account in (row.sender, row.receiver):
            if account not in ring_of:
                ring_of[account] = int(row.ring_id)
                type_of[account] = str(row.ring_type)

    features_df[TARGET_COL] = features_df["node"].isin(mule_nodes).astype(int)
    features_df["ring_id"] = (
        features_df["node"].map(ring_of).fillna(-1).astype(int)
    )
    features_df["ring_type"] = (
        features_df["node"].map(type_of).fillna("organic").astype(str)
    )
    return features_df


# ══════════════════════════════════════════════════════════════════
# Pipeline
# ══════════════════════════════════════════════════════════════════

def process_split(name: str, edges_path: Path) -> pd.DataFrame:
    """Process a single split into a labelled feature table."""
    print(f"\n{hr(50)}")
    print(f"Processing: {name}")
    print(hr(50))

    edges_df = pd.read_csv(edges_path, parse_dates=["timestamp"])
    span_days = (
        edges_df["timestamp"].max() - edges_df["timestamp"].min()
    ).total_seconds() / 86_400.0
    print(f"  Loaded {len(edges_df):,} edges over {span_days:.0f} days")

    G = build_graph(edges_df)
    print(f"  Graph: {G.number_of_nodes():,} nodes, "
          f"{G.number_of_edges():,} unique pairs")

    features_df = compute_node_features(G, verbose=True)
    features_df = label_nodes(features_df, edges_df)
    features_df["split"] = name.lower()

    n_pos = int(features_df[TARGET_COL].sum())
    print(f"  Features: {features_df.shape[0]:,} nodes x "
          f"{len(FEATURE_COLS)} model features")
    print(f"  Positives: {n_pos:,} / {len(features_df):,} "
          f"({features_df[TARGET_COL].mean():.2%})")
    by_type = (
        features_df.loc[features_df[TARGET_COL] == 1, "ring_type"]
        .value_counts().to_dict()
    )
    print(f"  By archetype: "
          + ", ".join(f"{k} {v}" for k, v in sorted(by_type.items())))

    # Deterministic column order: identity, features, metadata, ground truth.
    # Built from the declared contract rather than hard-coded, so adding a
    # feature to models/features.py is the only edit needed.
    ordered = (
        ["node"]
        + FEATURE_COLS
        + [c for c in METADATA_COLS if c != "node"]
        + LABEL_META_COLS
        + [TARGET_COL]
    )
    assert set(ordered) == set(features_df.columns), (
        f"column set mismatch: "
        f"missing {set(ordered) - set(features_df.columns)}, "
        f"unexpected {set(features_df.columns) - set(ordered)}"
    )
    return features_df[ordered]


def main() -> None:
    enable_utf8_stdout()
    print(banner("UPI Mule-Ring Sentinel: Feature Extractor (v3)"))

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    spans: dict[str, float] = {}
    for split in SPLITS:
        edges_path = RAW_DIR / f"{split}_edges.csv"
        if not edges_path.exists():
            raise FileNotFoundError(
                f"{edges_path} not found. Run `python -m data.generator` first."
            )

        features = process_split(split.upper(), edges_path)
        out_path = PROCESSED_DIR / f"{split}_features.csv"
        features.to_csv(out_path, index=False)
        print(f"  {sym('arrow')} {out_path.name} "
              f"({len(features):,} rows x {features.shape[1]} cols)")

    print(f"\n{sym('ok')} Feature extraction complete "
          f"({len(FEATURE_COLS)} model features + "
          f"{len(METADATA_COLS) + len(LABEL_META_COLS)} metadata columns).")


if __name__ == "__main__":
    main()
