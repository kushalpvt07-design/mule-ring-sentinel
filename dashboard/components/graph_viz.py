"""
dashboard/components/graph_viz.py
─────────────────────────────────
Interactive transaction-graph viewer.

─────────────────────────────────────────────────────────────────────────────
WHAT CHANGED FROM v2, AND WHY IT MATTERED
─────────────────────────────────────────────────────────────────────────────
1. IT ACCUSED THE VICTIMS.  v2 decided who was a mule from EDGE incidence:

       for u, v, data in G.edges(data=True):
           if data.get("is_mule", 0) == 1:
               mule_nodes.add(u); mule_nodes.add(v)

   A mule edge has two ends, and in a fan-in ring only one of them is the mule.
   The other is the account being drained. Measured on `data/raw/test_edges.csv`:
   285 nodes touch at least one `is_mule == 1` edge, but only 119 accounts carry
   a node-level mule label. So 166 nodes were painted "⚠️ MULE SUSPECT" in red,
   and ALL 166 of them are senders on `fan_in` edges — the victims paying into
   the collection hub. On a page whose whole subject is false-positive cost, the
   picture had a 58% false-accusation rate before the model was even consulted.

   Now truth comes from the node-level `is_mule` column in
   `data/processed/{split}_features.csv` — the same label the model is trained
   and scored against — and the model's opinion comes from the real booster via
   `dashboard/scoring.py`. Those are two different things, so they are drawn as
   two different things: ground truth sets the SHAPE, the model's decision sets
   the COLOUR, and victims get their own class that says in words that they are
   not suspects.

2. EVERY EDGE TOOLTIP SHOWED THE WRONG AMOUNT.  v2 looped `G.add_edge(...)` per
   transaction row, and a DiGraph keeps one edge per ordered pair, so each
   repeat silently overwrote the last. 16,158 of 18,700 ordered pairs in the test
   split carry more than one transaction (max 31), so 86% of edges displayed a
   single arbitrary transaction's amount as if it were the whole relationship.
   Pairs are now aggregated first — transaction count, total and mean — matching
   `data.extractor.build_graph`, so the picture and the features agree.

3. CYCLE ENUMERATION WAS UNBOUNDED.  `list(nx.simple_cycles(G))` materialises
   every simple cycle at any length; on a few hundred densely connected accounts
   that can be millions, and v2's `except Exception` would not catch a hang. The
   panel now uses the project's own definition of a cycle — the pruned
   repeated-edge core from `data/extractor.py`, lengths in
   [3, MAX_CYCLE_LEN], relationships of at least MIN_REPEATS transactions — with
   a hard cap on both results and search steps, and it says when it truncated.
   Using the extractor's core is also the point: the cycles shown here are the
   same cycles `cycle_participation` scores, not a second opinion.

4. THE TEMP FILE LEAKED, AND ON WINDOWS IT FAILED.  v2 opened a
   `NamedTemporaryFile(delete=False)` and re-opened `f.name` by path while its
   own handle was still open, which Windows refuses, and `delete=False` meant
   nothing was ever cleaned up. PyVis renders to a string now, with a
   `TemporaryDirectory` fallback that closes before reading and removes itself.

5. PAGERANK WAS RECOMPUTED ON THE SUBGRAPH.  It was labelled the same as the
   `pagerank` feature but computed over ~150 sampled accounts, so it was a
   different number wearing the feature's name. Tooltips now show the real
   feature values the model was actually given.

6. SELECTION WAS "HIGHEST DEGREE".  Trimming to the top-degree nodes centres the
   picture on hubs and can cut a ring in half. Selection is now intentional —
   a labelled ring, the model's top alerts, its worst false positives, or its
   misses — because a fraud reviewer's question is never "show me 150 accounts",
   it is "show me why this one fired".
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dashboard.scoring import (  # noqa: E402
    ModelUnavailable,
    load_features,
    load_metrics,
    resolve_threshold,
    score_split,
)
from data.extractor import (  # noqa: E402
    MAX_CYCLE_LEN,
    MIN_CYCLE_LEN,
    MIN_REPEATS,
    repeated_edge_cycle_core,
)
from models.cost_matrix import (  # noqa: E402
    DEFAULT_FN_COST,
    DEFAULT_FP_COST,
    CostEvaluator,
)

try:
    from pyvis.network import Network
    PYVIS_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    PYVIS_AVAILABLE = False

RAW_DIR = PROJECT_ROOT / "data" / "raw"

# Imported, not re-declared: if the feature's cycle bounds ever change, this
# page must change with them or it will illustrate a definition the model does
# not use.
MIN_CYCLE_NODES = MIN_CYCLE_LEN
MAX_CYCLES_SHOWN = 15
CYCLE_STEP_BUDGET = 200_000

# Features worth putting in a tooltip: the ones that carry the ring signal.
# Read from the split's feature file, so they are the values the model saw.
TOOLTIP_FEATURES = (
    "pagerank",
    "cycle_participation",
    "fan_in_concentration",
    "reciprocity",
    "repeat_ratio",
    "flow_passthrough",
    "community_internal_ratio",
)

# One definition of the colour scheme, used to paint nodes AND to draw the
# legend. v2 hard-coded the legend in a separate markdown string, which is how a
# legend ends up disagreeing with the picture.
NODE_CLASSES: dict[str, tuple[str, str]] = {
    "caught": ("#ff4757", "Mule, flagged — true positive"),
    "missed": ("#7d1f2b", "Mule, NOT flagged — false negative (the expensive one)"),
    "false_alarm": ("#ffa502", "Legitimate, flagged — false positive (analyst time + customer friction)"),
    "victim": ("#48dbfb", "Legitimate account that PAYS INTO a mule — a victim, not a suspect"),
    "clear": ("#3d5a80", "Legitimate, not flagged"),
    "unlabelled": ("#57606f", "Not present in this split's labelled accounts"),
}


# ══════════════════════════════════════════════════════════════════
# Loading
# ══════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def _load_edges(split: str) -> pd.DataFrame | None:
    """Raw transaction rows for one split, or None if the file is absent."""
    path = RAW_DIR / f"{split}_edges.csv"
    if not path.exists():
        return None
    frame = pd.read_csv(path)
    for column in ("sender", "receiver"):
        frame[column] = frame[column].astype(str)
    return frame


@st.cache_data(show_spinner="Loading accounts and scoring them with the real model...")
def _node_table(split: str) -> tuple[pd.DataFrame | None, str | None]:
    """
    Node-level truth for one split, plus the real model's risk score.

    Never raises: a missing model should cost the viewer the score column, not
    the whole page. `risk_score` is NaN when the model could not be loaded, and
    the second return value explains why so the caption can say so out loud
    rather than showing an uncoloured graph with no reason given.
    """
    try:
        scored = score_split(split)
        frame = scored.features.copy()
        frame["risk_score"] = scored.y_proba
        return frame, None
    except ModelUnavailable as exc:
        try:
            frame = load_features(split)
        except ModelUnavailable as inner:
            return None, str(inner)
        frame["risk_score"] = np.nan
        return frame, (
            "Model scores are unavailable, so nodes are coloured by label only "
            f"and nothing here reflects the detector's decisions. {exc}"
        )


def _default_threshold() -> tuple[float, str]:
    """
    The operating threshold, and a sentence saying where it came from.

    Falls back to the break-even probability implied by the default FN/FP costs
    rather than to 0.5. At ₹200,000 : ₹15,000 break-even is ≈0.07, so a 0.5
    default would grey out most of the model's true positives and make the
    picture look like a detector that finds nothing.
    """
    try:
        return resolve_threshold(load_metrics()), "the threshold selected on validation in metrics.json"
    except ModelUnavailable:
        evaluator = CostEvaluator(fn_cost=DEFAULT_FN_COST, fp_cost=DEFAULT_FP_COST)
        return (
            float(evaluator.break_even_probability),
            f"the break-even probability at the default ₹{DEFAULT_FN_COST:,.0f} : "
            f"₹{DEFAULT_FP_COST:,.0f} costs (metrics.json was unreadable)",
        )


# ══════════════════════════════════════════════════════════════════
# Subgraph construction
# ══════════════════════════════════════════════════════════════════

def _khop(
    edges: pd.DataFrame,
    seeds: set[str],
    hops: int,
    max_nodes: int,
) -> tuple[pd.DataFrame, set[str], bool]:
    """
    Expand `seeds` outward by `hops` and return the induced edge rows.

    Vectorised with `isin` masks; v2 walked `sample.iterrows()` over tens of
    thousands of rows to build the graph, which is the slowest way pandas offers.

    When a hop would exceed `max_nodes`, candidates are ranked by how many
    transactions link them to what is already kept, so the picture grows along
    the strongest relationships instead of being decapitated by a degree sort.
    Ties resolve by first appearance in the CSV, which is fixed, so the same
    controls always yield the same picture.
    """
    keep = set(seeds)
    truncated = False

    for _ in range(max(1, hops)):
        touching = edges["sender"].isin(keep) | edges["receiver"].isin(keep)
        if not touching.any():
            break
        sub = edges.loc[touching]
        candidates = pd.concat(
            [sub["sender"], sub["receiver"]], ignore_index=True
        )
        new = set(candidates.unique()) - keep
        if not new:
            break

        if len(keep) + len(new) > max_nodes:
            # Rank by transaction count against the kept set.
            linked = pd.concat(
                [
                    sub.loc[sub["receiver"].isin(keep), "sender"],
                    sub.loc[sub["sender"].isin(keep), "receiver"],
                ],
                ignore_index=True,
            ).value_counts()
            room = max(0, max_nodes - len(keep))
            keep.update([n for n in linked.index if n in new][:room])
            truncated = True
            break

        keep.update(new)

    induced = edges["sender"].isin(keep) & edges["receiver"].isin(keep)
    return edges.loc[induced], keep, truncated


def _pair_frame(edges: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse transaction rows into one row per ordered pair.

    `weight` is the transaction count, named to match
    `data.extractor.build_graph` so the same graph can be handed to
    `repeated_edge_cycle_core` without translation.

    `is_mule` aggregates with max, which is exact here: no ordered pair in any
    split carries a mixed mule label (verified — 0 of 18,700 pairs). `edge_role`
    genuinely does vary within a pair (3,484 pairs mix roles, e.g. an account
    that both trades organically with a hub and later feeds it), so roles are
    kept as a sorted set rather than an arbitrary first value.
    """
    return (
        edges.groupby(["sender", "receiver"], sort=True)
        .agg(
            weight=("amount", "size"),
            total_amount=("amount", "sum"),
            mean_amount=("amount", "mean"),
            is_mule=("is_mule", "max"),
            roles=("edge_role", lambda s: ", ".join(sorted(set(s)))),
        )
        .reset_index()
    )


def _build_graph(pairs: pd.DataFrame, nodes: set[str]) -> nx.DiGraph:
    """Directed graph from the aggregated pair frame, isolated nodes included."""
    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    if len(pairs):
        G.add_edges_from(
            (
                row.sender,
                row.receiver,
                {
                    "weight": int(row.weight),
                    "total_amount": float(row.total_amount),
                    "mean_amount": float(row.mean_amount),
                    "is_mule": int(row.is_mule),
                    "roles": str(row.roles),
                },
            )
            for row in pairs.itertuples(index=False)
        )
    return G


# ══════════════════════════════════════════════════════════════════
# Bounded cycle enumeration
# ══════════════════════════════════════════════════════════════════

def _bounded_cycles(
    succ: dict[str, set[str]],
    min_len: int = MIN_CYCLE_NODES,
    max_len: int = MAX_CYCLE_LEN,
    max_results: int = MAX_CYCLES_SHOWN,
    step_budget: int = CYCLE_STEP_BUDGET,
) -> tuple[list[list[str]], bool]:
    """
    Enumerate up to `max_results` directed cycles, with a hard work budget.

    Each cycle is reported exactly once, as the rotation beginning at its
    smallest member: the search only steps to nodes that sort after the start,
    so the |c| rotations of a cycle collapse to one. Recursion depth is bounded
    by `max_len` (8), so there is no stack risk.

    Returns (cycles, truncated). `truncated` is true when either cap was hit,
    and the caller must say so — "no more cycles found" and "we stopped looking"
    are very different claims to put in front of a fraud analyst.
    """
    order = {node: i for i, node in enumerate(sorted(succ))}
    results: list[list[str]] = []
    state = {"steps": 0, "truncated": False}

    def walk(start: str, node: str, path: list[str], on_path: set[str]) -> None:
        state["steps"] += 1
        if state["steps"] > step_budget:
            state["truncated"] = True
            return
        for nxt in sorted(succ.get(node, ())):
            if nxt == start:
                if min_len <= len(path) <= max_len:
                    results.append(list(path))
                    if len(results) >= max_results:
                        state["truncated"] = True
                        return
                continue
            if nxt in on_path or order.get(nxt, -1) < order[start]:
                continue
            if len(path) + 1 > max_len:
                continue
            path.append(nxt)
            on_path.add(nxt)
            walk(start, nxt, path, on_path)
            path.pop()
            on_path.discard(nxt)
            if state["truncated"]:
                return

    for start in sorted(succ):
        if state["truncated"]:
            break
        walk(start, start, [start], {start})

    return results, state["truncated"]


# ══════════════════════════════════════════════════════════════════
# Classification and rendering
# ══════════════════════════════════════════════════════════════════

def _classify(known: bool, is_mule: bool, flagged: bool, touches_mule: bool) -> str:
    """
    Which of NODE_CLASSES a node belongs to.

    Order is the whole point. `is_mule` is node-level ground truth and decides
    first; only then does the model's decision split true positives from misses.
    A legitimate account that the model flagged is a false positive even if it
    also touches a mule edge — being a victim does not make an alert correct,
    which is exactly the conflation v2 made.
    """
    if not known:
        return "unlabelled"
    if is_mule:
        return "caught" if flagged else "missed"
    if flagged:
        return "false_alarm"
    if touches_mule:
        return "victim"
    return "clear"


def _node_tooltip(
    node: str,
    klass: str,
    row: pd.Series | None,
    threshold: float,
    graph: nx.DiGraph,
    touches_mule: bool,
) -> str:
    """PyVis hover text: what the node is, what the model said, and why."""
    lines = [node, NODE_CLASSES[klass][1], ""]

    if row is None:
        lines.append("Not in this split's labelled accounts.")
    else:
        truth = "MULE (labelled)" if int(row.get("is_mule", 0)) == 1 else "legitimate (labelled)"
        lines.append(f"Ground truth: {truth}")
        ring_type = row.get("ring_type")
        ring_id = row.get("ring_id")
        if isinstance(ring_type, str) and ring_type not in ("", "organic"):
            lines.append(f"Ring: {ring_id} ({ring_type})")

        score = row.get("risk_score", np.nan)
        if pd.notna(score):
            verdict = "FLAGGED" if float(score) >= threshold else "not flagged"
            lines.append(f"Model risk: {float(score):.4f} → {verdict} (t={threshold:.4f})")
        else:
            lines.append("Model risk: unavailable")

        if touches_mule and int(row.get("is_mule", 0)) == 0:
            lines.append("Transacts with a mule — being paid from or paying into a ring.")

        lines.append("")
        for feature in TOOLTIP_FEATURES:
            if feature in row.index and pd.notna(row[feature]):
                lines.append(f"{feature}: {float(row[feature]):.4f}")

    lines.append("")
    lines.append(f"in-degree {graph.in_degree(node)} / out-degree {graph.out_degree(node)}")
    return "\n".join(lines)


def _create_pyvis_network(
    graph: nx.DiGraph,
    nodes: pd.DataFrame,
    seeds: set[str],
    threshold: float,
    touching_mule: set[str],
) -> Network:
    """Style the graph for PyVis. Colour = model decision × truth, shape = truth."""
    net = Network(
        height="620px",
        width="100%",
        bgcolor="#0e1117",
        font_color="white",
        directed=True,
    )
    net.set_options("""
    {
        "nodes": {
            "borderWidth": 2,
            "borderWidthSelected": 4,
            "font": {"size": 11, "color": "#cccccc"}
        },
        "edges": {
            "color": {"inherit": false},
            "smooth": {"type": "curvedCW", "roundness": 0.15},
            "arrows": {"to": {"enabled": true, "scaleFactor": 0.55}}
        },
        "physics": {
            "forceAtlas2Based": {
                "gravitationalConstant": -60,
                "centralGravity": 0.008,
                "springLength": 140,
                "springConstant": 0.08
            },
            "solver": "forceAtlas2Based",
            "stabilization": {"iterations": 150}
        },
        "interaction": {"hover": true, "tooltipDelay": 80}
    }
    """)

    by_node = nodes.set_index("node") if "node" in nodes.columns else nodes
    shapes = {"caught": "diamond", "missed": "diamond", "victim": "triangle",
              "unlabelled": "square"}

    for node in graph.nodes():
        row = by_node.loc[node] if node in by_node.index else None
        known = row is not None
        is_mule = known and int(row.get("is_mule", 0)) == 1
        score = float(row["risk_score"]) if known and pd.notna(row.get("risk_score", np.nan)) else np.nan
        flagged = bool(pd.notna(score) and score >= threshold)
        klass = _classify(known, is_mule, flagged, node in touching_mule)

        # Size carries the model's risk, which is the quantity a reviewer is
        # triaging on. v2 sized by a PageRank recomputed over the sample, which
        # measured the sample rather than the account.
        size = 10 + 30 * score if pd.notna(score) else 14

        net.add_node(
            node,
            label=node[:14],
            size=float(size),
            color=NODE_CLASSES[klass][0],
            shape=shapes.get(klass, "dot"),
            borderWidth=5 if node in seeds else 2,
            title=_node_tooltip(node, klass, row, threshold, graph, node in touching_mule),
        )

    for u, v, data in graph.edges(data=True):
        mule = int(data.get("is_mule", 0)) == 1
        weight = int(data.get("weight", 1))
        roles = str(data.get("roles", ""))
        if mule and "fan_in" in roles:
            colour, width = "#ffa502", 2.5      # money being drained INTO a hub
        elif mule:
            colour, width = "#ff4757", 3.0      # ring-internal movement
        else:
            colour, width = "#2f3b52", 1.0
        net.add_edge(
            u, v, color=colour,
            width=float(width + min(2.0, np.log1p(weight))),
            title=(
                f"{u} → {v}\n"
                f"{weight} transaction(s), total ₹{data.get('total_amount', 0):,.0f}\n"
                f"mean ₹{data.get('mean_amount', 0):,.0f}\n"
                f"role: {roles or 'unknown'}"
            ),
        )

    return net


def _render_html(net: Network) -> str:
    """
    PyVis output as a string, without leaking a temp file.

    v2 held a `NamedTemporaryFile(delete=False)` handle open and re-opened the
    same path, which fails on Windows and left the file behind on every rerun.
    `generate_html` avoids the disk entirely; the fallback writes into a
    directory that removes itself and is closed before it is read.
    """
    try:
        return net.generate_html(notebook=False)
    except (AttributeError, TypeError):  # older pyvis
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "graph.html"
            net.save_graph(str(path))
            return path.read_text(encoding="utf-8")


# ══════════════════════════════════════════════════════════════════
# Seed selection
# ══════════════════════════════════════════════════════════════════

def _ring_options(nodes: pd.DataFrame) -> dict[str, set[str]]:
    """Labelled rings in this split, as {label: member accounts}."""
    if "ring_id" not in nodes.columns:
        return {}
    mules = nodes[(nodes["is_mule"] == 1) & nodes["ring_id"].notna()]
    options: dict[str, set[str]] = {}
    for ring_id, group in mules.groupby("ring_id", sort=True):
        ring_type = str(group["ring_type"].iloc[0]) if "ring_type" in group else "?"
        label = f"{ring_id} — {ring_type}, {len(group)} mule account(s)"
        options[label] = set(group["node"].astype(str))
    return options


def _top_seeds(nodes: pd.DataFrame, mask: pd.Series, k: int, ascending: bool) -> set[str]:
    """The k most (or least) risky accounts satisfying `mask`."""
    subset = nodes.loc[mask & nodes["risk_score"].notna()]
    if subset.empty:
        return set()
    ordered = subset.sort_values("risk_score", ascending=ascending)
    return set(ordered["node"].astype(str).head(k))


# ══════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════

def render_graph_visualization(data: dict) -> None:
    """Main render function called from dashboard/app.py."""
    if not PYVIS_AVAILABLE:
        st.error(
            "PyVis is not installed, so the graph cannot be drawn. "
            "`pip install -r requirements.txt`."
        )
        return

    st.markdown("### Transaction network")
    st.markdown(
        "Ground truth sets the **shape**, the model's decision sets the "
        "**colour**. They are separate because a mule the model missed and a "
        "legitimate account it flagged are both failures, and neither is visible "
        "if one symbol has to mean both."
    )

    split = st.radio(
        "Split", ["test", "validation", "train"], index=0, horizontal=True,
        help="test is held out. Ring structure looks the same in all three; only "
             "the model's performance on it differs.",
        key="graph_split",
    )
    file_split = {"validation": "val"}.get(split, split)

    edges = data.get(f"{file_split}_edges")
    if edges is None:
        edges = _load_edges(file_split)
    if edges is None or len(edges) == 0:
        st.warning(
            f"No `data/raw/{file_split}_edges.csv`. Run `python -m data.generator`."
        )
        return
    edges = edges.copy()
    edges["sender"] = edges["sender"].astype(str)
    edges["receiver"] = edges["receiver"].astype(str)

    nodes, note = _node_table(file_split)
    if nodes is None:
        st.warning(note)
        return
    if note:
        st.info(note)
    nodes = nodes.copy()
    nodes["node"] = nodes["node"].astype(str)
    have_scores = bool(nodes["risk_score"].notna().any())

    threshold, threshold_source = _default_threshold()
    threshold = st.slider(
        "Decision threshold", 0.01, 0.99, float(np.clip(threshold, 0.01, 0.99)),
        step=0.005,
        help=f"Defaults to {threshold_source}. Accounts at or above this are flagged.",
        key="graph_threshold",
    )

    # ── The set that v2 mistook for the mules ─────────────────────────────
    mule_edges = edges[edges["is_mule"] == 1]
    touching_mule = set(mule_edges["sender"]) | set(mule_edges["receiver"])
    labelled_mules = set(nodes.loc[nodes["is_mule"] == 1, "node"])
    victims = touching_mule - labelled_mules

    # ── Selection ─────────────────────────────────────────────────────────
    rings = _ring_options(nodes)
    modes = ["A labelled ring"]
    if have_scores:
        modes += ["Top model alerts", "Worst false positives", "Missed mules"]
    modes += ["A specific account"]

    st.markdown("#### What to look at")
    mode = st.selectbox(
        "View", modes, index=0,
        help="A reviewer's question is never 'show me 150 accounts' — it is "
             "'show me why this one fired'.",
        key="graph_mode",
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        hops = st.slider("Hops around the selection", 1, 2, 1, key="graph_hops")
    with col2:
        max_nodes = st.slider("Node budget", 40, 400, 140, step=20, key="graph_max_nodes")
    with col3:
        seed_count = st.slider("Accounts to seed", 1, 10, 3, key="graph_seeds",
                               disabled=mode in ("A labelled ring", "A specific account"))

    seeds: set[str] = set()
    subtitle = ""
    if mode == "A labelled ring":
        if not rings:
            st.info("No labelled rings in this split.")
            return
        choice = st.selectbox("Ring", list(rings), key="graph_ring")
        seeds = rings[choice]
        subtitle = (
            "Every diamond is a labelled ring member. The pale blue triangles "
            "around it are the accounts feeding the ring — they are victims, and "
            "flagging one is a false positive."
        )
    elif mode == "Top model alerts":
        seeds = _top_seeds(nodes, pd.Series(True, index=nodes.index), seed_count, ascending=False)
        subtitle = "The accounts the model considers riskiest, whatever the truth turns out to be."
    elif mode == "Worst false positives":
        seeds = _top_seeds(nodes, nodes["is_mule"] == 0, seed_count, ascending=False)
        subtitle = (
            "The legitimate accounts the model scores highest. This is the "
            "false-positive cost the track's bar asks about, with faces on it — "
            "each one is a customer who would have been frozen or called."
        )
    elif mode == "Missed mules":
        seeds = _top_seeds(nodes, nodes["is_mule"] == 1, seed_count, ascending=True)
        subtitle = (
            "Labelled mules the model scores lowest. Look at what they lack: "
            "usually a closed cycle, which is the signal the detector leans on."
        )
    else:
        typed = st.text_input("Account id", key="graph_account").strip()
        if not typed:
            st.info("Enter an account id, e.g. one from the alerts table below.")
            return
        if typed not in set(nodes["node"]):
            st.warning(f"`{typed}` is not an account in the {split} split.")
            return
        seeds = {typed}
        subtitle = "The account and its counterparties."

    if not seeds:
        st.info("Nothing matched that selection.")
        return
    st.caption(subtitle)

    # ── Build ─────────────────────────────────────────────────────────────
    kept_edges, kept_nodes, trimmed = _khop(edges, seeds, hops, max_nodes)
    pairs = _pair_frame(kept_edges) if len(kept_edges) else pd.DataFrame(
        columns=["sender", "receiver", "weight", "total_amount",
                 "mean_amount", "is_mule", "roles"]
    )
    graph = _build_graph(pairs, kept_nodes)

    if trimmed:
        st.caption(
            f"Node budget reached: the view keeps the {len(kept_nodes)} accounts "
            f"most strongly connected to the selection. Edges to accounts outside "
            f"the view are not drawn, so degrees shown here are degrees WITHIN "
            f"this picture — the feature values in the tooltips are the real ones, "
            f"computed on the whole graph."
        )

    # ── Stats ─────────────────────────────────────────────────────────────
    view_nodes = nodes[nodes["node"].isin(kept_nodes)]
    in_view_mules = int((view_nodes["is_mule"] == 1).sum())
    if have_scores:
        flagged_mask = view_nodes["risk_score"] >= threshold
        in_view_flagged = int(flagged_mask.sum())
        in_view_fp = int((flagged_mask & (view_nodes["is_mule"] == 0)).sum())
        in_view_fn = int((~flagged_mask & (view_nodes["is_mule"] == 1)).sum())
    else:
        in_view_flagged = in_view_fp = in_view_fn = 0

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accounts shown", graph.number_of_nodes())
    col2.metric("Relationships", graph.number_of_edges(),
                help=f"{len(kept_edges):,} individual transactions collapsed into "
                     f"one edge per ordered pair.")
    col3.metric("Labelled mules in view", in_view_mules)
    col4.metric("Flagged in view", in_view_flagged,
                delta=f"{in_view_fp} false / {in_view_fn} missed",
                delta_color="inverse")

    # ── Legend, generated from the same dict that paints the nodes ────────
    legend = " &nbsp;|&nbsp; ".join(
        f"<span style='color:{colour}'>●</span> {label}"
        for _, (colour, label) in NODE_CLASSES.items()
    )
    st.markdown(
        f"<div style='font-size:0.82rem;line-height:1.8'>{legend}<br>"
        "◆ labelled mule &nbsp;|&nbsp; ▲ victim paying into a mule &nbsp;|&nbsp; "
        "● other &nbsp;|&nbsp; ■ unlabelled &nbsp;|&nbsp; thick border = the "
        "account you selected &nbsp;|&nbsp; node size = model risk"
        "</div>",
        unsafe_allow_html=True,
    )

    components.html(
        _render_html(_create_pyvis_network(
            graph, nodes, seeds, threshold, touching_mule
        )),
        height=640,
        scrolling=False,
    )

    # ── The v2 defect, stated with this split's own numbers ───────────────
    with st.expander(
        f"Why edge colour is not node colour — {len(victims):,} accounts in this "
        f"split transact with a mule but are not mules"
    ):
        st.markdown(
            f"`{file_split}_edges.csv` contains {len(mule_edges):,} mule "
            f"transactions, and {len(touching_mule):,} distinct accounts appear "
            f"on at least one of them. Only {len(labelled_mules):,} accounts "
            f"carry a node-level mule label. The remaining **{len(victims):,}** "
            f"are on the paying side of a fan-in edge: ordinary customers whose "
            f"money is moving into a collection hub.\n\n"
            f"The previous version of this page derived its red nodes from edge "
            f"incidence, so it marked all {len(touching_mule):,} as "
            f"\"⚠️ MULE SUSPECT\" — a "
            f"{len(victims) / max(len(touching_mule), 1):.0%} false-accusation "
            f"rate in the picture itself, on a submission whose bar is honest "
            f"false-positive cost. Node labels come from "
            f"`{file_split}_features.csv` now, and victims get a class of their "
            f"own that says what they are."
        )

    # ── Highest-risk accounts in view ─────────────────────────────────────
    if have_scores and len(view_nodes):
        table = view_nodes.sort_values("risk_score", ascending=False).head(12)
        display = pd.DataFrame({
            "Account": table["node"].to_numpy(),
            "Risk": [f"{v:.4f}" for v in table["risk_score"]],
            "Flagged": np.where(table["risk_score"] >= threshold, "yes", "no"),
            "Truth": np.where(table["is_mule"] == 1, "MULE", "legitimate"),
            "Ring": table.get("ring_type", pd.Series(["-"] * len(table))).fillna("-").to_numpy(),
            "cycle_participation": [
                f"{v:.3f}" for v in table.get("cycle_participation", pd.Series([np.nan] * len(table)))
            ],
        })
        st.markdown("#### Highest-risk accounts in this view")
        st.dataframe(display, use_container_width=True, hide_index=True)

    # ── Cycles, using the project's own definition ────────────────────────
    st.markdown("---")
    st.markdown("#### Circular flows in this view")
    st.caption(
        f"Same definition `cycle_participation` uses: relationships of at least "
        f"{MIN_REPEATS} transactions, cycle length {MIN_CYCLE_NODES}–"
        f"{MAX_CYCLE_LEN} accounts, over the pruned repeated-edge core from "
        f"`data/extractor.py`. A cycle built from one-off transfers is "
        f"coincidence, not laundering, which is why single transactions are "
        f"excluded here as well as in the feature."
    )

    succ, _pred = repeated_edge_cycle_core(graph, MIN_REPEATS)
    cycles, truncated = _bounded_cycles(succ)

    if not cycles:
        st.info(
            "No repeated-edge cycle of "
            f"{MIN_CYCLE_NODES}–{MAX_CYCLE_LEN} accounts in this view. For a "
            "labelled ring this usually means the node budget cut the loop, or "
            "the ring is an open chain that never closes — the feature scores "
            "those zero too, which is part of why recall is not 100%."
        )
    else:
        for i, cycle in enumerate(sorted(cycles, key=len, reverse=True), 1):
            mules_in = sum(1 for n in cycle if n in labelled_mules)
            shown = " → ".join(cycle[:7]) + (" → …" if len(cycle) > 7 else f" → {cycle[0]}")
            st.markdown(
                f"**{i}.** {len(cycle)} accounts, {mules_in} labelled mule(s): "
                f"`{shown}`"
            )
        if truncated:
            st.caption(
                f"Stopped after {len(cycles)} cycles (cap {MAX_CYCLES_SHOWN}, "
                f"step budget {CYCLE_STEP_BUDGET:,}). There may be more. The cap "
                f"exists because enumerating every simple cycle is unbounded in "
                f"output size — the previous version called "
                f"`nx.simple_cycles(G)` with no length limit and would hang on a "
                f"dense selection."
            )
