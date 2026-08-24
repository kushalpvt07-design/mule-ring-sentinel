"""
models/report.py
────────────────
Renders the README's Results section from models/saved_models/metrics.json.

WHY THIS EXISTS
───────────────
The most expensive bug in this repo's history was a *published number that
described a model which no longer existed*: v2 shipped a metrics.json claiming
ROC-AUC 0.9999 for a retired booster, and the README quoted figures nobody had
re-derived. Hand-typing metrics into a README recreates exactly that failure one
level up — the file drifts silently, and a stale README is a dishonest one.

So the README does not contain hand-typed metrics. It contains a delimited block

    <!-- METRICS:BEGIN -->  ...  <!-- METRICS:END -->

and this module regenerates that block from the metrics.json that the training
run actually wrote. If the numbers in the README are wrong, the fix is to re-run
training and re-run this — never to edit the prose.

The module deliberately imports only the standard library plus models.features,
so it runs on a bare checkout with no xgboost, no sklearn and no pandas. A
reporting tool that needs the training stack installed is a reporting tool that
does not get run.

Usage:
    python -m models.report            # print the block to stdout
    python -m models.report --write     # splice it into README.md in place
    python -m models.report --check     # exit 1 if README.md is out of date
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from models.features import FEATURE_COLS, MODEL_VERSION

ROOT = Path(__file__).resolve().parent.parent
METRICS_PATH = ROOT / "models" / "saved_models" / "metrics.json"
README_PATH = ROOT / "README.md"

BEGIN = "<!-- METRICS:BEGIN -->"
END = "<!-- METRICS:END -->"

PLACEHOLDER = f"""{BEGIN}
> **No metrics published yet.** `models/saved_models/metrics.json` is absent, so
> this section is intentionally empty rather than filled with numbers nobody
> measured. Run `python -m models.train` and then
> `python -m models.report --write`.
{END}"""


# ══════════════════════════════════════════════════════════════════
# Formatting helpers
# ══════════════════════════════════════════════════════════════════

def _rupees(value: float | None) -> str:
    """Indian-grouped rupees. ₹1234567 → ₹12,34,567."""
    if value is None:
        return "—"
    whole = f"{abs(round(float(value))):,}"          # 1,234,567
    digits = whole.replace(",", "")
    if len(digits) > 3:                              # regroup as 12,34,567
        head, tail = digits[:-3], digits[-3:]
        parts = []
        while len(head) > 2:
            parts.insert(0, head[-2:])
            head = head[:-2]
        if head:
            parts.insert(0, head)
        whole = ",".join(parts + [tail])
    return ("−" if float(value) < 0 else "") + "₹" + whole


def _pct(value: float | None, places: int = 1) -> str:
    return "—" if value is None else f"{100.0 * float(value):.{places}f}%"


def _num(value: float | None, places: int = 4) -> str:
    return "—" if value is None else f"{float(value):.{places}f}"


def _row(label: str, block: dict | None, *, bold: bool = False) -> str:
    """One line of the baseline comparison table."""
    name = f"**{label}**" if bold else label
    if not block or "test_f1" not in block:
        return f"| {name} | _skipped_ | | | | |"
    cells = [
        _pct(block.get("test_precision")),
        _pct(block.get("test_recall")),
        _num(block.get("test_f1"), 3),
        f"{block.get('test_alerts_per_1000', float('nan')):.1f}",
        _rupees(block.get("test_total_cost")),
    ]
    if bold:
        cells = [f"**{c}**" for c in cells]
    return f"| {name} | " + " | ".join(cells) + " |"


# ══════════════════════════════════════════════════════════════════
# The block
# ══════════════════════════════════════════════════════════════════

def render(metrics: dict) -> str:
    """Build the markdown Results block from a metrics.json payload."""
    lines: list[str] = [BEGIN, ""]

    version = metrics.get("model_version")
    stale = version != MODEL_VERSION
    if stale:
        lines += [
            f"> ⚠️ **These numbers describe `{version}`, but the code is at "
            f"`{MODEL_VERSION}`.** Re-run `python -m models.train`.",
            "",
        ]

    cost = metrics.get("cost_config", {})
    dataset = metrics.get("dataset", {})
    baselines = metrics.get("baselines", {})
    trivial = baselines.get("trivial", {})
    rule = baselines.get("best_single_feature_rule_by_cost", {})
    strongest = rule.get("strongest_single_feature_test_auc", {})

    # ── headline ──
    lines += [
        f"Held-out **test** split, threshold selected on **validation** "
        f"(never on test). Model `{version}`, "
        f"{metrics.get('n_features', len(FEATURE_COLS))} features, trained "
        f"{metrics.get('trained_at', 'unknown')}.",
        "",
        "| | |",
        "|---|---|",
        f"| ROC-AUC | {_num(metrics.get('roc_auc'))}"
        + (f" (95% CI {_num(metrics['roc_auc_ci_95'][0], 3)}–"
           f"{_num(metrics['roc_auc_ci_95'][1], 3)})"
           if isinstance(metrics.get("roc_auc_ci_95"), (list, tuple))
           and len(metrics["roc_auc_ci_95"]) == 2 else "") + " |",
        f"| Average precision | {_num(metrics.get('average_precision'))} |",
        f"| Precision | {_pct(metrics.get('precision'))} |",
        f"| Recall | {_pct(metrics.get('recall'))} |",
        f"| F1 | {_num(metrics.get('f1'), 3)} |",
        f"| Operating threshold | {_num(metrics.get('optimal_threshold'), 4)} |",
        f"| Total cost on test | {_rupees(metrics.get('total_cost'))} |",
        "",
    ]

    # ── the cost model, stated with its assumptions ──
    if cost:
        lines += [
            f"Costs are assumptions, not measurements: a missed mule is priced at "
            f"{_rupees(cost.get('fn_cost'))} and a false alert at "
            f"{_rupees(cost.get('fp_cost'))} — a ratio of "
            f"{cost.get('fn_fp_ratio', '—')} : 1. Only the ratio sets the "
            f"threshold. The break-even probability that follows from it is "
            f"**{_pct(cost.get('break_even_probability'), 2)}**, which is also the "
            f"break-even *precision* of the alert queue: any queue cleaner than "
            f"that is cheaper to work than to ignore.",
            "",
        ]

    # ── baselines: the honest-lift claim ──
    lines += [
        "### Does the model earn its complexity?",
        "",
        "Every row is priced on the same cost model, with each baseline's own "
        "threshold also selected on validation. This is the table that decides "
        "whether a graph pipeline was worth building.",
        "",
        "| | precision | recall | F1 | alerts/1k | total cost |",
        "|---|---|---|---|---|---|",
        _row("flag nothing", trivial.get("flag_nothing")),
        _row("flag everything", trivial.get("flag_everything")),
        _row(f"one-line rule: `{rule.get('rule', 'n/a')}`", rule),
        _row("logistic regression, same features",
             baselines.get("logistic_regression")),
        _row("XGBoost, no graph features",
             baselines.get("xgboost_without_graph_features")),
        _row("XGBoost, full model", {
            "test_precision": metrics.get("precision"),
            "test_recall": metrics.get("recall"),
            "test_f1": metrics.get("f1"),
            "test_alerts_per_1000": (metrics.get("test", {})
                                     .get("at_selected_threshold", {})
                                     .get("alerts_per_1000_accounts",
                                          float("nan"))),
            "test_total_cost": metrics.get("total_cost"),
        }, bold=True),
        "",
    ]

    gain = baselines.get("graph_feature_value", {})
    if gain:
        lines += [
            f"Graph features are worth "
            f"{_rupees(gain.get('cost_avoided_vs_no_graph'))} in avoided cost and "
            f"{gain.get('average_precision_gain', 0):+.4f} average precision "
            f"against the identical model trained without them.",
            "",
        ]

    # ── leakage ──
    if strongest:
        inverted = (" (inverted; raw "
                    f"{_num(strongest.get('raw_auc'), 4)})"
                    if strongest.get("inverted") else "")
        lines += [
            "### Is the task actually hard?",
            "",
            f"The strongest *single* feature on test is "
            f"`{strongest.get('feature', '—')}` at direction-corrected AUC "
            f"**{_num(strongest.get('auc'))}**{inverted}, against a leakage "
            f"ceiling of {strongest.get('ceiling', 0.99)}. If any one column "
            f"reached that ceiling the generator would be planting the label and "
            f"every number above would be theatre — `tests/test_baselines.py` "
            f"recomputes this from the shipped CSVs and fails the build if it "
            f"ever does.",
            "",
        ]

    # ── per-archetype recall ──
    # Lives at test.recall_breakdown, written by train.archetype_breakdown().
    breakdown = (metrics.get("test", {}) or {}).get("recall_breakdown") or {}
    by_account = breakdown.get("by_archetype_accounts") or {}
    by_ring = breakdown.get("by_archetype_rings") or {}
    if by_account:
        lines += [
            "### Recall by ring archetype",
            "",
            "Reported separately because one headline recall hides *which* "
            "laundering shapes the model misses. Ring recall counts a ring as "
            "caught if any member is flagged — the operationally relevant "
            "number, since one alert brings an analyst to the whole ring.",
            "",
            "| archetype | accounts | account recall | rings | ring recall |",
            "|---|---|---|---|---|",
        ]
        for name in sorted(by_account):
            acc = by_account[name] or {}
            ring = by_ring.get(name) or {}
            lines.append(
                f"| `{name}` "
                f"| {acc.get('detected', '—')}/{acc.get('n_accounts', '—')} "
                f"| {_pct(acc.get('account_recall'))} "
                f"| {ring.get('rings_with_an_alert', '—')}/"
                f"{ring.get('n_rings', '—')} "
                f"| {_pct(ring.get('ring_recall'))} |")
        if breakdown.get("overall_ring_recall") is not None:
            lines.append(
                f"| **all rings** | | "
                f"| **{breakdown.get('rings_with_an_alert', '—')}/"
                f"{breakdown.get('n_rings', '—')}** "
                f"| **{_pct(breakdown.get('overall_ring_recall'))}** |")
        lines.append("")

    # ── dataset shape ──
    if dataset:
        lines += [
            "### Data the numbers were measured on",
            "",
            "| split | accounts | mules | prevalence | rings |",
            "|---|---|---|---|---|",
        ]
        for split in ("train", "val", "test"):
            block = dataset.get(split) or {}
            if not isinstance(block, dict) or not block:
                continue
            lines.append(
                f"| {split} | {block.get('n_accounts', '—')} "
                f"| {block.get('n_positives', '—')} "
                f"| {_pct(block.get('prevalence'), 2)} "
                f"| {block.get('n_rings', '—')} |")
        lines += [
            "",
            "Splits are consecutive equal-length time windows — no account's "
            "future is used to predict its past, and no ring appears in two "
            "splits.",
            "",
        ]

    lines += [
        "_Generated from `models/saved_models/metrics.json` by "
        "`python -m models.report --write`. Do not hand-edit._",
        END,
    ]
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════
# Splice
# ══════════════════════════════════════════════════════════════════

def splice(readme_text: str, block: str) -> str:
    """Replace the delimited block, leaving every other byte untouched."""
    start = readme_text.find(BEGIN)
    end = readme_text.find(END)
    if start == -1 or end == -1:
        raise SystemExit(
            f"README.md has no {BEGIN} / {END} markers — nothing to fill.\n"
            "  The Results section is machine-generated; restore the markers "
            "rather than pasting numbers by hand."
        )
    if end < start:
        raise SystemExit(f"README.md has {END} before {BEGIN}.")
    return readme_text[:start] + block + readme_text[end + len(END):]


def load_metrics() -> dict | None:
    if not METRICS_PATH.exists():
        return None
    try:
        return json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"metrics.json is not valid JSON: {exc}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true",
                        help="splice the block into README.md in place")
    parser.add_argument("--check", action="store_true",
                        help="exit 1 if README.md does not match metrics.json")
    args = parser.parse_args()

    metrics = load_metrics()
    block = PLACEHOLDER if metrics is None else render(metrics)

    if metrics is None and (args.write or args.check):
        print(f"no metrics.json at {METRICS_PATH.relative_to(ROOT)} — "
              f"run `python -m models.train` first", file=sys.stderr)

    if not (args.write or args.check):
        print(block)
        return 0

    current = README_PATH.read_text(encoding="utf-8")
    updated = splice(current, block)

    if args.check:
        if current == updated:
            print("README.md Results section is up to date.")
            return 0
        print("README.md Results section is STALE — run "
              "`python -m models.report --write`.", file=sys.stderr)
        return 1

    if current == updated:
        print("README.md already up to date.")
        return 0
    README_PATH.write_text(updated, encoding="utf-8")
    print(f"README.md Results section updated from "
          f"{METRICS_PATH.relative_to(ROOT)}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
