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

ABSENT = "—"

# WHY A SHARED ABSENCE PREDICATE
# ──────────────────────────────
# A metrics.json can be *partially* written: a run that died between blocks, a
# schema that grew a key this code does not populate yet, a baseline that was
# skipped because scikit-learn was missing. Those arrive here in two different
# shapes — a MISSING key, which hits `.get()`'s default, and a key present with
# an explicit `null`, which does not. The second shape used to reach a raw
# f-string format spec (`f"{None:.1f}"`) and raise TypeError, so half a
# metrics.json crashed the reporter instead of rendering "no data" in one cell.
# A reporting tool that dies on an incomplete file is a reporting tool that
# stops being run, and this module's entire premise is that it always gets run.
#
# NaN counts as absent too, because `.get(key, float("nan"))` was the idiom used
# for missing floats and "nan" is not a number a README should publish.
def _absent(value: object) -> bool:
    """True for None and for NaN — the two ways "no measurement" arrives."""
    if value is None:
        return True
    return isinstance(value, float) and value != value      # NaN != NaN


def _plain(value: object) -> str:
    """A value quoted verbatim (a ratio, a count) rather than formatted."""
    return ABSENT if _absent(value) else str(value)


def _rupees(value: float | None) -> str:
    """Indian-grouped rupees. ₹1234567 → ₹12,34,567."""
    if _absent(value):
        return ABSENT
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
    return ABSENT if _absent(value) else f"{100.0 * float(value):.{places}f}%"


def _num(value: float | None, places: int = 4, *, signed: bool = False) -> str:
    if _absent(value):
        return ABSENT
    return f"{float(value):{'+' if signed else ''}.{places}f}"


# Small counts read better as words in prose, and deriving the word beats typing
# it: "the three archetypes" was a literal that would have gone stale the day a
# fourth archetype was added to the generator.
_COUNT_WORDS = ("no", "one", "two", "three", "four", "five", "six", "seven",
                "eight", "nine", "ten")


def _count_word(n: int) -> str:
    return _COUNT_WORDS[n] if 0 <= n < len(_COUNT_WORDS) else str(n)


# WHY DIRECTIONS ARE COMPUTED AND NEVER TYPED
# ───────────────────────────────────────────
# Every "rose" / "fell" / "fewer" / "cheaper" in this block is a *measurement of
# one training run*, not a property of the project. Typed as a literal it is
# correct on the day it is written and silently wrong afterwards — the same
# failure as a hand-typed number, except a reader cannot even spot it by
# comparing against metrics.json. So each one is derived here.
#
# The flat band matters. Directions are computed on raw floats but read next to
# values rendered to one decimal place of a percentage, so a difference smaller
# than half a displayed step would print "recall fell" beside two identical
# percentages. `flat_tol` defaults to that half-step.
def _which_way(before: float | None, after: float | None,
               *, up: str, down: str, flat: str,
               flat_tol: float = 5e-4) -> str | None:
    """The phrase for how `before` → `after` moved, or None if unmeasured."""
    if _absent(before) or _absent(after):
        return None
    delta = float(after) - float(before)
    if abs(delta) <= flat_tol:
        return flat
    return up if delta > 0 else down


def _moved(name: str, before: float | None, after: float | None,
           *, up: str = "rose", down: str = "fell", flat: str = "held",
           flat_tol: float = 5e-4) -> str | None:
    """'recall fell' / 'precision rose' / 'recall held', or None if unmeasured."""
    way = _which_way(before, after, up=up, down=down, flat=flat,
                     flat_tol=flat_tol)
    return None if way is None else f"{name} {way}"


def _row(label: str, block: dict | None, *, bold: bool = False) -> str:
    """One line of the baseline comparison table."""
    name = f"**{label}**" if bold else label
    if not block or "test_f1" not in block:
        return f"| {name} | _skipped_ | | | | |"
    cells = [
        _pct(block.get("test_precision")),
        _pct(block.get("test_recall")),
        _num(block.get("test_f1"), 3),
        # Via _num rather than a raw f-string: `f"{None:.1f}"` is a TypeError,
        # and a baseline whose alerts-per-1000 was never recorded must cost one
        # cell of the table, not the whole report.
        _num(block.get("test_alerts_per_1000"), 1),
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
    validation = metrics.get("validation", {}) or {}
    test = metrics.get("test", {}) or {}
    val_opt = validation.get("cost_optimal") or {}
    at_thr = test.get("at_selected_threshold") or {}
    oracle = test.get("oracle_threshold_diagnostic") or {}

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
            f"{_plain(cost.get('fn_fp_ratio'))} : 1. Only the ratio sets the "
            f"threshold. The break-even probability that follows from it is "
            f"**{_pct(cost.get('break_even_probability'), 2)}**, which is also the "
            f"break-even *precision* of the alert queue: any queue cleaner than "
            f"that is cheaper to work than to ignore.",
            "",
        ]

    # ── what honest threshold selection actually cost ──
    # The single most self-incriminating number available, so it leads rather
    # than hides: a project that quietly selected on test would publish the
    # oracle cost instead and look better for it.
    if oracle and at_thr and val_opt:
        gap = oracle.get("cost_of_not_peeking_at_test")
        headline_cost = at_thr.get("total_cost")
        share = (f" — {_pct(gap / headline_cost, 0)} of the reported total — "
                 if gap and headline_cost else " ")
        # Which way the two figures moved from validation to test is a property
        # of this run, not of the method: a different split, a different seed or
        # a wider plateau moves either one either way. This used to read "recall
        # fell, precision rose" as a literal — true of the run that was on disk
        # when the sentence was typed, and a lie from the next retrain onward.
        moves = [m for m in (_moved("recall", val_opt.get("recall"),
                                    at_thr.get("recall")),
                             _moved("precision", val_opt.get("precision"),
                                    at_thr.get("precision"))) if m]
        transfer = (" — " + ", ".join(moves) + ", and neither was tuned to make "
                    "that happen"
                    if moves else
                    ", and neither figure was tuned to transfer well")
        # `gap` cannot be negative — the oracle minimises cost over test — but it
        # CAN be exactly zero when the validation optimum happens to also be the
        # test optimum. "cost X instead of X" and "the cheaper number" would then
        # assert a saving that does not exist, so the zero case gets its own
        # sentence rather than a misleading rendering of the general one.
        peeked_cheaper = not _absent(gap) and float(gap) > 0
        if peeked_cheaper:
            comparison = f" instead of {_rupees(headline_cost)}. "
            price = (f"**That {_rupees(gap)} difference{share}is the price of "
                     f"not peeking**, and it is published because the "
                     f"alternative — quietly reporting the cheaper number — is "
                     f"the exact failure this repo already shipped once.")
        else:
            comparison = (f" — the same {_rupees(headline_cost)} the "
                          f"validation-chosen threshold costs. ")
            price = ("**The price of not peeking was zero on this run**, which "
                     "is luck and not method: the validation optimum happened "
                     "to land on the test optimum too. It is stated because a "
                     "later run where the gap is not zero must not be able to "
                     "quietly report it as though it were.")
        lines += [
            "### What it cost to pick the threshold honestly",
            "",
            f"The operating threshold {_num(val_opt.get('threshold'), 4)} was "
            f"chosen on validation, where it scored recall "
            f"{_pct(val_opt.get('recall'))} at precision "
            f"{_pct(val_opt.get('precision'))}. Applied unchanged to test it "
            f"gives recall {_pct(at_thr.get('recall'))} at precision "
            f"{_pct(at_thr.get('precision'))}{transfer}.",
            "",
            f"Had the threshold been chosen *with test labels in hand* it would "
            f"have been {_num(oracle.get('threshold'), 4)} and cost "
            f"{_rupees(oracle.get('total_cost'))}{comparison}{price}",
            "",
        ]
        width = val_opt.get("plateau_width")
        if width is not None:
            verdict = ("wide, so the choice is robust" if width > 0.05 else
                       "**narrow**, so the training run warned in advance that "
                       "this threshold might be fitted to validation noise")
            lines += [
                f"The run flagged this risk before test was ever touched: the "
                f"validation cost plateau spans "
                f"{_num(val_opt.get('plateau_lo'), 4)}–"
                f"{_num(val_opt.get('plateau_hi'), 4)}, width "
                f"{_num(width, 4)} — {verdict}. Total cost is a step function of "
                f"the threshold, so a narrow plateau means the minimum was a "
                f"knife-edge rather than a basin, which is precisely when a "
                f"validation-selected cutoff transfers poorly.",
                "",
            ]

    # ── calibration: why the operating threshold is not p* ──
    calib = test.get("calibration") or {}
    if calib:
        mean_p = calib.get("mean_predicted_probability")
        prev = calib.get("actual_prevalence")
        # `scale_pos_weight` is *intended* to push scores up, but whether it did
        # on this run is a measurement — and the word has to agree with the two
        # numbers printed immediately after it, or the sentence contradicts its
        # own evidence one line later.
        if _absent(mean_p) or _absent(prev):
            skew = "off the base rate by an unrecorded amount"
        elif float(mean_p) > float(prev):
            skew = "out inflated"
        elif float(mean_p) < float(prev):
            skew = "out deflated"
        else:
            skew = "out level with the base rate"
        lines += [
            "### Calibration, and why the threshold is not the break-even p\\*",
            "",
            f"Break-even p\\* is the correct score cutoff only for a *calibrated* "
            f"model, and this one is deliberately not calibrated: "
            f"`scale_pos_weight` trades calibration away to fight "
            f"{_pct(prev, 1)} prevalence, and the scores come {skew}. Mean "
            f"predicted probability is "
            f"{_num(calib.get('mean_predicted_probability'), 4)} against an "
            f"actual rate of {_num(calib.get('actual_prevalence'), 4)}; Brier "
            f"score {_num(calib.get('brier_score'), 4)}, expected calibration "
            f"error {_num(calib.get('expected_calibration_error'), 4)}.",
            "",
        ]
        bins = calib.get("reliability") or []
        if bins:
            lines += [
                "| score bin | accounts | mean predicted | observed rate |",
                "|---|---|---|---|",
            ]
            for b in bins:
                lines.append(
                    f"| `{b.get('bin','—')}` | {b.get('n','—')} "
                    f"| {_num(b.get('mean_predicted'), 4)} "
                    f"| {_num(b.get('observed_rate'), 4)} |")
            # This sentence used to read "over-states risk in every bin below the
            # top one" — and it was ALREADY false against the metrics.json in the
            # repo, where all five bins over-state (the top bin predicts 0.9593
            # against an observed 0.9222). A hand-typed summary of a table
            # printed directly above it is the worst kind of stale claim: the
            # reader can see the contradiction and cannot tell which is live. So
            # the summary is now read off the same rows the table renders.
            def _side(b: dict) -> str | None:
                mp, obs = b.get("mean_predicted"), b.get("observed_rate")
                if _absent(mp) or _absent(obs):
                    return None
                if float(mp) > float(obs):
                    return "over"
                if float(mp) < float(obs):
                    return "under"
                return "on"

            sides = [(b.get("bin", ABSENT), _side(b)) for b in bins]
            measured = [(name, s) for name, s in sides if s is not None]
            over = [name for name, s in measured if s == "over"]
            under = [name for name, s in measured if s == "under"]

            def _bins(names: list[str]) -> str:
                return ", ".join(f"`{n}`" for n in names)

            if not measured:
                claim = ("The reliability table records no comparable bin, so "
                         "nothing is claimed about the direction of the error")
            elif not over and not under:
                # Every measured bin landed exactly on its observed rate. Rare,
                # and the one case where the shared consequent below — "so the
                # scores are not probabilities" — would contradict the table
                # printed directly above it. Without this branch the code fell
                # through to the under-states one and rendered "under-states risk
                # in no of five bins ()", nonsense prose attached to a false
                # claim, which is the same failure as a typed literal reached by
                # a different route.
                claim = (f"Every one of the {_count_word(len(measured))} bins "
                         f"lands exactly on its observed rate")
            elif len(over) == len(measured):
                claim = ("The model over-states risk in **every** bin, the top "
                         "one included")
            elif len(under) == len(measured):
                claim = ("The model **under**-states risk in every bin, which is "
                         "the more dangerous direction: a score below the true "
                         "rate hides mules rather than wasting analyst time")
            elif over and under:
                claim = (f"The model over-states risk in "
                         f"{_count_word(len(over))} of "
                         f"{_count_word(len(measured))} bins ({_bins(over)}) and "
                         f"under-states it in {_count_word(len(under))} "
                         f"({_bins(under)})")
            elif over:
                claim = (f"The model over-states risk in "
                         f"{_count_word(len(over))} of "
                         f"{_count_word(len(measured))} bins ({_bins(over)}) and "
                         f"lands exactly on the observed rate in the rest")
            else:
                claim = (f"The model under-states risk in "
                         f"{_count_word(len(under))} of "
                         f"{_count_word(len(measured))} bins ({_bins(under)}) and "
                         f"lands exactly on the observed rate in the rest")
            # The consequent is a claim too. "so the scores are not
            # probabilities" is only supported when some bin actually missed its
            # observed rate, so it is derived from the same rows rather than
            # trailing the sentence unconditionally.
            if not measured:
                consequent = ("and nothing is claimed about whether they are "
                              "probabilities either")
            elif not over and not under:
                consequent = ("so on this run the binned scores are "
                              "indistinguishable from probabilities — which is a "
                              "measurement of one reliability table, not a "
                              "calibration guarantee")
            else:
                consequent = "so the scores are not probabilities"
            lines += [
                "",
                f"{claim}, {consequent}. That is why "
                f"p\\* is used here as a statement about the **alert queue** "
                f"— break-even precision — and never as a score cutoff. The "
                f"cutoff is the empirical validation optimum. Reporting p\\* as "
                f"though it were the operating threshold would be "
                f"arithmetically tidy and wrong.",
                "",
            ]

    # ── how far off the probabilities are, and proof the fix was not applied ──
    # The section above says the scores are not probabilities. This one says by
    # HOW MUCH, which is what turns a conceded limitation into a measured one.
    #
    # The load-bearing row is ROC-AUC. A calibrator printed in a report is an
    # invitation to wire it in, and the shipped threshold was selected on the RAW
    # scale — rescale the scores underneath it and every operating point in this
    # README moves without a word appearing anywhere. Publishing both AUCs makes
    # "it cannot re-rank the queue" checkable by a reader instead of asserted by
    # the author, and tests/test_baselines.py asserts it too.
    platt = metrics.get("probability_calibration") or {}
    scaler = platt.get("scaler") or {}
    invariance = platt.get("ranking_invariance") or {}
    raw_side = platt.get("test_raw") or {}
    cal_side = platt.get("test_calibrated") or {}
    if scaler and invariance:
        slope = scaler.get("slope")
        # Over- versus under-confident is a measurement of this run's slope, not
        # a property of `scale_pos_weight`, so the word is derived. The flat band
        # is a hair either side of 1.0, where both "sharpens" and "shrinks" would
        # overstate a difference the fourth decimal place cannot support.
        if _absent(slope):
            spread = "moved the spread of the scores by an unrecorded amount"
        elif abs(float(slope) - 1.0) <= 5e-3:
            spread = ("left the spread essentially untouched, so on this run the "
                      "scores were already about as sharp as the evidence "
                      "supports")
        elif float(slope) < 1.0:
            spread = ("shrinks the log-odds — the signature of "
                      "**over**-confident scores, spread wider than the evidence "
                      "supports, which is the expected effect of "
                      "`scale_pos_weight`")
        else:
            spread = ("sharpens the log-odds, meaning the raw scores were "
                      "**under**-confident")
        positives = scaler.get("n_positive_fit")
        lines += [
            "### How far off the probabilities are",
            "",
            f"A two-parameter Platt map (a logistic fit on the score log-odds) "
            f"was fitted on **{_plain(platt.get('fitted_on'))}** and measured on "
            f"**{_plain(platt.get('measured_on'))}**, the same split discipline "
            f"every threshold here follows. Slope "
            f"{_num(slope, 4)}, intercept "
            f"{_num(scaler.get('intercept'), 4, signed=True)}: it {spread}."
            + ("" if _absent(positives) else
               f" Two parameters rather than an isotonic fit because the "
               f"validation split carries only {_plain(positives)} positives, "
               f"and a free-form monotone fit on that many points memorises "
               f"them.")
            + ("" if scaler.get("converged", True) else
               " **The fit did not converge**, so the numbers below describe an "
               "unfinished optimisation and should not be quoted."),
            "",
            "| | raw | calibrated | change |",
            "|---|---|---|---|",
        ]
        for label, key, places in (
            ("Brier score", "brier_score", 5),
            ("Expected calibration error", "expected_calibration_error", 5),
        ):
            before, after = raw_side.get(key), cal_side.get(key)
            delta = (ABSENT if _absent(before) or _absent(after)
                     else _num(float(after) - float(before), places, signed=True))
            lines.append(f"| {label} | {_num(before, places)} "
                         f"| {_num(after, places)} | {delta} |")
        auc_raw = invariance.get("test_auc_raw")
        auc_cal = invariance.get("test_auc_calibrated")
        auc_delta = (ABSENT if _absent(auc_raw) or _absent(auc_cal)
                     else _num(float(auc_cal) - float(auc_raw), 6, signed=True))
        lines += [
            f"| ROC-AUC | {_num(auc_raw, 5)} | {_num(auc_cal, 5)} "
            f"| {auc_delta} |",
            "",
        ]
        # The verdict on the AUC row is derived, because it is the one row whose
        # meaning inverts: a delta at the tolerance is the guarantee holding, and
        # a delta above it is a silent change to the model. A fixed sentence here
        # would keep reassuring the reader on the day it stopped being true.
        delta_value = invariance.get("abs_delta")
        tolerance = invariance.get("tolerance")
        if _absent(delta_value) or _absent(tolerance):
            verdict = ("The AUC comparison was not recorded, so nothing is "
                       "claimed here about whether the map preserves the "
                       "ranking.")
        elif float(delta_value) <= float(tolerance):
            # An exact zero is the common case and "moves by 0.0e+00" reads as a
            # rounding artefact rather than the clean result it is, so the two
            # are worded apart.
            moved = ("does not move at all" if float(delta_value) == 0.0 else
                     f"moves by {float(delta_value):.1e}, inside the "
                     f"{float(tolerance):.0e} float-summation tolerance")
            verdict = (
                f"ROC-AUC {moved} — the map is monotone, so it can compress two "
                f"adjacent scores into a tie but cannot swap a pair. **The alert "
                f"queue is in the same order.** That is what makes this a "
                f"diagnostic and not an undisclosed model change: nothing above "
                f"or below this section is computed on the calibrated scale.")
        else:
            verdict = (
                f"⚠️ ROC-AUC moved by {float(delta_value):.2e}, **outside** the "
                f"{float(tolerance):.0e} tolerance. A calibrator that re-ranks "
                f"accounts is not a calibrator, and this block should not be "
                f"trusted until that is explained.")
        lines += [
            verdict,
            "",
            "The map is reported and never applied. The operating threshold in "
            "the headline table was selected on the raw scores, so calibrating "
            "underneath it would move the published precision, recall and cost "
            "without changing a single number in this README. If a downstream "
            "consumer needs scores that read as probabilities — expected-loss "
            "arithmetic, or blending with another model's output — apply this "
            "map at that boundary and re-select the threshold on the calibrated "
            "scale.",
            "",
        ]

    # ── sensitivity to the one unverifiable assumption ──
    # Rendered only for the split-tagged schema. The legacy schema's numbers were
    # test-optimised, and re-publishing them would restate the defect.
    sens = metrics.get("cost_ratio_sensitivity") or []
    if sens and all("val_threshold" in r for r in sens):
        lines += [
            "### Does the operating point survive a different cost ratio?",
            "",
            "The FN:FP ratio is the one number in the cost model nobody can "
            "verify, so a result quoted at a single ratio is not a result. Each "
            "row picks its threshold on validation and is then evaluated once on "
            "test at that frozen value — the same discipline as the headline.",
            "",
            "| FN:FP | break-even p\\* | threshold (val) | test precision "
            "| test recall | test cost |",
            "|---|---|---|---|---|---|",
        ]
        for r in sens:
            lines.append(
                f"| {r.get('fn_fp_ratio')} : 1 "
                f"| {_pct(r.get('break_even_p'), 2)} "
                f"| {_num(r.get('val_threshold'), 4)} "
                f"| {_pct(r.get('test_precision'))} "
                f"| {_pct(r.get('test_recall'))} "
                f"| {_rupees(r.get('test_total_cost'))} |")
        lines.append("")

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
                                     .get("alerts_per_1000_accounts")),
            "test_total_cost": metrics.get("total_cost"),
        }, bold=True),
        "",
    ]

    gain = baselines.get("graph_feature_value", {})
    ablation = baselines.get("xgboost_without_graph_features", {})
    if gain:
        avoided = gain.get("cost_avoided_vs_no_graph")
        # "are worth X in avoided cost" asserted the sign. A negative
        # cost_avoided_vs_no_graph — the graph pipeline being a net loss on test,
        # which tests/test_baselines.py exists precisely to catch — would have
        # rendered as "worth −₹4,90,000 in avoided cost", a defeat phrased as a
        # win. The verb is derived from the sign now.
        if _absent(avoided):
            cost_phrase = "change total cost by an amount this run did not record"
        elif float(avoided) > 0:
            cost_phrase = f"avoid {_rupees(avoided)} of cost"
        elif float(avoided) < 0:
            cost_phrase = f"**add** {_rupees(abs(float(avoided)))} of cost"
        else:
            cost_phrase = "leave total cost exactly unchanged"
        lines += [
            f"Against the identical model trained without them, graph features "
            f"{cost_phrase} and move average precision by "
            f"{_num(gain.get('average_precision_gain'), 4, signed=True)}.",
            "",
        ]
        # Total cost is nearly blind to precision at a 13:1 FN:FP ratio, and
        # precision is exactly where the graph features land. Reporting only the
        # cost delta makes them look pointless; reporting only the precision
        # delta oversells them. Both, then.
        test_block = dataset.get("test") or {}
        n = test_block.get("n_accounts")
        n_pos = test_block.get("n_positives")
        fp_full = at_thr.get("fp")
        if (n and n_pos and fp_full is not None
                and ablation.get("test_recall") is not None
                and ablation.get("test_alerts_per_1000") is not None):
            alerts_abl = round(ablation["test_alerts_per_1000"] / 1000.0 * n)
            fp_abl = alerts_abl - round(ablation["test_recall"] * n_pos)
            # fp_abl is RECONSTRUCTED from a rounded alerts-per-1000 and a rounded
            # recall, so on a queue that is almost all true positives it can come
            # out slightly negative. A negative false-positive count is not a
            # finding, it is arithmetic noise, and nothing honest can be built on
            # it — hence >= 0 rather than > 0, with the zero case handled below
            # instead of silently dropping the paragraph.
            if fp_abl >= 0:
                # THE DEFECT THIS GUARD REPLACES: the old condition was
                # `fp_abl > 0`, which only proved the denominator was safe — not
                # that false positives had gone DOWN. An ablation that beat the
                # full model on precision (fp_abl < fp_full) still rendered
                # "False positives fall from 12 to 20 (−67% fewer)": a precision
                # regression published as the section's headline argument. The
                # sign is now measured and the sentence follows it.
                if fp_abl > fp_full:
                    fp_clause = (f"False positives fall from {fp_abl} to "
                                 f"{fp_full} ({_pct(1 - fp_full / fp_abl, 0)} "
                                 f"fewer)")
                    lead = ("That cost figure understates them, and the reason "
                            "is worth stating")
                elif fp_abl < fp_full:
                    more = (f" ({_pct(fp_full / fp_abl - 1, 0)} more)"
                            if fp_abl else "")
                    fp_clause = (f"False positives **rise** from {fp_abl} to "
                                 f"{fp_full}{more}")
                    lead = ("That cost figure flatters them, and the reason is "
                            "worth stating")
                else:
                    fp_clause = (f"False positives are unchanged at {fp_full}")
                    lead = ("The cost figure is the whole of the effect, and the "
                            "reason is worth stating")
                # "at essentially unchanged recall" was the second literal here:
                # the precision argument only stands if recall did not pay for
                # it. The flat band is one positive account — below that the
                # difference is the rounding in the stored recall, not a mule.
                r_abl, r_full = ablation.get("test_recall"), at_thr.get("recall")
                recall_way = _which_way(
                    r_abl, r_full,
                    up="at higher recall", down="at lower recall",
                    flat="at unchanged recall",
                    flat_tol=1.0 / (2.0 * float(n_pos)))
                recall_clause = (
                    "" if recall_way is None else
                    f" {recall_way} ({_pct(r_abl)} → {_pct(r_full)})")
                # And the third: "shrinking the review queue" asserted the queue
                # got smaller, which is the same claim as the FP direction but
                # could disagree with it once recall moves too.
                q_abl = ablation.get("test_alerts_per_1000")
                q_full = at_thr.get("alerts_per_1000_accounts")
                q_way = _which_way(q_abl, q_full, up="growing",
                                   down="shrinking", flat="leaving",
                                   flat_tol=0.05)
                if q_way is None:
                    queue_clause = ""
                elif q_way == "leaving":
                    queue_clause = (f", leaving the review queue at "
                                    f"{_num(q_full, 1)} alerts per 1,000 "
                                    f"accounts")
                else:
                    queue_clause = (f", {q_way} the review queue from "
                                    f"{_num(q_abl, 1)} to {_num(q_full, 1)} "
                                    f"alerts per 1,000 accounts")
                lines += [
                    f"{lead}: with a miss priced at "
                    f"{_plain(cost.get('fn_fp_ratio'))}× a false alert, total "
                    f"cost is nearly insensitive to precision — and precision is "
                    f"exactly where the graph features land. {fp_clause}"
                    f"{recall_clause}{queue_clause}. An analyst feels that; the "
                    f"cost model barely does.",
                    "",
                ]
        if ablation.get("test_auc") is not None:
            # "most of the separating signal is reachable without a graph" is a
            # comparison, and it was typed rather than measured. The honest
            # yardstick is the share of ABOVE-CHANCE AUC the ablation retains —
            # comparing 0.9786 to 0.9837 directly makes almost any pair of good
            # models look identical, because both are dominated by the 0.5 floor.
            abl_auc = ablation.get("test_auc")
            full_auc = metrics.get("roc_auc")
            retained = None
            if not _absent(abl_auc) and not _absent(full_auc) \
                    and float(full_auc) > 0.5:
                retained = (float(abl_auc) - 0.5) / (float(full_auc) - 0.5)
            preamble = (f"{_plain(ablation.get('n_features'))} non-structural "
                        f"features reach test AUC {_num(abl_auc)} on their own, "
                        f"against {_num(full_auc)} with the full set")
            if retained is None:
                lines += [
                    f"{preamble} — but with one of the two AUCs at or below "
                    f"chance, no share-of-signal claim can be made from this "
                    f"pair, so none is made.",
                    "",
                ]
            elif retained >= 0.9:
                lines += [
                    f"Read the other way, this ablation is also the strongest "
                    f"argument against the framing at the top of this README. "
                    f"{preamble} — {_pct(retained, 1)} of the above-chance "
                    f"separation, so most of the separating signal is reachable "
                    f"without a graph at all. The structural features sharpen "
                    f"the queue; they do not find a fundamentally different "
                    f"population of mules. Anyone deciding whether to build this "
                    f"pipeline should weigh that.",
                    "",
                ]
            elif retained >= 0.5:
                lines += [
                    f"{preamble} — {_pct(retained, 1)} of the above-chance "
                    f"separation. So the non-structural features carry the bulk "
                    f"of the ranking, but the graph closes a real part of the "
                    f"remaining gap rather than only sharpening the queue.",
                    "",
                ]
            else:
                lines += [
                    f"{preamble} — only {_pct(retained, 1)} of the above-chance "
                    f"separation. On this run the structural features carry most "
                    f"of the separating signal, which is the opposite of the "
                    f"caveat the top of this README raises; the caveat, not this "
                    f"paragraph, is the one to re-check.",
                    "",
                ]

    # ── the capacity-constrained operating point ──
    # The least flattering block in this report, which is exactly why it is not
    # optional. The baseline table above gives every policy an unlimited review
    # queue — a luxury no risk team has — and under a cap the ordering can
    # invert. Publishing only the framing that favours the model would be the
    # same class of selective reporting as picking a threshold on test.
    cap = test.get("at_capacity_threshold") or {}
    budget = cost.get("alert_budget_per_1000")
    # Gated on `budget` as well as on the numbers. Every sentence in this section
    # is ABOUT a stated cap — "n× the capped budget", "a queue the budget forbids
    # outright" — and with alert_budget_per_1000 absent the old code fell back to
    # `max(budget or 1, 1)`, dividing an alerts-per-1000 figure by 1 and
    # publishing "roughly 200× the capped budget" against a budget nobody had
    # stated. A section whose subject is missing does not get rendered with a
    # placeholder in the subject's place; it gets left out, and the reader is not
    # invited to reason about a cap that was never declared.
    if cap.get("total_cost") is not None and budget:
        lines += [
            "### What happens when the review queue is capped",
            "",
            f"Every row above assumes an analyst queue of unlimited size. Cap it "
            f"at {_num(budget, 0)} alerts per 1,000 accounts "
            f"— a stated assumption, with the threshold that satisfies it chosen "
            f"on validation like every other threshold here — and the model is "
            f"pushed to the high-precision end of its own curve: "
            f"threshold {_num(cap.get('threshold'), 4)}, precision "
            f"{_pct(cap.get('precision'))} at recall {_pct(cap.get('recall'))}, "
            f"{_num(cap.get('alerts_per_1000_accounts'), 1)} alerts per 1,000 "
            f"accounts, total test cost {_rupees(cap.get('total_cost'))}.",
            "",
        ]
        rule_cost = rule.get("test_total_cost")
        rule_alerts = rule.get("test_alerts_per_1000")
        # `budget` is known truthy here, so this is a real multiple rather than a
        # division by a fallback of 1.
        multiple = ("" if _absent(rule_alerts) else
                    f"roughly {float(rule_alerts) / float(budget):.0f}× the "
                    f"capped budget, so it is not a policy the cap permits. ")
        if rule_cost is not None and cap["total_cost"] > rule_cost:
            lines += [
                f"On cost alone that loses to the one-line rule "
                f"({_rupees(rule_cost)}), and the inversion is a property of the "
                f"cost model rather than a defect. The rule reaches recall "
                f"{_pct(rule.get('test_recall'))} by issuing "
                f"{_num(rule_alerts, 1)} alerts per 1,000 accounts — "
                f"{multiple}With "
                f"a miss priced at {_plain(cost.get('fn_fp_ratio'))}× a false "
                f"alert, "
                f"misses dominate the total, and any policy free to flood the "
                f"queue wins on cost by construction. Note the rule's queue is "
                f"not economically absurd either — at "
                f"{_pct(rule.get('test_precision'))} precision it sits above the "
                f"{_pct(cost.get('break_even_probability'), 2)} break-even, so it "
                f"is worth working if you have the staff. The objection to it is "
                f"capacity, not arithmetic.",
                "",
                "The defensible claim is therefore the joint one: cheaper than "
                "the alternatives *and* a queue small enough for a human to "
                "actually work. Cost in isolation, under a cap the winning "
                "baseline could never satisfy, is not a claim this project "
                "makes. A team whose binding constraint is reviews-per-day "
                "should re-derive its own operating point from its own budget.",
                "",
            ]
        elif rule_cost is not None:
            # The clause about the rule's queue is CONDITIONAL on the rule's queue
            # actually breaching the cap. It used to read "a queue the budget
            # forbids outright" unconditionally, which is a claim about a number
            # printed two words earlier: on any run where the rule's uncapped
            # queue happened to fit, the README would have contradicted its own
            # table in the same sentence. The uncapped rule breaches the cap by a
            # wide margin on the shipped data, which is exactly why the sentence
            # survived unnoticed — it was right by luck, not by construction.
            if not _absent(rule_alerts) and float(rule_alerts) > float(budget):
                tail = (f", which the rule reaches only by issuing "
                        f"{_num(rule_alerts, 1)} alerts per 1,000 accounts — a "
                        f"queue the budget forbids outright.")
            elif not _absent(rule_alerts):
                tail = (f", at {_num(rule_alerts, 1)} alerts per 1,000 accounts "
                        f"against the same {_num(budget, 0)} cap — so on this run "
                        f"the rule is affordable and still more expensive.")
            else:
                tail = "."
            lines += [
                f"Even under the cap the model stays below the one-line rule "
                f"({_rupees(rule_cost)}){tail}",
                "",
            ]

    # ── the same cap, applied to everybody ──
    # The section above caps the MODEL and then compares it against baselines
    # still free to flood the queue. That is not a fair fight and the asymmetry
    # favours the baselines: at a 13:1 miss-to-false-alert ratio, misses dominate
    # the total, so any policy allowed unlimited alerts wins on cost by
    # construction. The paragraph above says so in words; this table is the
    # measurement, with the budget binding on every row.
    #
    # The uncapped table is NOT removed. Two framings of the same comparison,
    # both published, is the honest form — a reader who cares about cost alone and
    # a reader whose binding constraint is reviews-per-day are asking different
    # questions and both deserve an answer.
    fair = metrics.get("capacity_fair_comparison") or []
    if fair and budget:
        # A policy is only a candidate for the ⚠ overflow legend below if it HAD a
        # validation-selected threshold and that threshold bought a queue. An
        # infeasible row never had a queue to overflow, so counting it in the
        # legend's denominator would describe a risk it was never exposed to.
        scored = [r for r in fair
                  if not _absent(r.get("val_threshold"))
                  and r.get("feasible_under_budget") is not False]
        infeasible = [r for r in fair
                      if r.get("feasible_under_budget") is False
                      and not _absent(r.get("val_threshold"))]
        lines += [
            "#### The same cap, applied to every policy",
            "",
            f"Each threshold below was re-selected **on validation** to fit the "
            f"same {_num(budget, 0)} alerts per 1,000 accounts, then applied once "
            f"to test. The two trivial policies have no threshold to select and "
            f"are priced closed-form.",
            "",
            "| policy under the cap | val threshold | precision | recall | F1 "
            "| alerts / 1,000 (test) | total cost |",
            "|---|---|---|---|---|---|---|",
        ]
        for row in fair:
            name = str(row.get("policy", ABSENT))
            if row.get("is_model"):
                name = f"**{name}**"
            if row.get("feasible_under_budget") is False:
                # Not a footnote. Two different things land here and both mean
                # "this is not a policy you could run at this capacity":
                # flag-everything, which is 1,000 alerts per 1,000 by
                # construction; and a policy whose scores are too coarse to form
                # ANY non-empty queue that fits — a one-line rule on an
                # integer-valued feature has a large tie at the top, so its
                # strictest real cut can already exceed the cap. Both stay in the
                # table so a reader sees the excluded policy rather than
                # wondering whether it was quietly dropped, and neither is
                # eligible to be the rival the model beats.
                name += " _(infeasible)_"
            alerts = _num(row.get("test_alerts_per_1000"), 1)
            if (row.get("test_within_budget") is False
                    and not _absent(row.get("val_threshold"))):
                # A validation-frozen threshold CAN overflow on test: the score
                # distribution moved and the threshold did not. Marked rather
                # than re-solved — re-fitting the threshold until the constraint
                # held would be selection on test wearing a capacity argument as
                # a disguise.
                #
                # Gated on there BEING a validation threshold, because the legend
                # below describes exactly that case. Flag-everything is over the
                # cap by construction rather than by distribution shift, and it
                # already carries `(infeasible)`; marking it here too would put a
                # symbol on the page that its own legend does not explain.
                alerts += " ⚠"
            lines.append(
                f"| {name} | {_num(row.get('val_threshold'), 4)} "
                f"| {_pct(row.get('test_precision'))} "
                f"| {_pct(row.get('test_recall'))} "
                f"| {_num(row.get('test_f1'), 3)} "
                f"| {alerts} | {_rupees(row.get('test_total_cost'))} |")
        lines.append("")

        # An infeasible SCORED row needs saying out loud, because its cells look
        # like a policy's and are not. It reads as zero alerts at the cost of
        # every miss, which is what abstaining costs, and leaving that unexplained
        # invites a reader to score it as a rival the model crushed.
        for row in infeasible:
            floor = row.get("val_strictest_nonempty_alerts_per_1000")
            if _absent(floor):
                continue
            lines += [
                f"`{row.get('policy', ABSENT)}` has no feasible operating point "
                f"under this cap. Its scores tie: the strictest cut that flags "
                f"anything at all already raises {_num(floor, 1)} alerts per "
                f"1,000 accounts against a budget of {_num(budget, 0)}, so every "
                f"non-empty queue it can form is unaffordable and the only "
                f"affordable one is empty. The row below it is priced as an "
                f"abstention — no alerts, every miss paid for — and it is "
                f"excluded from the comparison rather than counted as a policy "
                f"the model beat.",
                "",
            ]

        # A row only counts as overflowing if its alert count is on record: the
        # flag and the number are two views of one measurement, and a row
        # carrying the flag without the number cannot be described (`max()` of an
        # empty sequence raises, which would take the whole report down over one
        # missing cell).
        overflowed = [r for r in scored
                      if r.get("test_within_budget") is False
                      and not _absent(r.get("test_alerts_per_1000"))]
        if overflowed:
            # "Slightly over" was the tempting word and it is a claim, not an
            # observation — on a run where the distributions moved further it
            # would be wrong beside a table the reader can check. The size of the
            # worst overflow is measured instead.
            worst = max(float(r["test_alerts_per_1000"]) for r in overflowed)
            lines += [
                f"⚠ marks a policy whose validation-selected threshold "
                f"overflowed the cap on test — {_count_word(len(overflowed))} of "
                f"{_count_word(len(scored))} here, the widest at "
                f"{_num(worst, 1)} alerts per 1,000 against a budget of "
                f"{_num(budget, 0)}. Each threshold was frozen on validation and "
                f"the test score distribution is not identical, so the queue "
                f"comes out a different size. It is reported rather than "
                f"corrected: re-solving the threshold on test until the "
                f"constraint held would be threshold selection on test, which is "
                f"the one thing this project refuses to do anywhere else.",
                "",
            ]

        # WHO WINS IS A MEASUREMENT, AND IT IS THE WHOLE POINT OF THE TABLE
        # ────────────────────────────────────────────────────────────────
        # Written as a fixed sentence this would be a claim about the project
        # rather than about this run — and it is precisely the claim most likely
        # to invert, because it is the one the capped comparison exists to test.
        # If the simpler policy wins here, the README says so.
        model_rows = [r for r in fair if r.get("is_model")
                      and not _absent(r.get("test_total_cost"))]
        rivals = [r for r in fair
                  if not r.get("is_model")
                  and r.get("feasible_under_budget") is not False
                  and not _absent(r.get("val_threshold"))
                  and not _absent(r.get("test_total_cost"))]
        if model_rows and rivals:
            model_row = model_rows[0]
            best = min(rivals, key=lambda r: float(r["test_total_cost"]))
            model_cost = float(model_row["test_total_cost"])
            best_cost = float(best["test_total_cost"])
            if model_cost <= best_cost:
                lines += [
                    f"With the cap binding on everyone, the model is the "
                    f"cheapest policy in the table at "
                    f"{_rupees(model_cost)} against {_rupees(best_cost)} for the "
                    f"next best ({best.get('policy', ABSENT)}) — "
                    f"{_rupees(best_cost - model_cost)} cheaper on the same "
                    f"queue size. That is the comparison the capped claim above "
                    f"rests on. The one-line rule's apparent win on cost alone "
                    f"was bought with a queue the cap does not permit, not with "
                    f"better detection.",
                    "",
                ]
            else:
                lines += [
                    f"⚠️ **Under the same cap the model is not the cheapest "
                    f"policy.** It costs {_rupees(model_cost)} against "
                    f"{_rupees(best_cost)} for {best.get('policy', ABSENT)}, a "
                    f"difference of {_rupees(model_cost - best_cost)} on an "
                    f"identical queue size. This is not an artefact of an unfair "
                    f"comparison — the budget binds on both rows — so the honest "
                    f"reading is that at this queue size the simpler policy is "
                    f"the better buy, and the graph pipeline earns its keep only "
                    f"on the other columns.",
                    "",
                ]

    # ── the same operating point at a realistic base rate ──
    # The generator elevates mule prevalence so the model has enough positive
    # signal to learn from, which makes every precision and rupee figure above a
    # measurement at a base rate no payment network has. Quoting them without the
    # correction invites a reviewer to find the problem for us. This table answers
    # it in advance, and it needs no retraining: TPR and FPR are within-class
    # rates, so only the mix changes — see cost_matrix.precision_at_prevalence
    # for the one assumption that buys.
    projection = metrics.get("prevalence_projection") or []
    if projection:
        observed = next((r for r in projection if r.get("is_observed")), None)
        lines += [
            "### The same operating point at a realistic base rate",
            "",
            "Precision and rupee cost depend on how many mules there are; "
            "recall, ROC-AUC and the leakage headroom do not. So the honest way "
            "to read the numbers above is to re-price them at base rates a real "
            "network would see. Nothing is retrained here — the class-conditional "
            "score distributions are held fixed and only the mix is re-weighted.",
            "",
            "| mule base rate | recall | precision | alerts / 1,000 | cost / "
            "1,000 accounts | queue pays for itself |",
            "|---|---|---|---|---|---|",
        ]
        for row in projection:
            rate = _pct(row.get("prevalence"), 3)
            if row.get("is_observed"):
                rate = f"**{rate}** _(measured)_"
            clears = row.get("clears_break_even")
            verdict = (ABSENT if clears is None
                       else ("yes" if clears else "**no**"))
            lines.append(
                f"| {rate} | {_pct(row.get('recall'))} "
                f"| {_pct(row.get('projected_precision'), 2)} "
                f"| {_num(row.get('projected_alerts_per_1000'), 1)} "
                f"| {_rupees(row.get('projected_cost_per_1000_accounts'))} "
                f"| {verdict} |")
        lines.append("")

        # The recall column being constant is the pedagogical point, and stating
        # it from the rendered rows rather than as prose means the sentence cannot
        # outlive the table. A drifting recall column would mean the projection
        # re-derived recall from projected counts and smuggled in an improvement
        # base rate cannot buy.
        recalls = {round(float(r["recall"]), 9) for r in projection
                   if not _absent(r.get("recall"))}
        if len(recalls) == 1:
            fixed_note = (
                f"Recall holds at {_pct(recalls.pop())} down the whole table, "
                f"which is not an approximation: it is a within-class rate, so "
                f"no change of base rate can touch it.")
        else:
            fixed_note = (
                f"⚠️ Recall varies across rows ({_count_word(len(recalls))} "
                f"distinct values). It should be constant — TPR is a within-class "
                f"rate — so this table was not built the way it claims.")

        # Where the queue stops paying for itself, read off the flags rather than
        # recomputed, so the sentence and the table cannot disagree. This is the
        # finding worth volunteering: below the crossing, working the queue costs
        # more than ignoring it, and no amount of ROC-AUC changes that.
        flagged = [(float(r["prevalence"]), bool(r["clears_break_even"]))
                   for r in projection if r.get("clears_break_even") is not None]
        flagged.sort()
        clearing = [p for p, ok in flagged if ok]
        failing = [p for p, ok in flagged if not ok]
        if clearing and failing:
            crossing = (
                f"The break-even line falls between "
                f"{_pct(max(failing), 3)} and {_pct(min(clearing), 3)}: below "
                f"roughly that base rate this operating point's queue costs more "
                f"to work than to ignore, because precision drops under the "
                f"{_pct(cost.get('break_even_probability'), 2)} break-even the "
                f"cost model implies. That is a real limit on the result and not "
                f"a fixable one at this threshold — a rarer mule population needs "
                f"a tighter cutoff, which trades recall for a queue worth "
                f"working.")
        elif clearing:
            crossing = (
                f"Every base rate in the table clears the "
                f"{_pct(cost.get('break_even_probability'), 2)} break-even "
                f"precision, so the queue pays for itself down to "
                f"{_pct(min(clearing), 3)} prevalence. That is a claim about the "
                f"rates shown; it says nothing about base rates below the bottom "
                f"row.")
        elif failing:
            crossing = (
                f"⚠️ No base rate in the table clears the "
                f"{_pct(cost.get('break_even_probability'), 2)} break-even "
                f"precision, including the rate this model was measured at. At "
                f"this threshold the alert queue costs more to work than to "
                f"ignore.")
        else:
            crossing = ("The break-even comparison was not recorded for any row, "
                        "so nothing is claimed about where the queue stops paying "
                        "for itself.")
        lines += [f"{fixed_note} {crossing}", ""]
        if observed:
            lines += [
                f"The bolded row is the one actually measured: "
                f"{_pct(observed.get('prevalence'), 3)} prevalence on the test "
                f"split. Every other row is arithmetic on that measurement, and "
                f"answers \"what would this queue look like if mules were "
                f"rarer\" — not \"what will this model do in production\", which "
                f"needs real data and is out of scope. See Limitations.",
                "",
            ]

    # ── leakage ──
    if strongest:
        inverted = (" (inverted; raw "
                    f"{_num(strongest.get('raw_auc'), 4)})"
                    if strongest.get("inverted") else "")
        # The reassurance is conditional prose ("if any one column reached that
        # ceiling…") sitting next to the number that decides it. Rendered as a
        # fixed sentence it would keep reassuring the reader on the day the
        # ceiling was actually breached, so the breach case gets said out loud.
        auc_1f = strongest.get("auc")
        ceiling = strongest.get("ceiling")
        breached = (not _absent(auc_1f) and not _absent(ceiling)
                    and float(auc_1f) >= float(ceiling))
        if breached:
            verdict = ("**That is at or above the ceiling, so a single column is "
                       "separating the classes and every number above is "
                       "theatre.** `tests/test_baselines.py` recomputes this from "
                       "the shipped CSVs and fails the build for exactly this "
                       "reason — fix `data/generator.py` before reading any "
                       "further.")
        else:
            verdict = ("If any one column reached that ceiling the generator "
                       "would be planting the label and every number above would "
                       "be theatre — `tests/test_baselines.py` recomputes this "
                       "from the shipped CSVs and fails the build if it ever "
                       "does.")
        lines += [
            "### Is the task actually hard?",
            "",
            f"The strongest *single* feature on test is "
            f"`{strongest.get('feature', '—')}` at direction-corrected AUC "
            f"**{_num(auc_1f)}**{inverted}, against a leakage "
            f"ceiling of {_num(ceiling, 2)}. {verdict}",
            "",
        ]

    # ── per-archetype recall ──
    # Lives at test.recall_breakdown, written by train.archetype_breakdown().
    breakdown = (metrics.get("test", {}) or {}).get("recall_breakdown") or {}
    by_account = breakdown.get("by_archetype_accounts") or {}
    by_ring = breakdown.get("by_archetype_rings") or {}
    if by_account:
        missed = [name for name, r in sorted(by_ring.items())
                  if (r or {}).get("ring_recall", 1.0) < 1.0]
        lines += [
            "### Recall by ring archetype",
            "",
            f"Reported separately because one headline recall hides *which* "
            f"laundering shapes the model misses, and the "
            f"{_count_word(len(by_account))} archetypes are of "
            f"deliberately unequal difficulty. Read the account column first: ring "
            f"recall counts a ring as caught if *any* member is flagged, so it "
            f"flatters a model whenever rings have several members, and a "
            f"saturated metric is no evidence of a good one. Ring recall is still "
            f"the operationally meaningful quantity — one alert brings an analyst "
            f"to the whole ring.",
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
        n_rings = breakdown.get("n_rings")
        flagged = breakdown.get("rings_with_an_alert")
        if missed and n_rings is not None and flagged is not None:
            escaped = n_rings - flagged
            noun = "ring" if escaped == 1 else "rings"
            lines += [
                "",
                f"{escaped} {noun} produced **no alert at all** "
                f"({', '.join(f'`{m}`' for m in missed)}). That is the failure "
                f"mode to watch, and it is not the same as a low account recall: "
                f"a ring with one flagged member still reaches an analyst, who "
                f"then has the whole component to pull on. A ring with zero "
                f"flagged members is invisible to the queue entirely.",
            ]
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
        # "no ring appears in two splits" is the claim every ring-level number
        # above rests on, and train.py already measures it into split_integrity.
        # Typed as a literal it would go on reassuring the reader for exactly as
        # long as it took someone to change the ring-seating code — which is a
        # change being made in data/generator.py right now.
        integrity = metrics.get("split_integrity") or {}
        rings_disjoint = integrity.get("rings_disjoint")
        if rings_disjoint is False:
            disjoint = ("but **at least one ring appears in two splits** "
                        "(`split_integrity.rings_disjoint` is false), so the "
                        "ring-level numbers above are not measured on unseen "
                        "rings")
        elif rings_disjoint is True:
            disjoint = "and no ring appears in two splits"
        else:
            disjoint = ("though nothing in metrics.json records whether a ring "
                        "appears in two splits")
        lines += [
            "",
            f"Splits are consecutive equal-length time windows — no account's "
            f"future is used to predict its past, {disjoint}.",
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
    # newline="\n" so `--write` does not rewrite every line of the README as a
    # side effect of which OS ran it. `read_text` above already normalised the
    # file's terminators in memory, so without this the output terminator is
    # whatever the running platform prefers, and the splice of one block shows
    # up as a diff against the entire document.
    with open(README_PATH, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(updated)
    print(f"README.md Results section updated from "
          f"{METRICS_PATH.relative_to(ROOT)}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
