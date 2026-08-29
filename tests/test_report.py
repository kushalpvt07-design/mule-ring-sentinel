"""
tests/test_report.py
────────────────────
The README is a published artefact, and until now it was the only one with no
staleness guard.

Everything else in this repo is defended. metrics.json cannot describe a retired
model (test_baselines.py). The sensitivity table cannot select a threshold on test
(test_baselines.py). Features cannot drift from the contract (test_contract.py).
Splits cannot leak (test_leakage.py). Meanwhile the file a judge actually reads
first could sit there quoting numbers from a training run three iterations old,
and nothing in the suite would notice.

That is v2's exact failure — a published claim about a model that no longer
existed — displaced one level up, from metrics.json into README.md. So:

  1. THE README MUST AGREE WITH metrics.json, BYTE FOR BYTE.  `models/report.py`
     regenerates the delimited Results block from the metrics the training run
     wrote. `test_results_block_matches_the_published_metrics` re-renders it and
     diffs. If it fails, the fix is `python -m models.report --write` — never
     editing the prose.

  2. NO MEASUREMENT MAY BE HAND-TYPED INTO PROSE.  A generated block is worthless
     if the introduction hand-quotes precision next to it, because that copy
     ages silently. `test_headline_numbers_appear_only_inside_the_generated_block`
     takes the live headline figures out of metrics.json and asserts each one
     appears nowhere in the README except inside the markers.

  3. ABSENT METRICS MUST PRODUCE A REFUSAL, NOT ZEROES.  A reporting tool that
     renders 0.0% precision when it has no data is worse than one that crashes.

The splice unit tests come first, for the same reason test_baselines.py checks its
own AUC helper before using it: a staleness guard built on a broken splice would
pass while corrupting the file it claims to protect.

Depends only on the standard library — models/report.py imports no xgboost, no
sklearn, no pandas — so this file runs on a bare checkout.

Usage:
    pytest tests/test_report.py -v
"""

from __future__ import annotations

import pytest

from models.features import MODEL_VERSION
from models.report import (
    BEGIN,
    END,
    PLACEHOLDER,
    README_PATH,
    _num,
    _pct,
    _rupees,
    render,
    splice,
)


def _readme() -> str:
    if not README_PATH.exists():
        pytest.skip("README.md not found")
    return README_PATH.read_text(encoding="utf-8")


def _generated_block(readme: str) -> str:
    """The text between the markers, inclusive. Skips if unmarked."""
    start = readme.find(BEGIN)
    end = readme.find(END)
    if start == -1 or end == -1:
        pytest.fail(
            f"README.md is missing the {BEGIN} / {END} markers. The Results "
            f"section is machine-generated; restore the markers rather than "
            f"hand-writing numbers into the prose."
        )
    return readme[start:end + len(END)]


def _prose(readme: str) -> str:
    """Everything OUTSIDE the generated block — where no metric may appear."""
    start = readme.find(BEGIN)
    end = readme.find(END)
    if start == -1 or end == -1:
        return readme
    return readme[:start] + readme[end + len(END):]


# ══════════════════════════════════════════════════════════════════
# 0. The splice, before it is trusted with the file
# ══════════════════════════════════════════════════════════════════

class TestTheSpliceIsCorrect:
    """
    `splice` is the only function in the repo that rewrites a tracked file in
    place. It gets pinned on the corners before anything relies on it.
    """

    def test_it_replaces_only_the_delimited_region(self):
        before = f"KEEP BEFORE\n{BEGIN}\nold numbers\n{END}\nKEEP AFTER\n"
        out = splice(before, f"{BEGIN}\nnew numbers\n{END}")
        assert out == f"KEEP BEFORE\n{BEGIN}\nnew numbers\n{END}\nKEEP AFTER\n"

    def test_it_is_idempotent(self):
        """Running --write twice must not accumulate markers or drop bytes."""
        block = f"{BEGIN}\nnumbers\n{END}"
        once = splice(f"A\n{BEGIN}\nold\n{END}\nB\n", block)
        assert splice(once, block) == once

    def test_it_refuses_a_readme_with_no_markers(self):
        """
        Silently appending would be the dangerous behaviour: the real Results
        section would stay stale while the tool reported success.
        """
        with pytest.raises(SystemExit):
            splice("no markers here\n", f"{BEGIN}\nx\n{END}")

    def test_it_refuses_reversed_markers(self):
        with pytest.raises(SystemExit):
            splice(f"{END} ... {BEGIN}", f"{BEGIN}\nx\n{END}")


# ══════════════════════════════════════════════════════════════════
# 1. Absent metrics must refuse, not fabricate
# ══════════════════════════════════════════════════════════════════

class TestNoMetricsMeansNoNumbers:

    def test_the_placeholder_states_the_absence_and_quotes_nothing(self):
        """
        On a fresh clone the block must say "no metrics published yet" rather
        than render a table of zeroes that reads like a terrible model, or worse,
        a table of last-run numbers that reads like a good one.
        """
        assert PLACEHOLDER.startswith(BEGIN) and PLACEHOLDER.endswith(END)
        assert "No metrics published yet" in PLACEHOLDER
        assert "%" not in PLACEHOLDER and "₹" not in PLACEHOLDER
        assert "models.train" in PLACEHOLDER, (
            "the placeholder must tell the reader how to produce the numbers"
        )


# ══════════════════════════════════════════════════════════════════
# 2. The README agrees with metrics.json
# ══════════════════════════════════════════════════════════════════

class TestReadmeIsNotStale:

    def test_results_block_matches_the_published_metrics(self, metrics):
        """
        The whole point of this file. A README quoting a superseded training run
        is a dishonest artefact, and it is the artefact people read first.
        """
        readme = _readme()
        expected = render(metrics)
        actual = _generated_block(readme)
        assert actual == expected, (
            "README.md's Results block does not match "
            "models/saved_models/metrics.json — the published numbers describe a "
            "different run than the one on disk.\n"
            "  Fix: python -m models.report --write\n"
            "  Do NOT hand-edit the block; it is regenerated."
        )

    def test_the_block_names_the_model_it_describes(self, metrics):
        """
        A number without a model version attached is unfalsifiable. The rendered
        block must name the version, and it must be the current one.
        """
        block = _generated_block(_readme())
        assert MODEL_VERSION in block, (
            f"the Results block does not name {MODEL_VERSION!r}, so a reader "
            f"cannot tell which model it describes."
        )

    def test_the_block_tells_the_reader_it_is_generated(self):
        """
        Without this line the next person to spot a wrong figure edits the prose,
        and the next --write silently reverts them.
        """
        block = _generated_block(_readme())
        assert "models.report" in block, (
            "the Results block must state that it is generated and how, or "
            "someone will hand-edit it."
        )

    def test_the_block_states_that_the_threshold_came_from_validation(self):
        """
        The single claim the track's bar turns on. It is rendered, not typed, so
        it cannot be quietly dropped while the numbers stay.
        """
        block = _generated_block(_readme())
        assert "validation" in block.lower(), (
            "the Results block no longer says which split chose the threshold; "
            "that provenance is the difference between a result and a guess."
        )


# ══════════════════════════════════════════════════════════════════
# 3. No measurement is hand-typed into the prose
# ══════════════════════════════════════════════════════════════════

class TestProseCarriesNoHandTypedMetrics:
    """
    The failure mode a generated block does NOT prevent.

    Generating the Results section is pointless if the introduction, the design
    notes or the limitations hand-quote the same figures: those copies are not
    regenerated, so they rot in place, and a reader has no way to tell which
    number is live. Cost *assumptions* (₹2,00,000, ₹15,000, the ratio and the
    break-even that follows from it) are deliberately exempt — they are inputs
    the author chose, stated as assumptions, not measurements of anything.
    """

    def _live_headline_strings(self, metrics) -> list[tuple[str, str]]:
        """
        (rendered form, what it is) for each live headline measurement.

        Formatting is borrowed from models.report rather than reimplemented. A
        second copy of the Indian-grouping logic here could disagree with the
        renderer, and then this test would look for a string the README never
        contains — a guard that always passes is worse than no guard.
        """
        out: list[tuple[str, str]] = []
        if metrics.get("total_cost") is not None:
            out.append((_rupees(metrics["total_cost"]), "total cost on test"))
        for key, what in (("precision", "test precision"),
                          ("recall", "test recall")):
            if metrics.get(key) is not None:
                out.append((_pct(metrics[key]), what))
        if metrics.get("roc_auc") is not None:
            out.append((_num(metrics["roc_auc"]), "test ROC-AUC"))
        threshold = ((metrics.get("validation") or {})
                     .get("cost_optimal", {}).get("threshold"))
        if threshold is not None:
            out.append((_num(threshold), "the operating threshold"))
        return out

    def test_headline_numbers_appear_only_inside_the_generated_block(
            self, metrics):
        prose = _prose(_readme())
        offenders = [(text, what)
                     for text, what in self._live_headline_strings(metrics)
                     if text in prose]
        assert not offenders, (
            "README.md hand-types live measurements outside the generated "
            "block: "
            + "; ".join(f"{text!r} ({what})" for text, what in offenders)
            + ".\n  Those copies are not regenerated, so they will silently "
              "describe an older run after the next `models.train`. Refer the "
              "reader to the Results section instead of restating the figure."
        )

    def test_the_prose_still_explains_the_cost_assumptions(self):
        """
        The mirror image, so the rule above cannot be satisfied by deleting all
        context. The cost figures are assumptions and MUST be argued for in
        prose — the track's bar is honest metrics *including false-positive
        cost*, and a bare number in a table does not discharge that.
        """
        prose = _prose(_readme()).lower()
        assert "assumption" in prose, (
            "the prose no longer states that the cost figures are assumptions"
        )
        assert "false" in prose and "positive" in prose, (
            "the prose no longer discusses false-positive cost"
        )
