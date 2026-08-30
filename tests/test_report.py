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

import re

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


# Currency and percent signs, the two places a hand-typed figure drifts from the
# renderer's spacing: `₹ 64,10,000` and `33.5 %` are the same claim as the forms
# models/report.py emits, and a plain `in` test misses both.
_UNIT_SIGN = re.compile(r"([₹%])")


def _spacing_tolerant(text: str) -> re.Pattern[str]:
    """
    A pattern matching `text` with optional whitespace either side of ₹ and %.

    Built from the rendered string rather than from a hand-written regex, for the
    same reason `_live_headline_strings` borrows its formatting from the renderer:
    a pattern assembled independently could stop matching what the tool actually
    writes, and a guard that cannot match is a guard that always passes.

    `\\s*` covers the non-breaking space too, which is what a copy-paste out of a
    rendered Markdown table tends to carry.
    """
    parts = [p for p in _UNIT_SIGN.split(text) if p]
    return re.compile(r"\s*".join(re.escape(p) for p in parts))


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
        block must name the version, and it must be the ONLY version it names.

        WHY THE SECOND HALF EXISTS. This test used to be `MODEL_VERSION in block`
        and could not fail. `models/report.py` injects the literal `sentinel_v3`
        exactly when the block describes a different model — the stale banner reads
        "These numbers describe `sentinel_v2`, but the code is at `sentinel_v3`" —
        so the current version is present precisely in the case the test was
        written to catch. Setting `"model_version": "sentinel_v2"` in metrics.json
        left this green while the headline read `Model `sentinel_v2``.

        Scanning for every `sentinel_vN` in the block and requiring the set to be
        exactly {MODEL_VERSION} fails on the stale render, because the stale render
        necessarily names both.
        """
        block = _generated_block(_readme())
        assert MODEL_VERSION in block, (
            f"the Results block does not name {MODEL_VERSION!r}, so a reader "
            f"cannot tell which model it describes."
        )

        named = set(re.findall(r"sentinel_v\d+", block))
        assert named == {MODEL_VERSION}, (
            f"the Results block names model version(s) {sorted(named)}; the code "
            f"is at {MODEL_VERSION!r}. A block that mentions two versions is "
            f"either the stale-metrics banner or a hand-edit — in both cases the "
            f"figures below it describe a model that is no longer the one on "
            f"disk.\n  Fix: python -m models.train, then python -m models.report "
            f"--write"
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

    WHAT THIS CATCHES, AND WHAT IT DOES NOT
    ───────────────────────────────────────
    It catches the realistic failure: someone copies a figure out of the generated
    block into prose, verbatim or with the spacing mangled (`33.5 %`, `₹ 64,10,000`).
    Copying is how this actually happens — nobody recomputes a metric by hand in
    order to quote it.

    It does not catch a *re-rendering*: `₹64.1L`, `about a third`, `~0.98 AUC`.
    Closing that would mean scanning the prose for any number within rounding
    distance of a metric, and the README legitimately contains a great many
    numbers — the 60-day window, the 13.3333:1 cost ratio, the archetype counts —
    so such a rule would fail on correct text and be deleted within a week. The
    byte-for-byte block comparison in `TestReadmeIsNotStale` plus
    `python -m models.report --check` in CI are what actually keep the published
    figures true; this class is the narrower guard against the copy that escapes
    the markers.
    """

    def _live_headline_strings(self, metrics) -> list[tuple[str, str]]:
        """
        (rendered form, what it is) for each live headline measurement.

        Formatting is borrowed from models.report rather than reimplemented. A
        second copy of the Indian-grouping logic here could disagree with the
        renderer, and then this test would look for a string the README never
        contains — a guard that always passes is worse than no guard.

        WHY THE KEYS ARE REQUIRED RATHER THAN SKIPPED OVER. Every figure used to be
        appended under `if metrics.get(key) is not None`, which made the scan's
        SIZE depend on the file it was scanning: rename `total_cost` and the guard
        quietly stops looking for the total cost, `offenders` comes back empty, and
        `assert not offenders` passes on a list that was never populated. That is
        the same failure this whole file exists to prevent, one level up. A missing
        headline key is now a failure with a name attached, and `_missing` is
        returned so the caller can report it rather than infer it from a short list.
        """
        HEADLINE_KEYS = (
            ("total_cost", "total cost on test", _rupees),
            ("precision", "test precision", _pct),
            ("recall", "test recall", _pct),
            ("roc_auc", "test ROC-AUC", _num),
        )
        out: list[tuple[str, str]] = []
        missing: list[str] = []
        for key, what, fmt in HEADLINE_KEYS:
            value = metrics.get(key)
            if value is None:
                missing.append(key)
            else:
                out.append((fmt(value), what))

        threshold = ((metrics.get("validation") or {})
                     .get("cost_optimal", {}).get("threshold"))
        if threshold is None:
            missing.append("validation.cost_optimal.threshold")
        else:
            out.append((_num(threshold), "the operating threshold"))

        assert not missing, (
            f"metrics.json is missing headline keys {missing}, so the prose scan "
            f"below would silently stop looking for those figures and pass on an "
            f"empty offender list. Either the keys were renamed — in which case "
            f"update HEADLINE_KEYS here, deliberately — or metrics.json predates "
            f"them and needs `python -m models.train`."
        )
        return out

    def test_headline_numbers_appear_only_inside_the_generated_block(
            self, metrics):
        prose = _prose(_readme())
        offenders = [(text, what)
                     for text, what in self._live_headline_strings(metrics)
                     if _spacing_tolerant(text).search(prose)]
        assert not offenders, (
            "README.md hand-types live measurements outside the generated "
            "block: "
            + "; ".join(f"{text!r} ({what})" for text, what in offenders)
            + ".\n  Those copies are not regenerated, so they will silently "
              "describe an older run after the next `models.train`. Refer the "
              "reader to the Results section instead of restating the figure."
        )

    def test_the_spacing_tolerant_scan_would_catch_a_mangled_copy(self, metrics):
        """
        The guard above, checked against a figure it used to miss.

        Substring matching passed a README that quoted `33.5 %` or `₹ 64,10,000`,
        because the space made it a different string from the one the renderer
        writes — and a hand-typed metric with a stray space rots exactly as
        silently as one without. Constructed here from the live metrics so this
        cannot drift out of step with what the scan looks for.

        THE ASSERTION THAT USED TO BE HERE COULD NOT FAIL: `assert spaced not in
        text`, where `spaced` is built by INSERTING characters into `text`. A longer
        string is never a substring of a shorter one, so no fixture edit could fire
        it. What it was reaching for is the two-sided claim below — the naive scan
        misses the mangled copy, and the spacing-tolerant one finds it. Without the
        first half, replacing `_spacing_tolerant` with a plain `in` check would
        leave this test green.
        """
        live = self._live_headline_strings(metrics)
        if not live:
            pytest.skip("metrics.json publishes no headline figures to mangle")

        mangled = [(text, text.replace("₹", "₹ ").replace("%", " %"))
                   for text, _ in live]
        detectable = [(text, spaced) for text, spaced in mangled
                      if text != spaced]
        if not detectable:
            pytest.skip("no headline figure carries a ₹ or % sign to space out")

        for text, spaced in detectable:
            prose = f"prose quoting {spaced} in passing"
            # Half one: the mangled copy really is invisible to substring matching,
            # so there is a defect here to catch.
            assert text not in prose, (
                f"{text!r} is a plain substring of prose containing {spaced!r}, so "
                f"the naive scan would already have caught this and the "
                f"spacing-tolerant one is not what makes the guard work. Check "
                f"what `_spacing_tolerant` is being credited with."
            )
            # Half two: the scan actually in use does catch it.
            assert _spacing_tolerant(text).search(prose), (
                f"a hand-typed {spaced!r} would slip past the prose scan, though "
                f"it states the same measurement as {text!r}."
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
