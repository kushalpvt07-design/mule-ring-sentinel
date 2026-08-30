"""
tests/test_responder.py
───────────────────────
The defense-only gate, and the tiering that must follow from the score.

The Razorpay Track 2 bar is explicit: "Strictly defense-only: anything
offense-capable is disqualified." This file is the machine-checked form of that
promise. The strongest action the Sentinel can emit is HOLD_FOR_REVIEW — a human
looks at the account. Nothing here bans, blocks, freezes, suspends or terminates.

─────────────────────────────────────────────────────────────────────────────
WHY THE OLD VERSION OF THIS FILE WAS NOT ENOUGH
─────────────────────────────────────────────────────────────────────────────
The previous batch test only ever fed `validate_response_batch` the OUTPUT of
`build_node_response`. Those pass by construction — the builder already computed
the tier and action from the score, so re-deriving them can only agree. It proved
the batch validator agrees with itself, not that it CATCHES a bad response.

The batch validator exists to stop a response assembled *somewhere else* — with a
CRITICAL score but an ALLOW action — from reaching the caller, because ALLOW is a
permitted value and a naive forbidden-action screen waves it through. So the real
test hand-forges exactly that: a NodeRiskScore whose action was downgraded below
what its score demands, and requires the validator to raise. See
`TestBatchValidationCatchesForgery`.

It also carried a stale call:
    build_node_response(..., top_features=["pagerank", "fan_in_concentration"])
`top_features: list[str]` is gone. api/main.py used to fill it with the model's
three highest GLOBAL gains — identical on every alert, so it described the model,
not the account. It is now `contributing_factors: list[ContributingFactor]`, the
per-account SHAP attributions. Both the parameter name and the element type
changed, and this file now builds real ContributingFactor objects.

─────────────────────────────────────────────────────────────────────────────
WHY THE TEST CASES NO LONGER COME FROM api/responder.py
─────────────────────────────────────────────────────────────────────────────
The bypass test used to be parametrized over the module's own blocklist:

    @pytest.mark.parametrize("verb", FORBIDDEN_VERBS)

Its cases were therefore generated from the code under test, so it could only
check verbs that code already knew about. That is why it stayed green for as long
as `LOCK_ACCOUNT` passed both rails: "BLOCK" is not a substring of "LOCK_ACCOUNT",
and nothing in the module said LOCK. The cases now come from
`ENFORCEMENT_ACTION_NAMES` below, written in this file from the outside. The sizes
of both blocklists are asserted directly as well, because emptying one yields an
empty parameter set and pytest reports that as a SKIP, not a failure — deleting a
rail by accident would have turned this class green too.

Everything here needs pydantic (api.schemas is a pydantic model) but no model,
data or network — so it runs on a bare checkout the moment pydantic is installed.
The one check that must not depend on pydantic — that no shipped action name reads
as enforcement — therefore cannot live in this file at all: the module-level
`importorskip` below deletes the whole suite, including it. It lives in
tests/test_contract.py as `TestActionsAreDefenseOnlyStatically`, which parses the
source with `ast` and imports nothing.

Usage:
    pytest tests/test_responder.py -v
"""

from __future__ import annotations

from enum import Enum

import pytest

# api.schemas is built on pydantic; importing api.responder pulls it in. Skip the
# whole module cleanly where pydantic is absent rather than erroring at collection.
pytest.importorskip("pydantic", reason="api.schemas is a pydantic model")

import api.responder as responder  # noqa: E402  (after importorskip, intentionally)
from api.responder import (  # noqa: E402
    FORBIDDEN_ACTIONS,
    FORBIDDEN_VERBS,
    PERMITTED_ACTIONS,
    DefenseOnlyViolation,
    _validate_action,
    build_node_response,
    classify_risk,
    critical_cutoff,
    determine_action,
    medium_cutoff,
    validate_response_batch,
)
from api.schemas import ActionCode, ContributingFactor, NodeRiskScore, RiskLevel  # noqa: E402

# The cost model's break-even probability, p* = fp_cost / (fp_cost + fn_cost) at
# ₹15,000 and ₹2,00,000. It is the floor of the step-up band.
#
# This one IS written out, and that is safe for a reason the operating threshold
# below does not share: p* is an identity over two prices this project chose, not
# an artefact of a fitting run. Retraining cannot move it. Two tests hold it in
# place from both sides — `test_the_step_up_floor_is_the_break_even_probability`
# recomputes it from 15,000 / 2,15,000, and
# `test_the_break_even_literal_matches_the_published_cost_model` requires it to
# equal what metrics.json publishes. So changing the cost model fails this file
# rather than diverging from it quietly.
BREAK_EVEN = 0.069767


def shipped_threshold(metrics: dict) -> float:
    """
    The operating threshold the loaded metrics.json actually publishes.

    THIS USED TO BE A MODULE CONSTANT, `SHIPPED_THRESHOLD = 0.5908352434635162`,
    and it was wrong: the shipped `optimal_threshold` had moved to roughly a third
    of that. The damage was not a stale comment. `SHIPPED_THRESHOLD` fed
    `classify_risk` in three tests, so those tests were checking the band geometry
    of a threshold no longer in service — and the regression test below was green
    ONLY because of the stale value. Corrected to the real threshold, five of its
    seven fixed scores move MEDIUM → HIGH, because the operating point had
    descended past them.

    A test that pins a fitted quantity by copying it cannot notice the fit moving.
    Read the number from the artefact that owns it, via the session-scoped
    `metrics` fixture in conftest.py, which skips on a fresh checkout.
    """
    threshold = metrics.get("optimal_threshold")
    if threshold is None:
        pytest.skip("metrics.json publishes no optimal_threshold; "
                    "run `python -m models.train`")
    threshold = float(threshold)
    if not (0.0 < threshold <= 1.0):
        pytest.fail(
            f"metrics.json publishes optimal_threshold={threshold!r}, which is "
            f"not an operating point in (0, 1]. classify_risk rejects it, so the "
            f"service could not serve this file."
        )
    return threshold

# Enforcement action names, written from the outside. Not one of these is read
# from api/responder.py — that is the entire point (see the header). `LOCK_ACCOUNT`
# is first because it is the name that used to pass both rails.
ENFORCEMENT_ACTION_NAMES = (
    "LOCK_ACCOUNT",
    "QUARANTINE_ACCOUNT",
    "RESTRICT_TRANSFERS",
    "HALT_PAYOUTS",
    "BLACKLIST_VPA",
    "DENY_TRANSACTION",
    "LIMIT_WITHDRAWALS",
    "REVERSE_TRANSACTION",
    "DISABLE_VPA",
    "REVOKE_MANDATE",
    "SEIZE_BALANCE",
    "CLOSE_ACCOUNT",
    "BAN_USER",
    "BLOCK_ACCOUNT",
    "FREEZE_FUNDS",
    "SUSPEND_ACCOUNT",
    "TERMINATE_USER",
)

# Every action a score is allowed to reach, with human review as the ceiling.
# Held as a literal so wiring a new, stronger member of ActionCode into
# `determine_action` fails this file instead of passing it. FLAG_FOR_REVIEW exists
# in the enum but no tier maps to it, so it is deliberately absent.
REACHABLE_ACTIONS = frozenset({
    "ALLOW",
    "REQUIRE_ADDITIONAL_AUTH",
    "HOLD_FOR_REVIEW",
})


def _factor(feature: str, contribution: float) -> ContributingFactor:
    """A minimal, valid ContributingFactor for building responses under test."""
    return ContributingFactor(
        feature=feature,
        description=f"{feature} (test fixture)",
        value=1.0,
        contribution=contribution,
        effect="raises_risk" if contribution >= 0 else "lowers_risk",
    )


# ══════════════════════════════════════════════════════════════════
# 1. Defense-only: no code path can emit an enforcement action
# ══════════════════════════════════════════════════════════════════

class TestDefenseOnlyConstraint:
    """The hard product constraint, checked from several directions."""

    def test_the_rails_have_not_been_emptied(self):
        """
        Sizes, asserted directly.

        Every other test in this class is parametrized, and an empty parameter set
        is a SKIP in pytest rather than a failure — so emptying `FORBIDDEN_VERBS`
        would have made this class green instead of red. The counts are floors, not
        equalities: adding a verb is welcome, losing one is a regression in a rail.
        `PERMITTED_ACTIONS` is pinned exactly, because widening it is the one way
        past the primary rail.
        """
        assert len(PERMITTED_ACTIONS) == 4, (
            f"PERMITTED_ACTIONS holds {sorted(PERMITTED_ACTIONS)}. The allowlist "
            f"is the primary defense-only rail; a fifth entry means some new "
            f"action can leave the API, and that is the change to justify."
        )
        assert len(FORBIDDEN_ACTIONS) >= 7
        assert len(FORBIDDEN_VERBS) >= 17

    @pytest.mark.parametrize("forbidden", sorted(FORBIDDEN_ACTIONS))
    def test_every_listed_forbidden_action_raises(self, forbidden):
        with pytest.raises(DefenseOnlyViolation):
            _validate_action(forbidden)

    def test_forbidden_check_is_case_insensitive(self):
        for spelling in ("ban_user", "Ban_User", "bAn_UsEr"):
            with pytest.raises(DefenseOnlyViolation):
                _validate_action(spelling)

    @pytest.mark.parametrize("name", ENFORCEMENT_ACTION_NAMES)
    def test_an_action_that_reads_as_enforcement_is_refused(self, name):
        """
        THE REGRESSION, and the reason the rail is now an allowlist.

        `LOCK_ACCOUNT` was in neither `FORBIDDEN_ACTIONS` nor caught by the
        substring scan — the banned verb is "BLOCK", and "BLOCK" is not a substring
        of "LOCK_ACCOUNT" — so the single guarantee this module exists to make was
        broken by a name a reviewer would spot instantly. These names are held in
        this file, so the test can fail; which rail catches them is not asserted,
        only that the action does not get out and that the message names it.
        """
        with pytest.raises(DefenseOnlyViolation) as excinfo:
            _validate_action(name)
        assert name in str(excinfo.value)

    def test_an_advisory_sounding_invention_is_also_refused(self):
        """
        Fail closed, not just fail on enforcement.

        A blocklist permits by default, so `ESCALATE_TO_FIU` — plausible, not on
        any list, and not this service's to send — would have passed. The allowlist
        refuses it for the honest reason: it is not one of the four actions this
        API is allowed to emit, and no rail can be relied on to recognise the
        wording of an action nobody has invented yet.
        """
        with pytest.raises(DefenseOnlyViolation, match="not one of the permitted"):
            _validate_action("ESCALATE_TO_FIU")

    def test_the_import_time_sweep_would_reject_a_forbidden_member(self,
                                                                   monkeypatch):
        """
        The sweep, made observable.

        Asserting that the shipped enum passes the sweep is vacuous — importing
        api.responder would already have raised, so the assert is unreachable. The
        only way to test a guard is to give it something bad, so a stand-in enum
        carrying `LOCK_ACCOUNT` is patched in and the sweep is required to refuse
        it. This fails if the allowlist check is ever inverted back into a
        blocklist, which is exactly the defect that shipped.
        """
        class ForgedActionCode(str, Enum):
            ALLOW = "ALLOW"
            LOCK_ACCOUNT = "LOCK_ACCOUNT"

        monkeypatch.setattr(responder, "ActionCode", ForgedActionCode)
        with pytest.raises(DefenseOnlyViolation, match="LOCK_ACCOUNT"):
            responder._assert_action_enum_is_defense_only()

    def test_the_sweep_notices_an_allowlist_that_has_drifted(self, monkeypatch):
        """
        The other direction: the allowlist naming actions the enum no longer has.

        `PERMITTED_ACTIONS` is a deliberate hand-written copy of the enum — that
        independence is what makes the sweep a real check rather than a tautology —
        and a copy is only worth having if it is required to stay in step. Here the
        enum has shrunk to one member and every member still passes, so only the
        equality check can catch it. It matters beyond tidiness: rename an action
        and a stale allowlist goes on permitting the old string.
        """
        class NarrowedActionCode(str, Enum):
            ALLOW = "ALLOW"

        monkeypatch.setattr(responder, "ActionCode", NarrowedActionCode)
        with pytest.raises(DefenseOnlyViolation, match="drifted"):
            responder._assert_action_enum_is_defense_only()

    @pytest.mark.parametrize("score", [0.0, 0.01, BREAK_EVEN, 0.1, 0.3, 0.5,
                                       0.7, 0.9, 0.99, 1.0])
    def test_the_strongest_action_any_score_can_reach_is_human_review(self, score):
        """
        The end-to-end product claim, as an upper bound rather than a
        non-membership check.

        `action.value not in FORBIDDEN_ACTIONS` could not fail: the value came from
        an enum whose every member was screened at import. What CAN fail is the
        mapping — point CRITICAL at some future stronger action and the set of
        actions reachable from a score changes, which this notices.
        """
        response = build_node_response(
            "test@upi", score, threshold=0.5, break_even_probability=BREAK_EVEN)
        assert response.action.value in REACHABLE_ACTIONS
        assert response.action != ActionCode.FLAG_FOR_REVIEW, (
            "FLAG_FOR_REVIEW is in the enum but no tier maps to it. If that "
            "changed deliberately, REACHABLE_ACTIONS is where to say so."
        )


# ══════════════════════════════════════════════════════════════════
# 2. Tiering: the boundaries, and the regressions each one fixed
# ══════════════════════════════════════════════════════════════════

class TestRiskTiering:
    """
    Tiers are anchored to two numbers, and to different ones on purpose. CRITICAL
    is the midpoint from the operating threshold to certainty — not a fixed 0.85
    and not `threshold * 1.5`. The step-up floor is the cost model's break-even
    probability — not a fraction of the threshold, which is the number the cost
    model has nothing to say about.
    """

    @pytest.mark.parametrize("score,expected", [
        (0.05, RiskLevel.LOW),           # below p* = 0.069767
        (BREAK_EVEN, RiskLevel.MEDIUM),  # the floor itself is inclusive
        (0.20, RiskLevel.MEDIUM),        # was LOW under `threshold * 0.6`
        (0.35, RiskLevel.MEDIUM),
        (0.50, RiskLevel.HIGH),          # in [0.50, 0.75)
        (0.85, RiskLevel.CRITICAL),      # >= 0.75 = 0.5 + (1-0.5)/2
        (0.99, RiskLevel.CRITICAL),
    ])
    def test_classification_at_threshold_half(self, score, expected):
        assert classify_risk(score, 0.5, BREAK_EVEN) == expected

    def test_no_score_above_break_even_is_ever_allowed(self, metrics):
        """
        THE REGRESSION, as the property it protects rather than as one tier.

        The step-up floor was `threshold * 0.6`, so a whole band above p* came back
        LOW and ALLOW — including the cutoff the test-set oracle picked and the mean
        predicted probability of the [0.2, 0.4) reliability bin. Above p* the cost
        model prices an account as worth more than nothing, and ALLOW is nothing.

        WHY THIS NO LONGER ASSERTS `== MEDIUM`. It used to, against a hardcoded
        threshold of 0.5908, over seven fixed scores. Both parts were wrong
        together and that is what kept it green: the real `optimal_threshold` is
        about a third of 0.5908, and at the real value five of those seven scores
        are above the operating point — so they are HIGH, and HIGH is a stronger
        answer than the test demanded. Pinning the tier turned a safety property
        into a snapshot of one fitting run, and the snapshot had already expired.

        The property is: no score at or above p* may tier LOW or earn ALLOW. It
        holds at every threshold, because the floor is `min(p*, t)` — at or above
        p* a score is at least MEDIUM when t >= p*, and HIGH when t < p*. So it can
        be swept across the band instead of sampled, and it needs no copy of the
        operating point to be meaningful.

        WHAT THE SWEEP CANNOT SEE. Its step is 0.001, so it detects an ALLOW hole
        opened anywhere in the band down to about that width and no narrower —
        checked by mutation, where holes of 0.01, 0.005 and 0.002 are all caught
        and one of 0.0005 is not. Sub-step defects are real: `build_node_response`
        rounds the score to six decimals before classifying while the threshold it
        compares against is unrounded, which is a band far below this resolution.
        This test does not cover that. It is a boundary-arithmetic question and
        belongs in a test that reasons about the rounding, not in a sweep.
        """
        threshold = shipped_threshold(metrics)

        # The band, densely. `threshold * 0.6` sits above p* at every threshold
        # this repo has shipped, so the mutant loses the low end of this sweep.
        scores = [BREAK_EVEN + i * 0.001 for i in range(int((1.0 - BREAK_EVEN) / 0.001))]
        # Plus the scores the old bug was caught on, kept by name so the history
        # stays checkable: the test-oracle cutoff, the reliability-bin mean, and
        # the two sides of the old `0.6 * 0.5908` floor.
        scores += [BREAK_EVEN, 0.1, 0.2665, 0.2903, 0.3544, 0.35450, 0.5, 1.0]

        for score in scores:
            level = classify_risk(score, threshold, BREAK_EVEN)
            action = determine_action(level)
            assert level != RiskLevel.LOW, (
                f"score {score:.6f} is at or above p* = {BREAK_EVEN} but tiered "
                f"LOW at threshold {threshold}. The cost model prices it as worth "
                f"more than nothing."
            )
            assert action != ActionCode.ALLOW, (
                f"score {score:.6f} tiered {level.value} and still earned ALLOW. "
                f"Whatever the tier, a score above break-even cannot be waved "
                f"through."
            )

    def test_the_break_even_literal_matches_the_published_cost_model(self, metrics):
        """
        The one thing `BREAK_EVEN` being a literal can get wrong.

        p* cannot move when the model is refitted, which is why it is written out
        above. It CAN move if someone reprices a miss or a false alert — and then
        every tiering test in this file would go on describing the old economics
        while the service applied the new ones. So the literal is required to equal
        what the cost model publishes.
        """
        published = (metrics.get("cost_config") or {}).get("break_even_probability")
        if published is None:
            pytest.skip("metrics.json has no cost_config.break_even_probability")
        assert float(published) == pytest.approx(BREAK_EVEN, abs=1e-6), (
            f"metrics.json publishes p* = {published}, this file assumes "
            f"{BREAK_EVEN}. The cost model was repriced; the step-up floor moved "
            f"with it, and the tiering cases here describe the old prices."
        )

    def test_the_step_up_floor_is_the_break_even_probability(self, metrics):
        """
        Stated directly, because it is an economic claim and not a tuning choice:
        p* = fp_cost / (fp_cost + fn_cost) = 15,000 / 2,15,000.

        Checked at the shipped threshold and at a spread of others above p*, so the
        claim is "the floor IS p* wherever the operating point lands", not "the
        floor happened to be p* at one fitted value".
        """
        assert BREAK_EVEN == pytest.approx(15_000 / (15_000 + 200_000), abs=1e-6)
        for threshold in (shipped_threshold(metrics), 0.07, 0.3, 0.6, 1.0):
            assert threshold >= BREAK_EVEN, (
                f"threshold {threshold} is below p*; that case is "
                f"test_the_step_up_band_cannot_invert_under_a_low_override"
            )
            assert medium_cutoff(threshold, BREAK_EVEN) == BREAK_EVEN

    def test_the_step_up_band_cannot_invert_under_a_low_override(self):
        """
        A caller may override the threshold to below p*. The floor is then the
        threshold itself, leaving the band empty rather than inverted — a MEDIUM
        floor above the HIGH floor would put scores in a band they are above.
        """
        assert medium_cutoff(0.05, BREAK_EVEN) == 0.05
        assert classify_risk(0.06, 0.05, BREAK_EVEN) == RiskLevel.HIGH
        assert classify_risk(0.04, 0.05, BREAK_EVEN) == RiskLevel.LOW

    def test_critical_cutoff_is_the_midpoint_to_certainty(self):
        for t in (0.07, 0.2, 0.5, 0.9):
            assert critical_cutoff(t) == pytest.approx(t + (1.0 - t) / 2.0)

    def test_critical_is_monotone_in_the_threshold(self):
        cutoffs = [critical_cutoff(t) for t in (0.05, 0.1, 0.3, 0.6, 0.9)]
        assert cutoffs == sorted(cutoffs)

    def test_the_tiers_are_ordered_across_the_whole_score_range(self, metrics):
        """
        One sweep, because the three boundaries are now set by two independent
        numbers and a mistake in either would show up as a tier that goes
        backwards. Scores rise, so tiers must never fall.
        """
        rank = {RiskLevel.LOW: 0, RiskLevel.MEDIUM: 1,
                RiskLevel.HIGH: 2, RiskLevel.CRITICAL: 3}
        for threshold in (0.07, shipped_threshold(metrics), 0.95):
            ranks = [rank[classify_risk(s / 1000.0, threshold, BREAK_EVEN)]
                     for s in range(0, 1001)]
            assert ranks == sorted(ranks), (
                f"tiers are not monotone in the score at threshold {threshold}"
            )

    def test_a_zero_threshold_is_rejected_not_all_critical(self):
        """
        THE REGRESSION. The old first branch was `score >= min(threshold*1.5,
        0.95)`; at threshold 0 that is `score >= 0`, true for every account —
        `classify_risk(0.0, 0.0)` returned CRITICAL. A zero threshold is now an
        error, so it can never again mark the entire population CRITICAL.
        """
        with pytest.raises(ValueError, match="threshold"):
            classify_risk(0.0, 0.0, BREAK_EVEN)

    @pytest.mark.parametrize("bad", [-0.1, 1.5, 2.0])
    def test_thresholds_outside_the_unit_interval_are_rejected(self, bad):
        with pytest.raises(ValueError, match="threshold"):
            classify_risk(0.5, bad, BREAK_EVEN)

    @pytest.mark.parametrize("bad", [0.0, -0.1, 1.5])
    def test_a_break_even_that_is_not_a_probability_is_rejected(self, bad):
        """
        The floor has to come from somewhere real. Zero would put every non-zero
        score into the step-up band, which is the mirror image of the zero-threshold
        defect above, so it is refused rather than defaulted.
        """
        with pytest.raises(ValueError, match="break_even_probability"):
            classify_risk(0.5, 0.5, bad)

    @pytest.mark.parametrize("bad", [-0.01, 1.01, 5.0])
    def test_scores_outside_the_unit_interval_are_rejected(self, bad):
        """A raw margin fed in as a probability is a bug worth catching loudly."""
        with pytest.raises(ValueError, match="risk_score"):
            classify_risk(bad, 0.5, BREAK_EVEN)

    def test_the_reported_score_and_its_tier_cannot_contradict(self):
        """
        v3 change 4: rounding happens BEFORE classification. A score of 0.0697999
        must not be reported as 0.0698 while its tier was computed from the longer
        value. Build a response near a tier boundary and confirm the tier follows
        from the number actually reported.
        """
        threshold = 0.0698
        response = build_node_response(
            "edge@upi", 0.06979994, threshold=threshold,
            break_even_probability=BREAK_EVEN)
        assert response.risk_score == round(0.06979994, 6)
        assert response.risk_level == classify_risk(
            response.risk_score, threshold, BREAK_EVEN)


class TestRiskLevelToAction:
    @pytest.mark.parametrize("level,action", [
        (RiskLevel.CRITICAL, ActionCode.HOLD_FOR_REVIEW),
        (RiskLevel.HIGH, ActionCode.HOLD_FOR_REVIEW),
        (RiskLevel.MEDIUM, ActionCode.REQUIRE_ADDITIONAL_AUTH),
        (RiskLevel.LOW, ActionCode.ALLOW),
    ])
    def test_each_level_maps_to_its_defense_only_action(self, level, action):
        assert determine_action(level) == action


# ══════════════════════════════════════════════════════════════════
# 3. build_node_response carries per-account attributions
# ══════════════════════════════════════════════════════════════════

class TestBuildNodeResponse:
    def test_high_score_holds_for_review_with_its_factors(self):
        factors = [_factor("pagerank", 1.4),
                   _factor("fan_in_concentration", 0.9)]
        response = build_node_response(
            node_id="suspect@upi",
            risk_score=0.95,
            threshold=0.5,
            break_even_probability=BREAK_EVEN,
            contributing_factors=factors,
        )
        assert response.risk_level in (RiskLevel.HIGH, RiskLevel.CRITICAL)
        assert response.action == ActionCode.HOLD_FOR_REVIEW
        # The factors are the account's own, passed straight through — not a
        # global top-3 list. Two in, two out, same features.
        assert len(response.contributing_factors) == 2
        assert [f.feature for f in response.contributing_factors] == [
            "pagerank", "fan_in_concentration"]

    def test_low_score_allows_and_needs_no_factors(self):
        response = build_node_response(
            "legit@upi", 0.05, threshold=0.5,
            break_even_probability=BREAK_EVEN)
        assert response.risk_level == RiskLevel.LOW
        assert response.action == ActionCode.ALLOW
        assert response.contributing_factors == []


# ══════════════════════════════════════════════════════════════════
# 4. Batch validation catches what a naive screen would not
# ══════════════════════════════════════════════════════════════════

class TestBatchValidationCatchesForgery:
    """
    The non-tautological half. These responses are NOT built by
    `build_node_response`; they are hand-forged with the score and the action out
    of step, exactly the shape a bug elsewhere would produce.
    """

    def test_batch_of_honest_responses_passes(self):
        responses = [
            build_node_response("a@upi", 0.9, threshold=0.5,
                                break_even_probability=BREAK_EVEN),
            build_node_response("b@upi", 0.1, threshold=0.5,
                                break_even_probability=BREAK_EVEN),
            build_node_response("c@upi", 0.5, threshold=0.5,
                                break_even_probability=BREAK_EVEN),
        ]
        assert len(validate_response_batch(responses, 0.5, BREAK_EVEN)) == 3

    def test_a_critical_score_wearing_an_allow_is_rejected(self):
        """
        The alert-suppression bug the validator exists to stop. ALLOW is a
        permitted action, so an action screen alone lets this through; only
        re-deriving the tier from the score catches it.
        """
        forged = NodeRiskScore(
            node_id="mule@upi",
            risk_score=0.96,                 # CRITICAL at threshold 0.5
            risk_level=RiskLevel.LOW,        # ← downgraded
            action=ActionCode.ALLOW,         # ← and waved through
            contributing_factors=[],
            seen_in_context=False,
        )
        with pytest.raises(DefenseOnlyViolation, match="does not follow"):
            validate_response_batch([forged], 0.5, BREAK_EVEN)

    def test_a_correct_tier_with_a_downgraded_action_is_rejected(self):
        """
        Subtler: the TIER is right for the score, but the action was swapped for a
        weaker one. Screening the action passes (ALLOW is permitted); re-deriving
        the action from the tier is what catches the downgrade.
        """
        forged = NodeRiskScore(
            node_id="mule2@upi",
            risk_score=0.96,
            risk_level=RiskLevel.CRITICAL,   # correct for the score
            action=ActionCode.ALLOW,         # ← should be HOLD_FOR_REVIEW
            contributing_factors=[],
            seen_in_context=False,
        )
        with pytest.raises(DefenseOnlyViolation, match="does not follow"):
            validate_response_batch([forged], 0.5, BREAK_EVEN)

    def test_validation_is_evaluated_at_the_batch_threshold(self):
        """
        The same score sits in a different tier under a different operating
        threshold. A response tiered honestly at 0.5 (HIGH) must fail validation if
        the batch is re-checked at 0.95, where 0.6 falls to MEDIUM — proving the
        threshold argument is actually used, not ignored.
        """
        response = build_node_response("x@upi", 0.6, threshold=0.5,
                                       break_even_probability=BREAK_EVEN)
        validate_response_batch([response], 0.5, BREAK_EVEN)      # consistent
        with pytest.raises(DefenseOnlyViolation):
            validate_response_batch([response], 0.95, BREAK_EVEN)  # now MEDIUM

    def test_validation_is_evaluated_at_the_batch_break_even_floor(self):
        """
        The same argument for the other boundary, which nothing checked before it
        existed. A score of 0.2 is MEDIUM at p* = 0.0698 and LOW at p* = 0.5, so
        re-validating the same response under a different cost model must raise —
        otherwise the floor could be ignored inside the validator and no test would
        notice.
        """
        response = build_node_response("y@upi", 0.2, threshold=0.5,
                                       break_even_probability=BREAK_EVEN)
        assert response.risk_level == RiskLevel.MEDIUM
        validate_response_batch([response], 0.5, BREAK_EVEN)      # consistent
        with pytest.raises(DefenseOnlyViolation, match="does not follow"):
            validate_response_batch([response], 0.5, 0.5)
