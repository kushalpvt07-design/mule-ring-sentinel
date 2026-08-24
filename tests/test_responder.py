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

Everything here needs pydantic (api.schemas is a pydantic model) but no model,
data or network — so it runs on a bare checkout the moment pydantic is installed.

Usage:
    pytest tests/test_responder.py -v
"""

from __future__ import annotations

import pytest

# api.schemas is built on pydantic; importing api.responder pulls it in. Skip the
# whole module cleanly where pydantic is absent rather than erroring at collection.
pytest.importorskip("pydantic", reason="api.schemas is a pydantic model")

from api.responder import (  # noqa: E402  (after importorskip, intentionally)
    FORBIDDEN_ACTIONS,
    FORBIDDEN_VERBS,
    DefenseOnlyViolation,
    _validate_action,
    build_node_response,
    classify_risk,
    critical_cutoff,
    determine_action,
    validate_response_batch,
)
from api.schemas import ActionCode, ContributingFactor, NodeRiskScore, RiskLevel  # noqa: E402


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

    @pytest.mark.parametrize("name", ["BAN_USER", "BLOCK_ACCOUNT", "FREEZE_FUNDS"])
    def test_named_enforcement_actions_are_forbidden(self, name):
        assert name in FORBIDDEN_ACTIONS

    @pytest.mark.parametrize("forbidden", sorted(FORBIDDEN_ACTIONS))
    def test_every_listed_forbidden_action_raises(self, forbidden):
        with pytest.raises(DefenseOnlyViolation):
            _validate_action(forbidden)

    def test_forbidden_check_is_case_insensitive(self):
        for spelling in ("ban_user", "Ban_User", "bAn_UsEr"):
            with pytest.raises(DefenseOnlyViolation):
                _validate_action(spelling)

    @pytest.mark.parametrize("verb", FORBIDDEN_VERBS)
    def test_an_unlisted_action_that_reads_as_enforcement_is_caught(self, verb):
        """
        The blocklist only catches names someone thought of. `DISABLE_VPA` or
        `REVOKE_MANDATE` are not in FORBIDDEN_ACTIONS, yet they are enforcement.
        The substring rail catches them by verb — this is the check that makes
        the guarantee hold against actions nobody has invented yet.
        """
        invented = f"{verb}_SOMETHING"
        assert invented not in FORBIDDEN_ACTIONS
        with pytest.raises(DefenseOnlyViolation, match="enforcement"):
            _validate_action(invented)

    def test_every_action_in_the_enum_passes(self):
        """
        The enum is swept at import time, so this cannot even reach a failing
        assert — importing api.responder would already have raised. Kept as the
        readable statement of intent: no shipped ActionCode is an enforcement verb.
        """
        for action in ActionCode:
            _validate_action(action.value)

    def test_no_score_anywhere_produces_a_forbidden_action(self):
        """The end-to-end guarantee across the whole score range and both edges."""
        for score in (0.0, 0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99, 1.0):
            response = build_node_response("test@upi", score, threshold=0.5)
            assert response.action.value not in FORBIDDEN_ACTIONS
            for verb in FORBIDDEN_VERBS:
                assert verb not in response.action.value.upper()


# ══════════════════════════════════════════════════════════════════
# 2. Tiering: the boundaries, and the two regressions v3 fixed
# ══════════════════════════════════════════════════════════════════

class TestRiskTiering:
    """
    Tiers are threshold-relative. CRITICAL is the midpoint from the operating
    threshold to certainty, not a fixed 0.85 and not `threshold * 1.5`.
    """

    @pytest.mark.parametrize("score,expected", [
        (0.05, RiskLevel.LOW),       # below 0.6 * threshold
        (0.20, RiskLevel.LOW),
        (0.35, RiskLevel.MEDIUM),    # in [0.30, 0.50)
        (0.50, RiskLevel.HIGH),      # in [0.50, 0.75)
        (0.85, RiskLevel.CRITICAL),  # >= 0.75 = 0.5 + (1-0.5)/2
        (0.99, RiskLevel.CRITICAL),
    ])
    def test_classification_at_threshold_half(self, score, expected):
        assert classify_risk(score, threshold=0.5) == expected

    def test_critical_cutoff_is_the_midpoint_to_certainty(self):
        for t in (0.07, 0.2, 0.5, 0.9):
            assert critical_cutoff(t) == pytest.approx(t + (1.0 - t) / 2.0)

    def test_critical_is_monotone_in_the_threshold(self):
        cutoffs = [critical_cutoff(t) for t in (0.05, 0.1, 0.3, 0.6, 0.9)]
        assert cutoffs == sorted(cutoffs)

    def test_a_zero_threshold_is_rejected_not_all_critical(self):
        """
        THE REGRESSION. The old first branch was `score >= min(threshold*1.5,
        0.95)`; at threshold 0 that is `score >= 0`, true for every account —
        `classify_risk(0.0, 0.0)` returned CRITICAL. A zero threshold is now an
        error, so it can never again mark the entire population CRITICAL.
        """
        with pytest.raises(ValueError, match="threshold"):
            classify_risk(0.0, threshold=0.0)

    @pytest.mark.parametrize("bad", [-0.1, 1.5, 2.0])
    def test_thresholds_outside_the_unit_interval_are_rejected(self, bad):
        with pytest.raises(ValueError, match="threshold"):
            classify_risk(0.5, threshold=bad)

    @pytest.mark.parametrize("bad", [-0.01, 1.01, 5.0])
    def test_scores_outside_the_unit_interval_are_rejected(self, bad):
        """A raw margin fed in as a probability is a bug worth catching loudly."""
        with pytest.raises(ValueError, match="risk_score"):
            classify_risk(bad, threshold=0.5)

    def test_the_reported_score_and_its_tier_cannot_contradict(self):
        """
        v3 change 4: rounding happens BEFORE classification. A score of 0.0697999
        must not be reported as 0.0698 while its tier was computed from the longer
        value. Build a response near a tier boundary and confirm the tier follows
        from the number actually reported.
        """
        threshold = 0.0698
        response = build_node_response("edge@upi", 0.06979994, threshold=threshold)
        assert response.risk_score == round(0.06979994, 6)
        assert response.risk_level == classify_risk(response.risk_score, threshold)


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
            contributing_factors=factors,
        )
        assert response.risk_level in (RiskLevel.HIGH, RiskLevel.CRITICAL)
        assert response.action == ActionCode.HOLD_FOR_REVIEW
        assert response.action.value not in FORBIDDEN_ACTIONS
        # The factors are the account's own, passed straight through — not a
        # global top-3 list. Two in, two out, same features.
        assert len(response.contributing_factors) == 2
        assert [f.feature for f in response.contributing_factors] == [
            "pagerank", "fan_in_concentration"]

    def test_low_score_allows_and_needs_no_factors(self):
        response = build_node_response("legit@upi", 0.05, threshold=0.5)
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
            build_node_response("a@upi", 0.9, threshold=0.5),
            build_node_response("b@upi", 0.1, threshold=0.5),
            build_node_response("c@upi", 0.5, threshold=0.5),
        ]
        assert len(validate_response_batch(responses, threshold=0.5)) == 3

    def test_a_critical_score_wearing_an_allow_is_rejected(self):
        """
        The alert-suppression bug the validator exists to stop. ALLOW is a
        permitted action, so a forbidden-action screen alone lets this through;
        only re-deriving the tier from the score catches it.
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
            validate_response_batch([forged], threshold=0.5)

    def test_a_correct_tier_with_a_downgraded_action_is_rejected(self):
        """
        Subtler: the TIER is right for the score, but the action was swapped for a
        weaker one. Screening actions against the forbidden set passes (ALLOW is
        fine); re-deriving the action from the tier is what catches the downgrade.
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
            validate_response_batch([forged], threshold=0.5)

    def test_validation_is_evaluated_at_the_batch_threshold(self):
        """
        The same score sits in a different tier under a different operating
        threshold. A response tiered honestly at 0.5 (HIGH) must fail validation if
        the batch is re-checked at 0.95, where 0.6 falls to MEDIUM — proving the
        threshold argument is actually used, not ignored.
        """
        response = build_node_response("x@upi", 0.6, threshold=0.5)  # HIGH at 0.5
        validate_response_batch([response], threshold=0.5)           # consistent
        with pytest.raises(DefenseOnlyViolation):
            validate_response_batch([response], threshold=0.95)      # now MEDIUM
