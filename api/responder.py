"""
api/responder.py
────────────────
Enforces the "Defense-Only" constraint.

This module is the SINGLE gatekeeper between the ML model's risk score and the
action sent back to the caller. It guarantees:

  1. The API can NEVER output "BAN_USER" or any automated enforcement action.
  2. High-risk accounts are always routed to human review.
  3. Threshold-based tiering is applied consistently.

This is both an ethical safeguard and a regulatory requirement.

─────────────────────────────────────────────────────────────────────────────
v2 → v3 CHANGES
─────────────────────────────────────────────────────────────────────────────
1. `classify_risk` NO LONGER RATES A ZERO-RISK ACCOUNT AS CRITICAL.
   The old first branch was `risk_score >= min(threshold * 1.5, 0.95)`. At
   threshold 0 that becomes `risk_score >= 0`, which is true for every account
   ever scored — including one the model gave 0.0. Verified against the old code:
   `classify_risk(0.0, 0.0)` returned CRITICAL. The threshold is now required to
   be in (0, 1], and a caller who passes 0 gets an error instead of an inbox with
   every account in it marked CRITICAL.

2. CRITICAL IS ANCHORED TO CERTAINTY, NOT TO A MULTIPLE OF THE THRESHOLD.
   `threshold * 1.5` conflates two unrelated things. The operating threshold is
   an *economic* quantity — with FN at ₹200k and FP at ₹15k the break-even sits
   near 0.07, so 1.5x put CRITICAL at 0.105. Telling an analyst that a 10.5%
   probability is CRITICAL burns the word on cases the model is not remotely
   confident about, and an alert queue whose top tier is noise gets ignored,
   which costs recall in practice however good the offline numbers look.
   CRITICAL is now the midpoint between the operating threshold and certainty,
   `t + (1 - t)/2`: at t = 0.07 that is 0.535, so CRITICAL means the model
   genuinely thinks the account is more likely a mule than not. The rule stays
   monotone in t and well-defined across the whole range.

3. THE ACTION ENUM IS SWEPT AT IMPORT TIME.
   `_validate_action` only fired on an action actually produced, so a forbidden
   member added to `ActionCode` would sit dormant until the day it was returned.
   The whole enum is now checked when this module loads, so the process refuses
   to start instead of failing in production.

4. TIERS ARE COMPUTED ON THE SCORE AS REPORTED.
   Rounding after classifying let a score of 0.0697999 be reported as `0.0698`
   with tier MEDIUM while the displayed threshold was also `0.0698` — a response
   that contradicts itself on its face. Rounding now happens first.

─────────────────────────────────────────────────────────────────────────────
WHAT THE MEASURED METRICS DO AND DON'T COVER
─────────────────────────────────────────────────────────────────────────────
The precision and recall in metrics.json describe ONE binary decision: score >=
threshold, or not. They say nothing about the four tiers below.

Concretely, MEDIUM sits BELOW the threshold, so accounts in it are ones the model
declined to flag. Its step-up-auth action is a cheap hedge on the borderline, not
a measured detection, and its precision is necessarily worse than the reported
figure. Splitting the flagged side into HIGH and CRITICAL is likewise a triage
ordering for a human queue, not four separately validated classifiers. Anyone
quoting a per-tier number needs to measure that tier.
"""

from __future__ import annotations

from api.schemas import ActionCode, ContributingFactor, NodeRiskScore, RiskLevel


# ──────────────────────────────────────────────────────────────────
# Forbidden actions (hard-coded safety rail)
# ──────────────────────────────────────────────────────────────────

FORBIDDEN_ACTIONS = frozenset({
    "BAN_USER",
    "BLOCK_ACCOUNT",
    "FREEZE_FUNDS",
    "SUSPEND_ACCOUNT",
    "TERMINATE_USER",
    "AUTO_BLOCK",
    "AUTO_BAN",
})

# Substrings that betray an enforcement action even under a name the frozenset
# above doesn't list. A blocklist of exact strings only catches what someone
# thought of; `DISABLE_VPA` or `REVOKE_MANDATE` would sail straight through.
FORBIDDEN_VERBS = ("BAN", "BLOCK", "FREEZE", "SUSPEND", "TERMINATE",
                   "DISABLE", "REVOKE", "SEIZE", "CLOSE")

# Tier boundary for MEDIUM, as a fraction of the operating threshold. Accounts
# here scored below the threshold but close to it, so they get a cheap step-up
# rather than an analyst's time.
MEDIUM_BAND_FRACTION = 0.6


class DefenseOnlyViolation(Exception):
    """Raised if any code path attempts to output a forbidden action."""
    pass


def _validate_action(action_str: str) -> None:
    """
    Hard check: if any code path ever produces a forbidden action,
    raise an exception rather than returning it to the caller.
    """
    upper = action_str.upper()

    if upper in FORBIDDEN_ACTIONS:
        raise DefenseOnlyViolation(
            f"BLOCKED: Action '{action_str}' violates the defense-only "
            f"constraint. The Sentinel API cannot autonomously ban, block, or "
            f"freeze accounts. All enforcement must go through human review."
        )

    for verb in FORBIDDEN_VERBS:
        if verb in upper:
            raise DefenseOnlyViolation(
                f"BLOCKED: Action '{action_str}' reads as an enforcement action "
                f"(contains '{verb}'). The Sentinel recommends review; it does "
                f"not act on accounts. If this action really is advisory, name "
                f"it so."
            )


def _assert_action_enum_is_defense_only() -> None:
    """
    Sweep every member of ActionCode at import time.

    The point of a safety rail is to fail before it matters. Checking only the
    action on its way out means a forbidden member added to the enum stays
    invisible until the first account unlucky enough to trigger it — in
    production, mid-request. This makes the process refuse to start.
    """
    for member in ActionCode:
        _validate_action(member.value)


_assert_action_enum_is_defense_only()


# ──────────────────────────────────────────────────────────────────
# Risk tiering
# ──────────────────────────────────────────────────────────────────

def critical_cutoff(threshold: float) -> float:
    """
    Score at or above which an account is CRITICAL: halfway from the operating
    threshold to certainty.

    Exposed rather than inlined so the dashboard can draw the same boundary the
    API applies, instead of a second copy that drifts.
    """
    return threshold + (1.0 - threshold) / 2.0


def classify_risk(risk_score: float, threshold: float) -> RiskLevel:
    """
    Map a continuous risk score to a discrete risk level.

      CRITICAL: score >= threshold + (1 - threshold)/2   (model is confident)
      HIGH:     score >= threshold                       (flagged: above operating point)
      MEDIUM:   score >= threshold * 0.6                 (borderline, below operating point)
      LOW:      everything else

    `threshold` must be in (0, 1]. Zero is rejected because every non-negative
    score satisfies `>= 0`, so a zero threshold silently rates the entire
    population CRITICAL — which is exactly what the previous version did.
    """
    if not (0.0 < threshold <= 1.0):
        raise ValueError(
            f"threshold must be in (0, 1]; got {threshold!r}. "
            "A threshold of 0 flags every account including zero-risk ones, and "
            "a threshold above 1 flags none, so neither describes an operating "
            "point."
        )
    if not (0.0 <= risk_score <= 1.0):
        raise ValueError(
            f"risk_score must be a probability in [0, 1]; got {risk_score!r}. "
            "If this came from a model, it is a raw margin rather than a "
            "probability."
        )

    if risk_score >= critical_cutoff(threshold):
        return RiskLevel.CRITICAL
    if risk_score >= threshold:
        return RiskLevel.HIGH
    if risk_score >= threshold * MEDIUM_BAND_FRACTION:
        return RiskLevel.MEDIUM
    return RiskLevel.LOW


def determine_action(risk_level: RiskLevel) -> ActionCode:
    """
    Map a risk level to a DEFENSE-ONLY action.

    CRITICAL / HIGH → HOLD_FOR_REVIEW (requires human analyst)
    MEDIUM          → REQUIRE_ADDITIONAL_AUTH (step-up auth, no block)
    LOW             → ALLOW
    """
    mapping = {
        RiskLevel.CRITICAL: ActionCode.HOLD_FOR_REVIEW,
        RiskLevel.HIGH: ActionCode.HOLD_FOR_REVIEW,
        RiskLevel.MEDIUM: ActionCode.REQUIRE_ADDITIONAL_AUTH,
        RiskLevel.LOW: ActionCode.ALLOW,
    }

    if risk_level not in mapping:
        # Unreachable while RiskLevel has four members, but a fifth added without
        # a mapping entry must not fall through to a default that permits.
        raise DefenseOnlyViolation(
            f"Risk level {risk_level!r} has no action mapping. Refusing to "
            f"guess: defaulting either way is wrong."
        )

    action = mapping[risk_level]

    # SAFETY RAIL: Validate the action is not in the forbidden set
    _validate_action(action.value)

    return action


# ──────────────────────────────────────────────────────────────────
# Response Builder
# ──────────────────────────────────────────────────────────────────

def build_node_response(
    node_id: str,
    risk_score: float,
    threshold: float,
    contributing_factors: list[ContributingFactor] | None = None,
    seen_in_context: bool = False,
) -> NodeRiskScore:
    """
    Build a complete, validated response for a single node.
    This is the ONLY function external code should call.

    `contributing_factors` are per-account SHAP attributions from
    models/explain.py. The parameter was `top_features: list[str]`, which
    api/main.py filled with the model's three highest global gains — identical on
    every alert, so it described the model rather than the account.

    Rounding happens before classification so the tier can never contradict the
    score printed next to it.
    """
    reported_score = round(float(risk_score), 6)
    risk_level = classify_risk(reported_score, threshold)
    action = determine_action(risk_level)

    return NodeRiskScore(
        node_id=node_id,
        risk_score=reported_score,
        risk_level=risk_level,
        action=action,
        contributing_factors=list(contributing_factors or []),
        seen_in_context=seen_in_context,
    )


def validate_response_batch(
    responses: list[NodeRiskScore],
    threshold: float,
) -> list[NodeRiskScore]:
    """
    Final validation pass on an entire batch of responses.

    Checks two things, not one. Beyond re-screening each action against the
    forbidden set, it recomputes the tier and action from the score and threshold
    and requires a match — otherwise a response assembled anywhere other than
    `build_node_response` could carry an ALLOW on a CRITICAL score, and the
    gatekeeper would wave it through because ALLOW is a permitted value.
    """
    for response in responses:
        _validate_action(response.action.value)

        expected_level = classify_risk(response.risk_score, threshold)
        if response.risk_level != expected_level:
            raise DefenseOnlyViolation(
                f"Account {response.node_id}: risk_level "
                f"{response.risk_level.value} does not follow from score "
                f"{response.risk_score} at threshold {threshold} "
                f"(expected {expected_level.value}). The tiering was bypassed."
            )

        expected_action = determine_action(expected_level)
        if response.action != expected_action:
            raise DefenseOnlyViolation(
                f"Account {response.node_id}: action {response.action.value} "
                f"does not follow from risk level {expected_level.value} "
                f"(expected {expected_action.value}). A response reaching the "
                f"caller with a downgraded action would suppress a real alert."
            )

    return responses
