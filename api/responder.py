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

5. THE ACTION RAIL IS AN ALLOWLIST, NOT A BLOCKLIST.
   The blocklist leaked. `LOCK_ACCOUNT` passed the exact-name set, because it is
   not in it, and passed the substring scan too, because the listed verb is
   "BLOCK" and "BLOCK" is not a substring of "LOCK_ACCOUNT". The single guarantee
   this module exists to make was therefore already broken by a name any reviewer
   would recognise on sight. No list of banned verbs can enumerate every way of
   saying "stop this account" in advance, so the rail now fails CLOSED:
   `PERMITTED_ACTIONS` names the only strings that may leave this service and
   everything else is refused. The blocklists are kept behind it as a second,
   independent check, with the verbs the leak exposed added to them.

6. THE MEDIUM FLOOR COMES FROM THE COST MODEL, NOT FROM A FRACTION.
   The floor was `threshold * 0.6`, which at the shipped threshold of 0.5908 sits
   at 0.3545. But the project's own cost model prices an account as worth
   attention above the break-even probability p* = fp/(fp + fn) = 0.0698, and the
   test-oracle cutoff is 0.2665 — both well inside the band the old floor called
   LOW. So the API answered ALLOW on scores its own economics called reviewable.
   The floor is now p* itself, passed in alongside the threshold it comes from.
   Note what this does NOT do: p* is used as the queue yardstick the README
   describes, never as the alert cutoff. HOLD_FOR_REVIEW still begins at the
   empirically-selected operating threshold, because this model is deliberately
   uncalibrated and p* applied to an inflated score would alert on a large
   multiple of the accounts it should. A step-up challenge is the cheap end of
   that distinction; an analyst's time is not.

─────────────────────────────────────────────────────────────────────────────
WHAT THE MEASURED METRICS DO AND DON'T COVER
─────────────────────────────────────────────────────────────────────────────
The precision and recall in metrics.json describe ONE binary decision: score >=
threshold, or not. They say nothing about the four tiers below.

Concretely, MEDIUM sits BELOW the threshold, so accounts in it are ones the model
declined to flag. It spans [p*, threshold) — on the shipped numbers 0.0698 to
0.5908, which is wide on purpose: every score in it is one the cost model says is
worth more than nothing and less than an analyst. Its step-up-auth action is a
cheap hedge, not a measured detection, and its precision is necessarily far worse
than the reported figure. Splitting the flagged side into HIGH and CRITICAL is
likewise a triage ordering for a human queue, not four separately validated
classifiers. Anyone quoting a per-tier number needs to measure that tier.
"""

from __future__ import annotations

from api.schemas import ActionCode, ContributingFactor, NodeRiskScore, RiskLevel


# ──────────────────────────────────────────────────────────────────
# Action rails
# ──────────────────────────────────────────────────────────────────

# THE PRIMARY RAIL. The only action strings that may ever leave this service —
# the four members of `ActionCode`, written out here by hand.
#
# An allowlist rather than a blocklist because a blocklist cannot enumerate every
# enforcement verb in advance, and this one demonstrably did not: `LOCK_ACCOUNT`
# was in neither FORBIDDEN_ACTIONS nor caught by the substring scan, since the
# banned verb is "BLOCK" and "BLOCK" is not a substring of "LOCK_ACCOUNT".
# Offense-capability is a disqualification criterion for this project, so the rail
# must fail closed: an action nobody anticipated is refused by default instead of
# permitted by default.
#
# Deliberately a hand-written copy and NOT `{m.value for m in ActionCode}`. A rail
# derived from the thing it is checking cannot fail. Written independently, it is
# what makes the import-time sweep below a real check on the enum — and the sweep
# requires the two to agree exactly, so this copy cannot quietly drift.
PERMITTED_ACTIONS = frozenset({
    "ALLOW",
    "FLAG_FOR_REVIEW",
    "HOLD_FOR_REVIEW",
    "REQUIRE_ADDITIONAL_AUTH",
})

# SECONDARY RAIL, belt and braces. Unreachable while PERMITTED_ACTIONS holds only
# the four advisory names above — and kept precisely for the case where it is not:
# these fire if someone widens the allowlist to admit an enforcement action, which
# is the one way past the primary rail.
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
# above doesn't list. LOCK, QUARANTINE, RESTRICT, HALT, BLACKLIST, DENY, LIMIT and
# REVERSE were all missing, which is how `LOCK_ACCOUNT` passed: the list is only
# ever as complete as the last person's imagination, which is the argument for the
# allowlist above rather than against keeping this.
FORBIDDEN_VERBS = ("BAN", "BLOCK", "FREEZE", "SUSPEND", "TERMINATE",
                   "DISABLE", "REVOKE", "SEIZE", "CLOSE", "LOCK",
                   "QUARANTINE", "RESTRICT", "HALT", "BLACKLIST", "DENY",
                   "LIMIT", "REVERSE")


class DefenseOnlyViolation(Exception):
    """Raised if any code path attempts to output a forbidden action."""
    pass


def _validate_action(action_str: str) -> None:
    """
    Hard check: refuse any action that is not one of the four permitted ones.

    Allowlist first, blocklists second. Both rails raise, and both are kept,
    because they fail in different directions: the allowlist catches every name
    nobody thought of, and the blocklists catch a permitted list someone widened.
    """
    upper = action_str.upper()

    if upper not in PERMITTED_ACTIONS:
        raise DefenseOnlyViolation(
            f"BLOCKED: Action '{action_str}' is not one of the permitted "
            f"defense-only actions {sorted(PERMITTED_ACTIONS)}. The Sentinel "
            f"recommends review; it does not act on accounts. Anything else — "
            f"including an enforcement action under a name no blocklist "
            f"anticipated — is refused here rather than returned to the caller."
        )

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

    Members are screened first, then the enum and PERMITTED_ACTIONS are required
    to match exactly. The second check is what keeps the hand-written allowlist
    honest: an entry left behind after an action was renamed makes the rail look
    like it is guarding a name that no longer exists.
    """
    for member in ActionCode:
        _validate_action(member.value)

    declared = {member.value for member in ActionCode}
    if declared != set(PERMITTED_ACTIONS):
        raise DefenseOnlyViolation(
            f"PERMITTED_ACTIONS and ActionCode have drifted apart: "
            f"only in the enum {sorted(declared - set(PERMITTED_ACTIONS))}, "
            f"only in the allowlist "
            f"{sorted(set(PERMITTED_ACTIONS) - declared)}. The allowlist is a "
            f"deliberate second copy, so it has to be updated with the enum — an "
            f"allowlist that no longer describes the enum is not evidence of "
            f"anything."
        )


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


def medium_cutoff(threshold: float, break_even_probability: float) -> float:
    """
    Score at or above which an account gets a step-up challenge: the cost model's
    break-even probability p* = fp_cost / (fp_cost + fn_cost).

    Above p*, the expected cost of acting on an account is below the expected cost
    of ignoring it, so it is worth *something* — and the cheapest something is a
    step-up, not an analyst. That is why the floor is p* and not a fraction of the
    threshold: `threshold * 0.6` was 0.3545 here, which returned ALLOW across a
    whole band this project's own economics price as reviewable.

    `min` because a caller may override the threshold to below p*, and a MEDIUM
    floor above the HIGH floor is not a band, it is a contradiction. Under such an
    override the band is simply empty and everything above the threshold is HIGH.
    """
    return min(break_even_probability, threshold)


def classify_risk(
    risk_score: float,
    threshold: float,
    break_even_probability: float,
) -> RiskLevel:
    """
    Map a continuous risk score to a discrete risk level.

      CRITICAL: score >= threshold + (1 - threshold)/2   (model is confident)
      HIGH:     score >= threshold                       (flagged: above operating point)
      MEDIUM:   score >= break-even p*                   (worth a step-up, per the cost model)
      LOW:      everything else

    `threshold` must be in (0, 1]. Zero is rejected because every non-negative
    score satisfies `>= 0`, so a zero threshold silently rates the entire
    population CRITICAL — which is exactly what the previous version did.

    `break_even_probability` is the queue yardstick from the cost model, and is
    required rather than defaulted: the tier boundary it sets is an economic
    statement, and a guessed value would put the ALLOW/step-up line somewhere
    nothing in this repo agrees with.
    """
    if not (0.0 < threshold <= 1.0):
        raise ValueError(
            f"threshold must be in (0, 1]; got {threshold!r}. "
            "A threshold of 0 flags every account including zero-risk ones, and "
            "a threshold above 1 flags none, so neither describes an operating "
            "point."
        )
    if not (0.0 < break_even_probability <= 1.0):
        raise ValueError(
            f"break_even_probability must be in (0, 1]; got "
            f"{break_even_probability!r}. It is fp_cost / (fp_cost + fn_cost), so "
            f"it is 0 only if a false positive is free — in which case there is "
            f"no cost model to speak of."
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
    if risk_score >= medium_cutoff(threshold, break_even_probability):
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
    break_even_probability: float,
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
    risk_level = classify_risk(reported_score, threshold, break_even_probability)
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
    break_even_probability: float,
) -> list[NodeRiskScore]:
    """
    Final validation pass on an entire batch of responses.

    Checks two things, not one. Beyond re-screening each action against the
    permitted set, it recomputes the tier and action from the score and threshold
    and requires a match — otherwise a response assembled anywhere other than
    `build_node_response` could carry an ALLOW on a CRITICAL score, and the
    gatekeeper would wave it through because ALLOW is a permitted value.

    The tier is re-derived at the same break-even floor the batch was built with,
    for the same reason it is re-derived at the same threshold: a boundary checked
    against a different number than the one applied proves nothing.
    """
    for response in responses:
        _validate_action(response.action.value)

        expected_level = classify_risk(
            response.risk_score, threshold, break_even_probability)
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
