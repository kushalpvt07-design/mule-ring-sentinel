"""
api/schemas.py
──────────────
Pydantic models for strict input/output typing on the Sentinel API.

All API contracts are defined here — no loose dicts anywhere in the codebase.

─────────────────────────────────────────────────────────────────────────────
v2 → v3 CHANGES
─────────────────────────────────────────────────────────────────────────────
1. `contributing_factors` IS NOW STRUCTURED, AND PER-ACCOUNT.
   It was `list[str]`, and api/main.py filled it with the model's top three
   GLOBAL features — the same three strings on every single alert, which tells an
   investigator nothing about the account in front of them. It is now a list of
   `ContributingFactor`, carrying the feature's value for this account, its
   signed SHAP contribution, and whether it pushed risk up or down.

2. SELF-TRANSFERS ARE ACTUALLY REJECTED.
   `TransactionEdge` carried a `no_self_transfer` field validator whose body was
   `return v` — it validated nothing, and could not have: a field validator on
   `sender` cannot see `receiver`. The check now runs as a model validator, where
   both fields are available. It matters beyond tidiness: a sender == receiver
   edge becomes a self-loop in the graph, and a self-loop inflates degree and
   makes `reciprocity` and `cycle_participation` describe a cycle of one.

3. TIMESTAMPS MUST BE COMPARABLE.
   Graph features are computed over a fixed-length observation window, so a
   request has to be locatable in time. Naive and timezone-aware datetimes cannot
   be compared without raising, so incoming timestamps are normalised to UTC
   here, once, rather than in the middle of a feature computation.

4. RESPONSES CARRY THEIR PROVENANCE.
   `ScoringContext` reports which model scored the batch, how much historical
   graph context was used, and the effective observation window — because a graph
   feature computed over a different window length than the model was trained on
   is silently wrong, and the caller deserves to be able to see that.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator


# ──────────────────────────────────────────────────────────────────
# Enums
# ──────────────────────────────────────────────────────────────────

class RiskLevel(str, Enum):
    """Risk classification tiers."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class ActionCode(str, Enum):
    """
    Allowed response actions.

    NOTE: There is intentionally NO 'BAN_USER' action, and no action anywhere in
    this enum blocks, freezes or terminates anything. The strongest thing the
    Sentinel can ask for is that a human look at the account. That is a hard
    product constraint, enforced again at runtime in api/responder.py.
    """
    ALLOW = "ALLOW"
    FLAG_FOR_REVIEW = "FLAG_FOR_REVIEW"
    HOLD_FOR_REVIEW = "HOLD_FOR_REVIEW"
    REQUIRE_ADDITIONAL_AUTH = "REQUIRE_ADDITIONAL_AUTH"


# ──────────────────────────────────────────────────────────────────
# Request Models
# ──────────────────────────────────────────────────────────────────

class TransactionEdge(BaseModel):
    """A single UPI transaction to be scored."""
    sender: str = Field(
        ...,
        min_length=3,
        max_length=50,
        description="UPI VPA of the sender",
        examples=["user123@upi"],
    )
    receiver: str = Field(
        ...,
        min_length=3,
        max_length=50,
        description="UPI VPA of the receiver",
        examples=["merchant456@upi"],
    )
    amount: float = Field(
        ...,
        gt=0,
        le=10_000_000,
        description="Transaction amount in INR",
        examples=[25000.0],
    )
    timestamp: datetime = Field(
        ...,
        description="ISO 8601 timestamp of the transaction",
        examples=["2025-06-15T14:30:00"],
    )

    @field_validator("timestamp")
    @classmethod
    def normalise_to_utc(cls, v: datetime) -> datetime:
        """
        Make every timestamp tz-aware UTC.

        Mixing naive and aware datetimes raises `TypeError` on comparison, and the
        comparison happens deep inside the observation-window trim where the
        traceback would be unhelpful. A naive timestamp is read as UTC.
        """
        return v.replace(tzinfo=timezone.utc) if v.tzinfo is None \
            else v.astimezone(timezone.utc)

    @model_validator(mode="after")
    def reject_self_transfer(self) -> "TransactionEdge":
        """
        A transaction cannot have the same account on both sides.

        Not merely invalid input: a self-loop inflates both degrees, and makes
        `reciprocity` and `cycle_participation` report a one-account cycle, so it
        would corrupt the graph features of a genuinely uninvolved account.
        """
        if self.sender == self.receiver:
            raise ValueError(
                f"sender and receiver are the same account ('{self.sender}'). "
                "A self-transfer is not a transfer between two parties, and as a "
                "graph self-loop it would distort the account's degree and cycle "
                "features."
            )
        return self


class ScoringRequest(BaseModel):
    """Batch scoring request — score one or more transactions."""
    transactions: list[TransactionEdge] = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="List of transactions to score",
    )
    threshold_override: float | None = Field(
        default=None,
        gt=0.0,
        lt=1.0,
        description=(
            "Optional: override the decision threshold. Must be strictly inside "
            "(0, 1). A threshold of exactly 0 flags every account including "
            "zero-risk ones; a threshold of exactly 1 is the mirror image and was "
            "admitted until recently. At t = 1 the CRITICAL cutoff is "
            "t + (1 - t)/2 = 1, so the HIGH band [1, 1) is empty and the only "
            "account that can be flagged at all is one scored at exactly 1.0. "
            "Neither endpoint describes an operating point, so neither is accepted "
            "here. `api/responder.py` still admits t = 1 because the tiering "
            "arithmetic is well-defined there; this is the caller-facing surface, "
            "and it is stricter on purpose."
        ),
    )
    include_context_accounts: bool = Field(
        default=False,
        description=(
            "If true, also return scores for accounts that appear only in the "
            "historical context graph. Off by default: a caller asking about 4 "
            "transactions wants 4 accounts back, not the whole ledger."
        ),
    )


# ──────────────────────────────────────────────────────────────────
# Response Models
# ──────────────────────────────────────────────────────────────────

class ContributingFactor(BaseModel):
    """
    One reason this specific account scored the way it did.

    From exact TreeSHAP, so the contributions across all features sum to the
    account's own log-odds margin. That is what makes this an explanation of the
    score rather than a description of the model — and models/train.py asserts
    the identity on the test split every run, so it cannot quietly drift.
    """
    feature: str = Field(..., description="Feature name, as in FEATURE_COLS")
    description: str = Field(
        ..., description="Plain-English meaning, for an analyst")
    value: float = Field(..., description="This account's value for the feature")
    contribution: float = Field(
        ...,
        description=(
            "Signed SHAP contribution in log-odds. Positive raises risk. "
            "All 18 contributions plus the model bias sum to this account's "
            "raw margin."
        ),
    )
    effect: Literal["raises_risk", "lowers_risk"] = Field(
        ..., description="Direction of this factor's influence")


class NodeRiskScore(BaseModel):
    """Risk assessment for a single account (node)."""
    node_id: str = Field(..., description="UPI VPA of the scored account")
    risk_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Probability of being a mule account [0, 1]",
    )
    risk_level: RiskLevel = Field(..., description="Classified risk tier")
    action: ActionCode = Field(..., description="Recommended action")
    contributing_factors: list[ContributingFactor] = Field(
        default_factory=list,
        description=(
            "Per-account SHAP attributions, risk-raising factors first. Empty "
            "only if attribution was unavailable."
        ),
    )
    seen_in_context: bool = Field(
        default=False,
        description=(
            "True if this account already had history in the observation "
            "window. False means it is new, so its graph features rest on the "
            "submitted transactions alone and the score is weaker evidence."
        ),
    )


class ScoringContext(BaseModel):
    """
    How this batch was scored — the provenance a reviewer needs.

    Present because graph features are only comparable to training if they were
    computed over a comparable window on a comparable graph. Without these
    fields, a score computed on a 4-edge graph is indistinguishable from one
    computed on the full ledger, and only one of them means anything.
    """
    model_version: str = Field(..., description="Model that produced the scores")
    partition_fingerprint: str | None = Field(
        default=None,
        description=(
            "The frozen community partition these scores were computed against. "
            "Two identical requests scored under different partitions can differ, "
            "so an alert is only reproducible together with this value."
        ),
    )
    threshold_source: Literal["metrics.json", "request_override"] = Field(
        ..., description="Where the decision threshold came from")
    break_even_probability: float | None = Field(
        default=None,
        description=(
            "The cost model's break-even probability p* = fp_cost / (fp_cost + "
            "fn_cost), which is the floor of the MEDIUM/step-up band. Published "
            "because it is the second number the tiering depends on and the "
            "response was previously unreadable without it: given only "
            "`threshold_used`, a caller could reconstruct the CRITICAL cutoff "
            "(t + (1 - t)/2) but had no way to derive where ALLOW ends and "
            "REQUIRE_ADDITIONAL_AUTH begins. It is NOT the alert cutoff — alerts "
            "still start at `threshold_used`."
        ),
    )
    context_provenance: str | None = Field(
        default=None,
        description=(
            "What the historical context graph is, stated plainly. On the shipped "
            "artefacts it is the validation split, edge for edge — so an account "
            "scored here against its own validation history is being scored "
            "in-sample with respect to the graph, and the precision and recall in "
            "the README are the honest out-of-sample numbers rather than anything "
            "measurable from this endpoint. Carried in the response so a reviewer "
            "reading one JSON body does not have to know that."
        ),
    )
    n_submitted_transactions: int = Field(...)
    n_context_transactions_used: int = Field(
        ..., description="Historical transactions included in the graph")
    n_duplicates_dropped: int = Field(
        default=0,
        description=(
            "Submitted transactions that exactly replayed a context transaction "
            "(same sender, receiver, amount and timestamp) and were counted once "
            "rather than twice. Non-zero means the request was redundant, not "
            "that anything was mis-scored."
        ),
    )
    n_nodes_in_graph: int = Field(
        ..., description="Accounts in the merged graph features were computed on")
    observation_window_days: float = Field(
        ..., description="Span of the merged edge set actually used")
    trained_window_days: float | None = Field(
        default=None,
        description="Span the model was trained on, for comparison")
    window_comparable: bool = Field(
        ...,
        description=(
            "False if the effective window differs enough from training that "
            "magnitude features are on a different scale, making scores "
            "unreliable. The features measured to scale with window length are "
            "in_amount_sum, out_amount_sum and repeat_ratio (2.00x, 2.00x and "
            "1.84x when the window doubles). `txn_velocity` is NOT one of them "
            "despite being named here previously — it divides by the account's "
            "own active span, so it is a rate."
        ),
    )
    warnings: list[str] = Field(default_factory=list)


class ScoringResponse(BaseModel):
    """Response from the scoring endpoint."""
    request_id: str = Field(..., description="Unique request identifier")
    scored_at: datetime = Field(..., description="Timestamp of scoring")
    threshold_used: float = Field(..., description="Decision threshold applied")
    node_scores: list[NodeRiskScore] = Field(
        ...,
        description="Risk scores for each unique account in the request",
    )
    context: ScoringContext = Field(
        ..., description="Provenance and graph context for this batch")
    summary: dict = Field(
        default_factory=dict,
        description="Aggregate statistics for the batch",
    )


class HealthResponse(BaseModel):
    """Health check response."""
    status: Literal["healthy", "degraded"] = "degraded"
    model_loaded: bool = False
    model_version: str = "unknown"
    model_file: str | None = None
    optimal_threshold: float | None = Field(
        default=None,
        description=(
            "The operating threshold in force, or None when the service could not "
            "read one and is therefore refusing to score. Optional rather than "
            "defaulted, because every concrete default is a lie in the degraded "
            "case: 0.0 names the one value `classify_risk` rejects outright (it "
            "flags every account), and 0.5 is a plausible-looking number that a "
            "reader would take for a real operating point. `api/main.py` passes "
            "`STATE.threshold`, which is `float | None`, so a non-optional "
            "annotation here also made /health raise a ValidationError in exactly "
            "the degraded state /health exists to report."
        ),
    )
    threshold_source: str = "default"
    n_features: int = 0
    feature_contract_verified: bool = Field(
        default=False,
        description=(
            "True only if the loaded booster's feature names match "
            "models/features.py exactly, in order. The service refuses to score "
            "when this is false."
        ),
    )
    context_graph_loaded: bool = False
    context_transactions: int = 0
    partition_fingerprint: str | None = Field(
        default=None,
        description=(
            "Label-invariant fingerprint of the frozen community partition used "
            "for community_internal_ratio, as 'hash/Nc/Mn'. Two replicas or two "
            "restarts reporting different values would score the same account "
            "differently, so this is published to make that visible."
        ),
    )
    detail: str | None = Field(
        default=None, description="Why the service is degraded, if it is")
