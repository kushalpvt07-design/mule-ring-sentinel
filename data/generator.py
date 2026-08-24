"""
data/generator.py
─────────────────
Generates the synthetic UPI transaction graph the whole project is measured on.

This file decides what every reported number means. If the generator plants a
signal, the model finds it, the metrics come out at 0.99, and the evaluation is
worthless — worse than worthless, because it looks like success. So the design
goal here is not "produce fraud that is detectable"; it is **produce legitimate
behaviour that is hard to tell apart from fraud**, and let the model earn
whatever separation it gets.

─────────────────────────────────────────────────────────────────────────────
WHAT THIS EMITS
─────────────────────────────────────────────────────────────────────────────
data/raw/train_edges.csv, val_edges.csv, test_edges.csv
data/raw/serving_context_edges.csv   (the val window — the API's lookback)

Columns: sender, receiver, amount, timestamp, is_mule, edge_role, ring_id,
         ring_type, split

`edge_role` is the label-critical column. data/extractor.py labels a node
positive iff it is an endpoint of an `edge_role == "ring"` edge. Everything
else — including accounts that unwittingly paid *into* a ring — stays 0.

The three splits are contiguous, non-overlapping and EQUAL LENGTH. Equal length
is load-bearing, not tidiness: see SPLIT_FRACTIONS below.

─────────────────────────────────────────────────────────────────────────────
v1 → v2: THE SPLIT DEFECT (fixed previously, kept here for the record)
─────────────────────────────────────────────────────────────────────────────
v1 generated every ring across the full 6-month range and then sliced the
*edge list* by time. Features and labels are per *node*, so all 25 rings landed
in both train and test: 73% of test mule nodes were also train mule nodes, and
82% of test nodes had been seen in training. "Held-out test set" meant "the same
accounts, later in the month". tests/test_leakage.py passed because it only
checked timestamp ordering — the one leak that was never happening.

Now each ring is born, launders and dies inside one window, and the accounts a
ring may hijack or use as feeders come from a pool reserved for that window:

    ring_members(train) ∩ ring_members(val) ∩ ring_members(test) == ∅

Organic accounts still recur across windows. That is deliberate: real accounts
persist, and pretending otherwise would be its own distortion.

─────────────────────────────────────────────────────────────────────────────
v2 → v3: THE DIFFICULTY DEFECT (what this rewrite fixes)
─────────────────────────────────────────────────────────────────────────────
v2 fixed leakage and then reported precision 0.98 / recall 0.97, which should
have been the tell. It was not leakage. It was that v2's organic traffic drew
every sender and receiver i.i.d. from a zipf popularity vector, so a legitimate
account essentially never paid the same counterparty twice:

    organic repeat_ratio (txns per distinct counterparty): p50 1.00, p99 2.25
    ring    repeat_ratio:                                  p50 ~13

`repeat_ratio` alone scored test AUC 0.9989, and the one-line rule
`repeat_ratio >= 3.0` scored precision 0.936 / recall 0.924 — within noise of
the XGBoost model. The model was not learning laundering topology. It was
learning that the generator forgot real people have habits.

Every fix below exists to remove one such shortcut. Each organic pattern here
is chosen specifically because it is the *legitimate* look-alike of a feature
in models/features.py:

    feature it defends against   organic look-alike introduced here
    ──────────────────────────   ─────────────────────────────────────────────
    repeat_ratio                 recurring counterparties: rent, EMI, subs,
                                 salary, and a daily-kirana relationship at
                                 3-8 day cadence
    amount_cv                    monthly obligations bill the *exact* same
                                 amount every cycle → naturally low CV
    counterparty_amount_cv       trader settlement groups pay each partner
                                 near-identical sums
    flow_passthrough             gig workers receive payouts and forward
                                 almost all of it
    fan_in_concentration         merchant hubs collect from hundreds of payers
    in_degree / pagerank         the same merchant hubs
    burst_ratio                  payday clustering: a salary credit is followed
                                 by several payments within hours
    reciprocity                  reciprocal social pairs (both directions)
    cycle_participation          organic B2B settlement cycles among traders —
                                 real money does go in circles
    community_internal_ratio     purpose-built mule accounts get camouflage
                                 traffic, so "no outside contacts" is not a
                                 free giveaway

And the rings are no longer one uniform shape. Three archetypes, with roughly
40% of positives deliberately *hard*:

    fast_cycle      loud: tight cycle, many near-identical repeats, one burst.
                    Should be easy. It is the case worth catching in minutes.
    stealth_cycle   low volume (2-5 txns per hop), ±30% amount jitter, spread
                    across the whole window, and 35% of them are a broken
                    cycle — a path, not a loop. Mostly hijacked real accounts.
    layered_fanin   collector with 15-40 feeders, then a 2-4 hop chain to a
                    cash-out account. Structurally near-identical to a
                    legitimate merchant with a settlement pipeline.

The intended consequence is that a single-rule baseline loses clearly and the
headline result becomes honest lift over a stated baseline, not a 0.99.
tests/test_baselines.py fails the build if any single feature reaches test
AUC >= 0.99, which is the automated form of this whole argument.

v3 also fixes a quieter defect that had nothing to do with difficulty: the
splits were 108 / 32 / 39 days long. Window length multiplies every count-,
sum- and rate-based feature, so the model fitted split points to a 108-day
scale and was then evaluated on a 39-day one — and the API, which rebuilt its
graph from train+val (140 days), was on a third scale again. All three windows
are now equal thirds, and the serving context is exactly one window.

Usage:
    python -m data.generator
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from console import banner, enable_utf8_stdout, hr, sym


# ══════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════

SEED = 42

NUM_ACCOUNTS = 3000

# Total organic volume is now *emergent* — it falls out of the relationship
# graph rather than being a knob. v2's NUM_ORGANIC_EDGES=30_000 is gone: fixing
# a target edge count is what forced i.i.d. pair sampling in the first place.
# This is the only remaining unstructured organic component.
ONE_OFF_EDGES = 10_000

TIME_START = datetime(2025, 1, 1)
TIME_END = datetime(2025, 6, 30)

# ── Three-way temporal split: contiguous, non-overlapping, EQUAL LENGTH ──
#
# The equal length is not cosmetic. Features are per-node counts, sums and
# rates computed over whatever window the edge file covers, so window length
# is a multiplier on almost every one of them. v2 used 60/18/22, giving windows
# of 108 / 32 / 39 days — so a train-split account showed roughly three times
# the in_degree, in_amount_sum and txn_velocity of the identical account in
# test, purely because the train window was three times longer. The model
# learned split points calibrated to a 108-day window and was then evaluated
# against 39-day features. That depresses measured performance and, worse,
# makes the threshold chosen on val (32 days) transfer incorrectly to test.
#
# With equal thirds every feature is directly comparable across splits, and
# the observation window becomes a fixed property of the system rather than an
# accident of the split ratio. tests/test_contract.py asserts it on the emitted
# CSVs; assert_equal_window_lengths() asserts it here at generation time.
SPLITS = ("train", "val", "test")
SPLIT_FRACTIONS = {"train": 1 / 3, "val": 1 / 3, "test": 1 / 3}

# Tolerance for the equal-length invariant, in days. Only absorbs the rounding
# the last window inherits when the total range is not divisible by three.
WINDOW_LENGTH_TOLERANCE_DAYS = 1.5

# Rings per split.
#
# val and test carry the SAME count deliberately. Threshold selection happens
# on val and is then applied unchanged to test (models/train.py), which is only
# valid if the two windows have comparable class balance — precision moves with
# base rate, so a threshold tuned at a 3.6% positive rate does not transfer to
# a 2.5% one. Train carries more because more labelled history is what you
# actually have in production.
#
# 24 test rings is roughly 110 positive accounts, which puts the smallest
# meaningful precision step near one percentage point. Fewer than that and the
# reported precision is quantisation noise.
RINGS_PER_SPLIT = {"train": 40, "val": 24, "test": 24}
NUM_MULE_RINGS = sum(RINGS_PER_SPLIT.values())  # derived, never hand-maintained

# Accounts eligible to be hijacked into a ring or used as a fan-in feeder,
# partitioned per split so no account can be mule-labelled twice. Sized for the
# worst case: a layered_fanin ring consumes up to 40 feeders.
HIJACK_POOL_SIZE = {"train": 800, "val": 550, "test": 550}

# Population mix. Percentages of NUM_ACCOUNTS.
ROLE_MIX = {
    "merchant": 0.045,   # fan-in hubs: kirana, restaurants, e-comm sellers
    "employer": 0.010,   # salary payers: fan-out hubs
    "landlord": 0.030,   # rent collectors: small fan-in, monthly, fixed amount
    "biller": 0.015,     # EMI / subscription / utility collectors
    "gig": 0.050,        # receive payouts, forward most of it out
    "trader": 0.040,     # B2B accounts that sit on settlement cycles
    # remainder → "consumer"
}

# Consumer activity tiers. Real payment activity is heavily skewed; modelling
# every consumer as equally busy both inflates volume and erases the long tail
# that the model has to survive.
ACTIVITY_MIX = {"light": 0.60, "medium": 0.30, "heavy": 0.10}

# Organic B2B settlement groups — traders paying each other in a loop. These
# exist purely so `cycle_participation` is not a free label.
SETTLEMENT_GROUPS = 60
SETTLEMENT_GROUP_SIZE = (3, 5)

# Reciprocal social pairs (both directions), the legitimate source of
# `reciprocity` and of short 2-cycles.
SOCIAL_PAIRS = 1_100

# Payday behaviour: after a salary credit, a burst of outbound payments.
PAYDAY_BURST_PROB = 0.35
PAYDAY_BURST_COUNT = (1, 4)
PAYDAY_BURST_WINDOW_H = 8

OUTPUT_DIR = Path(__file__).resolve().parent / "raw"

# Mule account prefixes, namespaced per split so they can never collide.
# retire_hijacked_accounts() uses these to tell a purpose-built mule (no
# organic life to freeze) from a hijacked real account.
MULE_PREFIX = {"train": "MULE_T", "val": "MULE_V", "test": "MULE_S"}
MULE_PREFIXES = tuple(MULE_PREFIX.values())

# Canonical column order for the emitted CSVs.
EDGE_COLUMNS = [
    "sender", "receiver", "amount", "timestamp",
    "is_mule", "edge_role", "ring_id", "ring_type", "split",
]

# Sentinels for non-ring edges.
NO_RING_ID = -1
NO_RING_TYPE = "organic"


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)


# ══════════════════════════════════════════════════════════════════
# Time helpers
# ══════════════════════════════════════════════════════════════════

# UPI activity is not uniform over the day. Weighting hours matters for
# `burst_ratio`: under a uniform hour distribution, every account's busiest hour
# holds ~1/24 of its traffic, which makes the feature trivially separable the
# moment rings burst. With a realistic diurnal shape, ordinary accounts already
# concentrate.
_HOUR_WEIGHTS = np.array([
    0.4, 0.2, 0.1, 0.1, 0.2, 0.5,   # 00-05 night
    1.2, 2.2, 3.4, 4.6, 5.2, 5.6,   # 06-11 morning
    5.4, 4.8, 4.4, 4.6, 5.0, 5.8,   # 12-17 afternoon
    6.6, 7.0, 6.2, 4.4, 2.6, 1.2,   # 18-23 evening peak
], dtype=float)
_HOUR_WEIGHTS /= _HOUR_WEIGHTS.sum()
_HOURS = np.arange(24)


def _daytime_hour() -> int:
    """An hour-of-day drawn from the diurnal UPI profile."""
    return int(np.random.choice(_HOURS, p=_HOUR_WEIGHTS))


def _random_timestamp(start: datetime, end: datetime) -> datetime:
    """A uniform random timestamp in [start, end)."""
    delta_s = int((end - start).total_seconds())
    if delta_s <= 0:
        return start
    return start + timedelta(seconds=random.randint(0, delta_s - 1))


def _diurnal_timestamp(start: datetime, end: datetime) -> datetime:
    """
    A random timestamp in [start, end) whose *hour* follows the diurnal profile.

    Used for one-off traffic so the unstructured component does not flatten the
    hour distribution that `burst_ratio` reads.
    """
    span_days = max(1, int((end - start).total_seconds() // 86400))
    day = start + timedelta(days=random.randrange(span_days))
    ts = day.replace(
        hour=_daytime_hour(),
        minute=random.randrange(60),
        second=random.randrange(60),
        microsecond=0,
    )
    return ts if start <= ts < end else _random_timestamp(start, end)


def build_split_windows(
    time_start: datetime,
    time_end: datetime,
) -> dict[str, tuple[datetime, datetime]]:
    """
    Carve the full range into contiguous train/val/test windows.

    Returns {split: (start, end)} with train.end == val.start and
    val.end == test.start.
    """
    total_s = (time_end - time_start).total_seconds()
    windows: dict[str, tuple[datetime, datetime]] = {}

    cursor = time_start
    for i, split in enumerate(SPLITS):
        if i == len(SPLITS) - 1:
            w_end = time_end  # absorb rounding into the last window
        else:
            w_end = cursor + timedelta(seconds=total_s * SPLIT_FRACTIONS[split])
        windows[split] = (cursor, w_end)
        cursor = w_end

    return windows


def split_for_timestamp(
    ts: datetime,
    windows: dict[str, tuple[datetime, datetime]],
) -> str:
    """Assign one timestamp to its split. Boundaries belong to the later split."""
    for split in SPLITS:
        start, end = windows[split]
        if start <= ts < end:
            return split
    return SPLITS[-1]  # exactly == time_end


def assign_splits_vectorised(
    timestamps: pd.Series,
    windows: dict[str, tuple[datetime, datetime]],
) -> np.ndarray:
    """
    Vectorised equivalent of split_for_timestamp over a whole column.

    searchsorted(side="right") on the two interior boundaries reproduces the
    "boundary belongs to the later split" rule exactly, and does it in one pass
    instead of ~160k Python calls.
    """
    interior = np.array(
        [windows["train"][1], windows["val"][1]], dtype="datetime64[ns]"
    )
    idx = np.searchsorted(interior, timestamps.values, side="right")
    return np.asarray(SPLITS, dtype=object)[idx]


# ══════════════════════════════════════════════════════════════════
# The account universe
# ══════════════════════════════════════════════════════════════════

@dataclass
class Population:
    """Accounts grouped by behavioural archetype."""
    all_accounts: list[str]
    by_role: dict[str, list[str]]
    role_of: dict[str, str]
    activity_of: dict[str, str] = field(default_factory=dict)


def build_population(num_accounts: int = NUM_ACCOUNTS) -> Population:
    """
    Assign every account a behavioural archetype.

    Archetypes are the whole point of v3: they are what give the negative class
    internal structure. A population of identical random payers has no
    legitimate look-alikes for fraud, so any structural feature separates the
    classes perfectly and the evaluation measures nothing.
    """
    accounts = [f"UPI_{i:05d}" for i in range(num_accounts)]
    shuffled = list(accounts)
    random.shuffle(shuffled)

    by_role: dict[str, list[str]] = {}
    role_of: dict[str, str] = {}

    cursor = 0
    for role, frac in ROLE_MIX.items():
        n = int(round(num_accounts * frac))
        members = shuffled[cursor:cursor + n]
        cursor += n
        by_role[role] = members
        for a in members:
            role_of[a] = role

    consumers = shuffled[cursor:]
    by_role["consumer"] = consumers
    for a in consumers:
        role_of[a] = "consumer"

    # Activity tier for anyone who behaves like a payer (consumers and gig
    # workers). Hubs are busy by construction.
    activity_of: dict[str, str] = {}
    tiers = list(ACTIVITY_MIX)
    weights = [ACTIVITY_MIX[t] for t in tiers]
    for a in consumers + by_role["gig"]:
        activity_of[a] = random.choices(tiers, weights=weights, k=1)[0]

    return Population(
        all_accounts=accounts,
        by_role=by_role,
        role_of=role_of,
        activity_of=activity_of,
    )


# ══════════════════════════════════════════════════════════════════
# Recurring relationships
# ══════════════════════════════════════════════════════════════════

@dataclass
class Relationship:
    """
    A standing payment relationship that fires repeatedly over time.

    `amount_jitter` is the relative standard deviation of the amount. Zero means
    the exact same rupee value every cycle, which is what a rent or EMI mandate
    actually looks like — and which is why `amount_cv` cannot be used as a
    fraud giveaway any more.

    `hour` of -1 means "draw a fresh diurnal hour each time"; a fixed hour
    models a scheduled mandate or a habitual daily payment.
    """
    sender: str
    receiver: str
    base_amount: float
    cadence_days: float
    amount_jitter: float
    hour: int
    role: str


def _round_to_nice(amount: float) -> float:
    """Round obligations to plausible billed values (₹15,000 not ₹14,983.41)."""
    if amount >= 10_000:
        return float(round(amount / 500) * 500)
    if amount >= 1_000:
        return float(round(amount / 50) * 50)
    return float(round(amount / 10) * 10) or 10.0


def _relationship_amount(rel: Relationship) -> float:
    if rel.amount_jitter <= 0.0:
        return rel.base_amount
    value = rel.base_amount * (1.0 + random.gauss(0.0, rel.amount_jitter))
    return round(max(10.0, value), 2)


def build_relationships(pop: Population) -> list[Relationship]:
    """
    Wire the standing relationship graph.

    Read this alongside models/features.py: nearly every entry exists to be the
    legitimate twin of one feature. The salary/rent/EMI mandates give organic
    accounts *identical repeated amounts*; the kirana relationship gives them
    *high repetition*; gig payouts give them *pass-through flow*; merchant hubs
    give them *fan-in concentration*.
    """
    rels: list[Relationship] = []

    merchants = pop.by_role["merchant"]
    employers = pop.by_role["employer"]
    landlords = pop.by_role["landlord"]
    billers = pop.by_role["biller"]
    gigs = pop.by_role["gig"]
    consumers = pop.by_role["consumer"]

    # How many discretionary merchant relationships each tier keeps, and how
    # tight the high-frequency one is.
    tier_spec = {
        # tier:    (n_frequent, freq_cadence,  n_medium, med_cadence, n_subs, emi_prob)
        "light":   (0,          (0, 0),        1,        (12.0, 25.0), 1,     0.20),
        "medium":  (1,          (5.0, 10.0),   1,        (10.0, 20.0), 2,     0.55),
        "heavy":   (1,          (3.0, 7.0),    2,        (8.0, 16.0),  2,     0.75),
    }

    for account in consumers:
        tier = pop.activity_of[account]
        n_freq, freq_cad, n_med, med_cad, n_subs, emi_prob = tier_spec[tier]

        # ── Salary in: fixed amount, fixed day, one employer ──
        employer = random.choice(employers)
        salary = _round_to_nice(random.uniform(18_000, 140_000))
        rels.append(Relationship(
            sender=employer, receiver=account,
            base_amount=salary,
            cadence_days=30.0,
            amount_jitter=0.0,          # payroll is exact
            hour=10,
            role="organic_salary",
        ))

        # ── Rent out: fixed amount, fixed day ──
        if random.random() < 0.72:
            rels.append(Relationship(
                sender=account, receiver=random.choice(landlords),
                base_amount=_round_to_nice(salary * random.uniform(0.15, 0.35)),
                cadence_days=30.0,
                amount_jitter=0.0,      # rent is exact
                hour=random.choice((9, 11, 20)),
                role="organic_rent",
            ))

        # ── EMI out: fixed amount, auto-debit ──
        if random.random() < emi_prob:
            rels.append(Relationship(
                sender=account, receiver=random.choice(billers),
                base_amount=_round_to_nice(salary * random.uniform(0.08, 0.22)),
                cadence_days=30.0,
                amount_jitter=0.0,
                hour=7,
                role="organic_emi",
            ))

        # ── Subscriptions out: small, exact, monthly ──
        for _ in range(n_subs):
            rels.append(Relationship(
                sender=account, receiver=random.choice(billers),
                base_amount=float(random.choice((99, 149, 199, 299, 499, 649, 799))),
                cadence_days=30.0,
                amount_jitter=0.0,
                hour=random.randrange(24),
                role="organic_subscription",
            ))

        # ── Utility out: monthly, mildly variable ──
        rels.append(Relationship(
            sender=account, receiver=random.choice(billers),
            base_amount=_round_to_nice(random.uniform(600, 4_500)),
            cadence_days=30.0,
            amount_jitter=0.18,         # usage varies month to month
            hour=random.choice((19, 20, 21)),
            role="organic_utility",
        ))

        # ── High-frequency merchant: the chai stall / kirana / auto ride ──
        # This single relationship is what moves organic repeat_ratio from
        # ~1.0 (v2) into the range rings occupy. Without it, `repeat_ratio`
        # alone scores test AUC 0.9989 and the model learns nothing.
        for _ in range(n_freq):
            rels.append(Relationship(
                sender=account, receiver=random.choice(merchants),
                base_amount=random.uniform(60, 900),
                cadence_days=random.uniform(*freq_cad),
                amount_jitter=0.45,
                hour=-1,
                role="organic_frequent",
            ))

        # ── Medium-frequency merchants: groceries, fuel, pharmacy ──
        for _ in range(n_med):
            rels.append(Relationship(
                sender=account, receiver=random.choice(merchants),
                base_amount=random.uniform(400, 9_000),
                cadence_days=random.uniform(*med_cad),
                amount_jitter=0.55,
                hour=-1,
                role="organic_merchant",
            ))

    # ── Gig workers: payout in, forward almost all of it out ──
    # The legitimate `flow_passthrough` population. A gig account receives from
    # one or two aggregators and moves the money on within days, which is
    # exactly the pass-through signature a mule has.
    for account in gigs:
        for _ in range(random.randint(1, 2)):
            payout = random.uniform(2_500, 22_000)
            rels.append(Relationship(
                sender=random.choice(merchants), receiver=account,
                base_amount=payout,
                cadence_days=random.uniform(3.0, 8.0),
                amount_jitter=0.30,
                hour=-1,
                role="organic_payout",
            ))
        # Forwards out: rent, family remittance, own savings mandate.
        for _ in range(random.randint(2, 3)):
            rels.append(Relationship(
                sender=account,
                receiver=random.choice(
                    pop.by_role["landlord"] + pop.by_role["consumer"][:400]
                ),
                base_amount=random.uniform(3_000, 25_000),
                cadence_days=random.uniform(6.0, 14.0),
                amount_jitter=0.25,
                hour=-1,
                role="organic_forward",
            ))

    # ── Merchant → biller: settlement, tax, supplier ──
    for account in merchants:
        for _ in range(random.randint(1, 3)):
            rels.append(Relationship(
                sender=account, receiver=random.choice(billers + pop.by_role["trader"]),
                base_amount=_round_to_nice(random.uniform(20_000, 250_000)),
                cadence_days=random.uniform(7.0, 20.0),
                amount_jitter=0.22,
                hour=random.choice((17, 18, 22)),
                role="organic_settlement",
            ))

    return rels


def emit_relationships(
    rels: list[Relationship],
    time_start: datetime,
    time_end: datetime,
) -> tuple[list[dict], list[tuple[str, datetime]]]:
    """
    Expand standing relationships into individual transactions.

    Cadence is jittered ±20% per interval so that repeat counts inside a short
    window are not deterministic — otherwise every account with a 30-day
    mandate contributes exactly one transaction to the 33-day validation
    window, and the val/test feature distributions drift away from train for a
    purely mechanical reason.

    Returns (records, salary_events) where salary_events feeds payday bursts.
    """
    records: list[dict] = []
    salary_events: list[tuple[str, datetime]] = []

    for rel in rels:
        # Random phase, so mandates are not all synchronised to day 0.
        cursor = time_start + timedelta(
            seconds=random.uniform(0.0, rel.cadence_days * 86_400.0)
        )
        while cursor < time_end:
            hour = rel.hour if rel.hour >= 0 else _daytime_hour()
            ts = cursor.replace(
                hour=hour,
                minute=random.randrange(60),
                second=random.randrange(60),
                microsecond=0,
            )
            if time_start <= ts < time_end:
                records.append({
                    "sender": rel.sender,
                    "receiver": rel.receiver,
                    "amount": _relationship_amount(rel),
                    "timestamp": ts,
                    "is_mule": 0,
                    "edge_role": rel.role,
                    "ring_id": NO_RING_ID,
                    "ring_type": NO_RING_TYPE,
                })
                if rel.role == "organic_salary":
                    salary_events.append((rel.receiver, ts))

            cursor += timedelta(
                days=rel.cadence_days * random.uniform(0.80, 1.20)
            )

    return records, salary_events


def emit_payday_bursts(
    rels: list[Relationship],
    salary_events: list[tuple[str, datetime]],
    time_start: datetime,
    time_end: datetime,
) -> list[dict]:
    """
    After a salary credit, fire several outbound payments within hours.

    This is the legitimate source of `burst_ratio`. Laundering is bursty, but so
    is the first evening after payday, and if only rings burst then "share of
    transactions in the busiest hour" is a label in disguise.
    """
    outbound: dict[str, list[Relationship]] = {}
    for rel in rels:
        if rel.role.startswith("organic_") and rel.role != "organic_salary":
            outbound.setdefault(rel.sender, []).append(rel)

    records: list[dict] = []
    for account, salary_ts in salary_events:
        if random.random() >= PAYDAY_BURST_PROB:
            continue
        candidates = outbound.get(account)
        if not candidates:
            continue

        n = random.randint(*PAYDAY_BURST_COUNT)
        for rel in random.sample(candidates, min(n, len(candidates))):
            ts = salary_ts + timedelta(
                seconds=random.uniform(600, PAYDAY_BURST_WINDOW_H * 3600)
            )
            if not (time_start <= ts < time_end):
                continue
            records.append({
                "sender": rel.sender,
                "receiver": rel.receiver,
                "amount": _relationship_amount(rel),
                "timestamp": ts,
                "is_mule": 0,
                "edge_role": "organic_burst",
                "ring_id": NO_RING_ID,
                "ring_type": NO_RING_TYPE,
            })

    return records


def emit_social_pairs(
    pop: Population,
    time_start: datetime,
    time_end: datetime,
) -> list[dict]:
    """
    Reciprocal peer-to-peer payments: splitting rent, settling dinner, lending.

    Both directions fire, which is the legitimate source of `reciprocity` and of
    short 2-cycles. Without this, "money came back to you" is only ever a
    laundering signature.
    """
    consumers = pop.by_role["consumer"]
    records: list[dict] = []

    for _ in range(SOCIAL_PAIRS):
        a, b = random.sample(consumers, 2)
        cadence = random.uniform(9.0, 26.0)
        base = random.uniform(150, 6_000)
        # Asymmetric intensity: friendships are rarely a perfect exchange.
        skew = random.uniform(0.35, 1.0)

        for sender, receiver, intensity in ((a, b, 1.0), (b, a, skew)):
            cursor = time_start + timedelta(
                seconds=random.uniform(0.0, cadence * 86_400.0)
            )
            while cursor < time_end:
                if random.random() < intensity:
                    ts = cursor.replace(
                        hour=_daytime_hour(),
                        minute=random.randrange(60),
                        second=random.randrange(60),
                        microsecond=0,
                    )
                    if time_start <= ts < time_end:
                        records.append({
                            "sender": sender,
                            "receiver": receiver,
                            "amount": round(base * random.uniform(0.4, 2.2), 2),
                            "timestamp": ts,
                            "is_mule": 0,
                            "edge_role": "organic_social",
                            "ring_id": NO_RING_ID,
                            "ring_type": NO_RING_TYPE,
                        })
                cursor += timedelta(days=cadence * random.uniform(0.7, 1.3))

    return records


def emit_settlement_cycles(
    pop: Population,
    time_start: datetime,
    time_end: datetime,
) -> list[dict]:
    """
    Organic B2B settlement loops: A pays B pays C pays A.

    THIS IS THE MOST IMPORTANT ORGANIC PATTERN IN THE FILE.

    The project's thesis is "mule rings are circular", and `cycle_participation`
    encodes it directly. If the only cycles in the graph are rings, that feature
    *is* the label and every reported metric is fiction. Real trade credit
    genuinely circulates — a distributor pays a supplier who pays a logistics
    firm who buys from the distributor — and those loops repeat on a settlement
    cadence with similar amounts to each partner, which also confounds
    `counterparty_amount_cv`.

    So these groups are not decoration. They are the reason a cycle-based
    detector has to do real work.
    """
    traders = pop.by_role["trader"]
    records: list[dict] = []
    if len(traders) < SETTLEMENT_GROUP_SIZE[1]:
        return records

    for _ in range(SETTLEMENT_GROUPS):
        size = random.randint(*SETTLEMENT_GROUP_SIZE)
        group = random.sample(traders, size)
        cadence = random.uniform(6.0, 16.0)
        base = random.uniform(25_000, 300_000)

        for i in range(size):
            sender = group[i]
            receiver = group[(i + 1) % size]
            if sender == receiver:
                continue
            cursor = time_start + timedelta(
                seconds=random.uniform(0.0, cadence * 86_400.0)
            )
            while cursor < time_end:
                ts = cursor.replace(
                    hour=_daytime_hour(),
                    minute=random.randrange(60),
                    second=random.randrange(60),
                    microsecond=0,
                )
                if time_start <= ts < time_end:
                    records.append({
                        "sender": sender,
                        "receiver": receiver,
                        # Similar-but-not-identical to each partner: the same
                        # low counterparty_amount_cv a ring produces.
                        "amount": round(base * random.uniform(0.80, 1.20), 2),
                        "timestamp": ts,
                        "is_mule": 0,
                        "edge_role": "organic_b2b_cycle",
                        "ring_id": NO_RING_ID,
                        "ring_type": NO_RING_TYPE,
                    })
                cursor += timedelta(days=cadence * random.uniform(0.8, 1.2))

    return records


def _one_off_amount() -> float:
    """
    Amount for unstructured one-off traffic.

    The large bands deliberately OVERLAP the ring range (₹8k-95k). In v1 organic
    amounts were mostly under ₹5k while every ring transaction was ₹8k-95k,
    which made `out_amount_sum` the #2 feature at 28.9% importance purely as a
    generator artefact.
    """
    r = random.random()
    if r < 0.55:
        amount = random.uniform(50, 2_000)
    elif r < 0.80:
        amount = random.uniform(2_000, 15_000)
    elif r < 0.95:
        amount = random.uniform(15_000, 90_000)     # overlaps rings
    else:
        amount = random.uniform(90_000, 400_000)
    return round(amount, 2)


def emit_one_off(
    pop: Population,
    n_edges: int,
    time_start: datetime,
    time_end: datetime,
) -> list[dict]:
    """
    The unstructured long tail: paying a stranger once and never again.

    Receivers are skewed toward merchant hubs, which is what builds high
    `in_degree` / `fan_in_concentration` on legitimate accounts.
    """
    accounts = np.array(pop.all_accounts)
    merchants = np.array(pop.by_role["merchant"] or pop.all_accounts)

    popularity = np.random.zipf(a=1.8, size=len(accounts)).astype(float)
    popularity /= popularity.sum()

    senders = np.random.choice(accounts, size=n_edges, p=popularity)

    # 45% of one-offs land on a merchant hub; the rest are true P2P.
    to_merchant = np.random.random(n_edges) < 0.45
    receivers = np.where(
        to_merchant,
        np.random.choice(merchants, size=n_edges),
        np.random.choice(accounts, size=n_edges, p=popularity),
    )

    keep = senders != receivers
    senders, receivers = senders[keep], receivers[keep]

    return [
        {
            "sender": s,
            "receiver": r,
            "amount": _one_off_amount(),
            "timestamp": _diurnal_timestamp(time_start, time_end),
            "is_mule": 0,
            "edge_role": "organic_oneoff",
            "ring_id": NO_RING_ID,
            "ring_type": NO_RING_TYPE,
        }
        for s, r in zip(senders, receivers)
    ]


def generate_organic_transactions(
    pop: Population,
    time_start: datetime = TIME_START,
    time_end: datetime = TIME_END,
) -> pd.DataFrame:
    """
    Assemble all legitimate traffic.

    Volume is emergent from the relationship graph rather than a target, which
    is the structural difference from v2. Fixing an edge count is what forced
    i.i.d. pair sampling, and i.i.d. pair sampling is what made `repeat_ratio`
    a label.
    """
    rels = build_relationships(pop)

    records, salary_events = emit_relationships(rels, time_start, time_end)
    records += emit_payday_bursts(rels, salary_events, time_start, time_end)
    records += emit_social_pairs(pop, time_start, time_end)
    records += emit_settlement_cycles(pop, time_start, time_end)
    records += emit_one_off(pop, ONE_OFF_EDGES, time_start, time_end)

    df = pd.DataFrame.from_records(records)
    # Self-transfers are meaningless in a payment graph and break cycle logic.
    return df[df["sender"] != df["receiver"]].reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════
# Mule rings
# ══════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class RingArchetype:
    """
    One laundering shape.

    A single ring shape is the second way a generator fakes a good score: if
    every positive looks the same, the model needs one decision path and recall
    saturates. Real ring behaviour spans "obvious in an hour" to "indefinitely
    plausible", and an evaluation that does not span that range is not
    measuring the thing the merchant cares about.
    """
    name: str
    share: float
    size_range: tuple[int, int]
    txns_per_pair: tuple[int, int]
    amount_jitter: float           # relative sd on the per-ring base amount
    hijack_prob: float
    burst_hours: tuple[int, int] | None   # None → spread across the window
    fan_in_range: tuple[int, int]
    open_chain_prob: float         # chance the loop is left open (a path)
    camouflage: bool               # give purpose-built mules a civilian life


RING_ARCHETYPES: tuple[RingArchetype, ...] = (
    RingArchetype(
        name="fast_cycle",
        share=0.35,
        size_range=(3, 6),
        txns_per_pair=(8, 18),
        amount_jitter=0.05,        # near-identical: the classic signature
        hijack_prob=0.35,
        burst_hours=(2, 12),       # placement → layering → integration, one push
        fan_in_range=(4, 10),
        open_chain_prob=0.00,
        camouflage=True,
    ),
    RingArchetype(
        name="stealth_cycle",
        share=0.40,
        size_range=(4, 8),
        # Low volume per hop is the whole trick: stay under the repetition
        # thresholds a rules engine watches.
        txns_per_pair=(2, 5),
        amount_jitter=0.30,        # jittered to defeat amount_cv screens
        hijack_prob=0.65,          # prefers accounts with real history
        burst_hours=None,          # smeared across weeks
        fan_in_range=(1, 3),
        open_chain_prob=0.35,      # a third never close the loop at all
        camouflage=True,
    ),
    RingArchetype(
        name="layered_fanin",
        share=0.25,
        size_range=(3, 5),
        txns_per_pair=(3, 8),
        amount_jitter=0.20,
        hijack_prob=0.45,
        burst_hours=(12, 72),
        # Large feeder count: structurally this is a merchant with customers.
        fan_in_range=(15, 40),
        open_chain_prob=0.50,
        camouflage=True,
    ),
)

RING_ARCHETYPE_BY_NAME = {a.name: a for a in RING_ARCHETYPES}


def allocate_archetypes(num_rings: int) -> list[str]:
    """
    Split `num_rings` across archetypes by share, largest-remainder.

    Deterministic allocation (rather than per-ring sampling) guarantees every
    split contains every archetype, which is what makes the per-archetype
    recall breakdown in models/train.py comparable across splits.
    """
    exact = {a.name: num_rings * a.share for a in RING_ARCHETYPES}
    counts = {name: int(v) for name, v in exact.items()}

    remaining = num_rings - sum(counts.values())
    order = sorted(exact, key=lambda n: exact[n] - counts[n], reverse=True)
    for i in range(remaining):
        counts[order[i % len(order)]] += 1

    names: list[str] = []
    for name, n in counts.items():
        names += [name] * n
    random.shuffle(names)
    return names


def build_hijack_pools(all_accounts: list[str]) -> dict[str, list[str]]:
    """
    Partition accounts into disjoint per-split pools eligible for hijacking or
    fan-in feeding.

    Disjointness is what makes the entity-level guarantee possible: an account
    hijacked by a train-window ring can never be hijacked by a test-window ring,
    so no node is mule-labelled in two splits.

    The capacity check runs BEFORE slicing. In v2 it ran after, so an
    over-subscribed configuration silently produced short or empty pools —
    Python slicing does not raise past the end of a list — and only then
    reported the error, by which point the pools had already been built and, in
    the val/test case, could have been empty.
    """
    required = sum(HIJACK_POOL_SIZE[s] for s in SPLITS)
    if required > len(all_accounts):
        raise ValueError(
            f"HIJACK_POOL_SIZE totals {required} but only {len(all_accounts)} "
            f"accounts exist. Raise NUM_ACCOUNTS or shrink the pools."
        )

    shuffled = list(all_accounts)
    random.shuffle(shuffled)

    pools: dict[str, list[str]] = {}
    cursor = 0
    for split in SPLITS:
        size = HIJACK_POOL_SIZE[split]
        pools[split] = shuffled[cursor:cursor + size]
        cursor += size

    return pools


def _ring_timestamp(
    arch: RingArchetype,
    window_start: datetime,
    window_end: datetime,
    burst_origin: datetime | None,
) -> datetime:
    """
    A timestamp for one ring transaction, always inside the window.

    Bursting archetypes draw from a per-ring burst; stealth archetypes draw from
    the whole window, which is exactly what makes them hard to see with a
    velocity rule.
    """
    if arch.burst_hours is None or burst_origin is None:
        return _diurnal_timestamp(window_start, window_end)

    span_h = random.randint(*arch.burst_hours)
    burst_end = min(burst_origin + timedelta(hours=span_h), window_end)
    return _random_timestamp(burst_origin, burst_end)


def _seat_ring(
    arch: RingArchetype,
    hijack_pool: list[str],
    prefix: str,
    mule_counter: int,
) -> tuple[list[str], int]:
    """
    Choose the accounts that make up one ring.

    Returns (ring_nodes, updated_mule_counter). Hijacked seats come from the
    split's pool; the rest are purpose-built accounts.
    """
    size = random.randint(*arch.size_range)
    nodes: list[str] = []
    seen: set[str] = set()

    for _ in range(size):
        if hijack_pool and random.random() < arch.hijack_prob:
            candidate = random.choice(hijack_pool)
            # Seating one account twice would wire a self-loop at node[i]→[i+1].
            if candidate not in seen:
                nodes.append(candidate)
                seen.add(candidate)
                continue
        name = f"{prefix}_{mule_counter:04d}"
        mule_counter += 1
        nodes.append(name)
        seen.add(name)

    return nodes, mule_counter


def _emit_camouflage(
    mule_accounts: list[str],
    hijack_pool: list[str],
    window_start: datetime,
    window_end: datetime,
) -> list[dict]:
    """
    Give purpose-built mule accounts a thin civilian transaction history.

    Without this, a fresh mule account's only counterparties are other ring
    members, so `community_internal_ratio` is exactly 1.0 for every
    purpose-built mule and 1.0 for essentially nobody else. That is a label
    wearing a feature's name. Real money mules are recruited people with phones
    and shops; they buy things.

    Labelled is_mule=0 with ring_id=-1: the *edge* is ordinary spending. The
    mule node itself stays positive via its `ring` edges.
    """
    if not hijack_pool:
        return []

    records: list[dict] = []
    for account in mule_accounts:
        for _ in range(random.randint(1, 3)):
            other = random.choice(hijack_pool)
            if other == account:
                continue
            outbound = random.random() < 0.7
            for _ in range(random.randint(2, 6)):
                records.append({
                    "sender": account if outbound else other,
                    "receiver": other if outbound else account,
                    "amount": round(random.uniform(80, 4_000), 2),
                    "timestamp": _diurnal_timestamp(window_start, window_end),
                    "is_mule": 0,
                    "edge_role": "organic_camouflage",
                    "ring_id": NO_RING_ID,
                    "ring_type": NO_RING_TYPE,
                })
    return records


def generate_mule_rings_for_split(
    split: str,
    num_rings: int,
    hijack_pool: list[str],
    window_start: datetime,
    window_end: datetime,
    ring_id_offset: int,
) -> pd.DataFrame:
    """
    Inject rings confined to a single temporal window.

    Two invariants matter here:

    1. Every timestamp is drawn from [window_start, window_end), so a ring
       cannot straddle a split boundary.

    2. Fan-in edges are tagged edge_role="fan_in" and their feeder is NOT
       labelled a mule. Feeders are ordinary accounts that paid into a ring.
       v1 labelled them positive, which put 100-250 accounts with entirely
       organic feature profiles into the positive class and was a large part of
       why precision sat at 0.499. The fan-in *topology* is still present for
       the model to learn; only the wrong node label is gone.
    """
    prefix = MULE_PREFIX[split]
    mule_counter = 0
    records: list[dict] = []
    fresh_mules: list[str] = []

    for local in range(num_rings):
        ring_id = ring_id_offset + local
        arch = RING_ARCHETYPE_BY_NAME[
            _ARCHETYPE_PLAN[split][local]
        ]

        ring_nodes, mule_counter = _seat_ring(
            arch, hijack_pool, prefix, mule_counter
        )
        if len(ring_nodes) < 2:
            continue
        size = len(ring_nodes)
        fresh_mules += [n for n in ring_nodes if n.startswith(MULE_PREFIXES)]

        base_amount = random.uniform(8_000, 95_000)
        burst_origin = (
            _random_timestamp(window_start, window_end)
            if arch.burst_hours is not None else None
        )

        # Hop list: node[i] → node[i+1], closing the loop unless left open.
        hops = [(ring_nodes[i], ring_nodes[i + 1]) for i in range(size - 1)]
        closes = random.random() >= arch.open_chain_prob
        if closes:
            hops.append((ring_nodes[-1], ring_nodes[0]))

        for sender, receiver in hops:
            if sender == receiver:
                continue
            n_txns = random.randint(*arch.txns_per_pair)
            for _ in range(n_txns):
                amount = base_amount * (
                    1.0 + random.gauss(0.0, arch.amount_jitter)
                )
                records.append({
                    "sender": sender,
                    "receiver": receiver,
                    "amount": round(max(100.0, amount), 2),
                    "timestamp": _ring_timestamp(
                        arch, window_start, window_end, burst_origin
                    ),
                    "is_mule": 1,
                    "edge_role": "ring",
                    "ring_id": ring_id,
                    "ring_type": arch.name,
                })

        # ── Fan-in: outside accounts feeding the ring's entry point ──
        entry = ring_nodes[0]
        n_feeders = random.randint(*arch.fan_in_range)
        pool = [a for a in hijack_pool if a not in set(ring_nodes)]
        if pool:
            feeders = random.sample(pool, min(n_feeders, len(pool)))
            for feeder in feeders:
                for _ in range(random.randint(1, 3)):
                    records.append({
                        "sender": feeder,
                        "receiver": entry,
                        "amount": round(random.uniform(4_000, 80_000), 2),
                        "timestamp": _ring_timestamp(
                            arch, window_start, window_end, burst_origin
                        ),
                        # is_mule=1 keeps the "this edge is ring activity"
                        # semantics, but edge_role="fan_in" means the feeder
                        # node is not labelled a mule.
                        "is_mule": 1,
                        "edge_role": "fan_in",
                        "ring_id": ring_id,
                        "ring_type": arch.name,
                    })

    # ── Camouflage traffic for purpose-built mule accounts ──
    records += _emit_camouflage(
        fresh_mules, hijack_pool, window_start, window_end
    )

    if not records:
        return pd.DataFrame(columns=EDGE_COLUMNS[:-1])

    df = pd.DataFrame.from_records(records)
    return df[df["sender"] != df["receiver"]].reset_index(drop=True)


# Archetype plan per split, fixed once per run so that both the ring generator
# and the reporting agree on which ring is which shape.
_ARCHETYPE_PLAN: dict[str, list[str]] = {}


def build_archetype_plan() -> dict[str, list[str]]:
    global _ARCHETYPE_PLAN
    _ARCHETYPE_PLAN = {
        s: allocate_archetypes(RINGS_PER_SPLIT[s]) for s in SPLITS
    }
    return _ARCHETYPE_PLAN


# ══════════════════════════════════════════════════════════════════
# Retirement
# ══════════════════════════════════════════════════════════════════

def retire_hijacked_accounts(
    organic_df: pd.DataFrame,
    mule_df: pd.DataFrame,
    windows: dict[str, tuple[datetime, datetime]],
) -> tuple[pd.DataFrame, int, int]:
    """
    Freeze hijacked accounts once their ring's window closes.

    A ring seats some members by hijacking real accounts, because an account
    with genuine history is exactly what a launderer wants and exactly what
    makes detection realistic. That history is kept: organic traffic before and
    during the ring stays untouched.

    What is removed is organic traffic *after* the ring window, for two reasons:

      1. It is what happens. An account caught mulling gets frozen, reported and
         closed. It does not carry on buying groceries next quarter.

      2. Without it, the same account is a labelled mule in `train` and a
         labelled legitimate account in `val`. That contradiction is label
         noise: the model is taught "this profile is fraud" and then penalised
         for saying so. It depressed measured performance rather than inflating
         it — so it was never a leak — but it is still a defect, and it hit 9.3%
         of train positives.

    Only ring *members* are retired. Fan-in feeders are victims: they keep their
    label of 0 and go on transacting normally.

    Returns (filtered_organic_df, n_accounts_retired, n_edges_removed).
    """
    ring_edges = mule_df[mule_df["edge_role"] == "ring"]
    if ring_edges.empty:
        return organic_df, 0, 0

    # For each hijacked account, the end of the earliest window whose ring
    # recruited it. Rings are window-confined, so deriving the split from the
    # timestamp is equivalent to reading the `split` column and does not depend
    # on the ring generator having set it.
    freeze_after: dict[str, datetime] = {}
    for row in ring_edges.itertuples(index=False):
        window_end = windows[split_for_timestamp(row.timestamp, windows)][1]
        for account in (row.sender, row.receiver):
            if account.startswith(MULE_PREFIXES):
                continue  # purpose-built mule: no organic life to freeze
            current = freeze_after.get(account)
            if current is None or window_end < current:
                freeze_after[account] = window_end

    if not freeze_after:
        return organic_df, 0, 0

    cutoff = pd.Series(freeze_after)
    sender_cut = organic_df["sender"].map(cutoff)
    receiver_cut = organic_df["receiver"].map(cutoff)
    edge_cut = pd.concat([sender_cut, receiver_cut], axis=1).min(axis=1)

    retired = edge_cut.notna() & (organic_df["timestamp"] > edge_cut)
    n_removed = int(retired.sum())

    assert n_removed < len(organic_df), "retirement filter removed everything"

    return organic_df[~retired].reset_index(drop=True), len(freeze_after), n_removed


# ══════════════════════════════════════════════════════════════════
# Invariants
# ══════════════════════════════════════════════════════════════════

def ring_member_nodes(df: pd.DataFrame) -> set[str]:
    """Accounts inside a ring cycle. Excludes fan-in feeders by design."""
    ring_edges = df[df["edge_role"] == "ring"]
    return set(ring_edges["sender"]) | set(ring_edges["receiver"])


def assert_no_entity_leakage(splits: dict[str, pd.DataFrame]) -> None:
    """No account is a ring member in two splits; no ring_id spans two splits."""
    members = {s: ring_member_nodes(df) for s, df in splits.items()}

    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = members[a] & members[b]
        assert not overlap, (
            f"ENTITY LEAKAGE: {len(overlap)} account(s) are ring members in "
            f"both '{a}' and '{b}'. Examples: {sorted(overlap)[:5]}"
        )

    rings = {
        s: set(df.loc[df["ring_id"] >= 0, "ring_id"].unique())
        for s, df in splits.items()
    }
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = rings[a] & rings[b]
        assert not overlap, (
            f"RING LEAKAGE: ring_id(s) {sorted(overlap)[:5]} appear in both "
            f"'{a}' and '{b}'."
        )


def assert_temporal_order(splits: dict[str, pd.DataFrame]) -> None:
    """train.max <= val.min and val.max <= test.min."""
    for earlier, later in (("train", "val"), ("val", "test")):
        e_max = splits[earlier]["timestamp"].max()
        l_min = splits[later]["timestamp"].min()
        assert l_min >= e_max, (
            f"TEMPORAL LEAKAGE: {later}.min ({l_min}) < {earlier}.max ({e_max})"
        )


def assert_equal_window_lengths(
    windows: dict[str, tuple[datetime, datetime]],
) -> None:
    """
    All three observation windows must be the same length.

    Window length multiplies every count-, sum- and rate-based feature, so
    unequal windows mean the same account has systematically different features
    depending on which split it lands in. That is a distribution shift the model
    cannot see and cannot correct for: it fits split points to the training
    window's scale and is then scored against a differently-scaled test window.

    v2's 60/18/22 fractions produced 108 / 32 / 39 day windows — a 3.4x spread.
    This assertion is why SPLIT_FRACTIONS is now equal thirds.
    """
    lengths = {
        s: (end - start).total_seconds() / 86_400.0
        for s, (start, end) in windows.items()
    }
    spread = max(lengths.values()) - min(lengths.values())
    assert spread <= WINDOW_LENGTH_TOLERANCE_DAYS, (
        "UNEQUAL OBSERVATION WINDOWS: "
        + ", ".join(f"{s}={d:.1f}d" for s, d in lengths.items())
        + f" (spread {spread:.1f}d > {WINDOW_LENGTH_TOLERANCE_DAYS}d). "
        "Every count/sum feature scales with window length, so the splits "
        "would not be comparable. Fix SPLIT_FRACTIONS."
    )


def assert_structural_sanity(splits: dict[str, pd.DataFrame]) -> None:
    """
    Cheap guards for defects that would silently corrupt every downstream number.

    Each split must carry positives (otherwise precision/recall are undefined),
    every archetype must appear in every split (otherwise the per-archetype
    recall table is not comparable), and there must be no self-loops or
    non-positive amounts.
    """
    for s, df in splits.items():
        assert not df.empty, f"split '{s}' is empty"
        assert (df["sender"] != df["receiver"]).all(), f"self-loop in '{s}'"
        assert (df["amount"] > 0).all(), f"non-positive amount in '{s}'"

        members = ring_member_nodes(df)
        assert members, f"split '{s}' has no ring members — no positive class"

        present = set(df.loc[df["edge_role"] == "ring", "ring_type"].unique())
        missing = {a.name for a in RING_ARCHETYPES} - present
        assert not missing, (
            f"split '{s}' is missing ring archetype(s) {sorted(missing)}; "
            f"per-archetype recall would not be comparable across splits"
        )


# ══════════════════════════════════════════════════════════════════
# Difficulty report
# ══════════════════════════════════════════════════════════════════

def repeat_ratio_report(df: pd.DataFrame) -> pd.DataFrame:
    """
    The generator's own honesty check, computed from the edge list alone.

    `repeat_ratio` (transactions per distinct counterparty) is the feature that
    made v2's metrics meaningless: organic sat at p50 1.00 / p99 2.25 while
    rings sat at ~13, so one threshold separated the classes and the model's
    0.98 precision measured nothing.

    This prints the distribution for legitimate accounts against each ring
    archetype. If the legitimate p90 does not reach into the ring range, the
    difficulty target has not landed and there is no point training — so it is
    reported here, before the CSVs are even written, rather than being
    discovered later in a metrics file.
    """
    # Undirected counterparty counts, computed without building a graph.
    pairs = pd.concat([
        df[["sender", "receiver"]].rename(
            columns={"sender": "node", "receiver": "other"}
        ),
        df[["receiver", "sender"]].rename(
            columns={"receiver": "node", "sender": "other"}
        ),
    ], ignore_index=True)

    txns = pairs.groupby("node").size()
    distinct = pairs.groupby("node")["other"].nunique()
    ratio = (txns / distinct.clip(lower=1)).rename("repeat_ratio")

    # Node label and ring archetype from the ring edges.
    ring_edges = df[df["edge_role"] == "ring"]
    node_type: dict[str, str] = {}
    for row in ring_edges[["sender", "receiver", "ring_type"]].itertuples(index=False):
        node_type.setdefault(row.sender, row.ring_type)
        node_type.setdefault(row.receiver, row.ring_type)

    group = ratio.index.map(lambda n: node_type.get(n, "legitimate"))
    frame = pd.DataFrame({"repeat_ratio": ratio.values, "group": group})

    out = frame.groupby("group")["repeat_ratio"].agg(
        n="size",
        p50=lambda s: s.quantile(0.50),
        p90=lambda s: s.quantile(0.90),
        p99=lambda s: s.quantile(0.99),
    )
    order = ["legitimate"] + [a.name for a in RING_ARCHETYPES]
    return out.reindex([g for g in order if g in out.index])


def print_difficulty_report(splits: dict[str, pd.DataFrame]) -> None:
    print()
    print(banner("Difficulty check: repeat_ratio by group (test split)"))
    report = repeat_ratio_report(splits["test"])
    print(report.to_string(float_format=lambda v: f"{v:8.2f}"))

    legit_p90 = float(report.loc["legitimate", "p90"])
    ring_rows = report.drop(index="legitimate", errors="ignore")
    if not ring_rows.empty:
        hardest = float(ring_rows["p50"].min())
        print()
        if hardest <= legit_p90:
            print(f"  {sym('ok')} the hardest archetype's median repeat_ratio "
                  f"({hardest:.2f}) sits below the legitimate p90 "
                  f"({legit_p90:.2f}) — a single repetition threshold cannot "
                  f"separate the classes.")
        else:
            print(f"  {sym('warn')} every archetype's median repeat_ratio "
                  f"({hardest:.2f}) exceeds the legitimate p90 "
                  f"({legit_p90:.2f}). A one-rule baseline may still win; "
                  f"consider lowering RING_ARCHETYPES txns_per_pair or raising "
                  f"consumer relationship cadence.")


# ══════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════

def main() -> None:
    enable_utf8_stdout()
    set_seed()

    windows = build_split_windows(TIME_START, TIME_END)
    assert_equal_window_lengths(windows)
    plan = build_archetype_plan()

    print(banner("UPI Mule-Ring Sentinel: Data Generator (v3)"))
    print(f"  Time window: {TIME_START.date()} {sym('arrow')} {TIME_END.date()}")
    for s in SPLITS:
        w0, w1 = windows[s]
        print(f"    {s:<5s} {w0.date()} {sym('arrow')} {w1.date()}  "
              f"({(w1 - w0).days} days)")

    # ── Step 1: population ──
    print("\n[1/7] Building account population...")
    pop = build_population()
    for role in list(ROLE_MIX) + ["consumer"]:
        print(f"    {role:<10s} {len(pop.by_role[role]):>5,}")

    # ── Step 2: organic traffic ──
    print("[2/7] Emitting organic traffic from the relationship graph...")
    organic_df = generate_organic_transactions(pop, TIME_START, TIME_END)
    by_role = organic_df["edge_role"].value_counts()
    print(f"  {len(organic_df):,} organic edges")
    for role, n in by_role.items():
        print(f"    {role:<24s} {n:>8,}")

    # ── Step 3: disjoint hijack/feeder pools ──
    print("[3/7] Partitioning disjoint hijack pools per split...")
    pools = build_hijack_pools(pop.all_accounts)
    for s in SPLITS:
        print(f"    {s:<5s} pool: {len(pools[s]):,} accounts")

    # ── Step 4: rings, confined to their own window ──
    print(f"[4/7] Injecting {NUM_MULE_RINGS} mule rings across 3 archetypes...")
    ring_frames: list[pd.DataFrame] = []
    offset = 0
    for s in SPLITS:
        n_rings = RINGS_PER_SPLIT[s]
        w0, w1 = windows[s]
        frame = generate_mule_rings_for_split(
            split=s,
            num_rings=n_rings,
            hijack_pool=pools[s],
            window_start=w0,
            window_end=w1,
            ring_id_offset=offset,
        )
        offset += n_rings
        ring_frames.append(frame)
        mix = pd.Series(plan[s]).value_counts().to_dict()
        mix_str = ", ".join(f"{k}:{v}" for k, v in sorted(mix.items()))
        print(f"    {s:<5s} {n_rings:>2} rings ({mix_str}) "
              f"{sym('arrow')} {len(frame):,} edges")

    mule_df = pd.concat(ring_frames, ignore_index=True)

    # ── Step 5: retire hijacked accounts after their ring window ──
    print("[5/7] Retiring hijacked accounts post-ring...")
    organic_df, n_retired, n_removed = retire_hijacked_accounts(
        organic_df, mule_df, windows
    )
    print(f"  {n_retired} accounts retired, {n_removed:,} organic edges removed")

    # ── Step 6: merge, assign splits, verify ──
    print("[6/7] Merging and verifying invariants...")
    combined = pd.concat([organic_df, mule_df], ignore_index=True)
    combined["ring_id"] = combined["ring_id"].fillna(NO_RING_ID).astype(int)
    combined["ring_type"] = combined["ring_type"].fillna(NO_RING_TYPE)
    combined = combined.sort_values("timestamp").reset_index(drop=True)
    combined["split"] = assign_splits_vectorised(combined["timestamp"], windows)
    combined = combined[EDGE_COLUMNS]

    print(f"  Total edges:  {len(combined):,}")
    print(f"  Ring edges:   {(combined['edge_role'] == 'ring').sum():,}")
    print(f"  Fan-in edges: {(combined['edge_role'] == 'fan_in').sum():,}")

    splits = {s: combined[combined["split"] == s].copy() for s in SPLITS}

    assert_temporal_order(splits)
    assert_no_entity_leakage(splits)
    assert_structural_sanity(splits)
    print(f"  {sym('ok')} temporal order, entity/ring disjointness and "
          f"structural sanity verified")

    # ── Step 7: save ──
    print("[7/7] Saving...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for s in SPLITS:
        df = splits[s]
        path = OUTPUT_DIR / f"{s}_edges.csv"
        df.to_csv(path, index=False)
        members = ring_member_nodes(df)
        nodes = set(df["sender"]) | set(df["receiver"])
        print(f"  {sym('arrow')} {path.name:<24s} {len(df):>8,} edges | "
              f"{len(nodes):>5,} accounts | "
              f"{len(members):>4,} ring members "
              f"({len(members) / max(len(nodes), 1):5.2%} positive)")

    # Everything the model is allowed to have seen by deployment time.
    #
    # This is the VAL window alone, not train+val. The API rebuilds its
    # historical graph from this file and computes features on it, so the file
    # must be exactly one observation window long — the same length the model
    # was trained on. train+val would be two windows (120 days), which would
    # hand the API doubled degrees, doubled sums and doubled velocities and make
    # every served score quietly wrong. Nothing would raise; the model would
    # just be reading features off a different scale than it was fitted to.
    #
    # Using the most recent completed window as the serving lookback is also
    # what a real deployment does: you score against recent history, not against
    # everything since inception.
    context = splits["val"].copy()
    context = context.sort_values("timestamp").reset_index(drop=True)
    context_path = OUTPUT_DIR / "serving_context_edges.csv"
    context.to_csv(context_path, index=False)
    ctx_days = (
        context["timestamp"].max() - context["timestamp"].min()
    ).total_seconds() / 86_400.0
    print(f"  {sym('arrow')} {context_path.name:<24s} {len(context):>8,} edges "
          f"({ctx_days:.0f}d = one observation window, the API's lookback)")

    print_difficulty_report(splits)

    print()
    print(hr())
    print(f"{sym('ok')} Data generation complete.")


if __name__ == "__main__":
    main()
