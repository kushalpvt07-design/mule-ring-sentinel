# UPI Mule-Ring Sentinel

**Detects circular mule-account rings in UPI transaction graphs, prices every
decision in rupees, and never enforces.** Built for Razorpay Buildathon 2026,
Track 2 — AI Risk Manager.

The loss class is **money-mule laundering rings**: accounts that receive fraud
proceeds and pass them on in circles to break the audit trail. A single mule hop
is unremarkable in isolation, so the design premise was that the pattern lives in
the *shape* of the network.

The ablation in [Results](#results) qualifies that premise rather than confirming
it, and the qualified version is the more useful claim. Strip all five structural
features and the remaining thirteen still rank almost as well — as a *ranking*
problem this is not one that requires a graph. What the structural features buy is
the alert queue: at effectively the same recall they cut false positives to a
fraction, because circulation is what separates a mule from a merely busy account.
So the honest framing is that graph structure is what makes the queue cheap enough
for a human to work, not what makes the signal visible in the first place.

The strongest action this system can emit is `HOLD_FOR_REVIEW`. It has no code
path that can ban, block, freeze, suspend, terminate, disable, revoke, seize or
close anything, and that is enforced at import time and by test, not by
convention. See [Defense-only, by construction](#defense-only-by-construction).

**The data is synthetic, and that is the first thing to know before reading any
number below.** No labelled UPI mule-ring dataset exists publicly, so the traffic
and the rings were generated here — which makes every metric exactly as
trustworthy as the generator's assumptions, and those assumptions are written
down rather than buried. What this establishes is a *mechanism*: these features
separate these ring archetypes, and the leakage gate demonstrates that no single
column carries the label. What it does not establish is that real mule rings take
these shapes, and no amount of work inside this repo could — that needs labelled
production data. See [Why a Custom Synthetic
Generator?](#why-a-custom-synthetic-generator) for why an off-the-shelf simulator
would not do, and [Limitations](#limitations) for the full scope boundary.

---

### Why a Custom Synthetic Generator?

Real-world payment graphs carry no ground-truth labels for unflagged accounts, which makes precise cost-matrix evaluation (₹2,00,000/miss vs. ₹15,000/false alert) impossible on production data — you cannot score a confusion matrix without knowing which accounts are truly mules. The AML/ML literature relies on synthetic injection for exactly this reason (IBM's AMLSim being the canonical example). Off-the-shelf simulators, however, could not be used here for four structural reasons:

* **Topological ground truth vs. single-row fraud.** Simulators like PaySim label individual transaction rows, not multi-hop network objects. A topological detector needs ring-level entities (`ring_id`, `ring_type`, `hijack_prob`) to build ring-disjoint temporal splits and to measure per-archetype recall (fast cycles vs. stealth cycles) — something row-level labels cannot express.
* **The UPI organic substrate.** Graph-typology simulators such as AMLSim do inject rings, but their transaction dynamics are not calibrated to UPI: instant, 24×7, zero-fee VPA-to-VPA micropayments. Detection difficulty lives in how well a ring hides inside *realistic* background traffic — salary-day bursts, kirana merchant fan-ins, high-velocity P2P reciprocity — and reproducing that substrate required a UPI-specific generator rather than adapting a simulator built for a different payment regime.
* **Temporal containment as a serving-scope boundary.** Constraining 100% of a ring's timeline to a single 60-day split guarantees zero positive-entity leakage across train/val/test (verified: ring-ID overlap across splits is 0) and mirrors the API's fixed 60-day serving context. Rings that straddle a window edge in production are an explicit out-of-scope boundary for v4, not an oversight.
* **Engineered prevalence.** Real money-mule prevalence is a fraction of a percent. Synthetic prevalence was deliberately elevated to ~4–7% to give XGBoost enough positive-class representation to learn from without extreme-imbalance degradation. The decision *economics* — not the base rate — are reintroduced downstream through the rupee cost matrix; absolute precision figures are therefore reported at the elevated prevalence and would compress at production base rates (see [Limitations](#limitations)).

## Results

<!-- METRICS:BEGIN -->

Held-out **test** split, threshold selected on **validation** (never on test). Model `sentinel_v4`, 18 features, trained 2026-08-30T16:14:21+00:00.

| | |
|---|---|
| ROC-AUC | 0.9739 (95% CI 0.952–0.990) |
| Average precision | 0.7997 |
| Precision | 42.9% |
| Recall | 87.3% |
| F1 | 0.575 |
| Operating threshold | 0.1257 |
| Total cost on test | ₹47,20,000 |

Costs are assumptions, not measurements: a missed mule is priced at ₹2,00,000 and a false alert at ₹15,000 — a ratio of 13.3333 : 1. Only the ratio sets the threshold. The break-even probability that follows from it is **6.98%**, which is also the break-even *precision* of the alert queue: any queue cleaner than that is cheaper to work than to ignore.

### What it cost to pick the threshold honestly

The operating threshold 0.1257 was chosen on validation, where it scored recall 87.4% at precision 50.9%. Applied unchanged to test it gives recall 87.3% at precision 42.9% — recall fell, precision fell, and neither was tuned to make that happen.

Had the threshold been chosen *with test labels in hand* it would have been 0.0700 and cost ₹45,00,000 instead of ₹47,20,000. **That ₹2,20,000 difference — 5% of the reported total — is the price of not peeking**, and it is published because the alternative — quietly reporting the cheaper number — is the exact failure this repo already shipped once.

The run flagged this risk before test was ever touched: the validation cost plateau spans 0.1256–0.1259, width 0.0002 — **narrow**, so the training run warned in advance that this threshold might be fitted to validation noise. Total cost is a step function of the threshold, so a narrow plateau means the minimum was a knife-edge rather than a basin, which is precisely when a validation-selected cutoff transfers poorly.

### Calibration, and why the threshold is not the break-even p\*

Break-even p\* is the correct score cutoff only for a *calibrated* model, and this one is deliberately not calibrated: `scale_pos_weight` trades calibration away to fight 3.6% prevalence, and the scores come out inflated. Mean predicted probability is 0.0461 against an actual rate of 0.0360; Brier score 0.0162, expected calibration error 0.0101.

| score bin | accounts | mean predicted | observed rate |
|---|---|---|---|
| `[0.0, 0.2)` | 2877 | 0.0090 | 0.0073 |
| `[0.2, 0.4)` | 61 | 0.2856 | 0.1967 |
| `[0.4, 0.6)` | 20 | 0.5095 | 0.2500 |
| `[0.6, 0.8)` | 22 | 0.7036 | 0.1818 |
| `[0.8, 1.0)` | 75 | 0.9579 | 0.9067 |

The model over-states risk in **every** bin, the top one included, so the scores are not probabilities. That is why p\* is used here as a statement about the **alert queue** — break-even precision — and never as a score cutoff. The cutoff is the empirical validation optimum. Reporting p\* as though it were the operating threshold would be arithmetically tidy and wrong.

### How far off the probabilities are

A two-parameter Platt map (a logistic fit on the score log-odds) was fitted on **validation** and measured on **test**, the same split discipline every threshold here follows. Slope 0.7801, intercept -0.8744: it shrinks the log-odds — the signature of **over**-confident scores, spread wider than the evidence supports, which is the expected effect of `scale_pos_weight`. Two parameters rather than an isotonic fit because the validation split carries only 127 positives, and a free-form monotone fit on that many points memorises them.

| | raw | calibrated | change |
|---|---|---|---|
| Brier score | 0.01621 | 0.01409 | -0.00212 |
| Expected calibration error | 0.01005 | 0.00496 | -0.00509 |
| ROC-AUC | 0.97386 | 0.97386 | +0.000000 |

ROC-AUC does not move at all — the map is monotone, so it can compress two adjacent scores into a tie but cannot swap a pair. **The alert queue is in the same order.** That is what makes this a diagnostic and not an undisclosed model change: nothing above or below this section is computed on the calibrated scale.

The map is reported and never applied. The operating threshold in the headline table was selected on the raw scores, so calibrating underneath it would move the published precision, recall and cost without changing a single number in this README. If a downstream consumer needs scores that read as probabilities — expected-loss arithmetic, or blending with another model's output — apply this map at that boundary and re-select the threshold on the calibrated scale.

### Does the operating point survive a different cost ratio?

The FN:FP ratio is the one number in the cost model nobody can verify, so a result quoted at a single ratio is not a result. Each row picks its threshold on validation and is then evaluated once on test at that frozen value — the same discipline as the headline.

| FN:FP | break-even p\* | threshold (val) | test precision | test recall | test cost |
|---|---|---|---|---|---|
| 2.0 : 1 | 33.33% | 0.5745 | 72.3% | 66.4% | ₹15,30,000 |
| 5.0 : 1 | 16.67% | 0.4094 | 65.8% | 70.0% | ₹30,75,000 |
| 10.0 : 1 | 9.09% | 0.2301 | 53.8% | 78.2% | ₹47,10,000 |
| 13.333333333333334 : 1 | 6.98% | 0.1257 | 42.9% | 87.3% | ₹47,20,000 |
| 25.0 : 1 | 3.85% | 0.0686 | 35.1% | 91.8% | ₹61,80,000 |
| 50.0 : 1 | 1.96% | 0.0284 | 25.1% | 94.5% | ₹91,50,000 |

### Does the model earn its complexity?

Every row is priced on the same cost model, with each baseline's own threshold also selected on validation. This is the table that decides whether a graph pipeline was worth building.

| | precision | recall | F1 | alerts/1k | total cost |
|---|---|---|---|---|---|
| flag nothing | — | 0.0% | — | 0.0 | ₹2,20,00,000 |
| flag everything | 3.6% | 100.0% | 0.070 | 1000.0 | ₹4,41,75,000 |
| one-line rule: `cycle_participation >= 0.0736` | 11.1% | 85.5% | 0.196 | 278.2 | ₹1,45,40,000 |
| logistic regression, same features | 17.1% | 78.2% | 0.281 | 164.3 | ₹1,10,40,000 |
| XGBoost, no graph features | 38.7% | 80.9% | 0.523 | 75.3 | ₹63,15,000 |
| **XGBoost, full model** | **42.9%** | **87.3%** | **0.575** | **73.3** | **₹47,20,000** |

Against the identical model trained without them, graph features avoid ₹15,95,000 of cost and move average precision by +0.0400.

That cost figure understates them, and the reason is worth stating: with a miss priced at 13.3333× a false alert, total cost is nearly insensitive to precision — and precision is exactly where the graph features land. False positives fall from 141 to 128 (9% fewer) at higher recall (80.9% → 87.3%), shrinking the review queue from 75.3 to 73.3 alerts per 1,000 accounts. An analyst feels that; the cost model barely does.

Read the other way, this ablation is also the strongest argument against the framing at the top of this README. 13 non-structural features reach test AUC 0.9572 on their own, against 0.9739 with the full set — 96.5% of the above-chance separation, so most of the separating signal is reachable without a graph at all. The structural features sharpen the queue; they do not find a fundamentally different population of mules. Anyone deciding whether to build this pipeline should weigh that.

### What happens when the review queue is capped

Every row above assumes an analyst queue of unlimited size. Cap it at 20 alerts per 1,000 accounts — a stated assumption, with the threshold that satisfies it chosen on validation like every other threshold here — and the model is pushed to the high-precision end of its own curve: threshold 0.9755, precision 97.9% at recall 41.8%, 15.4 alerts per 1,000 accounts, total test cost ₹1,28,15,000.

Even under the cap the model stays below the one-line rule (₹1,45,40,000), which the rule reaches only by issuing 278.2 alerts per 1,000 accounts — a queue the budget forbids outright.

#### The same cap, applied to every policy

Each threshold below was re-selected **on validation** to fit the same 20 alerts per 1,000 accounts, then applied once to test. The two trivial policies have no threshold to select and are priced closed-form.

| policy under the cap | val threshold | precision | recall | F1 | alerts / 1,000 (test) | total cost |
|---|---|---|---|---|---|---|
| one-line rule: cycle_participation >= 0.0736 | 0.9000 | 9.2% | 9.1% | 0.091 | 35.7 ⚠ | ₹2,14,85,000 |
| logistic regression, same features | 0.9494 | 66.2% | 42.7% | 0.519 | 23.2 ⚠ | ₹1,29,60,000 |
| XGBoost, no graph features | 0.9602 | 97.9% | 41.8% | 0.586 | 15.4 | ₹1,28,15,000 |
| **XGBoost, full model** | 0.9755 | 97.9% | 41.8% | 0.586 | 15.4 | ₹1,28,15,000 |
| flag nothing _(infeasible)_ | — | — | 0.0% | — | 0.0 | ₹2,20,00,000 |
| flag everything _(infeasible)_ | — | 3.6% | 100.0% | 0.070 | 1000.0 | ₹4,41,75,000 |

⚠ marks a policy whose validation-selected threshold overflowed the cap on test — two of four here, the widest at 35.7 alerts per 1,000 against a budget of 20. Each threshold was frozen on validation and the test score distribution is not identical, so the queue comes out a different size. It is reported rather than corrected: re-solving the threshold on test until the constraint held would be threshold selection on test, which is the one thing this project refuses to do anywhere else.

With the cap binding on everyone, the model is the cheapest policy in the table at ₹1,28,15,000 against ₹1,28,15,000 for the next best (XGBoost, no graph features) — ₹0 cheaper on the same queue size. That is the comparison the capped claim above rests on. The one-line rule's apparent win on cost alone was bought with a queue the cap does not permit, not with better detection.

### The same operating point at a realistic base rate

Precision and rupee cost depend on how many mules there are; recall, ROC-AUC and the leakage headroom do not. So the honest way to read the numbers above is to re-price them at base rates a real network would see. Nothing is retrained here — the class-conditional score distributions are held fixed and only the mix is re-weighted.

| mule base rate | recall | precision | alerts / 1,000 | cost / 1,000 accounts | queue pays for itself |
|---|---|---|---|---|---|
| **3.601%** _(measured)_ | 87.3% | 42.86% | 73.3 | ₹15,45,019 | yes |
| 2.000% | 87.3% | 29.07% | 60.0 | ₹11,48,004 | yes |
| 1.000% | 87.3% | 16.86% | 51.8 | ₹8,99,978 | yes |
| 0.500% | 87.3% | 9.17% | 47.6 | ₹7,75,965 | yes |
| 0.200% | 87.3% | 3.87% | 45.1 | ₹7,01,558 | **no** |
| 0.100% | 87.3% | 1.97% | 44.3 | ₹6,76,755 | **no** |

Recall holds at 87.3% down the whole table, which is not an approximation: it is a within-class rate, so no change of base rate can touch it. The break-even line falls between 0.200% and 0.500%: below roughly that base rate this operating point's queue costs more to work than to ignore, because precision drops under the 6.98% break-even the cost model implies. That is a real limit on the result and not a fixable one at this threshold — a rarer mule population needs a tighter cutoff, which trades recall for a queue worth working.

The bolded row is the one actually measured: 3.601% prevalence on the test split. Every other row is arithmetic on that measurement, and answers "what would this queue look like if mules were rarer" — not "what will this model do in production", which needs real data and is out of scope. See Limitations.

### Is the task actually hard?

The strongest *single* feature on test is `cycle_participation` at direction-corrected AUC **0.8183**, against a leakage ceiling of 0.99. If any one column reached that ceiling the generator would be planting the label and every number above would be theatre — `tests/test_baselines.py` recomputes this from the shipped CSVs and fails the build if it ever does.

### Recall by ring archetype

Reported separately because one headline recall hides *which* laundering shapes the model misses, and the three archetypes are of deliberately unequal difficulty. Read the account column first: ring recall counts a ring as caught if *any* member is flagged, so it flatters a model whenever rings have several members, and a saturated metric is no evidence of a good one. Ring recall is still the operationally meaningful quantity — one alert brings an analyst to the whole ring.

| archetype | accounts | account recall | rings | ring recall |
|---|---|---|---|---|
| `fast_cycle` | 38/38 | 100.0% | 8/8 | 100.0% |
| `layered_fanin` | 21/21 | 100.0% | 6/6 | 100.0% |
| `stealth_cycle` | 37/51 | 72.5% | 10/10 | 100.0% |
| **all rings** | | | **24/24** | **100.0%** |

### Data the numbers were measured on

| split | accounts | mules | prevalence | rings |
|---|---|---|---|---|
| train | 3091 | 190 | 6.15% | 40 |
| val | 3060 | 127 | 4.15% | 24 |
| test | 3055 | 110 | 3.60% | 24 |

Splits are consecutive equal-length time windows — no account's future is used to predict its past, and no ring appears in two splits.

_Generated from `models/saved_models/metrics.json` by `python -m models.report --write`. Do not hand-edit._
<!-- METRICS:END -->

Nothing in that section is typed by hand. `python -m models.report --write`
regenerates it from the `metrics.json` the training run wrote, and
`--check` fails if the two have drifted. The reason for the machinery is that
this repo has already shipped a dishonest number once: v2's `metrics.json`
advertised ROC-AUC 0.9999 for a booster that had been retired, and nothing in
the project noticed. A README that quotes metrics by hand is that same bug with
extra steps.

---

## What "honest metrics" is being taken to mean

The track bar asks for honest metrics including false-positive cost. Five things
follow from taking that literally, and all five are load-bearing here.

**A number is only honest next to its baseline.** ROC-AUC on an imbalanced
fraud task is easy to make look good and hard to interpret. So every headline
figure is published beside the trivial policies (flag everyone, flag nobody), the
best single-feature threshold rule, a logistic regression on identical features,
and the same XGBoost trained *without* graph features. If the one-line rule wins,
the graph pipeline was not worth building, and the table says so out loud.
`tests/test_baselines.py` fails the build if the full model does not beat both
the cheapest trivial policy and the best single-feature rule on cost.

**A number is only honest if the task was hard.** A generator can trivially
plant the label in a feature and produce a 0.999 model. So the strongest
direction-corrected single-feature AUC is recomputed from the shipped test CSV on
every test run and checked against a ceiling of 0.99. Direction correction
matters: a feature at raw AUC 0.13 is not weak, it is strongly inverted, and a
model free to flip a sign already has that separating power. The check is applied
to `max(auc, 1 − auc)`, and the AUC helper itself is pinned to 1.0 / 0.5 / 0.0 on
perfect / constant / inverted input first, because a ceiling built on a broken
ruler is worse than no ceiling.

**A false positive has a price, and it is not the price of a block.** Both cost
figures are stated as assumptions in `models/cost_matrix.py` rather than
presented as measurements, and because only their *ratio* selects a threshold,
`sensitivity_to_cost_ratio()` publishes how the operating point moves as that
ratio changes. A threshold that only looks good at one assumed ratio is not a
result. The false-positive cost is deliberately modelled as an analyst's
attention and a customer's friction — not a frozen account — because this system
never freezes accounts.

**The threshold is never selected on test.** Splits are strictly temporal:
train, then validation, then test, in time order, with zero positive-account and
zero `ring_id` overlap. The operating threshold is chosen on validation and
applied unchanged to test. Organic accounts *do* recur across splits, which is
realistic — the same person keeps banking — and is not leakage; what must not
recur is a ring or a labelled mule, and `tests/test_leakage.py` asserts both.

This claim held for the headline number and quietly failed elsewhere, which is
worth recording rather than tidying away. The cost-ratio sensitivity table used
to pick *each row's* threshold by optimising on test, so the row at the shipped
ratio had become an unlabelled copy of the deliberate test-peeking diagnostic and
understated cost there by 28%. The table now selects every row on validation and
evaluates once on test at that frozen value, and
`tests/test_baselines.py::TestSensitivityTableDoesNotPeekAtTest` fails the build
if any published operating point is ever re-optimised on test again — including
under renamed columns.

**The live demo is in-sample, and the response says so.** The threshold is
selected on validation, and the graph the API scores against —
`serving_context_edges.csv` — is the validation split, edge for edge. So a
`/score` call against a context account computes that account's features over the
exact transactions the threshold was tuned on: it is in-sample with respect to the
graph and the operating point, and a precision figure read off a running demo
would be optimistic. No labels leak — the model never trains on validation, and
the split-integrity checks are about labels — so nothing above is weakened; the
honest out-of-sample figures are the test-split ones in [Results](#results), not
anything measurable from the endpoint. Every `/score` response carries this in its
`context.context_provenance` field, so a reviewer reading one JSON body does not
have to infer it — and serving the context from a fourth, held-out window is the
documented fix if this is ever pointed at something real.

---

## How it works

Five stages, each a module you can run on its own.

**1. Generate** (`data/generator.py`) — 3,000 accounts over six months of
synthetic UPI traffic, split into three equal ~60-day windows. Legitimate traffic
is not i.i.d. noise: accounts have recurring counterparties, salary-driven payday
bursts, reciprocal social pairs, rotating settlement groups and a long tail of
one-off payments. That realism is not decoration. In v2, organic accounts drew
counterparties independently at random, so they essentially never paid anyone
twice, and `repeat_ratio` alone hit test AUC 0.9989 — the model was reading an
artifact of the generator.

Eighty-eight rings are seated across the three windows in three archetypes, and
the mix is deliberately uneven in difficulty: `fast_cycle` (35% — near-identical
amounts, one hard push over hours, the classic signature), `stealth_cycle` (40% —
two to five transactions per hop, 30% amount jitter, smeared over weeks, a third
never closing the loop) and `layered_fanin` (25% — 15 to 40 feeders, structurally
indistinguishable from a merchant with customers). Roughly half of all ring
members are hijacked accounts with genuine prior history rather than fresh
signups — the stealth archetype prefers them, at 65% — and purpose-built mules
are given a civilian transaction life alongside their ring traffic. If every
positive looked the same, one decision path would saturate recall and the
reported number would mean nothing.

**2. Extract** (`data/extractor.py`) — builds a directed multigraph per window
and computes 18 account-level features: degree structure, value flow and
pass-through, PageRank and clustering, cycle participation, reciprocity, fan-in
concentration, velocity and burst, amount dispersion (overall and
per-counterparty), repeat ratio, and community closure. The full contract, with
the design rules every feature must satisfy and the reasons `louvain_community`,
`community_size` and `net_flow` were each dropped, is at the top of
`models/features.py`.

**3. Train** (`models/train.py`) — XGBoost with early stopping on validation,
ring-grouped cross-validation (grouping on `ring_id`, not Louvain community —
community size is unbounded and unrelated to the label, and on v2's data one
group swallowed 52% of train accounts and zero positives, so folds were silently
skipped), a cost-optimal threshold sweep, all five baselines, gain and mean-|SHAP|
importance, and per-archetype recall. It refuses to save if TreeSHAP
attributions do not reconstruct the model's own scores to 1e-5 — because those
attributions are what the API returns as explanations, and an API explaining a
different model than it scored with would be wrong silently and forever.

**4. Serve** (`api/main.py`) — FastAPI. `POST /score` takes a batch of
transactions, merges them into a reference graph, recomputes features through the
*same* `compute_node_features` call training used, scores, and returns per-account
risk with SHAP-derived contributing factors in analyst language ("money in ≈
money out", not `flow_passthrough=0.98`). `GET /health` reports model, version,
threshold and context state.

**5. Inspect** (`dashboard/app.py`) — Streamlit: an overview keyed to the real
`metrics.json`, a PyVis ring explorer, a cost-curve page where you move the
FN:FP ratio and watch the operating point move, and a live scoring demo that
posts to the running API.

---

## Two things that are easy to get wrong, and how they are handled

### The scoring threshold, and the two different jobs one number is doing

Flagging an account is worth it when the expected cost of flagging is below the
expected cost of not flagging. For an account with probability *p* of being a
mule, that is `(1 − p)·fp_cost < p·fn_cost`, so the break-even probability is
`p* = fp_cost / (fp_cost + fn_cost)` — derived from the cost assumptions, not
fitted to a dataset. At the assumed ₹15,000 and ₹2,00,000 that is p\* ≈ 0.0698.

It is tempting to use that directly as the score cutoff, and this README used to
imply it. That is only valid for a *calibrated* model, and this one is
deliberately not calibrated: `scale_pos_weight` inflates scores to fight a ~4%
prevalence, so a predicted 0.29 corresponds to a much lower observed rate. The
reliability table in the Results block shows the model over-stating risk in every
bin but the top one. Feeding p\* to an uncalibrated score would flag a large
multiple of the accounts it should.

So the number does one job and not the other. As a statement about the **alert
queue** it stands unmodified: break-even precision is about 7%, and any queue
cleaner than that is cheaper to work than to ignore — which is a far more
defensible thing to report than a tuned cutoff, and it is the figure to compare
the Results block's precision against. As a **score cutoff** it is not used at
all. The cutoff is chosen empirically on validation and frozen before test.

That empirical sweep is exact rather than gridded (total cost is a step function
of the threshold, changing only when the threshold crosses an observed score) and
it prefers the *centre of the widest cost plateau* to a knife-edge minimum. It
also publishes the plateau's width and warns when it is narrow, because a narrow
plateau is the signature of a threshold fitted to validation noise. On this run
the warning fired, and the Results block reports what it cost on test rather than
burying it: the gap between the frozen threshold and the one test labels would
have chosen is stated in rupees.

### Feature stability, which is the requirement that actually costs something

Determinism is the easy half: score the same graph twice, get the same answer.
The half that costs money is stability — adding an unrelated account, or a payment
between two strangers, must not move *your* features.

`community_internal_ratio` failed this outright. Louvain's partition is
reproducible on a fixed node set but moves the moment the node set changes, and
because the ratio is a per-community scalar broadcast to every member, one
repartition moves the feature for accounts nowhere near the change. Measured on
the 2,954-account validation graph: two accounts transacting only with each other,
connected to nothing else, moved this feature for **100% of accounts** and flipped
84 decisions — 2.84% of the population — at the cost-optimal threshold. Somebody
else's transaction changed your risk score.

The fix is not to drop the feature. Serving computes the Louvain partition **once**
from the reference graph and passes it into `compute_node_features(..., partition=…)`,
which extends it deterministically for accounts it has not seen. With that path
in place the same perturbation moves the feature for **0 of 2,954** accounts.
`tests/test_features.py` asserts bit-identical values for 17 of 18 features under
perturbation, so deleting that plumbing breaks the build.

PageRank is the one exemption from bit-identity, and it is a principled one: it is
a global fixpoint over a normalised rank vector, computed by an iterative solver
that stops at a tolerance rather than at a fixed point. The invariant it is held to
is stronger than "small", though. v4 emits it as `pagerank × N` — mean 1.0 rather
than a value that tracks node count — and for *this* perturbation that scaling is
**exactly** cancelling, not approximately: two new accounts transacting only with
each other form a closed component, so the only thing that changes for an existing
account is PageRank's uniform teleport term, every raw value scales by exactly
N/N′, and multiplying by the new N undoes it in full. The expected drift is zero.

That derivation is what fixed the guard. An earlier absolute budget of 1e-3 was the
wrong *shape* rather than the wrong number — an absolute tolerance on a quantity
defined as a multiple of the graph's own baseline means something different in every
graph — and the retrain measured 6.9e-3 against it, which turned out to be the
solver's own residual read on a column N times larger. The guard is now a relative
magnitude budget, a rank-order assertion (Spearman ≥ 0.9999, which is what a
tree model actually consumes), and `mean == 1.0` to pin the ×N scaling itself so the
relative budget cannot pass vacuously. See rule 5 in `models/features.py`.

---

## Defense-only, by construction

> "Strictly defense-only: anything offense-capable is disqualified."

The four actions that exist are `ALLOW`, `FLAG_FOR_REVIEW`, `HOLD_FOR_REVIEW` and
`REQUIRE_ADDITIONAL_AUTH`. There is no fifth. A human analyst makes every
enforcement decision, which is both the ethical position and the practical one:
the false-positive cost in the model is calibrated to an analyst's attention
precisely because nothing is ever frozen automatically.

Three independent rails hold this, in `api/responder.py`:

A **blocklist** rejects named enforcement actions. A **substring rail** rejects
any action name containing an enforcement verb — `BAN`, `BLOCK`, `FREEZE`,
`SUSPEND`, `TERMINATE`, `DISABLE`, `REVOKE`, `SEIZE`, `CLOSE` — which is what
makes the guarantee hold against action names nobody has invented yet. A blocklist
only catches what someone thought of; `DISABLE_VPA` is not on any list and is
plainly enforcement. And the whole `ActionCode` enum is **swept at import time**,
so a future contributor adding `FREEZE_FUNDS` to the enum cannot start the
process, let alone serve a request.

Then `validate_response_batch` re-derives the tier and the action from each
score before anything leaves the API. This catches the failure a forbidden-action
screen cannot see: a response with a 0.96 risk score wearing `ALLOW`. `ALLOW` is a
permitted value, so a blocklist waves it through — only re-deriving the tier
catches the suppressed alert. `tests/test_responder.py` hand-forges exactly that
response and requires the validator to raise.

---

## Quick start

```bash
pip install -r requirements.txt
```

The generated CSVs and the trained booster are committed, so a fresh clone works
on arrival — no pipeline run needed:

```bash
uvicorn api.main:app --reload --port 8000    # docs at /docs
streamlit run dashboard/app.py               # http://localhost:8501
pytest tests/ -v                             # add -m "not slow" for a fast loop
python -m models.report --check               # exit 1 if Results is stale
```

To rebuild everything from scratch rather than trust the shipped artifacts, run
the pipeline in this order. `data.extractor` must precede `models.train`, because
train's ring-attribution step reads columns the extractor writes — run it the
other way and that figure publishes as `null`:

```bash
python -m data.generator     # → data/raw/{train,val,test,serving_context}_edges.csv
python -m data.extractor     # → data/processed/{train,val,test}_features.csv
python -m models.train       # → models/saved_models/<features.MODEL_NAME> + metrics.json
python -m models.report --write   # fills the Results section of this README
```

Regenerating will not reproduce the committed CSVs row-for-row. `SEED = 42` is
fixed, but a seed only reproduces data while the generator consumes the same
number of random draws, so the distributions match while the individual accounts
do not. Retrain after regenerating, or the committed `metrics.json` describes a
dataset that is no longer on disk — which the suite will tell you about rather
than let you ship.

`models.train` prints the baseline comparison table, per-archetype recall, the
leakage check and the SHAP-identity confirmation. Read that output rather than
trusting the model silently: it is where the pipeline tells you whether the run
is trustworthy.

```bash
uvicorn api.main:app --reload --port 8000    # docs at /docs
streamlit run dashboard/app.py               # http://localhost:8501
pytest tests/ -v                             # add -m "not slow" for a fast loop
python -m models.report --check               # exit 1 if Results is stale
```

A worked `POST /score` call, with a payload dated inside the serving window:

```bash
curl -X POST http://localhost:8000/score \
  -H "Content-Type: application/json" \
  -d '{"transactions": [
        {"sender": "victim@upi",  "receiver": "hop1@upi", "amount": 48000,
         "timestamp": "2025-04-14T02:31:00"},
        {"sender": "hop1@upi",    "receiver": "hop2@upi", "amount": 47500,
         "timestamp": "2025-04-14T03:02:00"},
        {"sender": "hop2@upi",    "receiver": "victim@upi", "amount": 47000,
         "timestamp": "2025-04-14T03:44:00"}
      ]}'
```

The timestamps matter. Features include raw rupee sums, so the API scores inside a
fixed-length window and warns when submitted transactions imply a different span
than the model was trained on. Dates inside 2025-03-02 … 2025-04-30 land in the
serving window; a payload from 2019 will score, and will tell you it is skewed.

---

## Repo map

```
data/generator.py       synthetic UPI traffic + 88 rings in 3 archetypes
data/extractor.py       graph construction → 18 account features
models/features.py      THE FEATURE CONTRACT + design rules (read this first)
models/cost_matrix.py   cost model, exact threshold sweep, plateau selection
models/train.py         training, CV, all five baselines, per-archetype recall
models/explain.py       exact TreeSHAP via xgboost pred_contribs
models/report.py        renders this README's Results block from metrics.json
api/main.py             FastAPI: /health, /score — train/serve parity enforced
api/responder.py        the defense-only gate and risk tiering
api/schemas.py          pydantic request/response contracts
dashboard/app.py        Streamlit: overview, ring explorer, cost curves, demo
console.py              UTF-8 safe console output (Windows cp1252 workaround)
.gitattributes          LF normalisation, so a retrain diffs one number not 1,900
tests/                  see below
```

`models/features.py` is the single source of truth for the feature list.
`tests/test_contract.py` walks every module's AST and fails if any other file
hard-codes a feature list, because three copies of a feature list is three
chances for training and serving to drift apart.

---

## Build Quality & Engineering Standards

* **Single source of truth & contract enforcement.** The 18-feature contract lives only in `models/features.py`. An AST-walking suite (`tests/test_contract.py`) parses every source module and fails the build if any file hard-codes its own feature list, if `api/main.py` doesn't actually *call* the contract check, or if a model's columns don't match the contract in name **and order** — the exact defect that once served a stale 12-feature model against the current threshold with nothing raising.
* **Defensive testing.** Beyond accuracy, the suite guards structural invariants:
  * *Stability:* adding unrelated accounts to the graph leaves 17 of 18 features bit-identical (pagerank moves by ≤1e-5) — the guarantee that a Louvain re-partition cannot silently shift every score.
  * *Determinism & arithmetic:* end-to-end recomputation is bit-identical across runs, and the extractor's math is cross-checked against an independent plain-pandas reimplementation.
  * *Leakage gate:* a strict 0.99 ceiling on direction-corrected single-feature AUC (`LEAKAGE_AUC_CEILING`) — no single column may separate the classes, because if one could, the generator would have planted the label. Screened on validation as a decision gate and reconciled against the shipped test table in `tests/test_baselines.py`.
  * *Demo surface:* `tests/test_dashboard.py` tests the dashboard's scoring path behaviourally rather than only contract-checking its feature names — the label cannot reach the model, column order is enforced, the threshold has no default to fall back on, a `metrics.json` describing another model version stops the page, and an AST walk fails the build if a simulated score or a random draw reappears in a module that computes displayed figures.
  * *Traceability:* tests are documented with the specific production defect or regression they exist to prevent, so the suite reads as a changelog of failures already fixed.
* **Platform-independent artifacts.** `metrics.json` and this README's generated block are written with an explicit `newline="\n"`, and `.gitattributes` normalises the repository to LF. Python's text mode otherwise translates line endings per platform, so the same artifact written on Windows and on Linux differs on every line — and a retrain that moved one number arrives as a 1,900-line diff with the actual change buried inside it.

---

## How this was built

Written with heavy AI assistance, which is worth stating plainly rather than
leaving a reader to wonder about the volume of commentary in these modules. The
useful question is not whether a model helped write it, but where its suggestions
were overruled — so, three that were cut. The streaming layer: Kafka and PySpark
for a dataset that fits in memory is résumé architecture, not engineering. The
vector database: there is nothing here to retrieve, the representation is
eighteen floats per account. The agent framework: LangGraph orchestrating a
single scoring call buys a dependency and a failure mode in exchange for nothing.

The larger override is the model itself. The reflexive 2026 move is to point a
language model at the problem, and it would have been the wrong tool. This is a
tabular problem over a graph; the label is structural rather than semantic; and
gradient boosting over eighteen engineered features is both stronger here and
auditable by exact TreeSHAP in a way a prompt is not. A language model cannot tell
you that `cycle_participation` contributed 0.11 to a particular account's score,
and on a decision that costs a merchant real money, that attribution is the point.

What the assistance did not do is decide what to trust. Every figure in
[Results](#results) is generated from `metrics.json` by `python -m models.report
--write` rather than typed by hand, the operating threshold is chosen on
validation and never on test, and the defects catalogued in
[`AUDIT-2026-08-30.md`](AUDIT-2026-08-30.md) were found by auditing this code
against its own claims — including the one where the graph viewer inferred mule
labels from edge incidence and so painted fan-in victims as suspects, on the one
page whose entire subject is the cost of false accusation.

---

## Limitations

Stated, not hidden — and separated into the two kinds, because a scope boundary chosen on purpose and an unfixed defect deserve different amounts of a reader's suspicion. No live measurement is restated here: every figure referred to below lives in the generated [Results](#results) block, which `python -m models.report --write` rewrites from `metrics.json`, and so cannot drift away from the run it describes.

* **Elevated prevalence base rate.** Absolute precision and rupee cost are measured at the deliberately elevated synthetic prevalence described under [Why a Custom Synthetic Generator?](#why-a-custom-synthetic-generator), and would compress at real base rates, where mule prevalence is a fraction of a percent. Recall, false-positive rate and ROC-AUC are within-class quantities and do not move with the base rate; precision and total cost do. Rather than leave that as a caveat, Results carries the shipped operating point re-projected down a ladder of prevalences, marking the row that was actually measured and the point at which projected precision falls below break-even p\* — below that base rate the alert queue costs more to work than to ignore, which is the number a reviewer should actually want. The projection is arithmetic on the measured recall and false-positive rate, not a second experiment: it assumes the ranking transfers to the new base rate, which is exactly the assumption a real deployment would have to test.
* **Synthetic topology scope.** The generator establishes a *mechanism* — these graph features separate these three ring archetypes, and the leakage gate demonstrates no single column carries the label. It does not establish that real mule rings take these shapes, and no amount of work inside this repo could; that needs labelled production data. So this is a scope boundary and is deliberately not "addressed". What is done instead: the archetypes are stated explicitly, recall is reported per archetype so a reader can see which shape is hardest, and no claim is made beyond "the mechanism holds on the stated shapes".
* **Uncalibrated probabilities.** `scale_pos_weight` buys recall under imbalance by inflating scores, so a predicted score is not a probability. The shipped threshold is therefore a rank cutoff chosen by cost on validation, *not* the break-even p\*, and conflating the two is the mistake this repo is most careful about. Results publishes the size of the gap — Brier score and expected calibration error, before and after a two-parameter Platt map fitted on validation and measured on test — alongside ROC-AUC on both scales. The map is reported and **not** applied: the shipped threshold was selected on the raw scale, and rescaling scores underneath a threshold chosen for the old scale would move the operating point while looking like a free improvement. A Platt map is monotone non-decreasing, so it cannot reorder accounts and the ranking metrics are unchanged — an invariance the suite asserts rather than assumes.
* **Fixed-length window dependence.** v4 divides the 60-day window length out of the three features that scaled with it directly: `in_amount_sum`, `out_amount_sum` and `repeat_ratio` are rescaled to a 60-day reference window, so on the shipped equal-length splits the factor is 1.0001–1.0003 and no number in Results moves. What the rescale cannot touch is structure — window length changes the graph itself, because distinct counterparties saturate sublinearly, and that moves `degree_balance`, `reciprocity` and the cycle core. So equal-length windows remain a correctness requirement rather than a convenience: `assert_equal_window_lengths()` in the generator enforces it, `tests/test_leakage.py` covers it, and anything that builds a graph to score against must span one window — including `serving_context_edges.csv`. The API still states the window it was fitted for and warns when a submitted payload implies a different span, because a graph built over a different length is a different measurement even after the magnitudes are normalised.
* **Queue capacity constraints.** The cost matrix prices a false alert but otherwise assumes an analyst team large enough to work whatever queue the threshold produces. A capacity-constrained operating point is reported next to the unconstrained one, and the cap is applied to **every** policy in the comparison rather than only to the model — the earlier version priced the model with a capped queue while the baselines it was measured against could alert without limit, which is not a comparison so much as a handicap applied to one side. The residual limitation is real: the cap is a single assumed number, and where a validation-selected threshold overflows it on test the overflow is *reported* rather than re-solved, because re-fitting a threshold on the test split would be selection on test wearing a capacity argument as a disguise.
* **Demo surface coverage.** The Streamlit page is a demo, and in v2 it was worse than untested — its cost panel added the ground-truth label into the score it displayed, so every precision, recall and rupee figure on screen was fiction, computed against a score that already knew the answer. That path is gone: the dashboard scores through the real booster or renders the reason it cannot, and `tests/test_dashboard.py` pins the guarantees behaviourally — the label never reaches the model, column order is part of the contract, the threshold has no safe default, a stale `metrics.json` stops the page rather than mislabelling it, and neither a simulated score nor a random draw has come back. What remains genuinely untested is the *rendering*: nothing here drives a browser, so a chart could be mislabelled or a panel could show the wrong split without a test noticing. The numbers behind the page are pinned; their presentation is not.

---

## License

MIT — see [LICENSE](LICENSE).
