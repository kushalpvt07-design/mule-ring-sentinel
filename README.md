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

---

## Results

<!-- METRICS:BEGIN -->

Held-out **test** split, threshold selected on **validation** (never on test). Model `sentinel_v3`, 18 features, trained 2026-08-27T09:15:43+00:00.

| | |
|---|---|
| ROC-AUC | 0.9693 (95% CI 0.947–0.988) |
| Average precision | 0.8037 |
| Precision | 33.5% |
| Recall | 87.1% |
| F1 | 0.484 |
| Operating threshold | 0.1836 |
| Total cost on test | ₹64,10,000 |

Costs are assumptions, not measurements: a missed mule is priced at ₹2,00,000 and a false alert at ₹15,000 — a ratio of 13.3333 : 1. Only the ratio sets the threshold. The break-even probability that follows from it is **6.98%**, which is also the break-even *precision* of the alert queue: any queue cleaner than that is cheaper to work than to ignore.

### What it cost to pick the threshold honestly

The operating threshold 0.1836 was chosen on validation, where it scored recall 91.2% at precision 37.5%. Applied unchanged to test it gives recall 87.1% at precision 33.5% — recall fell, precision fell, and neither was tuned to make that happen.

Had the threshold been chosen *with test labels in hand* it would have been 0.5408 and cost ₹58,60,000 instead of ₹64,10,000. **That ₹5,50,000 difference — 9% of the reported total — is the price of not peeking**, and it is published because the alternative — quietly reporting the cheaper number — is the exact failure this repo already shipped once.

The run flagged this risk before test was ever touched: the validation cost plateau spans 0.1834–0.1837, width 0.0004 — **narrow**, so the training run warned in advance that this threshold might be fitted to validation noise. Total cost is a step function of the threshold, so a narrow plateau means the minimum was a knife-edge rather than a basin, which is precisely when a validation-selected cutoff transfers poorly.

### Calibration, and why the threshold is not the break-even p\*

Break-even p\* is the correct score cutoff only for a *calibrated* model, and this one is deliberately not calibrated: `scale_pos_weight` trades calibration away to fight 4.3% prevalence, and the scores come out inflated. Mean predicted probability is 0.0872 against an actual rate of 0.0433; Brier score 0.0297, expected calibration error 0.0439.

| score bin | accounts | mean predicted | observed rate |
|---|---|---|---|
| `[0.0, 0.2)` | 2555 | 0.0202 | 0.0063 |
| `[0.2, 0.4)` | 90 | 0.2903 | 0.0778 |
| `[0.4, 0.6)` | 53 | 0.5044 | 0.0943 |
| `[0.6, 0.8)` | 54 | 0.7118 | 0.1481 |
| `[0.8, 1.0)` | 114 | 0.9386 | 0.7719 |

The model over-states risk in **every** bin, the top one included, so the scores are not probabilities. That is why p\* is used here as a statement about the **alert queue** — break-even precision — and never as a score cutoff. The cutoff is the empirical validation optimum. Reporting p\* as though it were the operating threshold would be arithmetically tidy and wrong.

### Does the operating point survive a different cost ratio?

The FN:FP ratio is the one number in the cost model nobody can verify, so a result quoted at a single ratio is not a result. Each row picks its threshold on validation and is then evaluated once on test at that frozen value — the same discipline as the headline.

| FN:FP | break-even p\* | threshold (val) | test precision | test recall | test cost |
|---|---|---|---|---|---|
| 2.0 : 1 | 33.33% | 0.7363 | 69.7% | 74.2% | ₹15,60,000 |
| 5.0 : 1 | 16.67% | 0.6614 | 60.9% | 76.6% | ₹30,90,000 |
| 10.0 : 1 | 9.09% | 0.2261 | 36.4% | 86.3% | ₹53,55,000 |
| 13.333333333333334 : 1 | 6.98% | 0.1836 | 33.5% | 87.1% | ₹64,10,000 |
| 25.0 : 1 | 3.85% | 0.1471 | 31.2% | 90.3% | ₹82,05,000 |
| 50.0 : 1 | 1.96% | 0.1028 | 26.6% | 93.5% | ₹1,08,00,000 |

### Does the model earn its complexity?

Every row is priced on the same cost model, with each baseline's own threshold also selected on validation. This is the table that decides whether a graph pipeline was worth building.

| | precision | recall | F1 | alerts/1k | total cost |
|---|---|---|---|---|---|
| flag nothing | — | 0.0% | — | 0.0 | ₹2,48,00,000 |
| flag everything | 4.3% | 100.0% | 0.083 | 1000.0 | ₹4,11,30,000 |
| one-line rule: `cycle_participation >= 0.0524` | 16.4% | 91.9% | 0.278 | 242.5 | ₹1,07,15,000 |
| logistic regression, same features | 24.6% | 80.7% | 0.377 | 141.7 | ₹93,90,000 |
| XGBoost, no graph features | 46.7% | 79.0% | 0.587 | 73.3 | ₹68,80,000 |
| **XGBoost, full model** | **33.5%** | **87.1%** | **0.484** | **112.4** | **₹64,10,000** |

Against the identical model trained without them, graph features avoid ₹4,70,000 of cost and move average precision by +0.0189.

That cost figure flatters them, and the reason is worth stating: with a miss priced at 13.3333× a false alert, total cost is nearly insensitive to precision — and precision is exactly where the graph features land. False positives **rise** from 112 to 214 (91% more) at higher recall (79.0% → 87.1%), growing the review queue from 73.3 to 112.4 alerts per 1,000 accounts. An analyst feels that; the cost model barely does.

Read the other way, this ablation is also the strongest argument against the framing at the top of this README. 13 non-structural features reach test AUC 0.9586 on their own, against 0.9693 with the full set — 97.7% of the above-chance separation, so most of the separating signal is reachable without a graph at all. The structural features sharpen the queue; they do not find a fundamentally different population of mules. Anyone deciding whether to build this pipeline should weigh that.

### What happens when the review queue is capped

Every row above assumes an analyst queue of unlimited size. Cap it at 20 alerts per 1,000 accounts — a stated assumption, with the threshold that satisfies it chosen on validation like every other threshold here — and the model is pushed to the high-precision end of its own curve: threshold 0.9776, precision 98.0% at recall 39.5%, 17.4 alerts per 1,000 accounts, total test cost ₹1,50,15,000.

On cost alone that loses to the one-line rule (₹1,07,15,000), and the inversion is a property of the cost model rather than a defect. The rule reaches recall 91.9% by issuing 242.5 alerts per 1,000 accounts — roughly 12× the capped budget, so it is not a policy the cap permits. With a miss priced at 13.3333× a false alert, misses dominate the total, and any policy free to flood the queue wins on cost by construction. Note the rule's queue is not economically absurd either — at 16.4% precision it sits above the 6.98% break-even, so it is worth working if you have the staff. The objection to it is capacity, not arithmetic.

The defensible claim is therefore the joint one: cheaper than the alternatives *and* a queue small enough for a human to actually work. Cost in isolation, under a cap the winning baseline could never satisfy, is not a claim this project makes. A team whose binding constraint is reviews-per-day should re-derive its own operating point from its own budget.

### Is the task actually hard?

The strongest *single* feature on test is `cycle_participation` at direction-corrected AUC **0.8678**, against a leakage ceiling of 0.99. If any one column reached that ceiling the generator would be planting the label and every number above would be theatre — `tests/test_baselines.py` recomputes this from the shipped CSVs and fails the build if it ever does.

### Recall by ring archetype

Reported separately because one headline recall hides *which* laundering shapes the model misses, and the three archetypes are of deliberately unequal difficulty. Read the account column first: ring recall counts a ring as caught if *any* member is flagged, so it flatters a model whenever rings have several members, and a saturated metric is no evidence of a good one. Ring recall is still the operationally meaningful quantity — one alert brings an analyst to the whole ring.

| archetype | accounts | account recall | rings | ring recall |
|---|---|---|---|---|
| `fast_cycle` | 36/36 | 100.0% | 8/8 | 100.0% |
| `layered_fanin` | 24/25 | 96.0% | 6/6 | 100.0% |
| `stealth_cycle` | 48/63 | 76.2% | 10/10 | 100.0% |
| **all rings** | | | **24/24** | **100.0%** |

### Data the numbers were measured on

| split | accounts | mules | prevalence | rings |
|---|---|---|---|---|
| train | 3096 | 209 | 6.75% | 40 |
| val | 2947 | 136 | 4.61% | 24 |
| test | 2866 | 124 | 4.33% | 24 |

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

The track bar asks for honest metrics including false-positive cost. Four things
follow from taking that literally, and all four are load-bearing here.

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

PageRank is the one exemption, and it is a principled one: it is a global fixpoint
over a normalised rank vector, so strictly every node shifts when any node is
added. That is the definition, not a bug. It is therefore held to a *bound*
rather than to identity — measured max |Δ| 9.5e-06 against a 1e-4 tolerance.

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

Steps 1 and 2 write the CSVs the rest of the pipeline reads. They are
gitignored — deterministic from `SEED = 42`, so the recipe is committed instead of
the output — which means a fresh clone must run them before anything else works.

```bash
python -m data.generator     # → data/raw/{train,val,test,serving_context}_edges.csv
python -m data.extractor     # → data/processed/{train,val,test}_features.csv
python -m models.train       # → models/saved_models/sentinel_v3.xgb + metrics.json
python -m models.report --write   # fills the Results section of this README
```

`models.train` prints the baseline comparison table, per-archetype recall, the
leakage check and the SHAP-identity confirmation. Read that output rather than
trusting the model silently: it is where the pipeline tells you whether the run
is trustworthy.

```bash
uvicorn api.main:app --reload --port 8000    # docs at /docs
streamlit run dashboard/app.py               # http://localhost:8501
pytest tests/ -v                             # add -m "not slow" for a fast loop
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
tests/                  see below
```

`models/features.py` is the single source of truth for the feature list.
`tests/test_contract.py` walks every module's AST and fails if any other file
hard-codes a feature list, because three copies of a feature list is three
chances for training and serving to drift apart.

---

## What the tests actually guard

`test_leakage.py` — temporal ordering, zero positive-account and zero `ring_id`
overlap across splits, equal window lengths, and one-way recruitment.

`test_features.py` — extractor arithmetic against an independent pandas
implementation, and the perturbation-stability suite described above.

`test_contract.py` — that the feature contract is enforced, and that
`assert_feature_contract` is *actually called* on the API's startup path. This
one is the reason the suite exists in this shape: before v3 that function was
written, imported, and never invoked, so the API loaded a stale 12-feature
`sentinel_v1.xgb` and applied the current model's threshold to it. Every score
was wrong and nothing raised. **A guard nobody calls is not a guard**, so the
test parses the AST and requires the name in a *call* position — a mention in a
docstring or an import cannot satisfy it.

`test_baselines.py` — the leakage ceiling, the honest-lift claims, and that
`metrics.json` describes the current model version.

`test_responder.py` — the three defense-only rails, the tiering boundaries, and
batch validation against hand-forged responses.

On a fresh clone, tests that need the CSVs or a trained model **skip** rather
than fail, and `conftest.py` prints a header naming exactly which artefacts are
missing — so a mostly-skipped suite cannot be mistaken for a green one.

---

## Limitations

**The data is synthetic, and that is the biggest limitation by a distance.**
Every number in the Results section measures this model against a generator whose
assumptions were written by the same person. The mitigations are real — an
adversarial leakage ceiling, five published baselines, three ring archetypes of
deliberately unequal difficulty, and organic traffic built to defeat the specific
artifacts that made v2 look good — but they bound the self-deception rather than
eliminate it. **None of these figures is a claim about production performance.**
The honest claim is narrower: on a held-out temporal split of data designed to be
hard, the graph model beats a one-line rule by a measured margin, and here is
that margin.

**Both cost figures are assumptions.** ₹2,00,000 per missed mule and ₹15,000 per
false alert are stand-ins, not measurements from a real book. Only the ratio
affects the threshold, and the sensitivity table publishes how the operating
point moves as the ratio changes, but a merchant with different economics has a
different threshold and should recompute it.

**Raw magnitudes couple the model to a fixed window length.** `in_amount_sum` and
`out_amount_sum` are kept deliberately — the cost model is in rupees and an
analyst needs absolute exposure — but a sum scales with how long you watched. All
four windows (train, validation, test, serving context) are therefore held to the
same length, and the API warns on drift instead of silently rescaling. A
deployment wanting 7-day scoring must retrain, not just re-window.

**The serving partition ages.** Louvain is computed once from the reference graph
and extended for new accounts, which is what buys stability. The cost is that the
partition slowly stops describing the live network, so it needs periodic
re-derivation from fresh data. There is no automated trigger for that yet.

**Accounts are VPAs, with no entity resolution.** One person with five VPAs looks
like five accounts, and a ring that rotates VPAs looks like several small rings
rather than one large one. Real deployments resolve identity first; this does not.

**Scoring is batch-per-request, not streaming.** Each `/score` call rebuilds
features over the merged graph. That is honest about what it costs and fine at
this scale, but it is not an incremental online system, and PageRank over a
merged graph is the dominant cost as the graph grows.

**Calibration is measured, and it is poor — deliberately.** An earlier version of
this section said calibration was "assumed, not verified"; that is no longer true,
and the verification is unflattering. `scale_pos_weight` inflates scores to fight a
low positive rate, so the outputs are useful *rankings* and bad *probabilities* —
the reliability table in [Results](#results) shows the direction and size of the
bias. Two consequences are load-bearing. The threshold cannot be derived from
break-even probability and is instead chosen empirically. And no score from this
model should be read to a customer, an analyst or an auditor as "an X% chance this
is a mule". Fixing it properly means an isotonic or Platt step fitted on
validation, which does not exist yet.

**The operating threshold transfers imperfectly, and the run said so in advance.**
The cost-optimal threshold sits on a plateau under 1pp wide, which the selector
flags as possibly fitted to noise; the same threshold then costs materially more on
test than on validation. Both figures are published side by side in
[Results](#results) rather than only the favourable one. That gap is the price of
not peeking, and it is the number to watch on any future data — a wide plateau
would be evidence the operating point is robust, and this one is not wide.

**Under a tight analyst-capacity budget the model can lose to the one-line rule.**
The headline comparison gives every policy the same cost model but no queue limit.
Cap the queue near the model's own alert rate and the picture inverts: the model
becomes very precise and much less sensitive, while the single-feature rule keeps
buying recall with a queue an order of magnitude larger, and on this cost ratio
missed mules dominate. Both operating points are published. The practical reading
is that this system is justified by cost *and* by queue size together, and a team
whose real constraint is a hard cap on reviews per day should re-derive its own
operating point rather than inherit this one.

---

## License

MIT — see [LICENSE](LICENSE).
