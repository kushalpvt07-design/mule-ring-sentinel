# UPI Mule-Ring Sentinel

**Detects circular mule-account rings in UPI transaction graphs, prices every
decision in rupees, and never enforces.** Built for Razorpay Buildathon 2026,
Track 2 — AI Risk Manager.

The loss class is **money-mule laundering rings**: accounts that receive fraud
proceeds and pass them on in circles to break the audit trail. A single mule hop
is unremarkable in isolation — the pattern only exists in the *shape* of the
network, which is why this is a graph problem and not a per-transaction one.

The strongest action this system can emit is `HOLD_FOR_REVIEW`. It has no code
path that can ban, block, freeze, suspend, terminate, disable, revoke, seize or
close anything, and that is enforced at import time and by test, not by
convention. See [Defense-only, by construction](#defense-only-by-construction).

---

## Results

<!-- METRICS:BEGIN -->
> **No metrics published yet.** `models/saved_models/metrics.json` is absent, so
> this section is intentionally empty rather than filled with numbers nobody
> measured. Run `python -m models.train` and then
> `python -m models.report --write`.
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

### The scoring threshold, and why it is not a tuned magic number

Flagging an account is worth it when the expected cost of flagging is below the
expected cost of not flagging. For an account with probability *p* of being a
mule, that is `(1 − p)·fp_cost < p·fn_cost`, so the cost-optimal cutoff is
`p* = fp_cost / (fp_cost + fn_cost)` — derived from the cost assumptions, not
fitted to a dataset. At the assumed ₹15,000 and ₹2,00,000 that is p\* ≈ 0.0698,
and the same number read the other way is the **break-even precision of the alert
queue**: about 7%. Any queue cleaner than 7% precision is cheaper to work than to
ignore. That is a far more defensible thing to report than "we tuned the
threshold to 0.42".

The empirical sweep still runs, because the model is not perfectly calibrated and
what actually happened is what matters; the sweep is exact rather than gridded
(total cost is a step function of the threshold, changing only when the threshold
crosses an observed score) and prefers the *centre of the widest cost plateau* to
a knife-edge minimum. Both the analytic and empirical thresholds are printed so a
calibration gap is visible instead of hidden.

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

**Calibration is assumed, not verified.** The break-even threshold argument
presumes reasonably calibrated probabilities. The empirical sweep is used instead
precisely because that presumption is shaky, and both numbers are printed so the
gap is visible — but no reliability diagram or isotonic calibration step exists
yet.

---

## License

MIT — see [LICENSE](LICENSE).
