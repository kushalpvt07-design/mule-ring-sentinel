"""
models/explain.py
─────────────────
Per-prediction attribution. Exactly one implementation, shared by training and
serving.

WHY THIS MODULE EXISTS
──────────────────────
A risk product that cannot say *why* it flagged an account does not get used: an
analyst who is handed a score and no reason has to redo the investigation from
scratch, and a queue nobody trusts is a queue nobody works. The API's response
schema promises `contributing_factors`, so something has to produce them
honestly.

"Honestly" rules out the tempting shortcut. Global gain-based feature importance
describes the *model*, not the account in front of you — every alert would come
back with the same three reasons in the same order. TreeSHAP gives the
per-account decomposition instead: for one account, the contributions sum
exactly to that account's raw margin, so the explanation and the score cannot
disagree. `models/train.py` asserts that identity on the test split at the end of
every training run, which is what makes it a guarantee rather than a hope.

Two subtleties are handled here once, so neither caller has to remember them:

  1. EARLY STOPPING AND `pred_contribs` DO NOT AGREE BY DEFAULT.
     `XGBClassifier.predict_proba` automatically truncates to `best_iteration`
     when early stopping was used, but a raw `Booster.predict(...,
     pred_contribs=True)` call uses every tree that was fitted — including the
     ~30 trees after the best round that early stopping rejected. The
     attributions would then explain a slightly different model than the one that
     produced the score. `iteration_range_for()` recovers the right range from
     either an sklearn wrapper or a Booster reloaded from disk.

  2. THE BIAS COLUMN.
     `pred_contribs=True` returns n_features + 1 columns; the last is the base
     margin, not a feature. Slicing it off silently would break the summation
     identity, so it is returned separately and named.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from models.features import FEATURE_COLS, FEATURE_DESCRIPTIONS


# ──────────────────────────────────────────────────────────────────
# Booster plumbing
# ──────────────────────────────────────────────────────────────────

def as_booster(model):
    """Accept either an `xgboost.XGBClassifier` or a raw `Booster`."""
    getter = getattr(model, "get_booster", None)
    return getter() if getter is not None else model


def iteration_range_for(model) -> tuple[int, int] | None:
    """
    The tree range that `predict_proba` would use, so SHAP matches the score.

    Returns None when the model carries no `best_iteration` — i.e. it was fitted
    without early stopping — in which case every tree is used and no range needs
    to be passed.
    """
    best = getattr(model, "best_iteration", None)

    if best is None:
        booster = as_booster(model)
        best = getattr(booster, "best_iteration", None)
        if best is None:
            # A Booster reloaded from JSON keeps it as a string attribute.
            try:
                raw = booster.attr("best_iteration")
            except Exception:                       # pragma: no cover
                raw = None
            best = int(raw) if raw not in (None, "") else None

    if best is None:
        return None
    return (0, int(best) + 1)


# ──────────────────────────────────────────────────────────────────
# TreeSHAP
# ──────────────────────────────────────────────────────────────────

def shap_contributions(model, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """
    Exact TreeSHAP contributions for every row of `X`.

    Returns `(contribs, bias)` where `contribs` is (n_rows, n_features) aligned
    to `X.columns`, and `bias` is (n_rows,) — the base margin. By construction:

        contribs.sum(axis=1) + bias  ==  raw margin
        sigmoid(raw margin)          ==  predict_proba(X)[:, 1]

    This is exact, not sampled: `pred_contribs=True` runs the polynomial-time
    TreeSHAP algorithm inside XGBoost, so there is no approximation error to
    reason about and no `shap` package to install.
    """
    import xgboost as xgb          # local: keeps the dashboard's import cheap

    if list(X.columns) != list(FEATURE_COLS):
        raise RuntimeError(
            "shap_contributions received columns in the wrong order.\n"
            f"  got:      {list(X.columns)}\n"
            f"  expected: {list(FEATURE_COLS)}\n"
            "  TreeSHAP output is positional, so the attributions would be "
            "assigned to the wrong feature names."
        )

    booster = as_booster(model)
    dmatrix = xgb.DMatrix(X, feature_names=list(X.columns))
    kwargs = {"pred_contribs": True}
    rng = iteration_range_for(model)
    if rng is not None:
        kwargs["iteration_range"] = rng

    raw = np.asarray(booster.predict(dmatrix, **kwargs), dtype=float)
    if raw.ndim != 2 or raw.shape[1] != X.shape[1] + 1:
        raise RuntimeError(                          # pragma: no cover
            f"unexpected pred_contribs shape {raw.shape}; expected "
            f"({X.shape[0]}, {X.shape[1] + 1})"
        )
    return raw[:, :-1], raw[:, -1]


def top_factors(
    contribs_row: np.ndarray,
    values_row: np.ndarray,
    feature_names: list[str] | None = None,
    k: int = 3,
) -> list[dict]:
    """
    The k reasons this one account scored the way it did, largest effect first.

    Risk-raising factors come first because that is what an investigator acts on;
    if fewer than k features pushed the score up, the strongest *mitigating*
    factors fill the remainder rather than padding the list with near-zeros. Each
    entry carries the plain-English description from models/features.py, since
    "flow_passthrough = 0.98" is not an explanation an analyst can use.
    """
    names = list(feature_names) if feature_names is not None else list(FEATURE_COLS)
    contribs = np.asarray(contribs_row, dtype=float).ravel()
    values = np.asarray(values_row, dtype=float).ravel()
    if not (len(names) == contribs.size == values.size):
        raise ValueError(
            f"length mismatch: {len(names)} names, {contribs.size} contributions, "
            f"{values.size} values"
        )

    order = np.argsort(-contribs)                    # most risk-raising first
    raising = [i for i in order if contribs[i] > 0]
    lowering = [i for i in order[::-1] if contribs[i] <= 0]
    chosen = (raising + lowering)[:k]

    # `effect` is derived from the SAME rounded number that goes into the
    # response, not from the raw contribution. Deriving them separately let a
    # contribution of +4e-7 be published as `contribution: 0.0` with
    # `effect: "raises_risk"` — a label the number beside it does not support,
    # which is precisely the kind of detail that makes an analyst stop trusting
    # the explanation pane.
    factors = []
    for i in chosen:
        contribution = round(float(contribs[i]), 6)
        factors.append({
            "feature": names[i],
            "description": FEATURE_DESCRIPTIONS.get(names[i], names[i]),
            "value": round(float(values[i]), 6),
            "contribution": contribution,
            "effect": "raises_risk" if contribution > 0 else "lowers_risk",
        })
    return factors


def mean_abs_shap(contribs: np.ndarray, feature_names: list[str] | None = None) -> dict:
    """
    Global importance as mean |SHAP| over a dataset, normalised to sum to 1.

    Preferred over XGBoost's gain-based importance for reporting: gain counts
    split quality inside the trees, which rewards features that get split on
    often in deep branches serving few rows, while mean |SHAP| measures actual
    influence on the predictions that were made. Both are written to
    metrics.json; where they disagree, this is the one to trust.
    """
    names = list(feature_names) if feature_names is not None else list(FEATURE_COLS)
    magnitude = np.abs(np.asarray(contribs, dtype=float)).mean(axis=0)
    total = float(magnitude.sum())
    scaled = magnitude / total if total > 0 else magnitude
    return {name: float(v) for name, v in zip(names, scaled)}
