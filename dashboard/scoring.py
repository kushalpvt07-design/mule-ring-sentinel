"""
dashboard/scoring.py
────────────────────
The dashboard's ONLY source of risk scores: the trained model, or nothing.

─────────────────────────────────────────────────────────────────────────────
WHY THIS FILE EXISTS
─────────────────────────────────────────────────────────────────────────────
The v2 cost page did not use the model. It called a function named
`_simulate_predictions` that built a score out of normalised feature values, a
seeded noise term, and this:

    y_proba = np.clip(raw_scores + noise + y_true * 0.3, 0.01, 0.99)

`y_true * 0.3` adds the ground-truth label into the score. Every precision,
recall, cost and confusion-matrix figure on the page was computed against a
score that already knew the answer, which inflates all of them and makes the
whole page fiction — including the numbers a viewer would most reasonably read
as results. The feature list it used still contained `net_flow`, dropped in v3,
so it was also silently scoring on a stale contract.

There is no honest version of a simulated score on a page labelled "financial
impact". So simulation is gone, and this module replaces it with one rule: the
dashboard either shows numbers the real booster produced, or it shows an
explanation of why it can't. Nothing in between.

─────────────────────────────────────────────────────────────────────────────
WHAT THIS GUARANTEES
─────────────────────────────────────────────────────────────────────────────
• Scores come from `sentinel_v3.xgb` via `predict_proba`, on the exact columns
  in FEATURE_COLS, in that order — XGBoost matches columns positionally, so
  order is part of the contract, not a style choice.
• The booster's own feature names are checked against FEATURE_COLS before any
  score is produced (`assert_feature_contract`). A mismatched model raises here
  instead of quietly producing wrong numbers.
• The threshold comes from metrics.json and nowhere else. No `or 0.5` fallback:
  with FN/FP at 200k/15k the real operating point is near 0.07, so a 0.5 default
  would silently discard most of the recall the model was tuned for and every
  figure on screen would understate the model.
• Split labels (`is_mule`) are read for EVALUATION only, and never joined into
  the feature frame that goes to `predict_proba`.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.features import (  # noqa: E402
    FEATURE_COLS,
    MODEL_NAME,
    MODEL_VERSION,
    TARGET_COL,
    assert_feature_contract,
)

MODEL_DIR = PROJECT_ROOT / "models" / "saved_models"
MODEL_PATH = MODEL_DIR / MODEL_NAME
METRICS_PATH = MODEL_DIR / "metrics.json"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


class ModelUnavailable(RuntimeError):
    """
    Raised when the dashboard cannot produce real scores.

    A distinct type so callers can render the reason to the user instead of
    catching everything and falling back to something invented.
    """


@dataclass(frozen=True)
class ScoredSplit:
    """Real model output for one split, plus the labels needed to score it."""

    split: str
    node: pd.Series
    y_true: np.ndarray
    y_proba: np.ndarray
    ring_type: pd.Series
    features: pd.DataFrame

    @property
    def n(self) -> int:
        return int(len(self.y_true))

    @property
    def n_positive(self) -> int:
        return int(self.y_true.sum())

    @property
    def prevalence(self) -> float:
        return float(self.y_true.mean()) if self.n else 0.0


def load_model():
    """
    Load the booster and verify its feature contract before returning it.

    The contract check is not decorative. XGBoost lines columns up by position,
    so a model trained on a different or reordered feature set produces
    plausible-looking scores from the wrong inputs, with nothing raised. Better
    to fail on load than to draw a cost curve from it.
    """
    if not MODEL_PATH.exists():
        raise ModelUnavailable(
            f"No trained model at {MODEL_PATH.relative_to(PROJECT_ROOT)}.\n\n"
            "Run the pipeline first:\n"
            "```\npython -m data.generator\npython -m data.extractor\n"
            "python -m models.train\n```"
        )
    try:
        import xgboost as xgb
    except ImportError as exc:  # pragma: no cover - environment problem
        raise ModelUnavailable(
            f"xgboost is not installed ({exc}). `pip install -r requirements.txt`."
        ) from exc

    model = xgb.XGBClassifier()
    model.load_model(str(MODEL_PATH))

    try:
        assert_feature_contract(model.get_booster().feature_names)
    except RuntimeError as exc:
        raise ModelUnavailable(
            f"The saved model does not match the current feature contract, so "
            f"any score it produced would be computed from the wrong columns:"
            f"\n\n{exc}"
        ) from exc

    return model


def load_metrics() -> dict:
    """Read metrics.json, or explain why there isn't one."""
    if not METRICS_PATH.exists():
        raise ModelUnavailable(
            f"No {METRICS_PATH.name} at "
            f"{METRICS_PATH.parent.relative_to(PROJECT_ROOT)}. "
            "Run `python -m models.train`."
        )
    with open(METRICS_PATH, encoding="utf-8") as fh:
        metrics = json.load(fh)

    version = metrics.get("model_version")
    if version and version != MODEL_VERSION:
        raise ModelUnavailable(
            f"metrics.json was written for model '{version}' but the code is on "
            f"'{MODEL_VERSION}'. Its threshold and metrics describe a different "
            f"model. Retrain before reading anything off this page."
        )
    return metrics


def resolve_threshold(metrics: dict) -> float:
    """
    The operating threshold, from metrics.json only.

    v2 used `metrics.get("optimal_threshold", 0.5)`. At FN/FP = 200k/15k the
    break-even probability is ~0.07, so defaulting a missing key to 0.5 would
    have thrown away most of the model's recall and quietly reported the result
    as the model's performance.
    """
    raw = metrics.get("optimal_threshold")
    if raw is None:
        raise ModelUnavailable(
            "metrics.json has no 'optimal_threshold'. There is no safe default "
            "— the threshold is an economic quantity derived from FN and FP "
            "costs. Run `python -m models.train`."
        )
    threshold = float(raw)
    if not 0.0 < threshold <= 1.0:
        raise ModelUnavailable(
            f"metrics.json carries an out-of-range threshold ({threshold}). "
            f"Expected a probability in (0, 1]."
        )
    return threshold


def load_features(split: str) -> pd.DataFrame:
    """Load one processed split, or explain what to run."""
    path = PROCESSED_DIR / f"{split}_features.csv"
    if not path.exists():
        raise ModelUnavailable(
            f"No {path.name} in {PROCESSED_DIR.relative_to(PROJECT_ROOT)}. "
            f"Run `python -m data.generator` then `python -m data.extractor`."
        )
    return pd.read_csv(path)


def score_frame(model, frame: pd.DataFrame) -> np.ndarray:
    """
    Real probabilities for a feature frame.

    Selecting `frame[FEATURE_COLS]` explicitly is what keeps this honest: it
    drops `is_mule`, `ring_id`, `ring_type` and `split` by construction, so a
    label cannot reach the model however the CSV is shaped. It also raises on a
    missing feature rather than scoring on whatever happens to be present.
    """
    missing = [c for c in FEATURE_COLS if c not in frame.columns]
    if missing:
        raise ModelUnavailable(
            f"The feature file is missing {missing}. data/extractor.py and "
            f"models/features.py have diverged — re-run the extractor."
        )
    X = frame[FEATURE_COLS].astype(float)
    return model.predict_proba(X)[:, 1]


def score_split(split: str = "test") -> ScoredSplit:
    """
    Load a split, score it with the real model, and return both.

    `test` is the held-out split and the only one whose numbers should be quoted
    as results. `validation` is where the threshold was chosen, so its figures
    are optimistic by construction; `train` is meaningless as a metric and is
    offered only for contrast.
    """
    frame = load_features(split)
    if TARGET_COL not in frame.columns:
        raise ModelUnavailable(
            f"{split}_features.csv has no '{TARGET_COL}' column, so nothing on "
            f"this page could be evaluated. Re-run `python -m data.extractor`."
        )

    model = load_model()
    proba = score_frame(model, frame)

    return ScoredSplit(
        split=split,
        node=frame["node"],
        y_true=frame[TARGET_COL].to_numpy(dtype=int),
        y_proba=proba,
        ring_type=frame.get("ring_type", pd.Series(["unknown"] * len(frame))),
        features=frame,
    )


def feature_importance(metrics: dict) -> tuple[pd.DataFrame, str]:
    """
    Feature importance, preferring mean |SHAP| over XGBoost's gain.

    Both are in metrics.json and they answer different questions. Gain measures
    how much the split criterion improved during TRAINING; mean |SHAP| measures
    how much each feature actually moves predictions on the test split. The
    second is what a viewer reading "feature importance" on a results page
    assumes they are looking at, so it is the default — and the caption says
    which one is on screen, because presenting either unlabelled invites the
    wrong reading.
    """
    shap = metrics.get("feature_importance_mean_abs_shap")
    if shap:
        source = "mean |SHAP| on the test split — how much each feature moves predictions"
        values = shap
    elif metrics.get("feature_importance"):
        source = "XGBoost gain during training — not the same as test-time influence"
        values = metrics["feature_importance"]
    else:
        raise ModelUnavailable("metrics.json carries no feature importance block.")

    frame = (
        pd.DataFrame(sorted(values.items(), key=lambda kv: kv[1], reverse=True),
                     columns=["feature", "importance"])
        .set_index("feature")
    )
    return frame, source
