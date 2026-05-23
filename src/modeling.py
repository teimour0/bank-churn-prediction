"""
Train and compare three churn classifiers.

I am not chasing the absolute best ROC-AUC here. The honest goal is to show
how three reasonable baselines stack up against each other and to keep the
training code small enough that a reviewer can read it in a sitting.

The class imbalance (~20% positives) is mild but real, so we use
class_weight="balanced" on the linear and tree-based models and
scale_pos_weight on XGBoost. Stratified split keeps the test set honest.
"""

from __future__ import annotations

from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier
    _HAS_XGB = True
except Exception:  # pragma: no cover
    _HAS_XGB = False


@dataclass
class ModelResult:
    name: str
    model: object
    metrics: dict[str, float]
    confusion: np.ndarray
    test_probas: np.ndarray
    test_y: np.ndarray
    test_X: pd.DataFrame


def _build_models(pos_weight: float) -> dict[str, object]:
    """Return the candidate models. Linear gets scaling, trees do not."""
    candidates: dict[str, object] = {
        "logistic_regression": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=1000,
                        class_weight="balanced",
                        random_state=7,
                    ),
                ),
            ]
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=400,
            max_depth=None,
            min_samples_leaf=5,
            class_weight="balanced",
            n_jobs=-1,
            random_state=7,
        ),
    }
    if _HAS_XGB:
        candidates["xgboost"] = XGBClassifier(
            n_estimators=500,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            scale_pos_weight=pos_weight,
            eval_metric="logloss",
            tree_method="hist",
            random_state=7,
            n_jobs=-1,
        )
    return candidates


def train_and_compare(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 7,
) -> tuple[ModelResult, list[ModelResult]]:
    """Train all candidate models. Returns (best, full list)."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    # scale_pos_weight for XGBoost: ratio of negatives to positives.
    pos_weight = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))
    candidates = _build_models(pos_weight=pos_weight)

    results: list[ModelResult] = []
    for name, model in candidates.items():
        model.fit(X_train, y_train)
        probas = model.predict_proba(X_test)[:, 1]
        preds = (probas >= 0.5).astype(int)
        metrics = {
            "roc_auc": roc_auc_score(y_test, probas),
            "precision": precision_score(y_test, preds, zero_division=0),
            "recall": recall_score(y_test, preds, zero_division=0),
            "f1": f1_score(y_test, preds, zero_division=0),
        }
        cm = confusion_matrix(y_test, preds)
        results.append(
            ModelResult(
                name=name, model=model, metrics=metrics, confusion=cm,
                test_probas=probas, test_y=y_test.to_numpy(),
                test_X=X_test.reset_index(drop=True),
            )
        )

    # Pick the best on ROC-AUC. Ties broken by F1.
    best = max(results, key=lambda r: (r.metrics["roc_auc"], r.metrics["f1"]))
    return best, results


def summarise(results: list[ModelResult]) -> pd.DataFrame:
    rows = []
    for r in results:
        row = {"model": r.name, **{k: round(v, 4) for k, v in r.metrics.items()}}
        rows.append(row)
    return pd.DataFrame(rows).sort_values("roc_auc", ascending=False)


def save_best(result: ModelResult, path: str = "best_model.joblib") -> str:
    joblib.dump({"name": result.name, "model": result.model}, path)
    return path


def detailed_report(result: ModelResult) -> str:
    preds = (result.test_probas >= 0.5).astype(int)
    return classification_report(result.test_y, preds, digits=3)
