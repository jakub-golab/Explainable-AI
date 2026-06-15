"""Train and evaluate binary classifiers on interaction fingerprints."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

ImbalanceStrategy = Literal["none", "class_weight", "undersample"]


@dataclass
class ModelResult:
    name: str
    cv_metrics: dict[str, float]
    model: Any
    imbalance_strategy: ImbalanceStrategy


def _make_models(y: pd.Series, imbalance_strategy: ImbalanceStrategy) -> dict[str, Any]:
    """Return model candidates with optional class weighting (not used with undersampling)."""
    use_class_weight = imbalance_strategy == "class_weight"
    n_active = int((y == 1).sum())
    n_inactive = int((y == 0).sum())
    scale_pos_weight = n_inactive / n_active if n_active else 1.0

    return {
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            class_weight="balanced" if use_class_weight else None,
            random_state=42,
            n_jobs=-1,
        ),
        "logistic_l1": LogisticRegression(
            penalty="l1",
            solver="saga",
            C=0.5,
            class_weight="balanced" if use_class_weight else None,
            max_iter=2000,
            random_state=42,
        ),
        "xgboost": XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            scale_pos_weight=scale_pos_weight if use_class_weight else 1.0,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
        ),
    }


def _build_pipeline(model, imbalance_strategy: ImbalanceStrategy) -> Pipeline | ImbPipeline:
    """
    Build sklearn/imblearn pipeline.

    Undersampling is applied only on the training portion of each CV fold
    (or on the full dataset for the final fit). Binary features stay 0/1.
    """
    steps: list[tuple[str, Any]] = [("scaler", StandardScaler())]
    if imbalance_strategy == "undersample":
        steps.append(("undersample", RandomUnderSampler(random_state=42)))
    steps.append(("clf", model))
    if imbalance_strategy == "undersample":
        return ImbPipeline(steps)
    return Pipeline(steps)


def resolve_imbalance_strategy(receptor: str, use_undersampling: bool = False) -> ImbalanceStrategy:
    """Pick imbalance handling: D4 can use undersampling; otherwise none or class_weight."""
    if receptor.lower() == "d4" and use_undersampling:
        return "undersample"
    if receptor.lower() == "d4":
        return "class_weight"
    return "none"


def train_and_evaluate(
    X: pd.DataFrame,
    y: pd.Series,
    receptor: str,
    use_undersampling: bool = False,
    n_splits: int = 5,
) -> list[ModelResult]:
    """
    Cross-validate classifiers and return fitted models on full data.

    For D4, set use_undersampling=True to balance classes by randomly dropping
    excess active samples in each training fold (keeps features binary).
    """
    imbalance_strategy = resolve_imbalance_strategy(receptor, use_undersampling)
    models = _make_models(y, imbalance_strategy)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scoring = {
        "accuracy": "accuracy",
        "balanced_accuracy": "balanced_accuracy",
        "f1": "f1",
        "roc_auc": "roc_auc",
    }

    results: list[ModelResult] = []
    for name, base_model in models.items():
        pipeline = _build_pipeline(base_model, imbalance_strategy)
        cv_out = cross_validate(
            pipeline,
            X,
            y,
            cv=cv,
            scoring=scoring,
            return_train_score=False,
            n_jobs=-1,
        )
        metrics = {k.replace("test_", ""): float(np.mean(v)) for k, v in cv_out.items() if k.startswith("test_")}

        final_pipeline = _build_pipeline(clone(base_model), imbalance_strategy)
        final_pipeline.fit(X, y)
        results.append(
            ModelResult(
                name=name,
                cv_metrics=metrics,
                model=final_pipeline,
                imbalance_strategy=imbalance_strategy,
            )
        )

    return results


def evaluate_on_holdout(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, Any]:
    """Compute holdout metrics for a fitted pipeline."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
    }


def select_best_model(results: list[ModelResult], metric: str = "balanced_accuracy") -> ModelResult:
    """Pick the model with the highest cross-validated metric."""
    return max(results, key=lambda r: r.cv_metrics.get(metric, 0.0))
