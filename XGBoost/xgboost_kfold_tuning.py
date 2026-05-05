"""XGBoost K-fold Bayesian tuning using GroupKFold and skopt.

Saves results to tuning_results_kfold.json and tuning_plots_kfold/.

Run a short pilot by default: k=3, n_calls=6. Increase `N_CALLS` for full runs.
"""
from __future__ import annotations

import json
import math
import os
from functools import partial
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, recall_score
from sklearn.model_selection import GroupKFold
try:
    from sklearn.model_selection import StratifiedGroupKFold
except ImportError:
    StratifiedGroupKFold = None
from skopt import gp_minimize
from skopt.space import Integer, Real
from xgboost import XGBClassifier


ROOT = os.path.dirname(os.path.dirname(__file__)) if __file__ else '.'
ARTIFACTS = os.path.join(ROOT, "artifacts")
TRAIN_FEATURES = os.path.join(ARTIFACTS, "train_features.csv")


def convert_native(x: Any) -> Any:
    if isinstance(x, (np.integer, np.floating)):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    return x


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def prepare_xy(df: pd.DataFrame, label_col: str = "SepsisLabel") -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    # Expect person_id to be present for grouping
    if "person_id" not in df.columns:
        raise ValueError("person_id column required for GroupKFold")
    if label_col not in df.columns:
        raise ValueError(f"label column {label_col} not present")

    groups = df["person_id"]
    y = df[label_col]
    # Remove label and person_id from features
    X = df.drop(columns=[label_col, "person_id"]) if "person_id" in df.columns else df.drop(columns=[label_col])
    # Keep only numeric columns to avoid dtype issues with XGBoost
    X = X.select_dtypes(include=[np.number])
    return X, y, groups


def eval_candidate(params: Dict[str, Any], X: pd.DataFrame, y: pd.Series, groups: pd.Series, n_splits: int = 5) -> Tuple[float, Dict[str, Any]]:
    # Use StratifiedGroupKFold if available (better for imbalanced data), else fall back to GroupKFold
    if StratifiedGroupKFold is not None:
        kf = StratifiedGroupKFold(n_splits=n_splits, random_state=42, shuffle=True)
    else:
        kf = GroupKFold(n_splits=n_splits)
    prs = []
    recs = []
    fold_details = []

    for fold, (tr_idx, va_idx) in enumerate(kf.split(X, y, groups)):
        X_tr, X_va = X.iloc[tr_idx].to_numpy(), X.iloc[va_idx].to_numpy()
        y_tr, y_va = y.iloc[tr_idx].to_numpy(), y.iloc[va_idx].to_numpy()

        pos = int(y_tr.sum())
        neg = len(y_tr) - pos
        scale_pos_weight = float(neg / pos) if pos > 0 else 1.0

        clf = XGBClassifier(
            **{k: v for k, v in params.items() if v is not None},
            scale_pos_weight=scale_pos_weight,
            tree_method="hist",
            eval_metric="aucpr",
            random_state=42,
            n_jobs=-1,
        )

        # Use early stopping to prevent overfitting on each fold
        # (Note: XGBoost sklearn wrapper version here doesn't support early_stopping_rounds in fit())
        # Instead, rely on validation set and regularization parameters
        clf.fit(
            X_tr,
            y_tr,
            eval_set=[(X_va, y_va)],
            verbose=False,
        )

        p_val = clf.predict_proba(X_va)[:, 1]
        p_bin = (p_val >= 0.5).astype(int)
        pr = float(average_precision_score(y_va, p_val))
        rec = float(recall_score(y_va, p_bin, zero_division=0))

        prs.append(pr)
        recs.append(rec)
        fold_details.append({"pr": pr, "recall": rec, "pos": int(pos), "neg": int(neg)})

    mean_pr = float(np.mean(prs))
    mean_rec = float(np.mean(recs))
    # Optimize PR-AUC only (threshold-independent, better for imbalanced medical classification)
    # Threshold selection is a deployment decision, not a tuning objective
    objective_score = mean_pr

    extras = {
        "folds": fold_details,
        "mean_pr": mean_pr,
        "mean_recall": mean_rec,
        "objective_score": objective_score,
    }
    return objective_score, extras


def objective(space_vals, X, y, groups, n_splits: int):
    # Unpack skopt values in order we define below
    max_depth, min_child_weight, gamma, subsample, colsample_bytree, reg_lambda, reg_alpha, learning_rate, n_estimators = space_vals

    params = {
        "max_depth": int(max_depth),
        "min_child_weight": int(min_child_weight),
        "gamma": float(gamma),
        "subsample": float(subsample),
        "colsample_bytree": float(colsample_bytree),
        "reg_lambda": float(reg_lambda),
        "reg_alpha": float(reg_alpha),
        "learning_rate": float(learning_rate),
        "n_estimators": int(n_estimators),
    }

    objective_score, extras = eval_candidate(params, X, y, groups, n_splits=n_splits)

    # gp_minimize minimizes; we want to maximize PR-AUC, so return negative
    return -objective_score


def run_tuning(n_splits, n_calls, random_state: int = 42):
    print("Loading data from", TRAIN_FEATURES)
    df = load_data(TRAIN_FEATURES)
    X, y, groups = prepare_xy(df)

    space = [
        Integer(3, 10, name="max_depth"),
        Integer(1, 10, name="min_child_weight"),
        Real(0.0, 5.0, name="gamma"),
        Real(0.5, 1.0, name="subsample"),
        Real(0.5, 1.0, name="colsample_bytree"),
        Real(0.0, 5.0, name="reg_lambda"),
        Real(0.0, 5.0, name="reg_alpha"),
        Real(0.01, 0.3, name="learning_rate"),
        Integer(100, 2000, name="n_estimators"),  # Increased upper bound; early stopping will limit actual trees
    ]

    func = partial(objective, X=X, y=y, groups=groups, n_splits=n_splits)
    print(f"Starting gp_minimize: n_calls={n_calls}, n_splits={n_splits}")

    res = gp_minimize(func, space, n_calls=n_calls, random_state=random_state, n_initial_points=10)

    best = res.x
    best_params = {
        "max_depth": int(best[0]),
        "min_child_weight": int(best[1]),
        "gamma": float(best[2]),
        "subsample": float(best[3]),
        "colsample_bytree": float(best[4]),
        "reg_lambda": float(best[5]),
        "reg_alpha": float(best[6]),
        "learning_rate": float(best[7]),
        "n_estimators": int(best[8]),
    }

    # Evaluate best candidate with full k-fold and collect details
    objective_score, extras = eval_candidate(best_params, X, y, groups, n_splits=n_splits)

    out = {
        "best_params": {k: convert_native(v) for k, v in best_params.items()},
        "objective_score_pr_auc": convert_native(extras.get("objective_score")),
        "mean_pr": convert_native(extras.get("mean_pr")),
        "mean_recall": convert_native(extras.get("mean_recall")),
        "folds": extras.get("folds"),
        "skopt_results": {
            "x": [convert_native(v) for v in res.x],
            "fun": convert_native(res.fun),
            "func_vals": convert_native(res.func_vals.tolist()) if hasattr(res, "func_vals") else None,
        },
    }

    os.makedirs("tuning_plots_kfold", exist_ok=True)
    with open("tuning_results_kfold.json", "w") as f:
        json.dump(out, f, indent=2)

    print("Tuning finished. Results saved to tuning_results_kfold.json")


if __name__ == "__main__":
    # Pilot defaults: 5-fold, 30 calls.
    run_tuning(n_splits=5, n_calls=30)
