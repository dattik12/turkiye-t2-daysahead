"""Stacking: per-horizon Ridge on top of LGBM+XGB+CatBoost (TimeSeriesSplit OOF, leak-free).
Produces horizon-specific alpha/weights; serializes to data/results/stacking_weights.json.
"""
from __future__ import annotations
import json, os
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import TimeSeriesSplit

from . import config as C

def fit_stacking(oof_preds: pd.DataFrame, y_true: pd.Series, horizon: pd.Series) -> dict:
    """oof_preds columns = model names, rows = validation oof predictions.
    Returns {1: {coef, intercept, alpha}, 2: {...}} plus global fallback.
    """
    out = {}
    for h in [1, 2]:
        m = horizon == h
        if m.sum() < 50:
            continue
        X = oof_preds.loc[m].values
        y = y_true.loc[m].values
        ridge = RidgeCV(alphas=C.STACKING_ALPHAS, cv=3)
        ridge.fit(X, y)
        out[str(h)] = dict(coef=ridge.coef_.tolist(), intercept=float(ridge.intercept_), alpha=float(ridge.alpha_), models=list(oof_preds.columns))
    # global
    ridge = RidgeCV(alphas=C.STACKING_ALPHAS, cv=3)
    ridge.fit(oof_preds.values, y_true.values)
    out["global"] = dict(coef=ridge.coef_.tolist(), intercept=float(ridge.intercept_), alpha=float(ridge.alpha_), models=list(oof_preds.columns))
    return out

def save_weights(weights: dict, path: str | None = None):
    p = path or os.path.join(C.RESULTS_DIR, "stacking_weights.json")
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(weights, f, indent=2, ensure_ascii=False)
    return p

def load_weights(path: str | None = None) -> dict | None:
    p = path or os.path.join(C.RESULTS_DIR, "stacking_weights.json")
    if not os.path.exists(p):
        return None
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def apply_stacking(preds: dict, weights: dict, horizon: int) -> np.ndarray:
    """preds: model_name->array (aligned). Returns stacked array."""
    key = str(horizon) if str(horizon) in weights else "global"
    w = weights[key]
    models = w["models"]
    X = np.column_stack([preds[m] for m in models])
    coef = np.array(w["coef"])
    intercept = w["intercept"]
    return X @ coef + intercept
