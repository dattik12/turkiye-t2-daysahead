"""Model katmani: LightGBM + XGBoost + CatBoost ensemble (per-model feature set).
- LGBM (leaf-wise): hizli, derin num_leaves, column sampling -> manyak feature zengin ensemble
- XGB (level-wise, hist): duzenli derinlik, reg_alpha/lambda ile bayramlarda daha stabil
- CatBoost (ordered boosting): kategorik hour/weekday/month/holiday native
Her model horizon'a gore ayri egitilir; cat gorur, lag setleri farklilastirilabilir.
Kaynaklar: Ke et al. NeurIPS2017, Chen&Guestrin KDD2016, Prokhorenkova et al. NeurIPS2018.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
import catboost as cb

from . import config as C
from .features import BASE_H1, BASE_H2, CONTRA

# per-model feature adjustments (intentionally different for diversity)
# LGBM: full BASE(+CONT if enabled) + holiday-tail/bridge columns are inside calendar_cols
# XGB: leaner lag set (robust to overfit), good for regularisation
# CatBoost: same as BASE but categorical columns are handled natively (see cat_features)
LGB_EXTRA = []  # rely on calendar_cols holiday flags naturally
XGB_DROP = ["lag336"]  # example: XGB drops longest lag (favours shorter horizon stability)
CAT_EXTRA = []  # CatBoost shines with cat handling, not extra numeric

# Curated holiday extra cols added by calendar_cols (not in BASE_Hx, but present in X via make_row)
HOLIDAY_COLS = ["is_holiday_tail", "is_bridge", "is_holiday_effect"]

CAT_FEATURES = ["hour", "weekday", "month", "is_holiday", "is_official", "is_arife",
                "is_after_holiday", "is_holiday_tail", "is_bridge", "is_holiday_effect", "is_ramadan"]

LGB_CAT = ["hour", "weekday", "month"]
XGB_CAT = []  # XGB handles via one-hot if needed, but we keep numeric

def _feats_for(model_name: str, base: list, cont: bool) -> list:
    feats = list(base)
    if cont:
        feats = feats + list(CONTRA)
    # per-model tweaks
    if model_name == "xgb":
        feats = [f for f in feats if f not in XGB_DROP]
    # holiday cols are outside base; they come from calendar_cols, so append if present
    # they are always added if cont else still present via calendar_cols -> include
    feats = feats + HOLIDAY_COLS
    return feats

def _filter_existing(feats: list, X: pd.DataFrame) -> list:
    return [f for f in feats if f in X.columns]

def fit_lgbm(X: pd.DataFrame, y: pd.Series, feats: list):
    feats = _filter_existing(feats, X)
    cat = {c: "category" for c in LGB_CAT if c in feats}
    m = lgb.LGBMRegressor(**C.LGB_PARAMS)
    m.fit(X[feats].astype(cat), y)
    return m, feats

def fit_xgb(X: pd.DataFrame, y: pd.Series, feats: list):
    feats = _filter_existing(feats, X)
    m = xgb.XGBRegressor(**C.XGB_PARAMS)
    m.fit(X[feats], y)
    return m, feats

def fit_cat(X: pd.DataFrame, y: pd.Series, feats: list):
    feats = _filter_existing(feats, X)
    cat_idx = [i for i, f in enumerate(feats) if f in CAT_FEATURES]
    pool_feats = feats
    m = cb.CatBoostRegressor(**C.CAT_PARAMS)
    # CatBoost wants cat_features as indices or names; use indices for stability
    m.fit(X[pool_feats], y, cat_features=cat_idx, verbose=False)
    return m, feats

def predict_lgbm(m, feats: list, X: pd.DataFrame) -> np.ndarray:
    cat = {c: "category" for c in LGB_CAT if c in feats}
    return m.predict(X[feats].astype(cat))

def predict_xgb(m, feats: list, X: pd.DataFrame) -> np.ndarray:
    return m.predict(X[feats])

def predict_cat(m, feats: list, X: pd.DataFrame) -> np.ndarray:
    return m.predict(X[feats])

FITTERS = {"lgbm": fit_lgbm, "xgb": fit_xgb, "catboost": fit_cat}
PREDICTORS = {"lgbm": predict_lgbm, "xgb": predict_xgb, "catboost": predict_cat}

def train_engine_multi(P, wnat, wdyn, dayfrac, cw, seg_urban, seg_ind, cons,
                       train_idx, horizon, d1_pred: pd.Series | None = None):
    """Per-model, per-horizon ensemble of 3 tree models. Returns dict model_name -> {(cont,m,feats)}.
    Backward compat: if ENSEMBLE_MODELS == ['lgbm'] -> single-model behaviour.
    """
    from .features import make_row
    models = getattr(C, "ENSEMBLE_MODELS", ["lgbm"])
    conts = [False] if not getattr(C, "USE_CONT", False) else [False, True]
    out = {}
    for name in models:
        per_cont = {}
        for cont in conts:
            F = make_row(P, wnat, wdyn, dayfrac, cw, seg_urban, seg_ind,
                         train_idx, horizon, d1_pred=d1_pred, cont=cont)
            F["target"] = cons.reindex(train_idx)
            F = F.dropna()
            base = BASE_H1 if horizon == 1 else BASE_H2
            feats = _feats_for(name, base, cont)
            feats = _filter_existing(feats, F)
            m, feats_used = FITTERS[name](F, F["target"], feats)
            per_cont[cont] = (m, feats_used)
        out[name] = per_cont
    return out

def predict_multi(models, P, wnat, wdyn, dayfrac, cw, seg_urban, seg_ind,
                  idx, horizon, d1_pred: pd.Series | None = None) -> dict:
    """Returns dict model_name -> prediction array (mean over cont variants)."""
    from .features import make_row
    out = {}
    for name, per_cont in models.items():
        preds = []
        for cont, (m, feats) in sorted(per_cont.items()):
            X = make_row(P, wnat, wdyn, dayfrac, cw, seg_urban, seg_ind,
                         idx, horizon, d1_pred=d1_pred, cont=cont)
            preds.append(PREDICTORS[name](m, feats, X))
        out[name] = np.mean(preds, axis=0) if len(preds) > 1 else preds[0]
    return out

# Backward-compatible shims for single-model callers
def fit(X: pd.DataFrame, y: pd.Series, feats: list) -> tuple:
    return fit_lgbm(X, y, feats)

def predict(m, feats: list, X: pd.DataFrame) -> np.ndarray:
    return predict_lgbm(m, feats, X)

def train_engine(P, wnat, wdyn, dayfrac, cw, seg_urban, seg_ind, cons,
                 train_idx, horizon, d1_pred: pd.Series | None = None):
    # single-model (lgbm) for old callers
    from .features import make_row
    conts = [False] if not getattr(C, "USE_CONT", False) else [False, True]
    out = {}
    for cont in conts:
        F = make_row(P, wnat, wdyn, dayfrac, cw, seg_urban, seg_ind,
                     train_idx, horizon, d1_pred=d1_pred, cont=cont)
        F["target"] = cons.reindex(train_idx)
        F = F.dropna()
        feats = [c for c in F.columns if c != "target"]
        out[cont] = fit_lgbm(F, F["target"], feats)
    return out

def predict_pair(models, P, wnat, wdyn, dayfrac, cw, seg_urban, seg_ind,
                 idx, horizon, d1_pred: pd.Series | None = None) -> np.ndarray:
    from .features import make_row
    conts = sorted(models.keys())
    preds = []
    for cont in conts:
        m, feats = models[cont]
        X = make_row(P, wnat, wdyn, dayfrac, cw, seg_urban, seg_ind,
                     idx, horizon, d1_pred=d1_pred, cont=cont)
        preds.append(predict_lgbm(m, feats, X))
    return np.mean(preds, axis=0)
