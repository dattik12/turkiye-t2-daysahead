"""Model katmani: LightGBM BASE (v4.3) + CONT (v5.2) iki turlu, her ufuk icin."""
from __future__ import annotations
import numpy as np
import pandas as pd
import lightgbm as lgb

from . import config as C

CAT = ["hour", "weekday", "month"]


def fit(X: pd.DataFrame, y: pd.Series, feats: list) -> tuple:
    """X: feature df (target yok); feats: kolon listesi. Donen: (model, feats)."""
    cat = {c: "category" for c in CAT if c in feats}
    m = lgb.LGBMRegressor(**C.LGB_PARAMS)
    m.fit(X[feats].astype(cat), y)
    return m, feats


def predict(m, feats: list, X: pd.DataFrame) -> np.ndarray:
    cat = {c: "category" for c in CAT if c in feats}
    return m.predict(X[feats].astype(cat))


def train_engine(P, wnat, wdyn, dayfrac, cw, seg_urban, seg_ind, cons,
                 train_idx, horizon, d1_pred: pd.Series | None = None):
    """Horizon basina BASE (+ opsiyonel CONT/manyak) modellerini egit. Donen dict cont -> (model, feats)."""
    from .features import make_row
    conts = [False] if not getattr(C, "USE_CONT", False) else [False, True]
    out = {}
    for cont in conts:
        F = make_row(P, wnat, wdyn, dayfrac, cw, seg_urban, seg_ind,
                     train_idx, horizon, d1_pred=d1_pred, cont=cont)
        F["target"] = cons.reindex(train_idx)
        F = F.dropna()
        feats = [c for c in F.columns if c != "target"]
        out[cont] = fit(F, F["target"], feats)
    return out


def predict_pair(models, P, wnat, wdyn, dayfrac, cw, seg_urban, seg_ind,
                 idx, horizon, d1_pred: pd.Series | None = None) -> np.ndarray:
    """BASE (+ CONT) ortalamasi; USE_CONT=False ise yalnizca BASE."""
    from .features import make_row
    conts = sorted(models.keys())
    preds = []
    for cont in conts:
        m, feats = models[cont]
        X = make_row(P, wnat, wdyn, dayfrac, cw, seg_urban, seg_ind,
                     idx, horizon, d1_pred=d1_pred, cont=cont)
        preds.append(predict(m, feats, X))
    return np.mean(preds, axis=0)
