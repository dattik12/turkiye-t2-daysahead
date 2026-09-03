"""Model katmani: LightGBM BASE (v4.3) + CONT (v5.2) iki turlu, her ufuk icin."""
from __future__ import annotations
import numpy as np
import pandas as pd
import lightgbm as lgb

from . import config as C

CAT = ["hour", "weekday", "month"]


def fit(X: pd.DataFrame, y: pd.Series, feats: list, seed: int | None = None) -> tuple:
    """X: feature df (target yok); feats: kolon listesi. Donen: (model, feats)."""
    cat = {c: "category" for c in CAT if c in feats}
    params = dict(C.LGB_PARAMS)
    if seed is not None:
        params["random_state"] = seed
    m = lgb.LGBMRegressor(**params)
    m.fit(X[feats].astype(cat), y)
    return m, feats


def residual_adjust(train_resid: pd.Series, train_idx: pd.DatetimeIndex,
                    target_idx: pd.DatetimeIndex, k: int = 50) -> np.ndarray:
    """v4.4: (saat, gun-tipi) hucreli shrinkage rezidu duzeltmesi.
    train_resid: egitim hedefi - egitim tahmini (train_idx hizali).
    Hucre ortalamasi n/(n+k) ile sifira cekilir; hedef satirlara map'lenir."""
    from .features import calendar_cols
    cal_tr = calendar_cols(train_idx)
    hol_tr = (cal_tr["is_holiday_effect"] > 0).to_numpy()
    wknd_tr = (cal_tr["weekday"] >= 5).to_numpy()
    dtype_tr = np.where(hol_tr, 2, np.where(wknd_tr, 1, 0))
    r = np.asarray(train_resid, dtype=float)
    sums = np.zeros((24, 3)); counts = np.zeros((24, 3))
    np.add.at(sums, (train_idx.hour, dtype_tr), r)
    np.add.at(counts, (train_idx.hour, dtype_tr), 1)
    cell = sums / np.maximum(counts, 1) * (counts / (counts + k))
    cal_tg = calendar_cols(target_idx)
    hol_tg = (cal_tg["is_holiday_effect"] > 0).to_numpy()
    wknd_tg = (cal_tg["weekday"] >= 5).to_numpy()
    dtype_tg = np.where(hol_tg, 2, np.where(wknd_tg, 1, 0))
    return cell[target_idx.hour, dtype_tg]


def predict(m, feats: list, X: pd.DataFrame) -> np.ndarray:
    cat = {c: "category" for c in CAT if c in feats}
    return m.predict(X[feats].astype(cat))


def train_engine(P, wnat, wdyn, dayfrac, cw, seg_urban, seg_ind, cons,
                 train_idx, horizon, d1_pred: pd.Series | None = None,
                 extra_cols: dict | None = None):
    """Horizon basina BASE (+ opsiyonel CONT/manyak) modellerini egit. Donen dict cont -> (model, feats).
    extra_cols: horizon'a ozel ek feature'lar (kolon->array), orn. lep_rel (H1)."""
    from .features import make_row
    conts = [False] if not getattr(C, "USE_CONT", False) else [False, True]
    seeds = list(getattr(C, "ENSEMBLE_SEEDS", [42])) or [42]
    out = {}
    for cont in conts:
        F = make_row(P, wnat, wdyn, dayfrac, cw, seg_urban, seg_ind,
                     train_idx, horizon, d1_pred=d1_pred, cont=cont)
        if extra_cols:
            for k, v in extra_cols.items():
                F[k] = v
        F["target"] = cons.reindex(train_idx)
        F = F.dropna()
        feats = [c for c in F.columns if c != "target"]
        for s in seeds:  # v4.4: seed ensemble (ayni veri, farkli bagging)
            out[(cont, s)] = fit(F, F["target"], feats, seed=s)
    return out


def predict_pair(models, P, wnat, wdyn, dayfrac, cw, seg_urban, seg_ind,
                 idx, horizon, d1_pred: pd.Series | None = None,
                 extra_cols: dict | None = None) -> np.ndarray:
    """BASE (+ CONT) ortalamasi; USE_CONT=False ise yalnizca BASE."""
    from .features import make_row
    conts = sorted(models.keys())
    preds = []
    for key in conts:
        cont = key[0] if isinstance(key, tuple) else key
        m, feats = models[key]
        X = make_row(P, wnat, wdyn, dayfrac, cw, seg_urban, seg_ind,
                     idx, horizon, d1_pred=d1_pred, cont=cont)
        if extra_cols:
            for k, v in extra_cols.items():
                X[k] = v
        preds.append(predict(m, feats, X))
    return np.mean(preds, axis=0)
