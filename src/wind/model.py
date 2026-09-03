"""RES guc modeli v1: LightGBM, RITM-tahmini bazli residual ogrenme.

f : [agirlikli w100, w100^2, w100^3, shear, saat, gun TIPLERI, ritm_fc]
hedef : saatlik RITM generation (MW)

Egitim: train(df_train) -> Booster (models/wind_lgbm.txt).
Tahmin: predict(booster, feat) -> MW (>=0 clip).
Model yoksa builder RITM tahminine duser (wind_status='ritm_fallback').
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
import lightgbm as lgb

FEATS = ["w100", "w100_v2", "w100_v3", "shear", "hour", "dow",
         "is_weekend", "ritm_fc"]


def build_features(speed: pd.Series, ritm_fc: pd.Series | None = None) -> pd.DataFrame:
    idx = speed.index
    f = pd.DataFrame(index=idx)
    f["w100"] = speed.clip(lower=0)
    f["w100_v2"] = f["w100"] ** 2
    f["w100_v3"] = f["w100"] ** 3
    f["shear"] = np.nan
    f["hour"] = idx.hour
    f["dow"] = idx.dayofweek
    f["is_weekend"] = (idx.dayofweek >= 5).astype(int)
    f["ritm_fc"] = ritm_fc.reindex(idx).ffill() if ritm_fc is not None else np.nan
    return f


def attach_shear(f: pd.DataFrame, w100: pd.Series, w10: pd.Series) -> pd.DataFrame:
    f = f.copy()
    f["shear"] = (w100.reindex(f.index).fillna(0) /
                  w10.reindex(f.index).fillna(0).replace(0, np.nan)).fillna(1.0)
    return f


def train(df: pd.DataFrame, params: dict | None = None) -> lgb.Booster:
    p = dict(n_estimators=800, learning_rate=0.05, num_leaves=63,
             colsample_bytree=0.8, subsample=0.8, subsample_freq=1,
             min_child_samples=40, random_state=42, verbose=-1)
    if params:
        p.update(params)
    d = lgb.Dataset(df[FEATS].fillna(0), label=df["wind_gen_mw"])
    return lgb.train(p, d)


def predict(booster: lgb.Booster, f: pd.DataFrame) -> pd.Series:
    s = pd.Series(booster.predict(f[FEATS].fillna(0)), index=f.index)
    return s.clip(lower=0)


def save(booster: lgb.Booster, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    booster.save_model(path)


def load(path: str) -> lgb.Booster | None:
    return lgb.Booster(model_file=path) if os.path.exists(path) else None
