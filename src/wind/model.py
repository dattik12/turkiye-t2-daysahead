"""RES guc modeli v2: LightGBM residual-reframe + yon sektoru (tam dosya)."""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
import lightgbm as lgb

FEATS = ["w100", "w100_v2", "w100_v3", "shear", "hour", "dow",
         "is_weekend", "wdir_sin", "wdir_cos", "wdir_sector", "ritm_fc",
         "gen_lag168", "gen_lag336", "gen_roll168", "err_lag168",
         "gfs_w100", "nwp_spread"]
# NOT: lag24/48 gun-oncesi ufukta (D+2 gec saatler) karar-aninda BILINMEZ -> yasak.
# Sadece t-168 ve otesi (her zaman bilinen) kullanilir.
CAT = ["wdir_sector"]
MONO_RAW = [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
SECTORS = list(range(8))


def _circ_deg(deg: pd.Series) -> pd.Series:
    return deg.fillna(0.0) % 360.0


def attach_direction(f: pd.DataFrame, wdir_weighted: pd.Series) -> pd.DataFrame:
    f = f.copy()
    deg = _circ_deg(pd.Series(wdir_weighted).reindex(f.index).ffill()).to_numpy(dtype=float)
    rad = np.deg2rad(deg)
    f["wdir_sin"] = np.sin(rad)
    f["wdir_cos"] = np.cos(rad)
    f["wdir_sector"] = pd.Categorical((deg // 45).astype(int), categories=SECTORS)
    return f


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
    f["wdir_sin"] = 0.0
    f["wdir_cos"] = 1.0
    f["wdir_sector"] = pd.Categorical([0] * len(idx), categories=SECTORS)
    f["ritm_fc"] = ritm_fc.reindex(idx).ffill() if ritm_fc is not None else np.nan
    f["gfs_w100"] = np.nan  # ikinci NWP (egitimde doldurulur)
    f["nwp_spread"] = np.nan  # |IFS - GFS| agirlikli (belirsizlik proxy'si)
    for c in ["gen_lag24", "gen_lag48", "gen_lag168", "gen_lag336",
              "gen_roll24", "gen_roll168", "err_lag24", "err_lag48", "err_lag168"]:
        f[c] = np.nan  # egitimde tarihceden doldurulur; sunumda son gerceklesmeden
    return f


# Ufuk guvenligi: D+1 (lead 1-24) icin N>=24, D+2 (lead 25-48) icin N>=72 bakis.
# Taze kolonlar D+2 satirlarinda maskelenir (egitim) / dogal NaN (sunum).
FRESH_COLS = ["gen_lag24", "gen_lag48", "gen_roll24", "err_lag24", "err_lag48"]
SAFE_FEATS = [c for c in FEATS if c not in FRESH_COLS]


def attach_shear(f: pd.DataFrame, w100: pd.Series, w10: pd.Series) -> pd.DataFrame:
    f = f.copy()
    r = (w10.reindex(f.index) / w100.reindex(f.index).clip(lower=0.5)).clip(0.3, 2.0)
    f["shear"] = r.fillna(1.0).values
    return f


def train(df_train: pd.DataFrame, target: str = "residual", monotone: bool = False,
          feats: list | None = None):
    feats = feats or FEATS
    X = df_train[feats].copy()
    for c in CAT:
        X[c] = X[c].astype("category")
    if target == "residual":
        y = (df_train["target_gen"] - df_train["ritm_fc"]).values
    else:
        y = df_train["target_gen"].values
    params = dict(n_estimators=800, learning_rate=0.05, num_leaves=63,
                  min_child_samples=100, subsample=0.8, subsample_freq=1,
                  colsample_bytree=0.8, random_state=42, verbose=-1)
    if monotone and target == "raw":
        params["monotone_constraints"] = MONO_RAW
    d = lgb.Dataset(X, label=y)
    booster = lgb.train(params, d)
    booster.target_mode = target
    booster.feat_list = list(feats)
    return booster


def save(booster, path: str | None = None, meta: dict | None = None) -> str:
    from .. import config as Cfg
    path = path or Cfg.WIND_MODEL_TXT
    os.makedirs(os.path.dirname(path), exist_ok=True)
    booster.save_model(path)
    with open(path + ".mode", "w") as fh:
        fh.write(getattr(booster, "target_mode", "residual") + "\n")
        fh.write(",".join(getattr(booster, "feat_list", FEATS)) + "\n")
    if meta is not None:
        import json
        with open(path + ".meta.json", "w") as fh:
            json.dump(meta, fh)
    return path


def load_meta(path: str) -> dict:
    import json
    try:
        with open(path + ".meta.json") as fh:
            return json.load(fh)
    except OSError:
        return {"blend_w": 0.0, "bias": {}}


def load(path: str | None = None):
    from .. import config as Cfg
    path = path or Cfg.WIND_MODEL_TXT
    if not os.path.exists(path):
        return None
    booster = lgb.Booster(model_file=path)
    try:
        with open(path + ".mode") as fh:
            lines = fh.read().strip().split("\n")
            booster.target_mode = lines[0] if lines else "residual"
            booster.feat_list = lines[1].split(",") if len(lines) > 1 else list(FEATS)
    except OSError:
        booster.target_mode = "residual"
        booster.feat_list = list(FEATS)
    return booster


def predict(booster, feat: pd.DataFrame) -> pd.Series:
    feats = getattr(booster, "feat_list", FEATS)
    X = feat[feats].copy()
    for c in CAT:
        if c in X:
            X[c] = X[c].astype("category")
    p = np.asarray(booster.predict(X), dtype=float)
    if getattr(booster, "target_mode", "residual") == "residual":
        base = feat["ritm_fc"].fillna(0).to_numpy(dtype=float) if "ritm_fc" in feat else 0.0
        p = p + np.asarray(base, dtype=float)
    return pd.Series(np.clip(p, 0, None), index=feat.index)
