"""Uretim ozellik setiyle spatial katki testi.

Uretim modeli su kolonlari kullanir (lag24/48 YASAK, D+2 icin bilinmez):
  w100, w100^2, w100^3, hour, dow, is_weekend, gen_lag168, gen_lag336,
  gen_roll168, err_lag168
(shear/gfs/yon bu testte yok — veri cekilmedi; iki kolda da yok, kiyas adil.)

Kollar:
  A) uretim ozellikleri (agirlikli tek seri)
  B) A + spatial istatistikler (mean/std/min/max/spread/q25/q75)
  C) A + 39 il hizi ayri ayri

Protokol: 60 gun egitim, 7 gun adim, ~90 gun degerlendirme. RITM ayni pencerede.
"""
import os, sys, glob
os.chdir(r"C:\projeler\turkiye-t2-daysahead")
sys.path.insert(0, r"C:\projeler\turkiye-t2-daysahead")
import numpy as np, pandas as pd
import lightgbm as lgb
from scipy.optimize import nnls
from src.wind import pull_actual as PA

TRAIN_D, STEP_D, EVAL_D = 60, 7, 90

try:
    tgt = PA.pull_generation("2026-03-01", "2026-11-03").rename("y")
    if getattr(tgt.index, "tz", None) is None:
        tgt.index = tgt.index.tz_localize("Europe/Istanbul")
except Exception as ex:
    print("RITM gerceklesme yok, rt-gen:", str(ex)[:100])
    g = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob("data/generation/rt_gen_*.parquet"))])
    g.index = pd.DatetimeIndex(g.index)
    tgt = g["wind"].rename("y")
tgt = tgt[~tgt.index.duplicated(keep="last")].sort_index()

W = pd.read_parquet(r"C:\projeler\prov_speed_long.parquet")
W.index = pd.DatetimeIndex(W.index)
if W.index.tz is None:
    W.index = W.index.tz_localize("Europe/Istanbul")
W = W[~W.index.duplicated(keep="last")].sort_index()

j = W.join(tgt, how="inner").dropna(subset=["y"])
j = j[j["y"] > 0]
print(f"eslesen {len(j)} saat | {j.index.min()} -> {j.index.max()}")

sp = j[W.columns]

def build(weighted, frame, use_sp_stats=False, use_sp_raw=False):
    idx = frame.index
    f = pd.DataFrame(index=idx)
    f["w100"] = weighted.reindex(idx).clip(lower=0)
    f["w100_2"] = f["w100"] ** 2
    f["w100_3"] = f["w100"] ** 3
    f["hour"] = idx.hour
    f["dow"] = idx.dayofweek
    f["is_weekend"] = (idx.dayofweek >= 5).astype(int)
    y = frame["y"]
    f["gen_lag168"] = y.shift(168)
    f["gen_lag336"] = y.shift(336)
    f["gen_roll168"] = y.shift(168).rolling(168).mean()
    f["err_lag168"] = (y - weighted).shift(168)
    if use_sp_stats:
        s = frame[W.columns]
        f["sp_mean"] = s.mean(axis=1)
        f["sp_std"] = s.std(axis=1)
        f["sp_min"] = s.min(axis=1)
        f["sp_max"] = s.max(axis=1)
        f["sp_spread"] = f["sp_max"] - f["sp_min"]
        f["sp_q25"] = s.quantile(0.25, axis=1)
        f["sp_q75"] = s.quantile(0.75, axis=1)
    if use_sp_raw:
        for c in W.columns:
            f[f"p_{c}"] = frame[c]
    return f

def wmape(p, a):
    return (p - a).abs().sum() / a.sum() * 100

days = pd.date_range(j.index.min().normalize(), j.index.max().normalize(), freq="D")
eval_start = days.max() - pd.Timedelta(days=EVAL_D)
print(f"degerlendirme {eval_start.date()} -> {days.max().date()}")

sinks = {"A": [], "B": [], "C": []}
d = days.min() + pd.Timedelta(days=TRAIN_D)
blocks = 0
while d <= days.max():
    te0, te1 = d, min(d + pd.Timedelta(days=STEP_D - 1), days.max())
    if te0 < eval_start:
        d += pd.Timedelta(days=STEP_D); continue
    tr0 = d - pd.Timedelta(days=TRAIN_D)
    mtr = (j.index >= tr0) & (j.index < d)
    mte = (j.index >= te0) & (j.index <= te1 + pd.Timedelta(hours=23))
    if mtr.sum() < 900 or mte.sum() < 100:
        d += pd.Timedelta(days=STEP_D); continue

    # agirlik yalnizca egitimden
    Xs = sp[mtr].fillna(sp[mtr].mean())
    yv = j.loc[mtr, "y"]
    wv, _ = nnls((Xs / yv.max()).to_numpy(), (yv / yv.max()).to_numpy())
    wv = wv / wv.sum() if wv.sum() > 0 else np.full(len(wv), 1 / len(wv))
    weighted = sp.mul(pd.Series(wv, index=W.columns), axis=1).sum(axis=1)

    for tag, kw in (("A", {}), ("B", dict(use_sp_stats=True)), ("C", dict(use_sp_raw=True))):
        f = build(weighted, j, **kw)
        feats = [c for c in f.columns if not f[c].isna().all()]
        m = lgb.LGBMRegressor(n_estimators=700, learning_rate=0.05, num_leaves=31,
                              min_child_samples=30, subsample=0.85, colsample_bytree=0.85,
                              random_state=42, verbose=-1)
        m.fit(f.loc[mtr, feats].fillna(0), j.loc[mtr, "y"])
        p = pd.Series(m.predict(f.loc[mte, feats].fillna(0)), index=j.loc[mte].index).clip(lower=0)
        sinks[tag].append(pd.DataFrame({"p": p, "y": j.loc[mte, "y"]}))
    blocks += 1
    d += pd.Timedelta(days=STEP_D)

print(f"\nblok: {blocks}")
print("=== SONUC ===")
scores = {}
for tag, nm in (("A", "uretim ozellikleri"), ("B", "+ spatial stats"), ("C", "+ 39 il hiz")):
    if not sinks[tag]:
        continue
    d0 = pd.concat(sinks[tag])
    scores[tag] = wmape(d0["p"], d0["y"])
    print("  %-3s %-22s WMAPE %6.2f%% | MAPE %6.2f%% | bias %+6.0f MW" % (
        tag, nm, scores[tag], ((d0["p"] - d0["y"]).abs() / d0["y"]).mean() * 100,
        (d0["p"] - d0["y"]).mean()))

try:
    r = PA.pull_ritm_forecast(str(eval_start.date()), str(days.max().date()))[["ritm_fc_mw"]]
    if getattr(r.index, "tz", None) is None:
        r.index = r.index.tz_localize("Europe/Istanbul")
    r = r[~r.index.duplicated(keep="last")]
    yy = tgt.reindex(r.index); ok = yy.notna() & (yy > 0)
    print("  RİTM                        WMAPE %6.2f%% | MAPE %6.2f%%" % (
        wmape(r["ritm_fc_mw"][ok], yy[ok]),
        ((r["ritm_fc_mw"][ok] - yy[ok]).abs() / yy[ok]).mean() * 100))
except Exception as ex:
    print("  RITM alinamadi:", str(ex)[:120])

if "A" in scores:
    for t in ("B", "C"):
        if t in scores:
            print("  %s - A = %+.2f pp" % (t, scores[t] - scores["A"]))
