"""W3 — RUZGAR SEGMENT ANALIZI.

Soru: ruzgarda NEREDE kaybediyoruz? Tek bir WMAPE sayisi yol gostermiyor.
Segment kirilimi: ay, ruzgar bandi, saat, yil.

Protokol (uretim benzeri): egitim 90 gun, adim 7 gun, degerlendirme tum pencere.
Agirliklar her blokta SADECE egitim penceresinden NNLS (sizinti yok).

Cikti: tahmin vs RITM vs gercek, segment segment.

Kullanim: python wind_segments.py
"""
import os, sys, json
os.chdir(r"C:\projeler\turkiye-t2-daysahead")
sys.path.insert(0, r"C:\projeler\turkiye-t2-daysahead")
import numpy as np, pandas as pd
import lightgbm as lgb
from scipy.optimize import nnls
from src.wind import pull_actual as PA

OUT = r"C:\projeler\autoloop"
STEP_D, TRAIN_D = 7, 90

tgt = PA.pull_generation("2026-03-01", "2026-11-03").rename("y")
tgt.index = pd.DatetimeIndex(tgt.index)
if tgt.index.tz is None:
    tgt.index = tgt.index.tz_localize("Europe/Istanbul")
tgt = tgt[~tgt.index.duplicated(keep="last")].sort_index()

W = pd.read_parquet(r"C:\projeler\prov_speed_long.parquet")
W.index = pd.DatetimeIndex(W.index)
if W.index.tz is None:
    W.index = W.index.tz_localize("Europe/Istanbul")
W = W[~W.index.duplicated(keep="last")].sort_index()

j = W.join(tgt, how="inner").dropna(subset=["y"])
j = j[j["y"] > 0]
sp = j[W.columns]
print(f"eslesen {len(j)} saat | {j.index.min()} -> {j.index.max()}", flush=True)

idx = j.index
f = pd.DataFrame(index=idx)
f["hour"] = idx.hour; f["dow"] = idx.dayofweek
f["gen_lag168"] = j["y"].shift(168)
f["gen_lag336"] = j["y"].shift(336)
f["gen_roll168"] = j["y"].shift(168).rolling(168).mean()
for c in W.columns:
    f[f"p_{c}"] = j[c]
s = j[W.columns]
for nm, fn in (("sp_mean", "mean"), ("sp_std", "std"), ("sp_min", "min"), ("sp_max", "max")):
    f[nm] = getattr(s, fn)(axis=1)
f["sp_spread"] = f["sp_max"] - f["sp_min"]


def wm(p, a):
    return float(np.abs(p - a).sum() / a.sum() * 100)


days = pd.date_range(j.index.min().normalize(), j.index.max().normalize(), freq="D")
eval_start = days.min() + pd.Timedelta(days=TRAIN_D)
print(f"degerlendirme {eval_start.date()} -> {days.max().date()}", flush=True)

preds = []
d = eval_start.normalize()
while d <= days.max():
    te0, te1 = d, min(d + pd.Timedelta(days=STEP_D - 1), days.max())
    mte = (idx >= te0) & (idx <= te1 + pd.Timedelta(hours=23))
    tr0 = d - pd.Timedelta(days=TRAIN_D)
    mtr = (idx >= tr0) & (idx < d)
    if mtr.sum() < 1000 or mte.sum() < 100:
        d += pd.Timedelta(days=STEP_D); continue
    Xs = sp[mtr].fillna(sp[mtr].mean()); yv = j.loc[mtr, "y"]
    wv, _ = nnls((Xs / yv.max()).to_numpy(), (yv / yv.max()).to_numpy())
    wv = wv / wv.sum() if wv.sum() > 0 else np.full(len(wv), 1 / len(wv))
    weighted = sp.mul(pd.Series(wv, index=W.columns), axis=1).sum(axis=1)
    fb = f.copy()
    fb["w100"] = weighted.clip(lower=0); fb["w100_2"] = fb["w100"] ** 2
    err = j["y"] - weighted
    fb["err_lag168"] = err.shift(168).reindex(fb.index)
    fc = [c for c in fb.columns if not fb[c].isna().all()]
    m = lgb.LGBMRegressor(n_estimators=700, learning_rate=0.05, num_leaves=31,
                          min_child_samples=30, subsample=0.85, colsample_bytree=0.85,
                          random_state=42, verbose=-1)
    m.fit(fb.loc[mtr, fc], j.loc[mtr, "y"])
    p = np.clip(m.predict(fb.loc[mte, fc]), 0, None)
    preds.append(pd.DataFrame({"p": p, "y": j.loc[mte, "y"].to_numpy()}, index=j.loc[mte].index))
    d += pd.Timedelta(days=STEP_D)

R = pd.concat(preds)
print(f"tahmin saati: {len(R)}", flush=True)

# RITM ayni saatlerde
try:
    r = PA.pull_ritm_forecast(str(R.index.min().date()), str(R.index.max().date()))[["ritm_fc_mw"]]
    r.index = pd.DatetimeIndex(r.index)
    if r.index.tz is None:
        r.index = r.index.tz_localize("Europe/Istanbul")
    R = R.join(r[~r.index.duplicated(keep="last")], how="left")
except Exception as ex:
    print("RITM yok:", str(ex)[:80]); R["ritm_fc_mw"] = np.nan

has_r = R["ritm_fc_mw"].notna()
print(f"RITM eslesen: {int(has_r.sum())}/{len(R)}\n")

def seg(name, key):
    print(f"=== {name} ===")
    print("%-16s %9s %9s %8s %7s" % ("segment", "BIZ", "RITM", "fark", "n"))
    g = R.groupby(key, observed=True)
    for k, s2 in g:
        if len(s2) < 40:
            continue
        a = s2["y"].to_numpy(); b = wm(s2["p"].to_numpy(), a)
        rr = s2["ritm_fc_mw"].dropna()
        rt = wm(rr.to_numpy(), s2.loc[rr.index, "y"].to_numpy()) if len(rr) > 20 else float("nan")
        fk = b - rt if not np.isnan(rt) else float("nan")
        print("%-16s %8.2f%% %8.2f%% %+7.2f %7d" % (str(k), b, rt, fk, len(s2)))
    print()

seg("AY", R.index.month)
seg("RUZGAR BANDI (MW)", pd.cut(R["y"], [0, 1000, 3000, 6000, 9000, 15000]))
seg("SAAT DILIMI", pd.cut(R.index.hour, [-1, 5, 11, 17, 23],
                          labels=["gece", "sabah", "ogleden_sonra", "aksam"]))
seg("CEYREK", R.index.to_period("Q").astype(str))

R.to_parquet(os.path.join(OUT, "wind_segments.parquet"))
print("segment verisi -> wind_segments.parquet", flush=True)
