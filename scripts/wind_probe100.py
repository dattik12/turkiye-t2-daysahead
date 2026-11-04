"""HIPOTEZ TESTI: 100m ruzgar (39 il) 10m'den iyi mi?

Elimizdeki 6.5 aylik 39-il 100m verisiyle kisa walk-forward.
Amac skor degil: 10m modeli %32.6 idi; 100m bunu %15 altina indiriyor mu?
Indiriyorsa yarin 5 yillik cekim hakli.
"""
import os, sys, glob
import numpy as np, pandas as pd, lightgbm as lgb

os.chdir(r"C:\projeler\turkiye-t2-daysahead")
sys.path.insert(0, r"C:\projeler\turkiye-t2-daysahead")

RAW = r"data\raw\epias_national_rt_gen_2021_2026.parquet"
CAND = [r"C:\projeler\prov_speed_long.parquet",
        r"C:\projeler\autoloop\prov_speed_long.parquet",
        r"C:\projeler\prov_lead\prov_speed_long.parquet"]
P = next((p for p in CAND if os.path.exists(p)), None)
if P is None:
    P = next(iter(glob.glob(r"C:\projeler\**\prov_speed_long.parquet", recursive=True)), None)
print(f"kaynak: {P}")
if P is None:
    raise SystemExit("prov_speed_long bulunamadi")

pv = pd.read_parquet(P)
pv.index = pd.DatetimeIndex(pv.index)
if pv.index.tz is not None:
    pv.index = pv.index.tz_localize(None)
print(f"kolon: {len(pv.columns)} | ornek: {pv.shape} | {pv.index.min()} -> {pv.index.max()}")

wnd = pd.read_parquet(RAW)["wind"].astype(float)
wnd.index = pd.DatetimeIndex(wnd.index).tz_localize(None)
wnd = wnd[~wnd.index.duplicated(keep="last")].sort_index()

# 39 il kolonunun TAMAMI (kolon adlari il adi, birim 100m hiz)
cols = list(pv.columns)
X0 = pv[cols].astype(float)
agg = pd.DataFrame({"ws_mean": X0.mean(axis=1), "ws_max": X0.max(axis=1),
                    "ws_min": X0.min(axis=1), "ws_std": X0.std(axis=1)})
agg["ws_cu"] = agg.ws_mean ** 3
print(f"kapsam: {agg.index.min()} -> {agg.index.max()} | {len(agg)} saat")

C = wnd.rolling(30 * 24, min_periods=15 * 24).quantile(0.99).shift(1).clip(lower=100.0)
cf = (wnd / C).clip(0, 1.5)

TRAIN_D, STEP_D = 75, 7
idx_all = agg.index
Xs, Ys = [], []
for t in idx_all:
    r = dict(ws_mean=t and agg.ws_mean[t], ws_max=agg.ws_max[t], ws_min=agg.ws_min[t],
             ws_std=agg.ws_std[t], ws_cu=agg.ws_cu[t],
             hour=t.hour, month=t.month,
             sin_h=np.sin(2*np.pi*t.hour/24), cos_h=np.cos(2*np.pi*t.hour/24))
    for L in (24, 48, 168):
        r[f"cf_lag{L}"] = cf.get(t - pd.Timedelta(hours=L), np.nan)
    r["C_cut"] = float(C.asof(t))
    Xs.append(r)
X = pd.DataFrame(Xs, index=idx_all)
Y = (wnd.reindex(idx_all) / X["C_cut"]).clip(0, 1.5).to_numpy()
FEAT = [c for c in X.columns if c != "C_cut"]
ok = np.isfinite(Y)
X, Y = X[ok], Y[ok]
print(f"ornek: {len(X)} | feature: {len(FEAT)}")

def wape(p, a):
    m = np.isfinite(p) & np.isfinite(a) & (a.sum() > 0)
    return float(np.abs(p - a)[m].sum() / a[m].sum() * 100)

days = pd.DatetimeIndex(sorted(set(X.index.normalize())))
preds, acts = [], []
d = days.min() + pd.Timedelta(days=TRAIN_D)
while d <= days.max():
    mtr = (X.index < d) & (X.index >= d - pd.Timedelta(days=TRAIN_D))
    mte = (X.index >= d) & (X.index < d + pd.Timedelta(days=STEP_D))
    if mtr.sum() < 1000 or mte.sum() < 24:
        d += pd.Timedelta(days=STEP_D); continue
    m = lgb.LGBMRegressor(n_estimators=600, learning_rate=0.04, num_leaves=15,
                          max_depth=5, min_child_samples=40, subsample=0.85,
                          colsample_bytree=0.85, random_state=42, verbose=-1)
    m.fit(X.loc[mtr, FEAT], Y[mtr])
    p = np.clip(m.predict(X.loc[mte, FEAT]), 0, 1.3) * X.loc[mte, "C_cut"].to_numpy()
    preds.append(pd.Series(p, index=X.index[mte]))
    acts.append(wnd.reindex(X.index[mte]))
    d += pd.Timedelta(days=STEP_D)

P = pd.concat(preds); A = pd.concat(acts); m = A.notna()
print("
=== 100m / 39 IL (6.5 ay, kisa WF) ===")
print(f"  WAPE {wape(P[m].to_numpy(), A[m].to_numpy()):5.2f}% | "
      f"nMAE {float((P[m]-A[m]).abs().mean()):,.0f} MW | "
      f"bias {float((P[m]-A[m]).mean()):+,.0f} | n={int(m.sum())}")
print("  Kiyas: 10m/10il = %32.59 | RITM ~%13")
