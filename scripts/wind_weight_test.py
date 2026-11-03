"""Ruzgar hava girdisi karsilastirmasi (ayni model, farkli cografi kapsam).

Sorulari:
  1. Mevcut 3-il agirligi mi, 38-il esit agirlik mi, 38-il NNLS mi daha iyi?
  2. Tum il hizlarini modele vermek (spatial) tek agirlikli seriden iyi mi?

Kural: egitim 2026-08-04..2026-09-30, test 2026-10-01..2026-11-01.
Hicbir kol test donemini gormez. RITM ayni test doneminde kiyas.
"""
import os, sys, glob
os.chdir(r"C:\projeler\turkiye-t2-daysahead")
sys.path.insert(0, r"C:\projeler\turkiye-t2-daysahead")
import numpy as np, pandas as pd
from scipy.optimize import nnls
import lightgbm as lgb

TR0, TR1 = "2026-08-04", "2026-09-30"
TE0, TE1 = "2026-10-01", "2026-11-01"

# --- gerceklesme
act = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob("data/generation/rt_gen_*.parquet"))])
act.index = pd.DatetimeIndex(act.index)
act = act[~act.index.duplicated(keep="last")].sort_index()
y = act["wind"].rename("y")

# --- il hizlari
W = pd.read_parquet(r"C:\projeler\prov_speed_90d.parquet")
W.index = pd.DatetimeIndex(W.index)
if W.index.tz is None:
    W.index = W.index.tz_localize("Europe/Istanbul")
W = W[~W.index.duplicated(keep="last")].sort_index()
print(f"il hizi: {W.shape} | {W.index.min()} -> {W.index.max()}")

j = W.join(y, how="inner").dropna(subset=["y"])
j = j[j["y"] > 0]
print(f"eslesen: {len(j)} saat | {j.index.min()} -> {j.index.max()}")

# mevcut uretim agirligi
cur = pd.read_csv("data/wind/WIND_FARMS.csv")
cur = cur[cur["w"] > 0].set_index("province")["w"]
cur = cur[cur.index.isin(W.columns)]
cur = cur / cur.sum()
print(f"\nmevcut etkin iller: {dict(cur.round(3))}")

def wavg(df, w):
    w = w.reindex(df.columns).fillna(0.0)
    if w.sum() <= 0:
        raise ValueError("agirlik yok")
    return df.mul(w, axis=1).sum(axis=1) / w.sum()

def fit_nnls(X, t):
    tm = t.max()
    A = (X / tm).to_numpy()
    w, _ = nnls(A, (t / tm).to_numpy())
    s = w.sum()
    return pd.Series(w / s if s > 0 else np.full(len(w), 1 / len(w)), index=X.columns)

tr_m = (j.index >= TR0) & (j.index <= TR1)
te_m = (j.index >= TE0) & (j.index <= TE1)
print(f"egitim {tr_m.sum()} saat | test {te_m.sum()} saat")

def run(frame, feat_cols, tag):
    Xtr, ytr = frame.loc[tr_m, feat_cols], frame.loc[tr_m, "y"]
    Xte, yte = frame.loc[te_m, feat_cols], frame.loc[te_m, "y"]
    m = lgb.LGBMRegressor(n_estimators=700, learning_rate=0.05, num_leaves=31,
                          min_child_samples=40, subsample=0.85, colsample_bytree=0.85,
                          random_state=42, verbose=-1)
    m.fit(Xtr.fillna(0), ytr)
    p = pd.Series(m.predict(Xte.fillna(0)), index=Xte.index)
    e = (p - yte).abs()
    wmape = e.sum() / yte.sum() * 100
    mape = (e / yte).mean() * 100
    print("  %-22s WMAPE %6.2f%%  MAPE %6.2f%%  bias %+7.0f MW" % (
        tag, wmape, mape, (p - yte).mean()))
    return p, wmape

print("\n=== TEK AGIRLIKLI SERI (1 feature) ===")
res = {}
sp = j[W.columns]
for tag, w in [("mevcut 3-il", cur),
               ("39-il esit", pd.Series(1.0, index=W.columns))]:
    jj = j.copy(); jj["ws"] = wavg(sp, w)
    p, sc = run(jj, ["ws"], tag)
    res[tag] = (p, sc)

# NNLS agirligi SADECE egitim doneminden
wn = fit_nnls(sp[tr_m].fillna(sp[tr_m].mean()), j.loc[tr_m, "y"])
print("  NNLS etkin il sayisi (w>0.01): %d | en buyuk 6: %s" % (
    (wn > 0.01).sum(), dict(wn.nlargest(6).round(3))))
jj = j.copy(); jj["ws"] = wavg(sp, wn)
p, sc = run(jj, ["ws"], "39-il NNLS (egt.fit)")
res["nnls"] = (p, sc)

print("\n=== TUM IL HIZLARI (spatial, 39 feature) ===")
p, sc = run(j, list(W.columns), "39-il spatial")
res["spatial"] = (p, sc)

# RITM kiyas
try:
    from src.wind import pull_actual as PA
    r = PA.pull_ritm_forecast(TE0, TE1)[["ritm_fc_mw"]]
    if getattr(r.index, "tz", None) is None:
        r.index = r.index.tz_localize("Europe/Istanbul")
    r = r[~r.index.duplicated(keep="last")]
    yy = y.reindex(r.index)
    ok = yy.notna() & (yy > 0)
    e = (r["ritm_fc_mw"][ok] - yy[ok]).abs()
    print("\n  %-22s WMAPE %6.2f%%  MAPE %6.2f%%" % (
        "RITM (ayni donem)", e.sum() / yy[ok].sum() * 100, (e / yy[ok]).mean() * 100))
except Exception as ex:
    print("  RITM alinamadi:", str(ex)[:120])

print("\n=== TEST DONEMI GUNLUK KAZANMA (en iyi kol vs RITM) ===")
if res:
    best = min(res.items(), key=lambda kv: kv[1][1])
    print(f"  en iyi kol: {best[0]} ({best[1][1]:.2f}%)")
