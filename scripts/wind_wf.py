"""Uretim benzeri walk-forward: 3-il agirlik mi, 39-il spatial mi?

Protokol: 60 gun egitim, 7 gun adim, son ~90 gun degerlendirme.
Hedef: RES gerceklesmesi (EPIAS). Ayni pencere icin RITM kiyasi.

Iki kol, ruzgar girdisi disinda HER SEY ayni:
  A) agirlikli tek seri (39 il, NNLS agirligi) + istatistikler
  B) 39 il hizi ayri ayri (spatial)
"""
import os, sys, glob
os.chdir(r"C:\projeler\turkiye-t2-daysahead")
sys.path.insert(0, r"C:\projeler\turkiye-t2-daysahead")
import numpy as np, pandas as pd
import lightgbm as lgb
from scipy.optimize import nnls
from src.wind import pull_actual as PA

TRAIN_D, STEP_D, EVAL_D = 60, 7, 90

# ---------------- hedef ----------------
try:
    tgt = PA.pull_generation("2026-03-01", "2026-11-03")
    tgt = tgt.rename("y")
    if getattr(tgt.index, "tz", None) is None:
        tgt.index = tgt.index.tz_localize("Europe/Istanbul")
    tgt = tgt[~tgt.index.duplicated(keep="last")].sort_index()
    print(f"hedef (RITM endpoint gerceklesme): {len(tgt)} saat | {tgt.index.min()} -> {tgt.index.max()}")
    if len(tgt) < 2000:
        raise RuntimeError("kisa")
except Exception as ex:
    print("RITM gerceklesme alinamadi, rt-gen kullanilacak:", str(ex)[:120])
    g = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob("data/generation/rt_gen_*.parquet"))])
    g.index = pd.DatetimeIndex(g.index)
    tgt = g["wind"].rename("y")
    tgt = tgt[~tgt.index.duplicated(keep="last")].sort_index()

# ---------------- ruzgar hizi ----------------
W = pd.read_parquet(r"C:\projeler\prov_speed_long.parquet")
W.index = pd.DatetimeIndex(W.index)
if W.index.tz is None:
    W.index = W.index.tz_localize("Europe/Istanbul")
W = W[~W.index.duplicated(keep="last")].sort_index()
print(f"il hizi: {W.shape} | {W.index.min()} -> {W.index.max()}")

j = W.join(tgt, how="inner").dropna(subset=["y"])
j = j[j["y"] > 0]
print(f"eslesen: {len(j)} saat | {j.index.min()} -> {j.index.max()}")

# ---------------- ortak ozellikler (ruzgar disi) ----------------
def common(df):
    f = pd.DataFrame(index=df.index)
    f["hour"] = df.index.hour
    f["dow"] = df.index.dayofweek
    f["month"] = df.index.month
    for L in (24, 48, 168):
        f[f"lag{L}"] = df["y"].shift(L)
    f["roll24"] = df["y"].shift(24).rolling(24).mean()
    f["roll168"] = df["y"].shift(24).rolling(168).mean()
    return f

base = common(j)

# ---------------- iki girdi kolu ----------------
sp = j[W.columns]
# agirlik SADECE egitim penceresinden cikarilacak; kolaylik icin tum seriyi
# kullanmiyoruz -> asagida her blokta yeniden fit edilir.
stats = pd.DataFrame(index=j.index)
stats["wstd"] = sp.std(axis=1)
stats["wmin"] = sp.min(axis=1)
stats["wmax"] = sp.max(axis=1)
stats["wspread"] = stats["wmax"] - stats["wmin"]

# ---------------- walk-forward ----------------
days = pd.date_range(j.index.min().normalize(), j.index.max().normalize(), freq="D")
eval_start = days.max() - pd.Timedelta(days=EVAL_D)
print(f"degerlendirme: {eval_start.date()} -> {days.max().date()}")

def wmape(p, a):
    return (p - a).abs().sum() / a.sum() * 100

resA, resB, ritm_rows = [], [], []
blocks = 0
d = days.min() + pd.Timedelta(days=TRAIN_D)
while d <= days.max():
    tr0, tr1 = d - pd.Timedelta(days=TRAIN_D), d - pd.Timedelta(days=1)
    te0, te1 = d, min(d + pd.Timedelta(days=STEP_D - 1), days.max())
    mtr = (j.index >= tr0) & (j.index <= tr1 + pd.Timedelta(hours=23))
    mte = (j.index >= te0) & (j.index <= te1 + pd.Timedelta(hours=23))
    if mtr.sum() < 800 or mte.sum() < 100:
        d += pd.Timedelta(days=STEP_D); continue
    if te0 < eval_start:
        d += pd.Timedelta(days=STEP_D); continue

    # agirlik: sadece egitim penceresi
    Xtr_sp = sp[mtr].fillna(sp[mtr].mean())
    yv = j.loc[mtr, "y"]
    wv, _ = nnls((Xtr_sp / yv.max()).to_numpy(), (yv / yv.max()).to_numpy())
    wv = wv / wv.sum() if wv.sum() > 0 else np.full(len(wv), 1 / len(wv))
    wser = pd.Series(wv, index=W.columns)
    wa = sp.mul(wser, axis=1).sum(axis=1)

    fA = base.join(pd.DataFrame({"ws": wa}).join(stats))
    fB = base.join(sp).join(stats[["wstd", "wmin", "wmax", "wspread"]])
    fa = [c for c in fA.columns if not fA[c].isna().all()]
    fb = [c for c in fB.columns if not fB[c].isna().all()]

    for tag, f, feats, sink in (("A", fA, fa, resA), ("B", fB, fb, resB)):
        m = lgb.LGBMRegressor(n_estimators=600, learning_rate=0.05, num_leaves=31,
                              min_child_samples=30, subsample=0.85, colsample_bytree=0.85,
                              random_state=42, verbose=-1)
        m.fit(f.loc[mtr, feats].fillna(0), j.loc[mtr, "y"])
        p = pd.Series(m.predict(f.loc[mte, feats].fillna(0)), index=j.loc[mte].index)
        p = p.clip(lower=0)
        sink.append(pd.DataFrame({"p": p, "y": j.loc[mte, "y"]}))
    blocks += 1
    d += pd.Timedelta(days=STEP_D)

print(f"blok: {blocks}")
def agg(lst, tag):
    if not lst:
        return None
    d = pd.concat(lst)
    print("  %-8s WMAPE %6.2f%% | MAPE %6.2f%% | bias %+7.0f MW | n=%d" % (
        tag, wmape(d["p"], d["y"]), ((d["p"] - d["y"]).abs() / d["y"]).mean() * 100,
        (d["p"] - d["y"]).mean(), len(d)))
    return d

print("\n=== WALK-FORWARD SONUC ===")
A = agg(resA, "A) agirlik")
B = agg(resB, "B) spatial")

# RITM ayni pencerede
try:
    r = PA.pull_ritm_forecast(str(eval_start.date()), str(days.max().date()))[["ritm_fc_mw"]]
    if getattr(r.index, "tz", None) is None:
        r.index = r.index.tz_localize("Europe/Istanbul")
    r = r[~r.index.duplicated(keep="last")]
    yy = tgt.reindex(r.index)
    ok = yy.notna() & (yy > 0)
    print("  %-8s WMAPE %6.2f%% | MAPE %6.2f%% | n=%d" % (
        "RITM", wmape(r["ritm_fc_mw"][ok], yy[ok]),
        ((r["ritm_fc_mw"][ok] - yy[ok]).abs() / yy[ok]).mean() * 100, ok.sum()))
except Exception as ex:
    print("  RITM alinamadi:", str(ex)[:150])

if A is not None and B is not None:
    print("\n  B - A = %+.2f pp (negatif = spatial daha iyi)" % (
        wmape(B["p"], B["y"]) - wmape(A["p"], A["y"])))
