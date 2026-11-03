"""GES/RES tahmin becerisi: bizim PTF tahmini vs EPIAS kaynak gerceklesmesi.

Gerceklesme: data/generation/rt_gen_*.parquet (sun, wind kolonlari, MW).
Tahmin     : data/forecast/ptf_features/archive/*.parquet (solar_pred_mw, wind_pred_mw)

KIYAS KURALI: her hedef gun icin o gunden ONCE uretilmis en son snapshot kullanilir
(horizon=1 satirlari). Boylece ileri-bakis sizintisi olmaz.

GES uyarisi: gerceklesme yalnizca lisansli santralleri kapsar (ulusal toplamin
alt kumesi). Bu yuzden GES icin SEVIYE hatasi degil SEKIL becerisi olculur:
  - normalize edilmis RMSE (her seri kendi gunluk tepesine bolunur)
  - tepe saati kaymasi (saat)
  - korelasyon
RES icin seviye hatasi da anlamli (kapsam makul): WMAPE + bias.
"""
from __future__ import annotations
import os, sys, glob
os.chdir(r"C:\projeler\turkiye-t2-daysahead")
sys.path.insert(0, r"C:\projeler\turkiye-t2-daysahead")
import numpy as np, pandas as pd

GEN = os.path.join("data", "generation")
ARC = os.path.join("data", "forecast", "ptf_features", "archive")

# ---------------- gerceklesme ----------------
files = sorted(glob.glob(os.path.join(GEN, "rt_gen_*.parquet")))
print(f"[cmp] {len(files)} gunluk gerceklesme dosyasi")
if not files:
    raise SystemExit("gerceklesme yok")
act = pd.concat([pd.read_parquet(f) for f in files])
act.index = pd.DatetimeIndex(act.index)
act = act[~act.index.duplicated(keep="last")].sort_index()
print(f"[cmp] gerceklesme: {act.index.min()} -> {act.index.max()} ({len(act)} saat)")

# ---------------- tahmin ----------------
snaps = []
for f in sorted(glob.glob(os.path.join(ARC, "*.parquet"))):
    try:
        d = pd.read_parquet(f)
    except Exception:
        continue
    if "solar_pred_mw" not in d.columns or "horizon" not in d.columns:
        continue
    d = d.copy()
    d["_file"] = os.path.basename(f)
    snaps.append(d)
if not snaps:
    raise SystemExit("arsiv snapshot yok")
fc = pd.concat(snaps, ignore_index=True)
fc["datetime"] = pd.to_datetime(fc["datetime"])
print(f"[cmp] snapshot satiri: {len(fc)} | {fc.datetime.min()} -> {fc.datetime.max()}")

# hedef gun basina: o gunden once uretilmis en son snapshot (T+1)
fc["target_day"] = fc["datetime"].dt.normalize()
best = (fc[fc["horizon"].astype(str) == "T+1"]
        .sort_values("_file")
        .groupby(["target_day", "datetime"]).last().reset_index())
best = best[["datetime", "solar_pred_mw", "wind_pred_mw"]].set_index("datetime")
print(f"[cmp] kullanilabilir tahmin saati: {len(best)}")
if len(best):
    md = best[best.index.hour == 12]
    print(f"[cmp] ogle ornekleri: GES {md['solar_pred_mw'].head(5).round(0).tolist()} | "
          f"RES {md['wind_pred_mw'].head(5).round(0).tolist()}")

j = best.join(act[["sun", "wind"]], how="inner")
print(f"[cmp] eslesen saat: {len(j)}")
if len(j) < 200:
    raise SystemExit("eslesme yetersiz")

# ---------------- GES: sekil becerisi ----------------
j["day"] = j.index.normalize()
rows = []
for d, g in j.groupby("day"):
    if g["sun"].max() < 200 or g["solar_pred_mw"].max() < 200:
        continue
    a = g["sun"].to_numpy(float)
    p = g["solar_pred_mw"].to_numpy(float)
    # gunes yalnizca gunduz: gece sifirlari kiyasa katma
    mask = (a > 0) | (p > 0)
    if mask.sum() < 8:
        continue
    an, pn = a / a.max(), p / p.max()
    rows.append({
        "day": d,
        "nrmse": float(np.sqrt(np.mean((pn[mask] - an[mask]) ** 2))),
        "mae_norm": float(np.mean(np.abs(pn[mask] - an[mask]))),
        "corr": float(np.corrcoef(an[mask], pn[mask])[0, 1]) if mask.sum() > 3 else np.nan,
        "tepe_kaymasi_h": int(np.argmax(p) - np.argmax(a)),
        "tahmin_tepe_mw": float(p.max()),
        "gercek_tepe_mw": float(a.max()),
    })
s = pd.DataFrame(rows)
print(f"\n=== GES (sekil) — {len(s)} gun ===")
if len(s):
    print(f"  normalize RMSE  : {s.nrmse.mean():.4f}")
    print(f"  normalize MAE   : {s.mae_norm.mean():.4f}")
    print(f"  korelasyon      : {s['corr'].mean():.4f} (medyan {s['corr'].median():.4f})")
    print(f"  tepe kaymasi    : {s.tepe_kaymasi_h.abs().mean():.2f} saat "
          f"| tam isabet {(s.tepe_kaymasi_h == 0).mean()*100:.0f}% | +/-1s icinde {(s.tepe_kaymasi_h.abs() <= 1).mean()*100:.0f}%")
    print(f"  tepe orani      : tahmin {s.tahmin_tepe_mw.mean():.0f} MW vs gercek {s.gercek_tepe_mw.mean():.0f} MW "
          f"(oran {s.tahmin_tepe_mw.mean()/s.gercek_tepe_mw.mean():.2f}x)")

# ---------------- RES: seviye becerisi ----------------
w = j[j["wind"].notna() & j["wind_pred_mw"].notna()].copy()
if len(w) > 100:
    w["err"] = w["wind_pred_mw"] - w["wind"]
    wmape = (w["err"].abs().sum() / w["wind"].abs().sum()) * 100
    mape = (w["err"].abs() / w["wind"].replace(0, np.nan)).mean() * 100
    print(f"\n=== RES (seviye) — {len(w)} saat ===")
    print(f"  WMAPE : {wmape:.2f}%")
    print(f"  MAPE  : {mape:.2f}% (sifir gerceklesmeler haric)")
    print(f"  bias  : {w['err'].mean():+.0f} MW ({w['err'].mean()/w['wind'].mean()*100:+.1f}%)")
    print(f"  korelasyon: {w['wind_pred_mw'].corr(w['wind']):.4f}")
    w["day"] = w.index.normalize()
    gd = w.groupby("day").apply(lambda g: (g["err"].abs().sum() / g["wind"].sum()) * 100, include_groups=False)
    print(f"  gunluk WMAPE: ort {gd.mean():.2f}% | medyan {gd.median():.2f}% | en iyi {gd.min():.2f}% | en kotu {gd.max():.2f}%")

s.to_csv(r"C:\projeler\solar_skill.csv", index=False)
print("\n[cmp] bitti")
