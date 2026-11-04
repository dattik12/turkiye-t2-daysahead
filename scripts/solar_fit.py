"""GES TESHIS — mevcut fiziksel model ne kadar sapiyor?

Mevcut model (src/solar/model.py):
    solar_mw = GHI/1000 * 27445 MW * 0.921 * sicaklik_derate
  - PR=0.921 SABIT, hic fit edilmedi (fit_scale() NotImplementedError)
  - kurulu guc nameplate (efektif degil)
  - radyasyon NUFUS agirlikli (10 il) -> GES'ler doguda/guneyde

Bu script lisansli GES gerceklesmesiyle orani olcer:
    r(t) = gercek(t) / tahmin(t)     (gunduz saatleri)
  r SABIT ise -> sekil dogru, sadece SEVIYE kayik
  r saat/aya gore degisiyorsa -> cografya (sekil) sorunu

Kullanim: python solar_fit.py [gun]
"""
import sys, glob
sys.path.insert(0, r"C:\projeler\turkiye-t2-daysahead")
import os
os.chdir(r"C:\projeler\turkiye-t2-daysahead")
import numpy as np, pandas as pd
from src import config as C
from src.solar import radiation as R, model as SM

N = int(sys.argv[1]) if len(sys.argv) > 1 else 90

g = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob("data/generation/rt_gen_*.parquet"))])
g.index = pd.DatetimeIndex(g.index)
g = g[~g.index.duplicated(keep="last")].sort_index()
act = g["sun"].dropna()
s, e = act.index.min().date().isoformat(), act.index.max().date().isoformat()
print(f"lisansli GES gerceklesmesi: {len(act)} saat | {s} -> {e}", flush=True)
print(f"  tepe {act.max():,.0f} MW | ort {act.mean():,.0f} MW")

rad = R.history_radiation(s, (act.index.max() + pd.Timedelta(days=1)).date().isoformat())
print(f"  radyasyon serisi: {len(rad)} saat | tz={rad.index.tz} | ort {rad.mean():.1f} W/m2", flush=True)
print(f"  gerceklesme tz={act.index.tz}", flush=True)
# SAAT DILIMI HIZALAMA — biri tz-aware biri naive olabilir
if (act.index.tz is None) != (rad.index.tz is None):
    if act.index.tz is None:
        act = act.tz_localize("Europe/Istanbul")
    if rad.index.tz is None:
        rad = rad.tz_localize("Europe/Istanbul")
else:
    if act.index.tz is not None:
        rad = rad.tz_convert(act.index.tz)
rad = rad.reindex(act.index)
print(f"  hizalandi: dolu {int(rad.notna().sum())}/{len(rad)}", flush=True)
temp = None
try:
    from src import data as D
    nat, _ = D.load_or_create_weather()
    temp = nat["temperature_2m"].reindex(act.index)
except Exception:
    pass

pred = SM.solar_from_radiation(rad, C.SOLAR_CAPACITY_MW, C.SOLAR_PR, temp)
j = pd.concat({"a": act, "p": pred, "r": rad}, axis=1).dropna()
day = j[j["r"] > 20]
print(f"\ngunduz saati (GHI>20): {len(day)}")

ratio = day["a"] / day["p"].replace(0, np.nan)
ratio = ratio.replace([np.inf, -np.inf], np.nan).dropna()
print(f"\nORAN r = gercek/tahmin:")
print(f"  medyan {ratio.median():.4f} | ort {ratio.mean():.4f} | std {ratio.std():.4f}")
print(f"  CV {ratio.std()/ratio.mean()*100:.1f}%  (dusuk = sekil dogru)")
print(f"  => model lisansli GES'in {ratio.mean()*100:.1f}%'ini tahmin ediyor")

print("\nSAAT BAZINDA oran (sekil bozuksa burada gorunur):")
h = ratio.groupby(ratio.index.hour).agg(["mean", "count"])
print(h.round(3).to_string())

print("\nAY BAZINDA oran:")
m = ratio.groupby(ratio.index.to_period("M")).agg(["mean", "count"])
print(m.round(3).to_string())

# modelin ulusal tahmini ile lisansli gercek arasindaki korelasyon + kayma
c = float(day["a"].corr(day["p"]))
print(f"\nkorelasyon (lisansli gercek vs model): {c:.4f}")

def wcenter(x):
    x = x.clip(lower=0)
    w = x / x.sum()
    return float((pd.Series(x.index.hour, index=x.index) * w).sum())

days = pd.DatetimeIndex(sorted(set(day.index.normalize())))
ax, px = [], []
for d in days:
    sub = day[day.index.normalize() == d]
    if sub["a"].sum() > 0 and sub["p"].sum() > 0:
        ax.append(wcenter(sub["a"])); px.append(wcenter(sub["p"]))
if ax:
    ax, px = np.array(ax), np.array(px)
    print(f"agirlik merkezi: gercek {ax.mean():.2f} | tahmin {px.mean():.2f} | fark {px.mean()-ax.mean():+.2f} saat")

print("\n--- YORUM ---")
if ratio.std() / ratio.mean() < 0.25:
    print("Oran nispeten sabit -> SEKIL dogru, sadece SEVIYE kayik.")
    print(f"Seviye duzeltmesi: PR 0.921 -> ~{0.921*ratio.mean():.3f} (veya kapasite carpani)")
else:
    print("Oran oynak -> SEKIL de bozuk (cografi agirliklandirma sorunu).")
