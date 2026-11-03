"""Gunes isinimi: nufus agirlikli mi, cografi ortalama mi?

Kiyas: lisansli GES gerceklesmesinin SEKLI (EPIAS rt-gen 'sun').
Olcum: agirlik merkezi saati, tepe saati, korelasyon, normalize RMSE.
"""
import os, sys, glob
os.chdir(r"C:\projeler\turkiye-t2-daysahead")
sys.path.insert(0, r"C:\projeler\turkiye-t2-daysahead")
import numpy as np, pandas as pd
from src import data as D

# --- lisansli gerceklesme
act = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob("data/generation/rt_gen_*.parquet"))])
act.index = pd.DatetimeIndex(act.index)
act = act[~act.index.duplicated(keep="last")].sort_index()
sun = act["sun"].rename("actual")

# --- il bazli isinim
R = pd.read_parquet(r"C:\projeler\solar_rad_39.parquet")
R.index = pd.DatetimeIndex(R.index)
if R.index.tz is None:
    R.index = R.index.tz_localize("Europe/Istanbul")
R = R[~R.index.duplicated(keep="last")].sort_index()
print(f"il isinim: {R.shape} | {R.index.min()} -> {R.index.max()}")

# --- nufus agirlikli (mevcut nat)
nat, _ = D.load_or_create_weather()
nat.index = pd.to_datetime(nat.index)
if getattr(nat.index, "tz", None) is None:
    nat.index = nat.index.tz_localize("Europe/Istanbul")
nat = nat[~nat.index.duplicated(keep="last")].sort_index()
pop = nat["shortwave_radiation"].rename("pop")

geo = R.mean(axis=1).rename("geo")          # 39 il esit agirlik = cografi ortalama
# guney/ic agirlikli (gunes ciftligi bolgeleri) — kabaca yuksek potansiyel iller
SOLAR_PROV = ["Konya", "Karaman", "Nigde", "Kayseri", "Sivas", "Sanliurfa" if False else "Gaziantep",
              "Adiyaman", "Malatya", "Elazig", "Mersin", "Adana", "Osmaniye", "Hatay",
              "Afyon", "Kutahya", "Kirikkale", "Kirsehir", "Corum"]
SOLAR_PROV = [p for p in SOLAR_PROV if p in R.columns]
solar_w = R[SOLAR_PROV].mean(axis=1).rename("solarw")
print(f"solar-bolge ili: {len(SOLAR_PROV)}")

j = pd.concat([sun, pop, geo, solar_w], axis=1, join="inner").dropna()
print(f"eslesen saat: {len(j)}")
j["day"] = j.index.normalize()

def cen(x, hr):
    s = x.sum()
    return float((x * hr).sum() / s) if s > 0 else np.nan

rows = []
for d, g in j.groupby("day"):
    if g["actual"].max() < 300 or g["pop"].max() < 50:
        continue
    hr = g.index.hour.to_numpy(float)
    r = {"day": d.date()}
    for c in ("actual", "pop", "geo", "solarw"):
        a = g[c].to_numpy(float)
        r[f"cen_{c}"] = cen(a, hr)
        r[f"pk_{c}"] = hr[int(np.argmax(a))]
    # normalize RMSE (yalniz gunduz)
    m = (g["actual"] > 0) | (g["geo"] > 0)
    an = g["actual"].to_numpy(float); an = an / an.max() if an.max() > 0 else an
    for c in ("pop", "geo", "solarw"):
        v = g[c].to_numpy(float); v = v / v.max() if v.max() > 0 else v
        r[f"nrmse_{c}"] = float(np.sqrt(np.mean((v[m] - an[m]) ** 2)))
        r[f"corr_{c}"] = float(np.corrcoef(v[m], an[m])[0, 1]) if m.sum() > 5 else np.nan
    rows.append(r)

s = pd.DataFrame(rows)
print(f"\n=== {len(s)} gun ===")
print("%-8s %8s %8s %8s | %8s %8s %8s" % ("", "merkez", "tepe", "kayma", "nRMSE", "korel", "tepe_fark"))
for c, lbl in (("actual", "GERCEK"), ("pop", "nufus-ag."), ("geo", "cografi-ort"), ("solarw", "gunes-bolge")):
    cen_m = s[f"cen_{c}"].mean(); pk_m = s[f"pk_{c}"].mean()
    if c == "actual":
        print("%-8s %8.2f %8.2f %8s | %8s %8s %8s" % (lbl, cen_m, pk_m, "-", "-", "-", "-"))
    else:
        print("%-8s %8.2f %8.2f %+8.2f | %8.4f %8.3f %+8.2f" % (
            lbl, cen_m, pk_m, cen_m - s["cen_actual"].mean(),
            s[f"nrmse_{c}"].mean(), s[f"corr_{c}"].mean(), pk_m - s["pk_actual"].mean()))

print("\nAGIRLIK MERKEZI farki (seri - gercek), saat:")
for c in ("pop", "geo", "solarw"):
    v = s[f"cen_{c}"] - s["cen_actual"]
    print("  %-12s ort %+.2f | medyan %+.2f | std %.2f" % (c, v.mean(), v.median(), v.std()))

s.to_csv(r"C:\projeler\solar_geo_cmp.csv", index=False)
print("\nkazanan (en dusuk nRMSE):", min(("pop", "geo", "solarw"), key=lambda c: s[f"nrmse_{c}"].mean()))
