"""GES kaymasinin kaynagi: hava girisi mi, donusum modeli mi?

Karsilastir (ayni gunler):
  A) gerceklesen gunes uretimi (lisansli, MW)      -> agirlik merkezi saati
  B) hava verisindeki shortwave_radiation (W/m2)   -> agirlik merkezi saati
  C) bizim solar_pred_mw                            -> agirlik merkezi saati

Yorum:
  B ~ A  ve C geç  -> donusum modeli (kapasite/PR/geometri) geç
  B geç ve C ~ B   -> hava girisi geç (nufus agirlikli bati egemenligi)
"""
import os, sys, glob
os.chdir(r"C:\projeler\turkiye-t2-daysahead")
sys.path.insert(0, r"C:\projeler\turkiye-t2-daysahead")
import numpy as np, pandas as pd
from src import data as D

nat, _ = D.load_or_create_weather()
nat.index = pd.to_datetime(nat.index)
nat = nat[~nat.index.duplicated(keep="last")].sort_index()
print("hava indeks tz:", nat.index.tz, "| aralik:", nat.index.min(), "->", nat.index.max())
print("kolonlar:", [c for c in nat.columns if "rad" in c.lower() or "temp" in c.lower()][:6])

RAD = "shortwave_radiation"
if RAD not in nat.columns:
    print("RADYASYON KOLONU YOK"); raise SystemExit

act = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob("data/generation/rt_gen_*.parquet"))])
act.index = pd.DatetimeIndex(act.index)
act = act[~act.index.duplicated(keep="last")].sort_index()
if act.index.tz is None:
    act.index = act.index.tz_localize("Europe/Istanbul")

snaps = []
for f in sorted(glob.glob("data/forecast/ptf_features/archive/*.parquet")):
    try:
        d = pd.read_parquet(f)
    except Exception:
        continue
    if "solar_pred_mw" not in d.columns:
        continue
    d = d.copy(); d["_file"] = os.path.basename(f); snaps.append(d)
fc = pd.concat(snaps, ignore_index=True)
fc["datetime"] = pd.to_datetime(fc["datetime"])
fc["target_day"] = fc["datetime"].dt.normalize()
best = (fc[fc["horizon"].astype(str) == "T+1"].sort_values("_file")
        .groupby(["target_day", "datetime"]).last().reset_index())
best = best[["datetime", "solar_pred_mw"]].set_index("datetime")
if best.index.tz is None:
    best.index = best.index.tz_localize("Europe/Istanbul")

rad = nat[[RAD]].copy()
if rad.index.tz is None:
    rad.index = rad.index.tz_localize("Europe/Istanbul")

j = best.join(act[["sun"]], how="inner").join(rad, how="inner")
j = j.dropna(subset=[RAD])
j["day"] = j.index.normalize()
print(f"eslesen saat: {len(j)} | gun: {j['day'].nunique()}")

rows = []
for d, g in j.groupby("day"):
    if g["sun"].max() < 300 or g["solar_pred_mw"].max() < 300 or g[RAD].max() < 50:
        continue
    hr = g.index.hour.to_numpy(float)
    def cen(x):
        x = np.asarray(x, float); s = x.sum()
        return float((x * hr).sum() / s) if s > 0 else np.nan
    rows.append(dict(day=d.date(),
                     cen_act=cen(g["sun"]), cen_rad=cen(g[RAD]), cen_fc=cen(g["solar_pred_mw"]),
                     peak_act=hr[int(np.argmax(g["sun"].to_numpy(float)))],
                     peak_rad=hr[int(np.argmax(g[RAD].to_numpy(float)))],
                     peak_fc=hr[int(np.argmax(g["solar_pred_mw"].to_numpy(float)))]))

s = pd.DataFrame(rows)
print(f"\n=== {len(s)} gun ===\n")
print("agirlik merkezi saati (TRT):")
print("  gerceklesen GES : %.2f" % s.cen_act.mean())
print("  radyasyon (hava): %.2f   (fark %.2f saat)" % (s.cen_rad.mean(), s.cen_rad.mean() - s.cen_act.mean()))
print("  bizim tahmin    : %.2f   (fark %.2f saat)" % (s.cen_fc.mean(), s.cen_fc.mean() - s.cen_act.mean()))
print("  tahmin - radyasyon: %+.2f saat" % (s.cen_fc.mean() - s.cen_rad.mean()))

print("\ntepe saati (TRT):")
print("  gerceklesen GES : %.2f" % s.peak_act.mean())
print("  radyasyon (hava): %.2f   (fark %+.2f)" % (s.peak_rad.mean(), s.peak_rad.mean() - s.peak_act.mean()))
print("  bizim tahmin    : %.2f   (fark %+.2f)" % (s.peak_fc.mean(), s.peak_fc.mean() - s.peak_act.mean()))
print("  tahmin - radyasyon: %+.2f saat" % (s.peak_fc.mean() - s.peak_rad.mean()))

print("\nYORUM:")
dr = s.cen_rad.mean() - s.cen_act.mean()
df_ = s.cen_fc.mean() - s.cen_rad.mean()
if abs(df_) > abs(dr):
    print("  -> HATA DONUSUM MODELINDE (tahmin, radyasyondan bile geç)")
elif abs(dr) > 0.4:
    print("  -> HATA HAVA GIRISINDE (nufus agirlikli radyasyon geç)")
else:
    print("  -> iki kaynak da kucuk; kayma baska yerde")
s.to_csv(r"C:\projeler\solar_source.csv", index=False)
