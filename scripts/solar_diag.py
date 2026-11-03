"""GES tepe kaymasinin teshisi.

Soru: 1.58 saatlik tepe kaymasi MODEL hatasi mi, yoksa lisansli santral
ornekleminin cografi/zamansal yanliligi mi?

Testler (gun basina):
  - gercek ve tahmin tepe saati
  - agirlikli ortalama saat (centroid) — gurultuye tepe kadar duyarli degil
  - dogum/batim gecisleri (tepenin %10 esigi) -> gunduz uzunlugu TRT ile uyumlu mu
  - capraz-korelasyonla EN IYI kaydirma (tum egriyi kaydirinca hata minimum nerede)
  - sabah/ogleden sonra enerji orani (bulut asimetrisi)
"""
import os, sys, glob
os.chdir(r"C:\projeler\turkiye-t2-daysahead")
sys.path.insert(0, r"C:\projeler\turkiye-t2-daysahead")
import numpy as np, pandas as pd

act = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob("data/generation/rt_gen_*.parquet"))])
act.index = pd.DatetimeIndex(act.index)
act = act[~act.index.duplicated(keep="last")].sort_index()

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

j = best.join(act[["sun"]], how="inner")
j["day"] = j.index.normalize()

rows = []
for d, g in j.groupby("day"):
    if g["sun"].max() < 300 or g["solar_pred_mw"].max() < 300:
        continue
    a = g["sun"].to_numpy(float); p = g["solar_pred_mw"].to_numpy(float)
    hr = g.index.hour.to_numpy(float)
    if a.max() <= 0 or p.max() <= 0:
        continue
    an, pn = a / a.max(), p / p.max()

    def centroid(x):
        s = x.sum()
        return float((x * hr).sum() / s) if s > 0 else np.nan

    # dogum/batim: tepenin %10'unu gecen ilk/son saat
    thr_a, thr_p = 0.10 * an.max(), 0.10 * pn.max()
    ia = np.where(an >= thr_a)[0]; ip = np.where(pn >= thr_p)[0]
    rise_a = hr[ia[0]] if len(ia) else np.nan
    set_a = hr[ia[-1]] if len(ia) else np.nan
    rise_p = hr[ip[0]] if len(ip) else np.nan
    set_p = hr[ip[-1]] if len(ip) else np.nan

    # en iyi kaydirma: tahmini k saat kaydirinca RMSE
    shifts = np.arange(-4, 5)
    rm = []
    for k in shifts:
        pk = np.roll(pn, int(k))
        rm.append(np.sqrt(np.mean((pk - an) ** 2)))
    kbest = float(shifts[int(np.argmin(rm))])

    # sabah/aksam enerji orani (12:00 oncesi / sonrasi)
    am = a[hr < 12].sum(); pm = a[hr >= 12].sum()

    rows.append(dict(day=d.date(), tepe_a=int(np.argmax(a)), tepe_p=int(np.argmax(p)),
                     tepe_shift=int(np.argmax(p) - np.argmax(a)),
                     cen_a=centroid(a), cen_p=centroid(p),
                     rise_a=rise_a, set_a=set_a, rise_p=rise_p, set_p=set_p,
                     kbest=kbest, am_pm=am / pm if pm > 0 else np.nan,
                     rmse0=rm[shifts.tolist().index(0)], rmse_best=min(rm)))

s = pd.DataFrame(rows)
print(f"=== GES teshisi: {len(s)} gun ===\n")

print("tepe kaymasi (tahmin - gercek), saat:")
print("  ort %+.2f | medyan %+.2f | std %.2f | dagilim:" % (s.tepe_shift.mean(), s.tepe_shift.median(), s.tepe_shift.std()))
print("  " + s.tepe_shift.value_counts().sort_index().to_string().replace("\n", "\n  "))

print("\ncentroid (agirlikli ortalama saat) farki:")
print("  ort %+.2f saat | medyan %+.2f" % ((s.cen_p - s.cen_a).mean(), (s.cen_p - s.cen_a).median()))

print("\ndogum/batim (tepenin %%10 esigi, TRT):")
print("  gercek  dogum %5.2f  batim %5.2f  gunuzunlugu %5.2f" % (
    s.rise_a.mean(), s.set_a.mean(), (s.set_a - s.rise_a).mean()))
print("  tahmin  dogum %5.2f  batim %5.2f  gunuzunlugu %5.2f" % (
    s.rise_p.mean(), s.set_p.mean(), (s.set_p - s.rise_p).mean()))

print("\ncapraz-korelasyon en iyi kaydirma:")
print("  ort %+.2f saat | medyan %+.0f | dagilim:" % (s.kbest.mean(), s.kbest.median()))
print("  " + s.kbest.astype(int).value_counts().sort_index().to_string().replace("\n", "\n  "))
print("  RMSE kaydirma yok %.4f -> en iyi kaydirma %.4f (kazanc %.1f%%)" % (
    s.rmse0.mean(), s.rmse_best.mean(), (1 - s.rmse_best.mean() / s.rmse0.mean()) * 100))

print("\ngerceklesen sabah/aksam enerji orani (12:00 oncesi/sonrasi):")
print("  ort %.3f | medyan %.3f | <0.8 olan gun: %d/%d" % (
    s.am_pm.mean(), s.am_pm.median(), (s.am_pm < 0.8).sum(), len(s)))
print("  (1.0 = simetrik; <1 = ogleden sonra daha guclu; >1 = sabah daha guclu)")

s.to_csv(r"C:\projeler\solar_diag.csv", index=False)
print("\nyazildi: C:\\projeler\\solar_diag.csv")
