"""Uctan uca dogrulama: sabah sozlesmesi + PTF kontrati + ruzgar modeli.

Hicbir sey degistirmez; sadece uretim ciktilarinin tutarliligini kontrol eder.
"""
import os, sys, glob
os.chdir(r"C:\projeler\turkiye-t2-daysahead")
sys.path.insert(0, r"C:\projeler\turkiye-t2-daysahead")
import numpy as np, pandas as pd

FAIL = []
def ok(cond, msg):
    print(("  OK   " if cond else "  FAIL ") + msg)
    if not cond:
        FAIL.append(msg)

print("=== 1) SABAH SOZLESMESI (09:00 -> yarin 24 saat) ===")
p = "data/forecast/morning/latest_yarin.parquet"
if os.path.exists(p):
    d = pd.read_parquet(p)
    ok(len(d) == 24, f"24 satir (gelen {len(d)})")
    ok(d["pred_mw"].notna().all(), "tahminler NaN'siz")
    ok((d["pred_mw"] > 5000).all() and (d["pred_mw"] < 80000).all(),
       f"makul aralik (min {d.pred_mw.min():.0f} / max {d.pred_mw.max():.0f})")
    ok(d["pred_mw"].max() / d["pred_mw"].min() < 3.0,
       f"tepe/dip orani {d.pred_mw.max()/d.pred_mw.min():.2f} (<3)")
    peak_h = int(d.loc[d.pred_mw.idxmax(), "datetime"].hour)
    ok(15 <= peak_h <= 20, f"tepe saati {peak_h}:00 makul (15-20)")
    ok(d["pred_mw"].mean() > 25000, f"ortalama {d.pred_mw.mean():.0f} MW")
else:
    ok(False, "sabah ciktisi yok")

print("\n=== 2) PTF KONTRATI (fiyat ekibine giden dosya) ===")
q = "data/forecast/ptf_features/latest.parquet"
NEED = ["datetime", "horizon", "consumption_pred_mw", "solar_pred_mw", "wind_pred_mw",
        "renewable_generation_mw", "residual_load_mw", "renewable_penetration",
        "residual_ramp_1h", "solar_ramp_1h", "is_peak_hour", "solar_status", "wind_status"]
if os.path.exists(q):
    f = pd.read_parquet(q)
    ok(len(f) == 48, f"48 satir (gelen {len(f)})")
    miss = [c for c in NEED if c not in f.columns]
    ok(not miss, f"13 kolon tam (eksik {miss})")
    for c in ["consumption_pred_mw", "solar_pred_mw", "wind_pred_mw", "residual_load_mw"]:
        if c in f.columns:
            ok(f[c].notna().all(), f"{c} NaN'siz")
    if "residual_load_mw" in f.columns:
        calc = f["consumption_pred_mw"] - f["solar_pred_mw"] - f["wind_pred_mw"]
        diff = (calc - f["residual_load_mw"]).abs().max()
        ok(diff < 1.0, f"reziduel = yuk - gunes - ruzgar (max fark {diff:.3f})")
    ok((f["solar_pred_mw"] >= 0).all(), "gunes negatif degil")
    ok((f["wind_pred_mw"] >= 0).all(), "ruzgar negatif degil")
    ok((f["residual_load_mw"] > 0).all(), "reziduel pozitif")
    ok(f["horizon"].nunique() == 2, f"iki horizon ({list(f.horizon.unique())})")
    print(f"  bilgi: ruzgar ort {f.wind_pred_mw.mean():.0f} tepe {f.wind_pred_mw.max():.0f} | "
          f"gunes tepe {f.solar_pred_mw.max():.0f} | reziduel min {f.residual_load_mw.min():.0f}")
else:
    ok(False, "PTF ciktisi yok")

print("\n=== 3) RUZGAR MODELI (spatial, 39 il) ===")
from src.wind import model as WMODEL
from src import config as C
for tag, path in (("D+1", C.WIND_MODEL_D1), ("D+2", C.WIND_MODEL_D2)):
    b = WMODEL.load(path)
    if b is None:
        ok(False, f"{tag} model yuklenemedi")
        continue
    feats = getattr(b, "feat_list", [])
    nsp = len([c for c in feats if c.startswith("p_")])
    ok(nsp >= 30, f"{tag} spatial kolon {nsp} (>=30)")
    meta = WMODEL.load_meta(path)
    print(f"  bilgi: {tag} blend_w={meta.get('blend_w')} bins={meta.get('blend_bins')}")

print("\n=== 4) SAATLIK TAKVIM ===")
from src import morning as MO
cal = MO.load_calendar()
ok(cal is not None and cal.shape[1] >= 30, f"takvim {None if cal is None else cal.shape}")
if cal is not None:
    idx = pd.date_range("2026-11-04", periods=24, freq="h")
    X = MO.attach_calendar(pd.DataFrame({"a": 1.0}, index=idx))
    ok(X.shape[1] == 32, f"attach_calendar 31 kolon ekliyor (toplam {X.shape[1]})")

print("\n=== 5) URETIM DEFTERLERI ===")
for name, path2, mn in (("forecast_results", "data/results/forecast_results.csv", 3400),
                        ("mape_history", "data/results/mape_history.csv", 100),
                        ("morning_history", "data/results/morning_history.csv", 24)):
    if os.path.exists(path2):
        d2 = pd.read_csv(path2)
        ok(len(d2) >= mn, f"{name}: {len(d2)} satir (>= {mn})")
    else:
        ok(False, f"{name} yok")

print("\n" + "="*50)
if FAIL:
    print(f"SONUC: {len(FAIL)} KONTROL BASARISIZ")
    for x in FAIL:
        print("  -", x)
    sys.exit(1)
print("SONUC: TUM KONTROLLER GECTI")
