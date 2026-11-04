"""Kontrattaki ruzgari RITM ile doldur (Open-Meteo'ya DOKUNMADAN).

latest.parquet 48x13 duruyor; yalniz wind + turetilen kolonlar yenilenir.
"""
import os, sys
import numpy as np, pandas as pd

os.chdir(r"C:\projeler\turkiye-t2-daysahead")
sys.path.insert(0, r"C:\projeler\turkiye-t2-daysahead")
from src.wind import pull_actual as PA

P = r"C:\projeler\turkiye-t2-daysahead\data\forecast\ptf_features\latest.parquet"
d = pd.read_parquet(P)
d["datetime"] = pd.to_datetime(d["datetime"])
idx = pd.DatetimeIndex(d["datetime"]).tz_localize(None)

s = idx.min().date().isoformat()
e = idx.max().date().isoformat()
r = PA.pull_ritm_forecast(s, e)["ritm_fc_mw"]
r.index = pd.DatetimeIndex(r.index)
if r.index.tz is not None:
    r.index = r.index.tz_localize(None)

w = r.reindex(idx).to_numpy(dtype=float)
print(f"RITM kapsama: {np.isfinite(w).sum()}/{len(w)} | tepe {np.nanmax(w):,.0f}")

d["wind_pred_mw"] = np.clip(w, 0, None).astype("float32")
d["renewable_generation_mw"] = (d.solar_pred_mw + d.wind_pred_mw).astype("float32")
d["residual_load_mw"] = (d.consumption_pred_mw - d.renewable_generation_mw).astype("float32")
d["renewable_penetration"] = (d.renewable_generation_mw / d.consumption_pred_mw).astype("float32")
d["residual_ramp_1h"] = d.residual_load_mw.diff().fillna(0).astype("float32")
d["solar_ramp_1h"] = d.solar_pred_mw.diff().fillna(0).astype("float32")
d["wind_status"] = "ritm"

ident = float((d.consumption_pred_mw - d.solar_pred_mw - d.wind_pred_mw - d.residual_load_mw).abs().max())
assert len(d) == 48 and len(d.columns) == 13, f"{len(d)}x{len(d.columns)}"
assert ident < 0.01, ident

d.to_parquet(P, index=False)
d.to_csv(P.replace(".parquet", ".csv"), index=False)
print(f"YAZILDI 48x13 | denge {ident:.6f} MW")
print(f"  tuketim ort {d.consumption_pred_mw.mean():,.0f} | gunes tepe {d.solar_pred_mw.max():,.0f} | "
      f"ruzgar tepe {d.wind_pred_mw.max():,.0f} | reziduel min {d.residual_load_mw.min():,.0f}")
