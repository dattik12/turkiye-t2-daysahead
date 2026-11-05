import os, sys
import numpy as np, pandas as pd
P = r"data\forecast\ptf_features\latest.parquet"
WL = r"data\forecast\wind_live.parquet"
d = pd.read_parquet(P)
idx = pd.DatetimeIndex(pd.to_datetime(d["datetime"])).tz_localize(None)
w = pd.read_parquet(WL)
w["datetime"] = pd.to_datetime(w["datetime"])
ws = pd.Series(w["wind_mw"].to_numpy(), index=pd.DatetimeIndex(w["datetime"]))
if getattr(ws.index, "tz", None) is not None:
    ws.index = ws.index.tz_localize(None)
v5 = ws.reindex(idx[24:])
print("v5 T+2 kapsama", v5.notna().sum(), "/24 | tepe", round(float(np.nanmax(v5.to_numpy()))))
arr = d["wind_pred_mw"].to_numpy(dtype=float).copy()
arr[24:] = v5.to_numpy()
d["wind_pred_mw"] = arr.astype("float32")
d["renewable_generation_mw"] = (d.solar_pred_mw + d.wind_pred_mw).astype("float32")
d["residual_load_mw"] = (d.consumption_pred_mw - d.renewable_generation_mw).astype("float32")
d["renewable_penetration"] = (d.renewable_generation_mw / d.consumption_pred_mw).astype("float32")
d["residual_ramp_1h"] = d.residual_load_mw.diff().fillna(0).astype("float32")
d["solar_ramp_1h"] = d.solar_pred_mw.diff().fillna(0).astype("float32")
d["wind_status"] = "blend0.70_t1_ml_v5_mnwp_t2"
bal = float((d.consumption_pred_mw - d.solar_pred_mw - d.wind_pred_mw - d.residual_load_mw).abs().max())
assert len(d) == 48 and len(d.columns) == 13 and bal < 0.01, (len(d), len(d.columns), bal)
d.to_parquet(P, index=False)
d.to_csv(P.replace(".parquet", ".csv"), index=False)
print("YAZILDI 48x13 | denge %.6f" % bal)
print("  T+1 ruzgar", round(d.wind_pred_mw[:24].max()), "| T+2 ruzgar", round(d.wind_pred_mw[24:].max()))
