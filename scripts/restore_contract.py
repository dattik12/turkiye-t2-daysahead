"""Bozulan kontrati onar: gunes = arsiv iklim egrisi (offline), ruzgar = RITM.

Kota kapaliyken canli hava yok; gunes icin ulusal `sun` arsivinden
ayni gun-of-year + saat ortalamasi (kapasite faktorune normalize) kullanilir.
solar_status = 'climatology_fallback' olarak ISARETLENIR.
"""
import os, sys
import numpy as np, pandas as pd

os.chdir(r"C:\projeler\turkiye-t2-daysahead")
sys.path.insert(0, r"C:\projeler\turkiye-t2-daysahead")
from src import config as C
from src.wind import pull_actual as PA

P = r"C:\projeler\turkiye-t2-daysahead\data\forecast\ptf_features\latest.parquet"
RAW = r"data\raw\epias_national_rt_gen_2021_2026.parquet"

d = pd.read_parquet(P)
idx = pd.DatetimeIndex(pd.to_datetime(d["datetime"])).tz_localize(None)

# --- ruzgar: RITM (EPiAS, ayri kota) ---
r = PA.pull_ritm_forecast(idx.min().date().isoformat(), idx.max().date().isoformat())["ritm_fc_mw"]
r.index = pd.DatetimeIndex(r.index)
if r.index.tz is not None:
    r.index = r.index.tz_localize(None)
w = np.array(r.reindex(idx).to_numpy(dtype=float), dtype=float)
print(f"RITM: {np.isfinite(w).sum()}/{len(w)} | tepe {np.nanmax(w):,.0f}")

# RITM yalnizca ~24 saat veriyor -> T+2 icin arsiv iklim egrisi
if np.isfinite(w).sum() < len(w):
    wnd = pd.read_parquet(RAW)["wind"].astype(float)
    wnd.index = pd.DatetimeIndex(wnd.index).tz_localize(None)
    wnd = wnd[~wnd.index.duplicated(keep="last")].sort_index()
    wp = wnd.groupby([wnd.index.dayofyear, wnd.index.hour]).mean()
    for i, (t, doy) in enumerate(zip(idx, idx.dayofyear)):
        if np.isfinite(w[i]):
            continue
        for dlt in (0, 1, -1, 2, -2):
            k = ((doy - 1 + dlt) % 365) + 1
            if (k, t.hour) in wp.index:
                w[i] = float(wp.loc[(k, t.hour)]); break
    print(f"T+2 ruzgar iklim ile dolduruldu | tepe {np.nanmax(w):,.0f}")

# --- gunes: arsiv iklim egrisi (offline) ---
sun = pd.read_parquet(RAW)["sun"].astype(float)
sun.index = pd.DatetimeIndex(sun.index).tz_localize(None)
sun = sun[~sun.index.duplicated(keep="last")].sort_index()
Cp = sun.rolling(30 * 24, min_periods=15 * 24).quantile(0.99).shift(1).clip(lower=50.0)
Cnow = float(Cp.iloc[-1])
cf = (sun / Cp).clip(0, 1.5)
prof = cf.groupby([cf.index.dayofyear, cf.index.hour]).mean()
tgt_doy = idx.dayofyear
s_vals = []
for t, doy in zip(idx, tgt_doy):
    for dlt in (0, 1, -1, 2, -2):
        k = ((doy - 1 + dlt) % 365) + 1
        if (k, t.hour) in prof.index:
            s_vals.append(float(prof.loc[(k, t.hour)]))
            break
    else:
        s_vals.append(0.0)
solar = np.clip(np.array(s_vals), 0, 1.2) * Cnow
solar = np.where(solar < 5, 0.0, solar)
print(f"gunes iklim: tepe {solar.max():,.0f} MW @ {idx[int(np.argmax(solar))]}")

d["solar_pred_mw"] = solar.astype("float32")
d["wind_pred_mw"] = np.clip(w, 0, None).astype("float32")
d["renewable_generation_mw"] = (d.solar_pred_mw + d.wind_pred_mw).astype("float32")
d["residual_load_mw"] = (d.consumption_pred_mw - d.renewable_generation_mw).astype("float32")
d["renewable_penetration"] = (d.renewable_generation_mw / d.consumption_pred_mw).astype("float32")
d["residual_ramp_1h"] = d.residual_load_mw.diff().fillna(0).astype("float32")
d["solar_ramp_1h"] = d.solar_pred_mw.diff().fillna(0).astype("float32")
d["solar_status"] = "climatology_fallback"
d["wind_status"] = np.where(pd.to_datetime(d["datetime"]).dt.date == idx.min().date(), "ritm", "climatology_fallback")

ident = float((d.consumption_pred_mw - d.solar_pred_mw - d.wind_pred_mw - d.residual_load_mw).abs().max())
assert len(d) == 48 and len(d.columns) == 13, f"{len(d)}x{len(d.columns)}"
assert ident < 0.01, ident
assert float(d.solar_pred_mw.max()) > 100 and float(d.wind_pred_mw.max()) > 1

d.to_parquet(P, index=False)
d.to_csv(P.replace(".parquet", ".csv"), index=False)
print(f"ONARILDI 48x13 | denge {ident:.6f}")
print(f"  yuk {d.consumption_pred_mw.mean():,.0f} | gunes tepe {d.solar_pred_mw.max():,.0f} | "
      f"ruzgar tepe {d.wind_pred_mw.max():,.0f} | reziduel min {d.residual_load_mw.min():,.0f}")
