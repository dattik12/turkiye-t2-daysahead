"""P1.1 — ARSIV HANGI VINTAGE? multinwp arsivi vs Previous Runs API day1/day2.

Kritik soru: arsivdeki deger, T+1 zamani mi (day1) yoksa daha kisa lead mi?
Kisa lead ise hem T+1 hem T+2 backtest'i iyimser.
"""
import os
import numpy as np
import pandas as pd
import requests

ARC = r"C:\projeler\multinwp\wind\Balikesir_2026Q4.parquet"
LAT, LON = 39.65, 27.89
MODEL = "ecmwf_ifs025"
VAR = "wind_speed_100m"
URL = "https://previous-runs-api.open-meteo.com/v1/forecast"

d = pd.read_parquet(ARC)
d.index = pd.DatetimeIndex(d.index)
if d.index.tz is not None:
    d.index = d.index.tz_localize(None)
col = f"{VAR}_{MODEL}"
print("arsiv:", d.shape, "| kolon var:", col in d.columns, "| son:", d.index.max())

r = requests.get(URL, params=dict(
    latitude=LAT, longitude=LON,
    hourly=f"{VAR},{VAR}_previous_day1,{VAR}_previous_day2,{VAR}_previous_day3",
    models=MODEL, past_days=60, forecast_days=1, timezone="UTC"), timeout=90)
print("HTTP", r.status_code)
h = r.json().get("hourly", {})
if "time" not in h:
    print("YANIT:", r.json()); raise SystemExit(1)

A = pd.DataFrame({(k.replace(f"{VAR}_", "") or "current"): v for k, v in h.items() if k != "time"},
                 index=pd.to_datetime(h["time"]))
print("api kolonlari:", list(A.columns), "| n:", len(A))

j = pd.DataFrame({col: d[col]}).join(A, how="inner").dropna()
print(f"ortak saat: {len(j)} | {j.index.min()} -> {j.index.max()}")
if len(j) == 0:
    raise SystemExit("ortak saat yok")

print("\n=== ARSIV HANGI VINTAGE'A ESIT? (ortalama mutlak fark) ===")
for c in A.columns:
    diff = (j[col] - j[c]).abs()
    print(f"  {c:16s} ort fark {diff.mean():6.2f} km/h | "
          f"birebir esit {100*(diff < 1e-6).mean():5.1f}% | medyan {diff.median():6.2f}")

print("\n=== KONTROL: day1 vs day2 farki (vintage'lar gercekten ayri mi) ===")
if "previous_day1" in j and "previous_day2" in j:
    dd = (j["previous_day1"] - j["previous_day2"]).abs()
    print(f"  ort {dd.mean():.2f} | birebir esit {100*(dd<1e-6).mean():.1f}%")
