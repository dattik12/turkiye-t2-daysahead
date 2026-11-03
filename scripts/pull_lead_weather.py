"""LEAD'LI HAVA ARSIVI — previous-runs API'den 10 sehir, lead1 + lead2.

Amac: yuk modeli ANALIZ havasiyla egitiliyor ama CANLI tahminde lead-1 hava
goruyor. Bu dagilim farki canli skoru bozuyor olabilir.

Bu script gecmis lead-1/lead-2 havasini parquet'e cikarir; ardindan
  (a) analiz havasi        -> referans
  (b) lead-1 havasi        -> canliya benzer
  (c) duzeltilmis lead-1   -> bias correction sonrasi
kiyasini yapabiliriz.

Kullanim: python pull_lead_weather.py 2021-01-01 2026-11-03
Cikti: C:\\projeler\\lead_weather\\<sehir>.parquet
"""
import os, sys, time
import numpy as np, pandas as pd, requests
sys.path.insert(0, r"C:\projeler\turkiye-t2-daysahead")
from src import config as C

START = sys.argv[1] if len(sys.argv) > 1 else "2021-01-01"
END = sys.argv[2] if len(sys.argv) > 2 else "2026-11-03"
OUTDIR = r"C:\projeler\lead_weather"
os.makedirs(OUTDIR, exist_ok=True)
VARS = list(C.OM_VARS)   # 7 degisken — ulusal cerceve ile BIREBIR ayni olmali,
                         # yoksa build_features eksik kolonla calisamaz.
URL = "https://previous-runs-api.open-meteo.com/v1/forecast"


def chunks(s, e, days=120):
    a = pd.Timestamp(s)
    while a < pd.Timestamp(e):
        b = min(a + pd.Timedelta(days=days - 1), pd.Timestamp(e))
        yield a.date().isoformat(), b.date().isoformat()
        a = b + pd.Timedelta(days=1)


def fetch(lat, lon, vars_, s, e):
    for att in range(3):
        try:
            r = requests.get(URL, params=dict(latitude=lat, longitude=lon,
                                              hourly=",".join(vars_), start_date=s, end_date=e,
                                              models="ecmwf_ifs025", timezone="Europe/Istanbul"),
                             timeout=90)
            if r.status_code != 200:
                raise RuntimeError(f"HTTP {r.status_code}: {r.text[:100]}")
            d = r.json()
            if d.get("error"):
                raise RuntimeError(d.get("reason", "?"))
            df = pd.DataFrame(d["hourly"])
            df["dt"] = pd.to_datetime(df["time"])
            return df.drop(columns=["time"]).set_index("dt")
        except Exception:
            if att == 2:
                raise
            time.sleep(2 * (att + 1))


for c in C.CITIES:
    p = os.path.join(OUTDIR, f"{c['city']}.parquet")
    if os.path.exists(p):
        print(f"  {c['city']}: var, atlandi", flush=True)
        continue
    parts = []
    for s, e in chunks(START, END):
        vv = [f"{v}_previous_day1" for v in VARS] + [f"{v}_previous_day2" for v in VARS]
        try:
            parts.append(fetch(c["lat"], c["lon"], vv, s, e))
        except Exception as ex:
            print(f"  {c['city']} {s}: ILK DENEME HATA {str(ex)[:70]} — sicaklik-only", flush=True)
            try:
                parts.append(fetch(c["lat"], c["lon"],
                                   ["temperature_2m_previous_day1", "temperature_2m_previous_day2"], s, e))
            except Exception as ex2:
                print(f"  {c['city']} {s}: ATLANDI {str(ex2)[:70]}", flush=True)
        time.sleep(0.35)
    if not parts:
        continue
    d = pd.concat(parts)
    d = d[~d.index.duplicated(keep="last")].sort_index()
    d.to_parquet(p)
    print(f"  {c['city']}: {d.shape} -> {p}", flush=True)

print("[lead-weather] bitti", flush=True)
