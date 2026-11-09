"""GUNLUK NWP CEKIMI — canli forecast API'sinden ruzgar (100m, 3 model).

Neden: sabah isi yeni hava cekmedigi icin `multinwp/wind` arsivi tukenmisti
(son 2026-11-04) -> T+2 ruzgar SIFIR cikiyordu.

Bu script bugunun canli tahminini (T+1/T+2/T+3) cekip ayni ceyreklik arsive
birlestirir. Kota dostu: il basina TEK istek, 3 model tek cagrida.

Cikti: C:\\projeler\\multinwp\\wind\\<Il>_<YYYY>Q<n>.parquet  (mevcutla birlestirilir)
"""
import os, sys, time, json
import numpy as np, pandas as pd
import requests

OUT = r"C:\projeler\multinwp"
URL = "https://api.open-meteo.com/v1/forecast"
MODELS = "ecmwf_ifs025,gfs_seamless,icon_seamless"
FORECAST_DAYS = int(os.environ.get("FORECAST_DAYS", "3"))

WIND_P = {"Balikesir": (39.65, 27.89), "Izmir": (38.42, 27.14), "Canakkale": (40.15, 26.41),
          "Manisa": (38.62, 27.43), "Hatay": (36.40, 36.35), "Aydin": (37.85, 27.84),
          "Istanbul": (41.01, 28.98), "Kirklareli": (41.74, 27.22),
          "Afyon": (38.75, 30.54), "Kayseri": (38.73, 35.49)}

VARS = {"wind": ("wind_speed_100m", "wind_speed_100m")}


def qtag(ts):
    return f"{ts.year}Q{(ts.month - 1)//3 + 1}"


def fetch(lat, lon, hourly):
    r = requests.get(URL, params=dict(latitude=lat, longitude=lon, hourly=hourly,
                                      models=MODELS, forecast_days=FORECAST_DAYS,
                                      timezone="UTC"), timeout=60)
    if r.status_code == 429:
        raise RuntimeError("KOTA 429")
    r.raise_for_status()
    return r.json()


def main():
    kind = "wind"
    outdir = os.path.join(OUT, kind)
    os.makedirs(outdir, exist_ok=True)
    now = pd.Timestamp.now().normalize()
    ok_n, skip_n, fail_n = 0, 0, 0

    for nm, (la, lo) in WIND_P.items():
        try:
            j = fetch(la, lo, "wind_speed_100m")
        except Exception as e:
            print(f"  {nm}: HATA {e}", flush=True)
            fail_n += 1
            time.sleep(3)
            continue
        if isinstance(j, list):
            j = j[0]
        h = j.get("hourly", {})
        if "time" not in h:
            print(f"  {nm}: bos yanit", flush=True)
            fail_n += 1
            continue
        idx = pd.to_datetime(h["time"])
        cols = {}
        for c, v in h.items():
            if c == "time" or not isinstance(v, list):
                continue
            if c.startswith("wind_speed_100m"):
                cols[c] = v
        if not cols:
            print(f"  {nm}: model kolonu yok", flush=True)
            fail_n += 1
            continue
        d = pd.DataFrame(cols, index=idx)
        d = d[~d.index.duplicated(keep="last")].sort_index()

        tgt = os.path.join(outdir, f"{nm}_{qtag(now)}.parquet")
        applied = False
        for path in (tgt,):
            if os.path.exists(path):
                old = pd.read_parquet(path)
                if getattr(old.index, "tz", None) is not None:
                    old.index = old.index.tz_localize(None)
                n0 = len(old)
                m = pd.concat([old, d])
                m = m[~m.index.duplicated(keep="last")].sort_index()
                m.to_parquet(path)
                applied = True
                print(f"  {nm}: +{len(m)-n0} saat -> {os.path.basename(path)} "
                      f"(son {m.index.max()})", flush=True)
            else:
                d.to_parquet(path)
                applied = True
                print(f"  {nm}: YENI {os.path.basename(path)} ({len(d)} saat)", flush=True)
        ok_n += 1 if applied else 0
        time.sleep(0.4)

    print(f"[pull_daily] ok={ok_n} fail={fail_n}", flush=True)
    return 0 if fail_n == 0 else 1




# --- LEAD_WEATHER (yuk + gunes) — 10 il, 7 degisken ---
LEAD_DIR = r"C:\projeler\lead_weather"
CITY_P = {"istanbul": (41.01, 28.98), "ankara": (39.93, 32.86), "izmir": (38.42, 27.14),
          "bursa": (40.19, 29.06), "antalya": (36.90, 30.70), "adana": (37.00, 35.32),
          "konya": (37.87, 32.48), "gaziantep": (37.07, 37.38),
          "kayseri": (38.73, 35.49), "trabzon": (41.00, 39.72)}
LEAD_VARS = ["temperature_2m", "apparent_temperature", "relative_humidity_2m",
             "shortwave_radiation", "wind_speed_10m", "precipitation", "cloud_cover"]


def pull_lead():
    os.makedirs(LEAD_DIR, exist_ok=True)
    ok = fail = 0
    for nm, (la, lo) in CITY_P.items():
        try:
            r = requests.get(URL, params=dict(
                latitude=la, longitude=lo, hourly=",".join(LEAD_VARS),
                forecast_days=FORECAST_DAYS, timezone="UTC"), timeout=60)
            if r.status_code == 429:
                raise RuntimeError("KOTA 429")
            r.raise_for_status()
            j = r.json()
        except Exception as e:
            print(f"  lead {nm}: HATA {e}", flush=True)
            fail += 1
            time.sleep(3)
            continue
        if isinstance(j, list):
            j = j[0]
        h = j.get("hourly", {})
        if "time" not in h:
            print(f"  lead {nm}: bos", flush=True)
            fail += 1
            continue
        idx = pd.to_datetime(h["time"])
        cols = {}
        for v in LEAD_VARS:
            if v in h:
                # T+1 ve T+2 semantigi: ayni canli tahminle doldurulur
                cols[f"{v}_previous_day1"] = h[v]
                cols[f"{v}_previous_day2"] = h[v]
        d = pd.DataFrame(cols, index=idx)
        d = d[~d.index.duplicated(keep="last")].sort_index()
        path = os.path.join(LEAD_DIR, f"{nm}.parquet")
        if os.path.exists(path):
            old = pd.read_parquet(path)
            if getattr(old.index, "tz", None) is not None:
                old.index = old.index.tz_localize(None)
            n0 = len(old)
            m = pd.concat([old, d])
            m = m[~m.index.duplicated(keep="last")].sort_index()
            m.to_parquet(path)
            print(f"  lead {nm}: +{len(m)-n0} saat (son {m.index.max()})", flush=True)
        else:
            d.to_parquet(path)
            print(f"  lead {nm}: YENI ({len(d)} saat)", flush=True)
        ok += 1
        time.sleep(0.4)
    print(f"[pull_lead] ok={ok} fail={fail}", flush=True)
    return fail

if __name__ == "__main__":
    rc = main()
    rc += pull_lead()
    sys.exit(0 if rc == 0 else 1)
