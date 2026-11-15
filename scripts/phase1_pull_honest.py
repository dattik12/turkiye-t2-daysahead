"""P1.2 — HONEST ARSIV: Previous Runs API'den day1/day2 vintage.

Cikti: data/weather/honest/<kind>/<Il>_<YYYY>Q<n>.parquet
Kolonlar: <var>_day1, <var>_day2  (3 model icin ayri dosya YOK; model boyutu kolonda)
  -> <var>__<model>__day1 / <var>__<model>__day2

Kullanim:
  python scripts/phase1_pull_honest.py wind
  python scripts/phase1_pull_honest.py solar
"""
import os, sys, time, json
import numpy as np
import pandas as pd
import requests

URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
MODELS = ["ecmwf_ifs025", "gfs_seamless", "icon_seamless"]
OUT = r"C:\projeler\turkiye-t2-daysahead\data\weather\honest"

WIND_P = {"Balikesir": (39.65, 27.89), "Izmir": (38.42, 27.14), "Canakkale": (40.15, 26.41),
          "Manisa": (38.62, 27.43), "Hatay": (36.40, 36.35), "Aydin": (37.85, 27.84),
          "Istanbul": (41.01, 28.98), "Kirklareli": (41.74, 27.22),
          "Afyon": (38.75, 30.54), "Kayseri": (38.73, 35.49)}
WIND_V = ["wind_speed_100m", "wind_direction_100m", "wind_speed_10m",
          "temperature_2m", "boundary_layer_height"]

SOLAR_P = {"Konya": (37.87, 32.48), "Sanliurfa": (37.16, 38.79), "Ankara": (39.93, 32.86),
           "Kayseri": (38.73, 35.49), "Izmir": (38.42, 27.14), "Antalya": (36.90, 30.70),
           "Mersin": (36.80, 34.63), "Kahramanmaras": (37.58, 36.93)}
# DIKKAT: cloud_cover_low/mid/high Previous Runs API'de HIC gelmiyor (%0 doluluk).
# Toplam cloud_cover var (%91-100). Operasyonel model de toplam bulutu kullaniyor.
SOLAR_V = ["shortwave_radiation", "direct_radiation", "diffuse_radiation",
           "cloud_cover", "temperature_2m"]

START = os.environ.get("START", "2024-01-01")
END = os.environ.get("END", pd.Timestamp.now().strftime("%Y-%m-%d"))


def qtag(ts):
    return f"{ts.year}Q{(ts.month - 1)//3 + 1}"


def month_chunks(s, e):
    cur = pd.Timestamp(s).normalize()
    e = pd.Timestamp(e).normalize()
    while cur <= e:
        nxt = (cur + pd.offsets.MonthBegin(1)) - pd.Timedelta(days=1)
        yield cur, min(nxt, e)
        cur = nxt + pd.Timedelta(days=1)


def fetch(lat, lon, hourly, s, e, tries=3):
    for k in range(tries):
        try:
            r = requests.get(URL, params=dict(
                latitude=lat, longitude=lon, hourly=hourly,
                models=",".join(MODELS), start_date=s, end_date=e,
                timezone="UTC"), timeout=180)
            if r.status_code == 429:
                print("    KOTA 429 — 60s bekle", flush=True)
                time.sleep(60); continue
            r.raise_for_status()
            return r.json()
        except Exception as ex:
            print(f"    hata ({k+1}/{tries}): {str(ex)[:70]}", flush=True)
            time.sleep(5)
    return None


def run(kind):
    provs, vars_ = (WIND_P, WIND_V) if kind == "wind" else (SOLAR_P, SOLAR_V)
    odir = os.path.join(OUT, kind)
    os.makedirs(odir, exist_ok=True)
    # Ruzgar degiskenleri tek istekte cakistiriyor (400 SurfacePressureAndHeightVariable),
    # tek tek istendiginde hepsi 200. Bu yuzden degisken basina ayri istek.
    per_var = (kind == "wind")
    print(f"[{kind}] {len(provs)} il | {len(vars_)} degisken | {len(MODELS)} model x2 vintage"
          f" | mod: {'degisken basina istek' if per_var else 'tek istek'}")

    ok = fail = 0
    for nm, (la, lo) in provs.items():
        frames = []
        for s, e in month_chunks(START, END):
            got = False
            batch = [[v] for v in vars_] if per_var else [list(vars_)]
            for grp in batch:
                hv = ",".join(f"{v}_previous_day{d}" for v in grp for d in (1, 2))
                j = fetch(la, lo, hv, s.date().isoformat(), e.date().isoformat())
                time.sleep(0.3)
                if not j or "time" not in j.get("hourly", {}):
                    fail += 1; continue
                h = j["hourly"]
                frames.append(pd.DataFrame({k: v for k, v in h.items() if k != "time"},
                                           index=pd.to_datetime(h["time"])))
                got = True
            if not got:
                continue
        if not frames:
            print(f"  {nm}: VERI YOK", flush=True); continue
        d = pd.concat(frames).sort_index()
        d = d[~d.index.duplicated(keep="last")]
        # ceyreklik dosyalara bol
        for q, g in d.groupby(pd.PeriodIndex(d.index, freq="Q")):
            path = os.path.join(odir, f"{nm}_{q.year}Q{q.quarter}.parquet")
            g.to_parquet(path)
        print(f"  {nm}: {len(d)} saat -> {d.index.min()} .. {d.index.max()}", flush=True)
        ok += 1
    print(f"[{kind}] ok={ok} fail={fail}")


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "wind")
