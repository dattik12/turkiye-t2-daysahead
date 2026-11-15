"""P1.2 — HONEST ARSIV (v2) — OpenMeteoClient ile, kota sayacli.

Esik: Previous Runs API, previous_day1 + previous_day2, 3 model.
Cikti: data/weather/honest/<kind>/<Il>_<YYYY>Q<n>.parquet
Kolonlar: {var}_previous_day{1,2}_{model}
"""
import os
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from openmeteo_client import OpenMeteoClient  # noqa: E402

BASE = "https://previous-runs-api.open-meteo.com/v1/forecast"
OUT = r"C:\projeler\turkiye-t2-daysahead\data\weather\honest"
MODELS = "ecmwf_ifs025,gfs_seamless,icon_seamless"

WIND_P = {"Balikesir": (39.65, 27.89), "Izmir": (38.42, 27.14), "Canakkale": (40.15, 26.41),
          "Manisa": (38.62, 27.43), "Hatay": (36.40, 36.35), "Aydin": (37.85, 27.84),
          "Istanbul": (41.01, 28.98), "Kirklareli": (41.74, 27.22),
          "Afyon": (38.75, 30.54), "Kayseri": (38.73, 35.49)}
WIND_V = ["wind_speed_100m", "wind_direction_100m", "wind_speed_10m",
          "temperature_2m", "boundary_layer_height"]

SOLAR_P = {"Konya": (37.87, 32.48), "Sanliurfa": (37.16, 38.79), "Ankara": (39.93, 32.86),
           "Kayseri": (38.73, 35.49), "Izmir": (38.42, 27.14), "Antalya": (36.90, 30.70),
           "Mersin": (36.80, 34.63), "Kahramanmaras": (37.58, 36.93)}
SOLAR_V = ["shortwave_radiation", "direct_radiation", "diffuse_radiation",
           "cloud_cover", "temperature_2m"]

START_Y, START_Q = 2024, 1
END = pd.Timestamp.now()


def quarters():
    y, q = START_Y, START_Q
    while True:
        sy = pd.Timestamp(year=y, month=3 * (q - 1) + 1, day=1)
        ey = (sy + pd.offsets.QuarterEnd(0))
        if sy > END:
            break
        yield f"{y}Q{q}", sy, min(ey, END)
        q += 1
        if q == 5:
            q, y = 1, y + 1


def url(la, lo, hv, s, e):
    return (f"{BASE}?latitude={la}&longitude={lo}&hourly={hv}"
            f"&models={MODELS}&start_date={s}&end_date={e}&timezone=UTC")


def run(kind):
    provs, vars_ = (WIND_P, WIND_V) if kind == "wind" else (SOLAR_P, SOLAR_V)
    odir = os.path.join(OUT, kind)
    os.makedirs(odir, exist_ok=True)

    cli = OpenMeteoClient(max_workers=4, use_proxy=False, cooldown_on_429=1.5,
                          max_retries=8,
                          rate_limits={"minute": 550, "hour": 4800, "day": 9500},
                          log_level=30, log_file=os.path.join(odir, "_pull.log"))

    QS = list(quarters())   # generator tek seferlik — materialize sart
    jobs = []
    for nm, (la, lo) in provs.items():
        for qtag, s, e in QS:
            for v in vars_:
                hv = f"{v}_previous_day1,{v}_previous_day2"
                jobs.append((nm, qtag, v, url(la, lo, hv, s.date().isoformat(),
                                              e.date().isoformat())))

    print(f"[{kind}] {len(provs)} il | {len(vars_)} degisken | istek: {len(jobs)}", flush=True)

    # 40'lik partiler halinde kos, her partide birikimi diske yaz (idempotent)
    B = 40
    buckets = {}
    for i in range(0, len(jobs), B):
        chunk = jobs[i:i + B]
        res = cli.fetch_bulk([j[3] for j in chunk])
        for (nm, qtag, v, u), r in zip(chunk, res):
            if not r.get("ok") or not r.get("data"):
                continue
            h = r["data"].get("hourly", {})
            if "time" not in h:
                continue
            d = pd.DataFrame({k: x for k, x in h.items() if k != "time"},
                             index=pd.to_datetime(h["time"]))
            buckets.setdefault((nm, qtag), []).append(d)
        # her partiden sonra yaz
        for (nm, qtag), frames in list(buckets.items()):
            m = pd.concat(frames, axis=1)
            m = m.loc[:, ~m.columns.duplicated()]
            m.to_parquet(os.path.join(odir, f"{nm}_{qtag}.parquet"))
        buckets = {}
        print(f"  {i + len(chunk)}/{len(jobs)} istek | {len(os.listdir(odir))} dosya",
              flush=True)
        time.sleep(1)

    print(cli.report(), flush=True)
    print(f"[{kind}] BITTI | dosya: {len([f for f in os.listdir(odir) if f.endswith('.parquet')])}",
          flush=True)


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "wind")
