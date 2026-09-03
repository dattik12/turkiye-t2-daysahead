"""IFS tahmin snapshot toplayici: lead-time hizali egitim verisi biriktirir.

Problem: egitim satirlari hedef-saat gerceklesen havasini gorur, sunumda IFS
tahmini gelir (train/serve skew). Cozum: her gun D+1..D+4 IFS tahminini
(run tarihi etiketli) arsivle; ilerde egitim bu anlik-goruntulerden beslenir.

Kullanim: python -m scripts.collect_fcast_snapshot
Cikti: data/weather/fcst_snapshots.parquet (run_date x target_dt x var)."""
from __future__ import annotations
import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
from src import config as C
from src import data as D

SNAP_VARS = ["temperature_2m", "relative_humidity_2m", "shortwave_radiation",
             "cloud_cover", "wind_speed_10m"]
SNAP_PATH = os.path.join(C.WEATHER_DIR, "fcst_snapshots.parquet")


def main() -> None:
    run = pd.Timestamp.now(tz="Europe/Istanbul").normalize()
    s = (run + pd.Timedelta(days=1)).date().isoformat()
    e = (run + pd.Timedelta(days=4)).date().isoformat()
    frames = []
    for c in C.CITIES:
        h = D.om_fetch(C.OM_FC_URL, c["lat"], c["lon"], s, e, vars=SNAP_VARS)
        f = pd.DataFrame({v: h[v] for v in SNAP_VARS}, index=pd.to_datetime(h["time"]))
        f["city"] = c["city"]
        frames.append(f)
    df = pd.concat(frames)
    df["run_date"] = run.date().isoformat()
    df = df.reset_index().rename(columns={"index": "target_dt"})
    if os.path.exists(SNAP_PATH):
        old = pd.read_parquet(SNAP_PATH)
        df = pd.concat([old, df], ignore_index=True)
        df = df.drop_duplicates(subset=["run_date", "city", "target_dt"], keep="last")
    df.to_parquet(SNAP_PATH, index=False)
    print(f"SNAPSHOT {run.date()}: {len(df)} satir -> {SNAP_PATH}")


if __name__ == "__main__":
    main()
