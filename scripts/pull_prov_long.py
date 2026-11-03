"""39 il ruzgar hizi — uzun pencere (walk-forward icin).

Cikti: C:\\projeler\\prov_speed_long.parquet
Onbellek: C:\\projeler\\prov_cache_long\\<il>.parquet  (tarih araligi ayri klasorde)
"""
import os, sys, time
os.chdir(r"C:\projeler\turkiye-t2-daysahead")
sys.path.insert(0, r"C:\projeler\turkiye-t2-daysahead")
import pandas as pd, numpy as np
from src import data as D, config as C

PROV = {
    "Balikesir": (39.65, 27.89), "Canakkale": (40.16, 26.41), "Izmir": (38.42, 27.14),
    "Manisa": (38.62, 27.43), "Kirklareli": (41.74, 27.23), "Tekirdag": (40.98, 27.51),
    "Bursa": (40.19, 29.06), "Yalova": (40.65, 29.27), "Sakarya": (40.78, 30.40),
    "Edirne": (41.68, 26.56), "Bilecik": (40.15, 29.98),
    "Mugla": (37.22, 28.36), "Aydin": (37.85, 27.84), "Afyon": (38.76, 30.54),
    "Kutahya": (39.42, 29.98), "Usak": (38.68, 29.41),
    "Hatay": (36.20, 36.16), "Osmaniye": (37.07, 36.25), "Mersin": (36.80, 34.63),
    "Kahramanmaras": (37.58, 36.93), "Adana": (37.00, 35.32),
    "Sivas": (39.75, 37.02), "Kayseri": (38.73, 35.49), "Konya": (37.87, 32.48),
    "Karaman": (37.18, 33.22), "Nigde": (37.97, 34.68), "Kirikkale": (39.85, 33.52),
    "Kirsehir": (39.15, 34.16),
    "Tokat": (40.31, 36.55), "Amasya": (40.65, 35.83), "Sinop": (42.03, 35.15),
    "Corum": (40.55, 34.95), "Kastamonu": (41.38, 33.78), "Ordu": (40.98, 37.88),
    "Malatya": (38.35, 38.31), "Elazig": (38.68, 39.22), "Gaziantep": (37.07, 37.38),
    "Adiyaman": (37.76, 38.28), "Erzurum": (39.90, 41.27),
}
START, END = "2026-04-15", "2026-11-03"
OUT = r"C:\projeler\prov_speed_long.parquet"
CACHE = r"C:\projeler\prov_cache_long"
os.makedirs(CACHE, exist_ok=True)

cols, fails = {}, []
t0 = time.time()
for i, (name, (lat, lon)) in enumerate(PROV.items()):
    cf = os.path.join(CACHE, f"{name}.parquet")
    if os.path.exists(cf):
        cols[name] = pd.read_parquet(cf).iloc[:, 0]
        continue
    got = None
    for attempt in range(3):
        try:
            h = D.om_fetch(C.OM_HISTO_URL, lat, lon, START, END,
                           vars=["wind_speed_100m", "wind_speed_10m"])
            arr = np.asarray(h["wind_speed_100m"], dtype=float) / 3.6
            ser = pd.Series(arr, index=pd.to_datetime(h["time"]), name=name)
            ser = ser[~ser.index.duplicated(keep="last")].sort_index()
            got = ser
            break
        except Exception as ex:
            print(f"  {name} d{attempt+1}: {type(ex).__name__} {str(ex)[:80]}", flush=True)
            time.sleep(2 + 2 * attempt)
    if got is None:
        fails.append(name); continue
    got.to_frame().to_parquet(cf)
    cols[name] = got
    time.sleep(0.35)
    if (i + 1) % 10 == 0:
        print(f"  {i+1}/{len(PROV)} | {len(cols)} ok ({time.time()-t0:.0f}s)", flush=True)

print(f"\nbasarili {len(cols)} | basarisiz {len(fails)} {fails}")
if not cols:
    raise SystemExit(1)
df = pd.DataFrame(cols).sort_index()
df = df[~df.index.duplicated(keep="last")]
df.to_parquet(OUT)
print(f"yazildi {OUT} {df.shape} | {df.index.min()} -> {df.index.max()}")
