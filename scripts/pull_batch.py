"""KOTA KORUMALI CEKIM: 39 il x 100m ruzgar (2023-2026) + 10 il radyasyon (2021-2026).

Kurallar:
  - Tek URL'de BATCH koordinat (paralel baglanti cezasi yok)
  - Yillik parca, her parca aninda diske
  - 2023-2026 once (yuksek kaldiraç), kota kalirsa 2021-2022
  - Her adimda kota durumu basilir
"""
import os, sys, json, time
import numpy as np, pandas as pd, requests

OUTW = r"C:\projeler\prov_lead100"
OUTS = r"C:\projeler\lead_rad"
URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"

PROV = {
 "Balikesir": (39.65,27.89), "Canakkale": (40.15,26.41), "Izmir": (38.42,27.14),
 "Manisa": (38.62,27.43), "Kirklareli": (41.74,27.22), "Tekirdag": (40.98,27.51),
 "Bursa": (40.19,29.06), "Yalova": (40.65,29.27), "Sakarya": (40.77,30.40),
 "Edirne": (41.68,26.56), "Bilecik": (40.15,29.98), "Mugla": (37.21,28.36),
 "Aydin": (37.85,27.84), "Afyon": (38.75,30.54), "Kutahya": (39.42,29.98),
 "Usak": (38.68,29.41), "Hatay": (36.40,36.35), "Osmaniye": (37.07,36.25),
 "Mersin": (36.80,34.63), "Kahramanmaras": (37.58,36.93), "Adana": (37.00,35.32),
 "Sivas": (39.75,37.02), "Kayseri": (38.73,35.49), "Konya": (37.87,32.48),
 "Karaman": (37.18,33.22), "Nigde": (37.97,34.68), "Kirikkale": (39.85,33.52),
 "Kirsehir": (39.15,34.16), "Tokat": (40.32,36.55), "Amasya": (40.65,35.83),
 "Sinop": (42.03,35.15), "Corum": (40.55,34.95), "Kastamonu": (41.38,33.78),
 "Ordu": (40.98,37.88), "Malatya": (38.35,38.31), "Elazig": (38.67,39.22),
 "Gaziantep": (37.06,37.38), "Adiyaman": (37.76,38.28), "Erzurum": (39.90,41.27)}

SOLAR = {"konya": (37.87,32.48), "ankara": (39.93,32.86), "sanliurfa": (37.16,38.79),
         "kayseri": (38.73,35.49), "izmir": (38.42,27.14), "antalya": (36.90,30.70),
         "mersin": (36.80,34.63), "kahramanmaras": (37.58,36.93),
         "istanbul": (41.01,28.98), "adana": (37.00,35.32)}

WV = "wind_speed_100m,wind_direction_100m,temperature_2m"


def quota():
    try:
        r = requests.get(URL, params=dict(latitude=39, longitude=35, hourly="temperature_2m",
                                          start_date="2026-11-04", end_date="2026-11-04"),
                         timeout=20)
        return r.status_code == 200
    except Exception:
        return False


def fetch_batch(names, coords, hourly, s, e, tries=4):
    for k in range(tries):
        try:
            r = requests.get(URL, params=dict(
                latitude=",".join(f"{c[0]}" for c in coords),
                longitude=",".join(f"{c[1]}" for c in coords),
                hourly=hourly, start_date=s, end_date=e,
                timezone="Europe/Istanbul"), timeout=300)
            if r.status_code != 200:
                raise RuntimeError(f"HTTP {r.status_code}: {r.text[:100]}")
            j = r.json()
            return j if isinstance(j, list) else [j]
        except Exception as ex:
            if k == tries - 1:
                raise
            print(f"    tekrar {k+1}: {str(ex)[:70]}", flush=True)
            time.sleep(8 * (k + 1))


def pull_wind(years):
    os.makedirs(OUTW, exist_ok=True)
    names = list(PROV)
    coords = [PROV[n] for n in names]
    for y in years:
        done = [n for n in names if os.path.exists(os.path.join(OUTW, f"{n}_{y}.parquet"))]
        if len(done) == len(names):
            print(f"  {y}: zaten tam ({len(names)} il)", flush=True)
            continue
        if not quota():
            print(f"  {y}: KOTA KAPALI — dur", flush=True); return False
        s, e = f"{y}-01-01", f"{y}-12-31"
        print(f"  {y} cekiliyor ({len(names)} il batch)...", flush=True)
        blocks = fetch_batch(names, coords, WV, s, e)
        for n, b in zip(names, blocks):
            d = pd.DataFrame(b["hourly"])
            d["time"] = pd.to_datetime(d["time"])
            d = d.set_index("time")
            d.to_parquet(os.path.join(OUTW, f"{n}_{y}.parquet"))
        print(f"  {y}: OK ({len(blocks)} il)", flush=True)
        time.sleep(2)
    return True


def pull_solar(years):
    os.makedirs(OUTS, exist_ok=True)
    names = list(SOLAR)
    coords = [SOLAR[n] for n in names]
    hv = "shortwave_radiation,direct_normal_irradiance,diffuse_radiation,cloud_cover,temperature_2m"
    for y in years:
        done = [n for n in names if os.path.exists(os.path.join(OUTS, f"{n}_{y}.parquet"))]
        if len(done) == len(names):
            print(f"  {y}: zaten tam", flush=True); continue
        if not quota():
            print(f"  {y}: KOTA KAPALI — dur", flush=True); return False
        s, e = f"{y}-01-01", f"{y}-12-31"
        print(f"  {y} radyasyon ({len(names)} il batch)...", flush=True)
        blocks = fetch_batch(names, coords, hv, s, e)
        for n, b in zip(names, blocks):
            d = pd.DataFrame(b["hourly"])
            d["time"] = pd.to_datetime(d["time"])
            d.set_index("time").to_parquet(os.path.join(OUTS, f"{n}_{y}.parquet"))
        print(f"  {y}: OK", flush=True)
        time.sleep(2)
    return True


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    print(f"kota: {'ACIK' if quota() else 'KAPALI'}", flush=True)
    if what in ("wind", "all"):
        print("\n=== RUZGAR 100m (2023-2026 once) ===", flush=True)
        ok = pull_wind([2023, 2024, 2025, 2026])
        if ok and quota():
            print("\n=== RUZGAR 100m (2021-2022) ===", flush=True)
            pull_wind([2021, 2022])
    if what in ("solar", "all"):
        print("\n=== RADYASYON (2021-2026) ===", flush=True)
        pull_solar([2021, 2022, 2023, 2024, 2025, 2026])
    print("\nBITTI", flush=True)
