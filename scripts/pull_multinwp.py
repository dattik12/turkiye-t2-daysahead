"""MULTI-NWP CEKIMI — 3 model (IFS/GFS/ICON), kota korumali.

Ruzgar : 10 RES ili x wind_speed_100m      x 3 model x ceyreklik 2023-2026
Gunes  :  8 GES ili x GHI/DNI/cloud_cover  x 3 model x ceyreklik 2023-2026

Kota kurallari:
  - Tek URL'de batch koordinat (paralel ceza yok)
  - Ceyreklik parca, her parca aninda diske
  - Her adimda kota probe; 429 -> DUR (yarin 09:00 canli kosusu icin pay birak)
  - BIRIM: km/h (Open-Meteo varsayilani). Model tarafinda /3.6 ile m/s'ye cevrilir.

Cikti: C:\\projeler\\multinwp\\{wind,solar}\\<il>_<yil>Q<q>.parquet
"""
import os, sys, time
import numpy as np, pandas as pd, requests

OUT = r"C:\projeler\multinwp"
URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"
MODELS = "ecmwf_ifs025,gfs_seamless,icon_seamless"
YEARS = [2023, 2024, 2025, 2026]
QS = [(1, "-01-01", "-03-31"), (2, "-04-01", "-06-30"),
      (3, "-07-01", "-09-30"), (4, "-10-01", "-12-31")]

WIND_P = {"Balikesir": (39.65, 27.89), "Izmir": (38.42, 27.14), "Canakkale": (40.15, 26.41),
          "Manisa": (38.62, 27.43), "Hatay": (36.40, 36.35), "Aydin": (37.85, 27.84),
          "Istanbul": (41.01, 28.98), "Kirklareli": (41.74, 27.22),
          "Afyon": (38.75, 30.54), "Kayseri": (38.73, 35.49)}

SOLAR_P = {"Konya": (37.87, 32.48), "Sanliurfa": (37.16, 38.79), "Ankara": (39.93, 32.86),
           "Kayseri": (38.73, 35.49), "Izmir": (38.42, 27.14), "Antalya": (36.90, 30.70),
           "Mersin": (36.80, 34.63), "Kahramanmaras": (37.58, 36.93)}


def probe():
    for k in range(3):
        try:
            r = requests.get(URL, params=dict(latitude=39, longitude=35, hourly="temperature_2m",
                                              start_date="2026-11-04", end_date="2026-11-04"),
                             timeout=25)
            if r.status_code == 200:
                return True, "ACIK"
            if r.status_code == 429:
                return False, "KOTA DOLU (429)"
        except Exception:
            pass
        time.sleep(5 * (k + 1))
    return True, "PROBE BELIRSIZ (devam)"


def fetch(coords, hourly, s, e, tries=3):
    lat = ",".join(f"{c[0]}" for c in coords)
    lon = ",".join(f"{c[1]}" for c in coords)
    for k in range(tries):
        try:
            r = requests.get(URL, params=dict(latitude=lat, longitude=lon, hourly=hourly,
                                              models=MODELS, start_date=s, end_date=e,
                                              timezone="Europe/Istanbul"), timeout=300)
            if r.status_code == 429:
                raise RuntimeError("KOTA429")
            if r.status_code != 200:
                raise RuntimeError(f"HTTP {r.status_code}: {r.text[:80]}")
            j = r.json()
            return j if isinstance(j, list) else [j]
        except Exception as ex:
            if "KOTA429" in str(ex) or k == tries - 1:
                raise
            time.sleep(10 * (k + 1))


def run(kind, provs, hourly):
    outdir = os.path.join(OUT, kind)
    os.makedirs(outdir, exist_ok=True)
    names = list(provs)
    coords = [provs[n] for n in names]
    done = 0
    for y in YEARS:
        for q, ms, me in QS:
            tag = f"{y}Q{q}"
            ok = [n for n in names if os.path.exists(os.path.join(outdir, f"{n}_{tag}.parquet"))]
            if len(ok) == len(names):
                print(f"  {kind} {tag}: zaten tam", flush=True)
                done += 1
                continue
            live, msg = probe()
            if not live:
                print(f"  {kind} {tag}: {msg} -> DURULDU (kota yarin icin saklandi)", flush=True)
                return done
            today = pd.Timestamp.now().normalize()
            e = min(pd.Timestamp(f"{y}{me}"), today - pd.Timedelta(days=1))
            if pd.Timestamp(f"{y}{ms}") > e:
                print(f"  {kind} {tag}: gelecek — atlandi", flush=True)
                continue
            # DEGISKEN BASINA AYRI ISTEK: cok degiskenli istek timeout yiyordu
            varlist = hourly.split(",")
            acc = {}
            failed = False
            for v in varlist:
                print(f"  {kind} {tag} [{v}] ({len(names)} il x 3 model)...", flush=True)
                try:
                    blocks = fetch(coords, v, f"{y}{ms}", e.strftime("%Y-%m-%d"))
                except Exception as ex:
                    print(f"  {kind} {tag} [{v}]: HATA {str(ex)[:70]}", flush=True)
                    failed = True
                    break
                for n, b in zip(names, blocks):
                    d = pd.DataFrame(b["hourly"])
                    d["time"] = pd.to_datetime(d["time"])
                    d = d.set_index("time")
                    acc.setdefault(n, []).append(d)
                time.sleep(1)
            if failed:
                print(f"  {kind} {tag}: yarim kaldi -> DURULDU", flush=True)
                return done
            for n, lst in acc.items():
                pd.concat(lst, axis=1).to_parquet(os.path.join(outdir, f"{n}_{tag}.parquet"))
            print(f"  {kind} {tag}: OK", flush=True)
            done += 1
            time.sleep(2)
    return done


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    live, msg = probe()
    print(f"kota: {msg}", flush=True)
    if not live:
        sys.exit(0)
    if what in ("wind", "all"):
        print("\n=== RUZGAR multi-NWP (10 RES ili) ===", flush=True)
        run("wind", WIND_P, "wind_speed_100m")
    if what in ("solar", "all"):
        print("\n=== GUNES multi-NWP (8 GES ili) ===", flush=True)
        run("solar", SOLAR_P, "shortwave_radiation,direct_normal_irradiance,cloud_cover")
    print("\nBITTI", flush=True)
