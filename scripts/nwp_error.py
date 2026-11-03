"""NWP HATA OLCUMU — gercek tahmin hatasi, lead'e ve aya gore.

Amac: sabah hattinin hava girdisi ne kadar bozuk? Backtest'te "analiz kalitesinde"
hava kullaniyoruz (w0 1.71), canlida lead'li hava geliyor (w24 1.94). Aradaki fark
gercek NWP hatasindan mi geliyor, yoksa muhafazakar bir varsayim mi?

Yontem:
  - previous-runs API: ayni gun icin day-1 ve day-2 onceden uretilmis tahmin
  - historical-forecast API: analiz (referans)
  - 10 sehir, nufus agirlikli ulusal seri
  - hata = tahmin - analiz; ay ve saat bazinda bias / MAE / RMSE

Kullanim: python nwp_error.py [baslangic] [bitis]
"""
import sys, time, json
import numpy as np, pandas as pd, requests

sys.path.insert(0, r"C:\projeler\turkiye-t2-daysahead")
from src import config as C

START = sys.argv[1] if len(sys.argv) > 1 else "2025-01-01"
END = sys.argv[2] if len(sys.argv) > 2 else "2026-11-03"
OUT = r"C:\projeler\autoloop"
VARS = ["temperature_2m", "relative_humidity_2m", "shortwave_radiation",
        "wind_speed_100m", "cloud_cover", "precipitation"]
PREV = "https://previous-runs-api.open-meteo.com/v1/forecast"
HIST = "https://historical-forecast-api.open-meteo.com/v1/forecast"


def chunks(s, e, days=90):
    a = pd.Timestamp(s)
    while a < pd.Timestamp(e):
        b = min(a + pd.Timedelta(days=days - 1), pd.Timestamp(e))
        yield a.date().isoformat(), b.date().isoformat()
        a = b + pd.Timedelta(days=1)


def fetch(url, lat, lon, vars_, s, e, model="ecmwf_ifs025"):
    for attempt in range(3):
        try:
            r = requests.get(url, params=dict(
                latitude=lat, longitude=lon, hourly=",".join(vars_),
                start_date=s, end_date=e, models=model, timezone="Europe/Istanbul"),
                timeout=60)
            if r.status_code != 200:
                raise RuntimeError(f"HTTP {r.status_code}: {r.text[:120]}")
            d = r.json()
            if d.get("error"):
                raise RuntimeError(d.get("reason", "bilinmeyen"))
            h = d["hourly"]
            df = pd.DataFrame(h)
            df["dt"] = pd.to_datetime(df["time"])
            return df.drop(columns=["time"]).set_index("dt")
        except Exception as ex:
            if attempt == 2:
                raise
            time.sleep(2 * (attempt + 1))
    return None


def pull(kind):
    """kind: 'prev' (lead'li) veya 'hist' (analiz). Nufus agirlikli ulusal seri."""
    frames = []
    tot = sum(c["pop"] for c in C.CITIES)
    for c in C.CITIES:
        got = []
        for s, e in chunks(START, END):
            if kind == "prev":
                vv = [f"{v}_previous_day1" for v in VARS] + [f"{v}_previous_day2" for v in VARS]
            else:
                vv = VARS
            try:
                d = fetch(PREV if kind == "prev" else HIST, c["lat"], c["lon"], vv, s, e)
            except Exception as ex:
                print(f"  {c['city']} {s} HATA: {str(ex)[:90]}", flush=True)
                # degisken seti desteklenmiyorsa sadece sicaklikla dene
                try:
                    if kind == "prev":
                        vv2 = ["temperature_2m_previous_day1", "temperature_2m_previous_day2"]
                    else:
                        vv2 = ["temperature_2m"]
                    d = fetch(PREV if kind == "prev" else HIST, c["lat"], c["lon"], vv2, s, e)
                    print(f"  {c['city']} {s}: sicaklik-only ile alindi", flush=True)
                except Exception as ex2:
                    print(f"  {c['city']} {s} ATLANDI: {str(ex2)[:90]}", flush=True)
                    continue
            got.append(d)
            time.sleep(0.4)
        if not got:
            continue
        d = pd.concat(got)
        d = d[~d.index.duplicated(keep="last")].sort_index()
        d.columns = [f"{c['city']}__{x}" for x in d.columns]
        frames.append(d * 1.0)
        print(f"  {kind} {c['city']}: {len(d)} saat", flush=True)
    if not frames:
        raise RuntimeError("hicbir sehir gelmedi")
    wide = pd.concat(frames, axis=1)
    out = pd.DataFrame(index=wide.index)
    suffixes = [""] if kind == "hist" else ["_previous_day1", "_previous_day2"]
    for v in VARS:
        for sfx in suffixes:
            src = f"{v}{sfx}"
            cols = [f"{c['city']}__{src}" for c in C.CITIES if f"{c['city']}__{src}" in wide.columns]
            if not cols:
                continue
            w = np.array([next(c["pop"] for c in C.CITIES if c["city"] == col.split("__")[0])
                          for col in cols], dtype=float)
            out[src] = (wide[cols] * (w / w.sum())).sum(axis=1, min_count=max(1, len(cols) // 2))
    return out, wide


print(f"[nwp] {START} -> {END} | {len(C.CITIES)} sehir", flush=True)
an, an_w = pull("hist")
print(f"[nwp] analiz: {an.shape}", flush=True)
fc, fc_w = pull("prev")
print(f"[nwp] tahmin: {fc.shape}", flush=True)

report = {}
for lead in (1, 2):
    print(f"\n=== LEAD {lead} GUN ===", flush=True)
    print("%-20s %8s %8s %8s %8s" % ("degisken", "bias", "MAE", "RMSE", "n"), flush=True)
    for v in VARS:
        a = an.get(v)
        f = fc.get(f"{v}_previous_day{lead}")
        if a is None or f is None:
            continue
        j = pd.concat({"a": a, "f": f}, axis=1).dropna()
        if len(j) < 100:
            continue
        e = j["f"] - j["a"]
        row = dict(bias=float(e.mean()), mae=float(e.abs().mean()),
                   rmse=float(np.sqrt((e ** 2).mean())), n=int(len(j)))
        report[f"L{lead}_{v}"] = row
        print("%-20s %+8.3f %8.3f %8.3f %8d" % (v, row["bias"], row["mae"], row["rmse"], row["n"]), flush=True)

# ay bazinda sicaklik bias'i (mevsimsel duzeltme icin)
print("\n=== SICAKLIK BIAS — AY BAZINDA (lead1 / lead2) ===", flush=True)
a = an["temperature_2m"]
for lead in (1, 2):
    f = fc[f"temperature_2m_previous_day{lead}"]
    j = pd.concat({"a": a, "f": f}, axis=1).dropna()
    g = (j["f"] - j["a"]).groupby(j.index.month).agg(["mean", "std", "count"])
    print(f"-- lead{lead}: " + " | ".join(f"{m}: {r['mean']:+.2f}±{r['std']:.2f}({int(r['count'])})"
                                          for m, r in g.iterrows()), flush=True)

# saat bazinda
print("\n=== SICAKLIK BIAS — SAAT BAZINDA (lead1) ===", flush=True)
j = pd.concat({"a": a, "f": fc["temperature_2m_previous_day1"]}, axis=1).dropna()
g = (j["f"] - j["a"]).groupby(j.index.hour).mean()
print(" ".join(f"{h}:{v:+.2f}" for h, v in g.items()), flush=True)

json.dump(report, open(rf"{OUT}\nwp_error.json", "w"), indent=1)
print("\n[nwp] bitti", flush=True)
