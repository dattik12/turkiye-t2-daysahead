"""Ulusal saatlik kaynak bazli uretim gerceklesmesi (EPIAS rt-gen-bulk).

Her gun icin tum santralleri cekip saatlik kaynak toplamlari uretir.
Kesintisiz: gunluk parquet yazar, tekrar kosunca var olanlari atlar.

Cikti: data/generation/rt_gen_YYYY-MM-DD.parquet
       kolonlar: sun, wind, total, naturalGas, ... (MW, saatlik toplam)

KAPSAM UYARISI: pp-list yalnizca EPIAS'a kayitli (cogunlukla lisansli)
santralleri icerir. Gunes icin bu ULUSAL TOPLAMIN ALT KUMESIDIR (~%2-3);
seviye degil SEKIL olarak kullanilmalidir. Ruzgar kapsami makuldur.
"""
from __future__ import annotations
import os, sys, time, datetime as dt

os.chdir(r"C:\projeler\turkiye-t2-daysahead")
sys.path.insert(0, r"C:\projeler\turkiye-t2-daysahead")

import pandas as pd
from eptr2 import EPTR2

OUTDIR = r"C:\projeler\turkiye-t2-daysahead\data\generation"
SRC = ["naturalGas", "dammedHydro", "lignite", "river", "importCoal", "wind", "sun",
       "fueloil", "geothermal", "asphaltiteCoal", "blackCoal", "biomass", "naphta",
       "lng", "importExport", "wasteheat"]
BATCH = 450

os.makedirs(OUTDIR, exist_ok=True)
START = sys.argv[1] if len(sys.argv) > 1 else "2026-08-04"
END = sys.argv[2] if len(sys.argv) > 2 else "2026-11-01"

e = EPTR2()
ids = [int(x) for x in e.call("pp-list")["id"].tolist()]
print(f"[gen] {len(ids)} santral | {START} -> {END} | batch {BATCH}", flush=True)

days = pd.date_range(START, END, freq="D")
done = 0
for day in days:
    ds = day.strftime("%Y-%m-%d")
    path = os.path.join(OUTDIR, f"rt_gen_{ds}.parquet")
    if os.path.exists(path):
        continue
    parts = []
    ok = True
    for i in range(0, len(ids), BATCH):
        chunk = ids[i:i + BATCH]
        got = False
        for attempt in range(3):
            try:
                parts.append(e.call("rt-gen-bulk", date=ds, pp_ids=chunk))
                got = True
                break
            except Exception as ex:
                msg = str(ex)[:110]
                if attempt == 2:
                    print(f"  {ds} batch {i//BATCH+1} BASARISIZ: {msg}", flush=True)
                    ok = False
                else:
                    time.sleep(2 + attempt * 3)
        if not got:
            break
        time.sleep(0.45)
    if not ok or not parts:
        continue
    df = pd.concat(parts, ignore_index=True)
    for c in SRC + ["total"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    df["dt"] = pd.to_datetime(df["date"], utc=True).dt.tz_convert("Europe/Istanbul")
    g = df.groupby("dt")[[c for c in SRC + ["total"] if c in df.columns]].sum()
    g.to_parquet(path)
    done += 1
    if done % 10 == 0:
        print(f"  [{done}] {ds} | gunes tepe {g['sun'].max():.0f} MW | ruzgar {g['wind'].mean():.0f} MW", flush=True)

print(f"[gen] bitti: {done} gun eklendi", flush=True)
