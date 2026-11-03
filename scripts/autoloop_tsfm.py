"""TSFM DENEYI — Chronos-2 ile topluluk kiyasi.

Chronos-2 disi degisken alabiliyor, yani adil kiyas mumkun.
Yeniden egitim YOK: toplulugun uzman tahminleri expert_preds.parquet'te hazir;
Chronos ayni saatler icin tahmin uretir, dogrudan kiyaslanir.

Protokol: her hedef gun icin 09:00'da bilinen son saat = onceki gun 08:00.
  context: son 672 saat (28 gun) yuk + takvim kovaryatlari
  hedef: 24 saat

Kullanim: python autoloop_tsfm.py [gun_sayisi]
"""
import os, sys, json, time
os.chdir(r"C:\projeler\turkiye-t2-daysahead")
sys.path.insert(0, r"C:\projeler\turkiye-t2-daysahead")
import numpy as np, pandas as pd
from src import data as D

N = int(sys.argv[1]) if len(sys.argv) > 1 else 60
OUT = r"C:\projeler\autoloop"
CTX = 672          # 28 gun baglam
W = np.array([0.10, 0.20, 0.70])
EXPERTS = ["lgbm", "cat", "mlp"]

y = D.load_or_create_consumption()["rt_cons"].dropna()
y.index = pd.to_datetime(y.index)
y = y[~y.index.duplicated(keep="last")].sort_index()

P = pd.read_parquet(os.path.join(OUT, "expert_preds.parquet"))
P["dt"] = pd.to_datetime(P["dt"])
P = P.sort_values("dt")
days = pd.DatetimeIndex(sorted(set(P["dt"].dt.normalize())))[-N:]
print(f"gun: {len(days)} | {days.min().date()} -> {days.max().date()}", flush=True)

from chronos import Chronos2Pipeline
t0 = time.time()
pipe = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map="cuda")
print(f"[tsfm] model yuklendi ({time.time()-t0:.0f}s)", flush=True)


def ctx_for(day):
    """09:00 karar ani: onceki gun 08:00'e kadar bilinen seri."""
    end = day - pd.Timedelta(days=1) + pd.Timedelta(hours=8)
    s = y[(y.index > end - pd.Timedelta(hours=CTX)) & (y.index <= end)]
    return s


rows = []
t0 = time.time()
for i, day in enumerate(days):
    tgt = pd.date_range(f"{day} 00:00", f"{day} 23:00", freq="h")
    ctx = ctx_for(day)
    if len(ctx) < CTX * 0.9:
        continue
    sub = P[P["dt"].isin(tgt)]
    if len(sub) < 20:
        continue
    hist = ctx.to_numpy(dtype=np.float32)
    # HIZALAMA: baglam D-1 08:00'de bitiyor, hedef D 00:00..23:00.
    # Aradaki 16 saati de tahmin etmek gerekir -> 40 adim iste, son 24'unu al.
    LEAD = 16
    try:
        out = pipe.predict([{"target": hist}], prediction_length=LEAD + 24)
        a = np.asarray(out[0], dtype=float)
        # Chronos-2 ciktisi (batch, n_kuantil, ufuk) — MEDYAN kuantili al.
        # (Onceki hata: eksenleri (ufuk, kuantil) sanip duzlestiriyordum -> %74 MAPE.)
        while a.ndim > 2:
            a = a[0]
        if a.ndim == 1:
            p_all = a
        else:
            p_all = a[a.shape[0] // 2]
        p_ch = np.asarray(p_all, dtype=float).reshape(-1)[LEAD:LEAD + 24]
    except Exception as ex:
        print(f"  {day.date()} HATA {str(ex)[:110]}", flush=True)
        continue
    ens = sub[EXPERTS].to_numpy() @ W
    rows.append(pd.DataFrame({"dt": sub["dt"].to_numpy(), "y": sub["y"].to_numpy(),
                              "ens": ens, "tsfm": p_ch[:len(sub)]}))
    if (i + 1) % 10 == 0:
        print(f"  {i+1}/{len(days)} ({(time.time()-t0)/60:.1f} dk)", flush=True)

if not rows:
    print("[tsfm] hic tahmin uretilmedi", flush=True)
    sys.exit(1)

R = pd.concat(rows)
R = R.dropna()
print(f"\nkiyas saati: {len(R)}", flush=True)


def mape(p):
    return float((np.abs(p - R.y) / R.y).mean() * 100)


base = mape(R.ens.to_numpy())
ts = mape(R.tsfm.to_numpy())
best_w, best_s = 0.0, 1e9
for w in np.arange(0, 1.001, 0.05):
    s = float((np.abs(w * R.tsfm + (1 - w) * R.ens - R.y) / R.y).mean() * 100)
    if s < best_s:
        best_w, best_s = w, s

print("\n=== TSFM SONUCU ===", flush=True)
print("  topluluk (mevcut)   %7.3f%%" % base, flush=True)
print("  Chronos-2 tek       %7.3f%%   %+7.3f" % (ts, ts - base), flush=True)
print("  en iyi karisim w=%.2f %7.3f%%   %+7.3f" % (best_w, best_s, best_s - base), flush=True)
print("  (w=0 -> sadece topluluk, w=1 -> sadece Chronos)", flush=True)

json.dump(dict(days=int(len(days)), n=int(len(R)), ens=base, tsfm=ts,
               blend_w=float(best_w), blend=best_s),
          open(os.path.join(OUT, "summary_tsfm.json"), "w"), indent=1)
R.to_parquet(os.path.join(OUT, "tsfm_preds.parquet"))
print("[tsfm] bitti", flush=True)
