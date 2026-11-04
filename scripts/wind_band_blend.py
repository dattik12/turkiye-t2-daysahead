"""W3b — RUZGAR BANDINA GORE OPTIMAL KARISIM.

Segment analizi gosterdi: dusuk ruzgarda model RITM'den cok kotu
 (0-1000 MW: biz 95.3%, RITM 84.9% | 1000-3000: biz 32.3%, RITM 25.2%)
Yuksek ruzgarda ise ondeyiz (>9000: biz 10.7%, RITM 11.1%).

Mevcut uretim ayari: blend_w=1.0 (tam model), blend_bins.ws={0:0.95,...}
Yani dusuk ruzgarda bile model %95 agirlikta.

Test: bant basina optimal w (w = modelin payi). Veri hazir, egitim YOK.
Kisit: agirliklar SADECE gecmis veriden secilir (ilk yari secim, ikinci yari test).

Kullanim: python wind_band_blend.py
"""
import sys
import numpy as np, pandas as pd

R = pd.read_parquet(r"C:\projeler\autoloop\wind_segments.parquet")
R = R.dropna(subset=["ritm_fc_mw"])
R["band"] = pd.cut(R["y"], [0, 1000, 3000, 6000, 9000, 15000])
print(f"saat: {len(R)} | {R.index.min()} -> {R.index.max()}\n")

GRID = np.arange(0, 1.0001, 0.05)


def wm(p, a):
    return float(np.abs(p - a).sum() / a.sum() * 100)


def best_w(sub):
    a = sub["y"].to_numpy(); p = sub["p"].to_numpy(); r = sub["ritm_fc_mw"].to_numpy()
    errs = [np.abs(w * p + (1 - w) * r - a).sum() for w in GRID]
    return float(GRID[int(np.argmin(errs))])


# --- yariya bol: agirliklar ilk yaridan secilir, ikinci yarida test ---
mid = R.index.min() + (R.index.max() - R.index.min()) / 2
train, test = R[R.index < mid], R[R.index >= mid]
print(f"secim {len(train)} saat | test {len(test)} saat\n")

print("BANT BASINA OPTIMAL w (w = modelin payi; 1.0 = tam model, 0.0 = tam RITM)")
print("%-16s %8s %8s %9s %9s" % ("bant", "opt_w", "n_sec", "bizim", "RITM"))
tbl = {}
for b in R["band"].cat.categories:
    st = train[train["band"] == b]
    if len(st) < 100:
        tbl[b] = 1.0
        print("%-16s %8.2f %8d %9s %9s  (yetersiz -> 1.0)" % (str(b), 1.0, len(st), "-", "-"))
        continue
    w = best_w(st)
    tbl[b] = w
    print("%-16s %8.2f %8d %8.2f%% %8.2f%%" % (
        str(b), w, len(st), wm(st["p"], st["y"]), wm(st["ritm_fc_mw"], st["y"])))

print("\n--- TEST (ikinci yari, secimde gorulmemis veri) ---")
a = test["y"].to_numpy()
fixed = test["p"].to_numpy()
banded = np.array([tbl[b] for b in test["band"]])
adapt = banded * test["p"].to_numpy() + (1 - banded) * test["ritm_fc_mw"].to_numpy()
ritm = test["ritm_fc_mw"].to_numpy()
half = 0.5 * test["p"].to_numpy() + 0.5 * ritm

rows = [("tam model (mevcut)", wm(fixed, a)),
        ("RITM tek", wm(ritm, a)),
        ("sabit 50/50", wm(half, a)),
        ("BANT BAZLI (aday)", wm(adapt, a))]
base = rows[0][1]
print("%-22s %8s %9s" % ("kol", "WMAPE", "fark"))
for nm, v in sorted(rows, key=lambda x: x[1]):
    print("%-22s %7.2f%% %+8.2f" % (nm, v, v - base))

print(f"\nmevcut uretim: blend_w=1.0, blend_bins.ws={{0:0.95, 1:1.0, 2:1.0}}")
print(f"aday: bant bazli w -> {({str(k): round(v,2) for k,v in tbl.items()})}")
