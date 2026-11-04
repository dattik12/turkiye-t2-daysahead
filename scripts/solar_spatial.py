"""GES FILO COGRAFYASI — agirliklari VERIDEN fit et.

Sorun: model tek ulusal GHI kullaniyor, agirlik NUFUS (bati). GES filosu doguda.
Sonuc: egrinin sividigi yanlis — sabah/aksam az, oglen fazla (olculdu: oran
0.41 @06:00 vs 0.21 @12:00).

Cozum: lisansli GES gerceklesmesinden il agirliklarini NNLS ile fit et.
  gercek(t) ~ sum_i w_i * GHI_i(t)
Bulunan w = filonun ETKIN cografi dagilimi.

Sonra ayni RRMSE ile karsilastir:
  (a) nufus agirlikli tek seri     (mevcut)
  (b) fit edilmis filo agirliklari (aday)

Kullanim: python solar_spatial.py
"""
import sys, glob
sys.path.insert(0, r"C:\projeler\turkiye-t2-daysahead")
import os
os.chdir(r"C:\projeler\turkiye-t2-daysahead")
import numpy as np, pandas as pd
from scipy.optimize import nnls

RAD = r"C:\projeler\solar_rad_39.parquet"
rad = pd.read_parquet(RAD)
rad.index = pd.DatetimeIndex(rad.index)
if rad.index.tz is None:
    rad.index = rad.index.tz_localize("Europe/Istanbul")
rad = rad[~rad.index.duplicated(keep="last")].sort_index()
print(f"isinim: {rad.shape} | {rad.index.min()} -> {rad.index.max()}")
print(f"kolon ornegi: {list(rad.columns)[:6]}")

g = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob("data/generation/rt_gen_*.parquet"))])
g.index = pd.DatetimeIndex(g.index)
if g.index.tz is None:
    g.index = g.index.tz_localize("Europe/Istanbul")
g = g[~g.index.duplicated(keep="last")].sort_index()
act = g["sun"].dropna()

idx = rad.index.intersection(act.index)
print(f"eslesen saat: {len(idx)}")
R = rad.loc[idx].fillna(0)
a = act.loc[idx]

# gunduz maskesi
mask = R.mean(axis=1) > 20
R = R[mask]; a = a[mask]
print(f"gunduz saati: {len(R)}")

# --- NNLS: agirliklar >= 0 ---
W, res = nnls(R.to_numpy(), a.to_numpy())
pred_fit = R.to_numpy() @ W
print(f"\nNNLS: kullanilan il {int((W>1e-6).sum())}/{len(W)} | RRMSE {np.sqrt(res)/a.mean()*100:.2f}%")
top = pd.Series(W, index=R.columns).sort_values(ascending=False)
print("en yuksek agirlikli 10 il:")
for k, v in top.head(10).items():
    print(f"   {k}: {v:.1f}")

# --- karsilastir: (a) nufus agirlikli, (b) fit ---
from src import config as C
pop = {c["city"]: c["pop"] for c in C.CITIES}
# nufus agirlikli seri: rad kolonlari il adi ise eslesen sehirleri bul
cols_norm = {c.lower().replace(" ", "").replace("ı", "i").replace("ş", "s")
             .replace("ğ", "g").replace("ü", "u").replace("ö", "o").replace("ç", "c"): c
             for c in R.columns}
match = {}
for city, p in pop.items():
    key = city.lower()
    for cn, orig in cols_norm.items():
        if key in cn or cn.startswith(key[:5]):
            match[orig] = p
            break
if match:
    wpop = pd.Series(match); wpop = wpop / wpop.sum()
    pred_pop = R[list(wpop.index)].to_numpy() @ wpop.to_numpy()
else:
    wpop = pd.Series(1.0, index=R.columns) / len(R.columns)
    pred_pop = R.to_numpy() @ wpop.to_numpy()
    print("(nufus eslesmedi -> esit agirlik kullanildi)")

print(f"\n{'kol':22s} {'korel':>7s} {'oran_med':>9s} {'oran_CV':>8s}")
for nm, pr in (("nufus agirlikli", pred_pop), ("fit filo agirligi", pred_fit)):
    rr = pd.Series(a.to_numpy() / np.maximum(pr, 1e-6), index=R.index)
    c = float(np.corrcoef(pr, a.to_numpy())[0, 1])
    print(f"{nm:22s} {c:7.4f} {rr.median():9.4f} {rr.std()/rr.mean()*100:7.1f}%")

# --- saat bazinda artik (sekil duzeldi mi) ---
print("\nSAAT BAZINDA oran (1.0'a yakin = sekil dogru):")
rows = []
for nm, pr in (("nufus", pred_pop), ("fit", pred_fit)):
    rr = pd.Series(a.to_numpy() / np.maximum(pr, 1e-6), index=R.index)
    g2 = rr.groupby(rr.index.hour).mean()
    rows.append((nm, g2))
    print(f"  {nm}: " + " ".join(f"{h}:{v:.2f}" for h, v in g2.items() if 6 <= h <= 19))

if rows:
    p_norm = rows[0][1] / rows[0][1].mean()
    f_norm = rows[1][1] / rows[1][1].mean()
    print(f"\nsekil sapmasi (std of normalized hourly ratio):")
    print(f"  nufus {p_norm.std()*100:.1f}%  |  fit {f_norm.std()*100:.1f}%  "
          f"(dusuk daha iyi)")

pd.Series(W, index=R.columns).to_csv(r"C:\projeler\autoloop\solar_fleet_weights.csv")
print("\nagirliklar -> C:\\projeler\\autoloop\\solar_fleet_weights.csv")
