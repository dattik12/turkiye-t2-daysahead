"""BIAS DUZELTMESI — sistematik eksik tahmin.

Uzman denetimi: 24 saatin HEPSINDE negatif bias, aksak zirvesinde -500 MW.
Ortalama bias -297 MW; MAPE %1.557 (ort yuk 40.242) -> bias, hatanin ~yarisi.

Test: gecmis pencerede (ay x saat) ortalama hatayi kestir, sonraki blokta duzelt.
Sizinti yok: duzeltme SADECE onceki gerceklesmis tahminlerden.

Kollar:
  ham                 mevcut topluluk
  saat_bias           + saat bazli bias (24 parametre)
  ay_saat_bias        + ay x saat (288 parametre)
  gun_saat_bias       + hafta gunu x saat (168)
  kayan_7g            + son 7 gunun saat bazli hatasi
"""
import numpy as np, pandas as pd

P = pd.read_parquet(r"C:\projeler\autoloop\expert_preds.parquet")
P["dt"] = pd.to_datetime(P["dt"])
P = P.sort_values("dt").reset_index(drop=True)
E = ["lgbm", "cat", "mlp"]
P["p"] = P[E].to_numpy() @ np.array([0.10, 0.20, 0.70])
P["err"] = P["p"] - P["y"]
P["h"] = P.dt.dt.hour
P["mo"] = P.dt.dt.month
P["dw"] = P.dt.dt.dayofweek
P["day"] = P.dt.dt.normalize()

def mape(p, a):
    return float((np.abs(p - a) / a).mean() * 100)

print(f"saat {len(P)} | ort yuk {P.y.mean():,.0f} | ham MAPE {mape(P.p, P.y):.3f}%")
print(f"ham bias {P.err.mean():+.0f} MW\n")

BLOCK = 30
blocks = pd.date_range(P.day.min(), P.day.max(), freq=f"{BLOCK}D")
res = {k: [] for k in ("ham", "saat_bias", "ay_saat_bias", "gun_saat_bias", "kayan_7g")}

for b in blocks[1:]:
    past = P[P.dt < b]
    cur = P[(P.dt >= b) & (P.dt < b + pd.Timedelta(days=BLOCK))]
    if len(past) < 2000 or len(cur) < 100:
        continue

    res["ham"].append(cur[["p", "y"]].rename(columns={"p": "pp"}))
    for nm, key in (("saat_bias", ["h"]), ("ay_saat_bias", ["mo", "h"]),
                    ("gun_saat_bias", ["dw", "h"])):
        tbl = past.groupby(key)["err"].mean()
        corr = cur.set_index(key).index.map(tbl).to_numpy(dtype=float)
        corr = np.nan_to_num(corr, nan=0.0)
        res[nm].append(pd.DataFrame({"pp": cur["p"].to_numpy() - corr, "y": cur["y"].to_numpy()}))
    # son 7 gun: her saat icin ortalama hata
    rec = past[past.dt >= b - pd.Timedelta(days=7)]
    tbl = rec.groupby("h")["err"].mean()
    corr = np.nan_to_num(cur["h"].map(tbl).to_numpy(dtype=float), nan=0.0)
    res["kayan_7g"].append(pd.DataFrame({"pp": cur["p"].to_numpy() - corr, "y": cur["y"].to_numpy()}))

print("=== SONUC (yuruyen duzeltme, sizintisiz) ===")
print("%-14s %8s %9s %9s" % ("kol", "MAPE", "fark", "bias"))
base = None
out = []
for k, v in res.items():
    if not v:
        continue
    d = pd.concat(v)
    m = mape(d.pp, d.y)
    bi = float((d.pp - d.y).mean())
    out.append((k, m, bi))
    if base is None:
        base = m
for k, m, bi in sorted(out, key=lambda x: x[1]):
    print("%-14s %7.3f%% %+8.3f %+9.0f" % (k, m, m - base, bi))
