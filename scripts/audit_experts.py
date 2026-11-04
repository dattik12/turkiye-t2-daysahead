"""SISTEM DENETIMI — uzman bazli reziduel matrisi ve cesitlilik.

expert_preds.parquet: lgbm / cat / mlp tahminleri + gercek (720 gun, 18720 saat).
"""
import numpy as np, pandas as pd, json, os

P = pd.read_parquet(r"C:\projeler\autoloop\expert_preds.parquet")
P["dt"] = pd.to_datetime(P["dt"])
P = P.sort_values("dt").reset_index(drop=True)
E = ["lgbm", "cat", "mlp"]
W = np.array([0.10, 0.20, 0.70])
P["ens"] = P[E].to_numpy() @ W
print(f"saat: {len(P)} | {P.dt.min().date()} -> {P.dt.max().date()}\n")

def mape(p, a):
    return float((np.abs(p - a) / a).mean() * 100)

print("=== 1. UZMAN BAZLI SKOR ===")
print("%-8s %8s %9s %9s %9s" % ("uzman", "MAPE", "bias_MW", "MAE", "RMSE"))
for c in E + ["ens"]:
    e = P[c] - P.y
    print("%-8s %7.3f%% %+9.0f %9.0f %9.0f" % (c, mape(P[c], P.y), e.mean(), e.abs().mean(),
                                               np.sqrt((e ** 2).mean())))

print("\n=== 2. HATA KORELASYONU (cesitlilik) ===")
R = pd.DataFrame({c: P[c] - P.y for c in E})
print(R.corr().round(3).to_string())
print("(dusuk korelasyon = uzmanlar farkli hata yapiyor = topluluk kazanir)")

print("\n=== 3. REZIDUEL — SAAT BAZINDA (bias MW) ===")
h = P.groupby(P.dt.dt.hour).apply(
    lambda d: pd.Series({c: (d[c] - d.y).mean() for c in E + ["ens"]}), include_groups=False)
print(h.round(0).to_string())

print("\n=== 4. REZIDUEL — AY BAZINDA (MAPE %) ===")
m = P.groupby(P.dt.dt.month).apply(
    lambda d: pd.Series({c: mape(d[c], d.y) for c in E + ["ens"]}), include_groups=False)
print(m.round(3).to_string())

print("\n=== 5. SICAKLIK BANDINA GORE MAPE % (yuk seviyesi proxy) ===")
band = pd.cut(P.y, [0, 25000, 32000, 38000, 44000, 60000])
b = P.groupby(band, observed=True).apply(
    lambda d: pd.Series({c: mape(d[c], d.y) for c in E + ["ens"]}), include_groups=False)
print(b.round(3).to_string())

print("\n=== 6. OPTIMAL AGIRLIK (in-sample vs out-of-sample) ===")
GRID = np.array([[w1, w2, 1 - w1 - w2]
                 for w1 in np.arange(0, 1.01, 0.05)
                 for w2 in np.arange(0, 1.01 - w1 + 1e-9, 0.05)])
A = P[E].to_numpy(); y = P.y.to_numpy()
mid = len(P) // 2
def bestw(sl):
    a, yy = A[sl], y[sl]
    errs = [np.abs(a @ w - yy).sum() for w in GRID]
    return GRID[int(np.argmin(errs))]
w_in = bestw(slice(0, mid))
w_out = bestw(slice(mid, None))
print(f"  ilk yari optimal : lgbm {w_in[0]:.2f} cat {w_in[1]:.2f} mlp {w_in[2]:.2f}")
print(f"  ikinci yari opt. : lgbm {w_out[0]:.2f} cat {w_out[1]:.2f} mlp {w_out[2]:.2f}")
print(f"  URETIM           : lgbm 0.10 cat 0.20 mlp 0.70")
print(f"  ilk yari optimal -> ikinci yari skoru: {mape(A[mid:] @ w_in, y[mid:]):.3f}%")
print(f"  uretim agirligi  -> ikinci yari skoru: {mape(A[mid:] @ W, y[mid:]):.3f}%")

print("\n=== 7. KAPSAMA (gercek dagilimi) ===")
q = P.y.describe(percentiles=[.01, .05, .25, .5, .75, .95, .99])
print(q.round(0).to_string())
