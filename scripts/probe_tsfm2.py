import sys, os
os.chdir(r"C:\projeler\turkiye-t2-daysahead")
sys.path.insert(0, r"C:\projeler\turkiye-t2-daysahead")
import numpy as np, pandas as pd
from src import data as D
from chronos import Chronos2Pipeline

y = D.load_or_create_consumption()["rt_cons"].dropna()
y.index = pd.to_datetime(y.index)
y = y[~y.index.duplicated(keep="last")].sort_index()

day = pd.Timestamp("2026-10-20")
tgt = pd.date_range(f"{day} 00:00", f"{day} 23:00", freq="h")
end = day - pd.Timedelta(days=1) + pd.Timedelta(hours=8)
ctx = y[(y.index > end - pd.Timedelta(hours=672)) & (y.index <= end)]
print("baglam:", len(ctx), ctx.index.min(), "->", ctx.index.max(), "ort", round(float(ctx.mean())))
print("gercek:", [round(float(v)) for v in y.reindex(tgt).values[:8]])

p = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map="cuda")
hist = ctx.to_numpy(np.float32)

for tag, inp in [
    ("tek degisken", {"target": hist}),
    ("kovaryatli", {"target": hist,
                    "past_covariates": {"hour": (ctx.index.hour / 23.0).astype(np.float32),
                                        "dow": (ctx.index.dayofweek / 6.0).astype(np.float32)},
                    "future_covariates": {"hour": (tgt.hour / 23.0).astype(np.float32),
                                          "dow": (tgt.dayofweek / 6.0).astype(np.float32)}}),
]:
    try:
        out = p.predict([inp], prediction_length=24)
        a = np.asarray(out[0], float)
        print(f"\n[{tag}] shape={a.shape}")
        if a.ndim == 2:
            print("  satir0:", [round(float(x), 1) for x in a[0][:6]])
            print("  ort:", [round(float(x), 1) for x in a.mean(axis=0)[:6]])
        arr = a.reshape(-1)[:24] if a.size >= 24 else None
        if arr is not None:
            print("  duz ilk6:", [round(float(x), 1) for x in arr[:6]])
    except Exception as ex:
        print(f"[{tag}] HATA: {str(ex)[:200]}")
