"""OTONOM DENEY — amac fonksiyonu / MAPE uyumu.

Hipotez: model L2 (kare hata) optimize ediyor ama skor MAPE. Yuksek yuklu saatler
L2'de baskin; MAPE ise her saati 1/y ile agirliklandirir. Amaci MAPE'ye hizalamak
dogrudan skoru iyilestirmeli.

Kollar:
  O00  L2                          (mevcut uretim)
  O10  L1 / MAE
  O20  Huber
  O30  L2 + ornek agirligi 1/y
  O40  L1 + ornek agirligi 1/y
  O50  L2 + agirlik 1/sqrt(y)      (yumusatilmis)

720 gun, aylik yeniden egitim, tek LightGBM (uretim parametreleri), sabah protokolu.

Kullanim: python autoloop_objective.py [eval_gun]
"""
import os, sys, json, time
os.chdir(r"C:\projeler\turkiye-t2-daysahead")
sys.path.insert(0, r"C:\projeler\turkiye-t2-daysahead")
import numpy as np, pandas as pd
import lightgbm as lgb
from src import data as D, morning as MO, features as F

EVAL = int(sys.argv[1]) if len(sys.argv) > 1 else 720
RETRAIN = 30
OUT = r"C:\projeler\autoloop"
os.makedirs(OUT, exist_ok=True)

cons = D.load_or_create_consumption()
nat, _ = D.load_or_create_weather(); nat.index = pd.to_datetime(nat.index)
nat = nat[~nat.index.duplicated(keep="last")].sort_index()
y = cons["rt_cons"].dropna(); y.index = pd.to_datetime(y.index)
fd = y.groupby(y.index.normalize()).size()
last_full = fd[fd == 24].index.max()
cal = MO.load_calendar()

WARM = 60
targets = pd.date_range(last_full - pd.Timedelta(days=EVAL + WARM - 1), last_full, freq="D")
EVAL_START = targets[WARM]
print(f"degerlendirme {EVAL_START.date()} -> {last_full.date()}", flush=True)

LGB = dict(n_estimators=900, learning_rate=0.05, num_leaves=63, colsample_bytree=0.8,
           subsample=0.8, min_child_samples=40, random_state=42, verbose=-1)


def sw_of(yv, kind):
    if kind is None:
        return None
    if kind == "inv":
        return (1.0 / yv).to_numpy()
    if kind == "inv_sqrt":
        return (1.0 / np.sqrt(yv)).to_numpy()
    return None


VARIANTS = [
    ("O00_L2", "Kare hata — mevcut uretim", dict(), None),
    ("O10_L1", "Mutlak hata (MAE)", dict(objective="regression_l1"), None),
    ("O20_huber", "Huber", dict(objective="huber", alpha=0.9), None),
    ("O30_L2_w", "L2 + agirlik 1/y (MAPE uyumlu)", dict(), "inv"),
    ("O40_L1_w", "L1 + agirlik 1/y", dict(objective="regression_l1"), "inv"),
    ("O50_L2_ws", "L2 + agirlik 1/kok(y)", dict(), "inv_sqrt"),
]

sinks = {v[0]: [] for v in VARIANTS}
t0 = time.time()
fit_until, fits = None, {}
for day in targets:
    tgt = pd.date_range(f"{day} 00:00", f"{day} 23:00", freq="h")
    if not fits or tgt[0] >= fit_until:
        dec = day - pd.Timedelta(days=1)
        tr_end = dec + pd.Timedelta(hours=8)
        tr = pd.date_range(tr_end - pd.Timedelta(days=2000), tr_end, freq="h")
        X = MO.build_features(y, nat, tr)
        X = X[y.reindex(X.index).notna()]
        X = MO.attach_calendar(X, cal)
        yv = y.reindex(X.index)
        feats = list(X.columns)
        fits = {}
        for vid, hyp, kw, wk in VARIANTS:
            sw = sw_of(yv, wk)
            m = lgb.LGBMRegressor(**{**LGB, **kw})
            m.fit(X.to_numpy(), yv.to_numpy(), sample_weight=sw)
            fits[vid] = (m, feats)
        fit_until = tgt[0] + pd.Timedelta(days=RETRAIN)
        print(f"    retrain @{day.date()} ({(time.time()-t0)/60:.1f} dk)", flush=True)

    a = y.reindex(tgt)
    ok = a.notna() & (a > 0)
    if ok.sum() < 20:
        continue
    Xp = MO.build_features(y, nat, tgt)      # tum kollar ayni matrisi kullanir
    Xp = MO.attach_calendar(Xp, cal)
    for vid, hyp, kw, wk in VARIANTS:
        m, feats = fits[vid]
        p = np.clip(m.predict(Xp[feats].to_numpy()), 1e3, None) * F.halfday_factor(tgt)
        sinks[vid].append(pd.DataFrame({"p": p[ok], "y": a[ok].to_numpy()}, index=tgt[ok]))

print(f"\n[{time.time()-t0:.0f}s] === AMAC FONKSIYONU SONUCU ===", flush=True)
rows = []
for vid, hyp, kw, wk in VARIANTS:
    d0 = pd.concat(sinks[vid])
    d0 = d0[pd.DatetimeIndex(d0.index).normalize() >= EVAL_START]
    e = np.abs(d0["p"] - d0["y"]) / d0["y"] * 100
    rows.append(dict(id=vid, hyp=hyp, mape=float(e.mean()),
                     bias=float((d0["p"] - d0["y"]).mean()), n=int(len(d0))))
base = rows[0]
print("%-12s %8s %9s %9s" % ("kol", "MAPE", "fark", "bias"))
for r in sorted(rows, key=lambda x: x["mape"]):
    print("%-12s %7.3f%% %+8.3f %+9.0f" % (r["id"], r["mape"], r["mape"] - base["mape"], r["bias"]))

with open(os.path.join(OUT, "ledger_objective.jsonl"), "a", encoding="utf-8") as fh:
    fh.write(json.dumps(dict(ts=pd.Timestamp.now().isoformat(timespec="seconds"),
                             eval_days=EVAL, rows=rows), ensure_ascii=False) + "\n")
json.dump(dict(eval_days=EVAL, rows=rows), open(os.path.join(OUT, "summary_objective.json"), "w"),
          indent=1, default=str)
print("[objective-loop] bitti", flush=True)
