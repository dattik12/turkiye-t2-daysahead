"""HAVA GIRDISI DENEYI — egitim/tahmin dagilim uyusmazligi.

Soru: model ANALIZ havasiyla egitiliyor ama CANLI tahminde lead-1 hava goruyor.
Bu uyusmazlik canli skoru ne kadar bozuyor, ve duzeltilebilir mi?

Kollar (hepsi AYNI hedefi, AYNI sabah protokolunu tahmin eder):
  A_analiz    egitim=analiz,  tahmin=analiz    -> mevcut backtest (iyimser)
  B_canli     egitim=analiz,  tahmin=lead1     -> BUGUNKU canli gercek
  C_uyumlu    egitim=lead1,   tahmin=lead1     -> dagilim eslesmis
  D_duzeltme  egitim=analiz,  tahmin=lead1+bias duzeltmesi (walk-forward)

bias duzeltmesi SADECE gecmis veriden kestirilir (ay x saat), sizinti yok.

Kullanim: python autoloop_weather.py [eval_gun]
"""
import os, sys, glob, json, time
os.chdir(r"C:\projeler\turkiye-t2-daysahead")
sys.path.insert(0, r"C:\projeler\turkiye-t2-daysahead")
import numpy as np, pandas as pd
import lightgbm as lgb
from src import data as D, morning as MO, config as C, features as F

EVAL = int(sys.argv[1]) if len(sys.argv) > 1 else 720
RETRAIN = 30
OUT = r"C:\projeler\autoloop"
os.makedirs(OUT, exist_ok=True)
LWDIR = r"C:\projeler\lead_weather"


def national_from_cities(sub: str) -> pd.DataFrame:
    """Sehir parquet'lerinden nufus agirlikli ulusal cerceve (pull_weather_national ile ayni)."""
    frames, pops = {}, {}
    for c in C.CITIES:
        p = os.path.join(LWDIR, f"{c['city']}.parquet")
        if not os.path.exists(p):
            continue
        d = pd.read_parquet(p)
        d.index = pd.to_datetime(d.index)
        for v in C.OM_VARS:
            col = f"{v}_{sub}"
            if col in d.columns:
                frames.setdefault(v, {})[c["city"]] = d[col]
                pops[c["city"]] = c["pop"]
    if not frames:
        raise RuntimeError(f"lead hava yok ({sub}) — once pull_lead_weather.py kos")
    # frames: {var: {city: Series}} — indeksi ilk seriden al
    any_series = next(iter(next(iter(frames.values())).values()))
    out = pd.DataFrame(index=any_series.index.sort_values())
    for v, per in frames.items():
        t = pd.DataFrame(per)
        w = pd.Series({k: pops[k] for k in t.columns})
        out[v] = t.mul(w, axis=1).sum(axis=1, min_count=max(1, len(w) // 2)) / w[t.columns].sum()
    out["HDD"] = (18 - out["temperature_2m"]).clip(lower=0)
    out["CDD"] = (out["temperature_2m"] - 22).clip(lower=0)
    return out.sort_index()


cons = D.load_or_create_consumption()
nat, _ = D.load_or_create_weather(); nat.index = pd.to_datetime(nat.index)
nat = nat[~nat.index.duplicated(keep="last")].sort_index()
y = cons["rt_cons"].dropna(); y.index = pd.to_datetime(y.index)
fd = y.groupby(y.index.normalize()).size()
last_full = fd[fd == 24].index.max()
cal = MO.load_calendar()

L1 = national_from_cities("previous_day1") if len(sys.argv) < 3 else national_from_cities(sys.argv[2])
print(f"lead1 hava: {L1.shape} | {L1.index.min()} -> {L1.index.max()}", flush=True)
# eksik degisken varsa analizle doldur (dokumante edilmis yaklasim)
missing = [c for c in nat.columns if c not in L1.columns]
if missing:
    print(f"  UYARI eksik kolon analizle dolduruldu: {missing}", flush=True)
    for c in missing:
        L1[c] = nat[c].reindex(L1.index)
L1 = L1[[c for c in nat.columns if c in L1.columns]]
gap = (L1["temperature_2m"] - nat["temperature_2m"].reindex(L1.index)).dropna()
print(f"  sicaklik lead1-analiz: ort {gap.mean():+.3f} | MAE {gap.abs().mean():.3f} | n {len(gap)}", flush=True)

WARM = 60
targets = pd.date_range(last_full - pd.Timedelta(days=EVAL + WARM - 1), last_full, freq="D")
EVAL_START = targets[WARM]
print(f"degerlendirme {EVAL_START.date()} -> {last_full.date()}", flush=True)

LGB = dict(n_estimators=900, learning_rate=0.05, num_leaves=63, colsample_bytree=0.8,
           subsample=0.8, min_child_samples=40, random_state=42, verbose=-1)


def bias_table(fit_end):
    """Ay x saat bazinda (lead1 - analiz) kestirimi; SADECE fit_end oncesi veri."""
    a = nat["temperature_2m"].reindex(L1.index)
    g = (L1["temperature_2m"] - a).dropna()
    g = g[g.index < fit_end]
    if len(g) < 2000:
        return None
    return g.groupby([g.index.month, g.index.hour]).mean()


def pred_weather(base, bias, tgt):
    """base: nat veya L1. bias verilirse sicaklik duzeltilir ve HDD/CDD YENIDEN hesaplanir."""
    if bias is None:
        return base.reindex(tgt)
    out = base.reindex(tgt).copy()
    b = pd.Series(list(zip(tgt.month, tgt.hour)), index=tgt).map(bias)
    out["temperature_2m"] = out["temperature_2m"] - b.fillna(0).values
    # KRITIK: HDD/CDD sicakliktan turetilir; duzeltmeden sonra yeniden hesaplanmali,
    # yoksa model celiskili girdi gorur (duzeltilmis sicaklik + eski HDD/CDD).
    out["HDD"] = (18 - out["temperature_2m"]).clip(lower=0)
    out["CDD"] = (out["temperature_2m"] - 22).clip(lower=0)
    return out


ARMS = ["A_analiz", "B_canli", "C_uyumlu", "D_duzeltme"]
sinks = {a: [] for a in ARMS}
t0 = time.time()
fit_until, fits = None, {}
for day in targets:
    tgt = pd.date_range(f"{day} 00:00", f"{day} 23:00", freq="h")
    if not fits or tgt[0] >= fit_until:
        dec = day - pd.Timedelta(days=1)
        tr_end = dec + pd.Timedelta(hours=8)
        tr = pd.date_range(tr_end - pd.Timedelta(days=2000), tr_end, freq="h")
        yv_all = y.reindex(tr)
        fits = {}
        for arm in ARMS:
            wsrc = L1 if arm == "C_uyumlu" else nat
            X = MO.build_features(y, wsrc, tr)
            X = X[y.reindex(X.index).notna()]
            X = MO.attach_calendar(X, cal)
            m = lgb.LGBMRegressor(**LGB)
            m.fit(X.to_numpy(), y.reindex(X.index).to_numpy())
            fits[arm] = (m, list(X.columns))
        bias = bias_table(tr_end)
        fits["_bias"] = bias
        fit_until = tgt[0] + pd.Timedelta(days=RETRAIN)
        print(f"    retrain @{day.date()} ({(time.time()-t0)/60:.1f} dk) bias={'var' if bias is not None else 'YOK'}", flush=True)

    a = y.reindex(tgt)
    ok = a.notna() & (a > 0)
    if ok.sum() < 20:
        continue
    for arm in ARMS:
        m, cols = fits[arm]
        if arm == "A_analiz":
            w = nat
        elif arm == "B_canli":
            w = L1
        elif arm == "C_uyumlu":
            w = L1
        else:
            w = pred_weather(L1, fits["_bias"], tgt)
        Xp = MO.build_features(y, w, tgt)
        Xp = MO.attach_calendar(Xp, cal)
        p = np.clip(m.predict(Xp[cols].to_numpy()), 1e3, None) * F.halfday_factor(tgt)
        sinks[arm].append(pd.DataFrame({"p": p[ok], "y": a[ok].to_numpy()}, index=tgt[ok]))

print(f"\n[{time.time()-t0:.0f}s] === HAVA GIRDISI SONUCU ===", flush=True)
rows = []
for arm in ARMS:
    d0 = pd.concat(sinks[arm])
    d0 = d0[pd.DatetimeIndex(d0.index).normalize() >= EVAL_START]
    e = np.abs(d0["p"] - d0["y"]) / d0["y"] * 100
    rows.append(dict(arm=arm, mape=float(e.mean()), bias=float((d0["p"] - d0["y"]).mean()), n=int(len(d0))))
base = rows[0]
print("%-12s %8s %10s %9s" % ("kol", "MAPE", "fark", "bias"))
for r in rows:
    print("%-12s %7.3f%% %+9.3f %+9.0f" % (r["arm"], r["mape"], r["mape"] - base["mape"], r["bias"]))
print("\nA->B = dagilim uyusmazliginin maliyeti | B->C = uyumlu egitimin kazanci | B->D = duzeltmenin kazanci")

with open(os.path.join(OUT, "ledger_weather.jsonl"), "a", encoding="utf-8") as fh:
    fh.write(json.dumps(dict(ts=pd.Timestamp.now().isoformat(timespec="seconds"),
                             eval_days=EVAL, rows=rows), ensure_ascii=False) + "\n")
json.dump(dict(eval_days=EVAL, rows=rows), open(os.path.join(OUT, "summary_weather.json"), "w"),
          indent=1, default=str)
print("[weather-loop] bitti", flush=True)
