"""L2 — BOLGESEL HAVA (yuk modeli).

Yuk modeli tek ulusal hava serisi kullaniyor (10 il, NUFUS agirlikli).
Ruzgar modeli 39 il kullaniyor ve bu orada kazandirdi. Yukte hic denenmedi.

Elestiri: yuk bolgesel — Marmara sanayi, Ege/Akdeniz tarim, Dogu isitma.
Tek ortalama bu sinyalleri birbirine karistiriyor. Ayrica nufus agirligi
tuketim agirligiyla ayni sey degil (Istanbul nufusun %39'u).

Kollar (720 gun, tek LGBM tarama; kazanan toplulukla dogrulanir):
  A_base       mevcut 83
  B_il_temp    + 10 il sicakligi (ayri kolonlar)
  C_spread     + mekansal yayilim (max-min, std, isitma farki)
  D_altweight  + esit agirlikli ulusal sicaklik (nufus yerine)
  E_tarim      + tarim bolgesi sicakligi (Konya/Gaziantep/Adana) + kuraklik

Kullanim: python autoloop_l2.py [eval_gun]
"""
import os, sys, json, time
os.chdir(r"C:\projeler\turkiye-t2-daysahead")
sys.path.insert(0, r"C:\projeler\turkiye-t2-daysahead")
import numpy as np, pandas as pd
import lightgbm as lgb
from src import data as D, morning as MO, features as F, config as C

EVAL = int(sys.argv[1]) if len(sys.argv) > 1 else 720
RETRAIN = 30
OUT = r"C:\projeler\autoloop"

cons = D.load_or_create_consumption()
nat, _ = D.load_or_create_weather(); nat.index = pd.to_datetime(nat.index)
nat = nat[~nat.index.duplicated(keep="last")].sort_index()
y = cons["rt_cons"].dropna(); y.index = pd.to_datetime(y.index)
fd = y.groupby(y.index.normalize()).size()
last_full = fd[fd == 24].index.max()
cal = MO.load_calendar()

# --- sehir bazli hava ---
ci = pd.read_parquet("data/weather/histfc_cities.parquet")
ci["dt"] = pd.to_datetime(ci["dt"])
if ci["dt"].dt.tz is None:
    ci["dt"] = ci["dt"].dt.tz_localize("Europe/Istanbul")
T = ci.pivot_table(index="dt", columns="city", values="temperature_2m")
T = T[~T.index.duplicated(keep="last")].sort_index()
P = ci.pivot_table(index="dt", columns="city", values="precipitation")
P = P.reindex(T.index)
CITY_COLS = list(T.columns)
print(f"sehir havasi: {T.shape} | {T.index.min()} -> {T.index.max()}", flush=True)


def _nv(i):
    """Saat dilimini at — model indeksi ile hava indeksi ayni eksende hizalansin.
    (Onceki hata: T tz-aware, hedef indeks naive -> reindex %100 NaN -> feature'lar
     modele hic girmiyordu, uc kol birebir ayni skoru veriyordu.)"""
    i = pd.DatetimeIndex(i)
    return i.tz_localize(None) if i.tz is not None else i


T.index = _nv(T.index)
if P is not None:
    P.index = _nv(P.index)


def add_il_temp(idx, base):
    out = T.reindex(_nv(idx)).copy()
    out.index = idx
    out.columns = [f"il_{c}" for c in out.columns]
    return out


def add_spread(idx, base):
    t = T.reindex(_nv(idx))
    out = pd.DataFrame(index=idx)
    out["sp_tmax"] = t.max(axis=1).to_numpy()
    out["sp_tmin"] = t.min(axis=1).to_numpy()
    out["sp_range"] = out["sp_tmax"] - out["sp_tmin"]
    out["sp_std"] = t.std(axis=1).to_numpy()
    out["sp_hdd_spread"] = ((16 - t).clip(lower=0).max(axis=1)
                            - (16 - t).clip(lower=0).min(axis=1)).to_numpy()
    out["sp_cdd_spread"] = ((t - 22).clip(lower=0).max(axis=1)
                            - (t - 22).clip(lower=0).min(axis=1)).to_numpy()
    return out


def add_altweight(idx, base):
    t = T.reindex(_nv(idx))
    out = pd.DataFrame(index=idx)
    pop = ci.groupby("city")["pop"].first()
    w = pop.reindex(CITY_COLS).fillna(0).to_numpy()
    eq = t.mean(axis=1).to_numpy()
    pw = (t.to_numpy() * (w / w.sum())).sum(axis=1)
    out["alt_temp_equal"] = eq
    out["alt_temp_pop"] = pw
    out["alt_temp_diff"] = eq - pw
    return out


def add_tarim(idx, base):
    out = pd.DataFrame(index=idx)
    agri = [c for c in ("konya", "gaziantep", "adana") if c in T.columns]
    if agri:
        d = T.reindex(_nv(idx))[agri].mean(axis=1)
        out["agri_temp"] = d.to_numpy()
        out["agri_cdd"] = (d - 22).clip(lower=0).to_numpy()
        out["agri_hdd"] = (16 - d).clip(lower=0).to_numpy()
    if P is not None and len(P.columns):
        dry = P.reindex(_nv(idx)).sum(axis=1)
        out["dry_7d"] = dry.rolling(168, min_periods=120).sum().to_numpy()
        out["dry_30d"] = dry.rolling(720, min_periods=480).sum().to_numpy()
    return out


BLOCKS = {"B_il_temp": add_il_temp, "C_spread": add_spread,
          "D_altweight": add_altweight, "E_tarim": add_tarim}
ARMS = ["A_base"] + list(BLOCKS)

LGB = dict(n_estimators=900, learning_rate=0.05, num_leaves=63, colsample_bytree=0.8,
           subsample=0.8, min_child_samples=40, random_state=42, verbose=-1)

WARM = 60
targets = pd.date_range(last_full - pd.Timedelta(days=EVAL + WARM - 1), last_full, freq="D")
EVAL_START = targets[WARM]
print(f"degerlendirme {EVAL_START.date()} -> {last_full.date()} | {len(ARMS)} kol", flush=True)

sinks = {a: [] for a in ARMS}
t0 = time.time()
fit_until, fits = None, {}
_dbg = {"done": False}
for day in targets:
    tgt = pd.date_range(f"{day} 00:00", f"{day} 23:00", freq="h")
    if not fits or tgt[0] >= fit_until:
        dec = day - pd.Timedelta(days=1)
        tr_end = dec + pd.Timedelta(hours=8)
        tr = pd.date_range(tr_end - pd.Timedelta(days=2000), tr_end, freq="h")
        X0 = MO.build_features(y, nat, tr)
        X0 = X0[y.reindex(X0.index).notna()]
        X0 = MO.attach_calendar(X0, cal)
        yv = y.reindex(X0.index)
        fits = {}
        for arm in ARMS:
            X = X0.copy()
            if arm != "A_base":
                blk = BLOCKS[arm](X.index, X)
                if not _dbg["done"]:
                    print(f"    [DEBUG] {arm}: blok {blk.shape} dolu={int(blk.notna().sum().sum())}"
                          f"/{blk.size} | kolonlar {list(blk.columns)[:3]}", flush=True)
                X = pd.concat([X, blk], axis=1)
            m = lgb.LGBMRegressor(**LGB)
            m.fit(X.to_numpy(), yv.to_numpy())
            fits[arm] = (m, list(X.columns))
        fit_until = tgt[0] + pd.Timedelta(days=RETRAIN)
        print(f"    retrain @{day.date()} ({(time.time()-t0)/60:.1f} dk)", flush=True)

    a = y.reindex(tgt)
    ok = a.notna() & (a > 0)
    if ok.sum() < 20:
        continue
    Xp0 = MO.build_features(y, nat, tgt)
    Xp0 = MO.attach_calendar(Xp0, cal)
    for arm in ARMS:
        m, cols = fits[arm]
        Xp = Xp0.copy()
        if arm != "A_base":
            Xp = pd.concat([Xp, BLOCKS[arm](Xp.index, Xp)], axis=1)
        p = np.clip(m.predict(Xp[cols].to_numpy()), 1e3, None) * F.halfday_factor(tgt)
        sinks[arm].append(pd.DataFrame({"p": p[ok], "y": a[ok].to_numpy()}, index=tgt[ok]))

print(f"\n[{time.time()-t0:.0f}s] === L2 BOLGESEL HAVA ===", flush=True)
rows = []
for arm in ARMS:
    d0 = pd.concat(sinks[arm])
    d0 = d0[pd.DatetimeIndex(d0.index).normalize() >= EVAL_START]
    e = np.abs(d0.p - d0.y) / d0.y * 100
    rows.append(dict(id=arm, mape=float(e.mean()), bias=float((d0.p - d0.y).mean()), n=int(len(d0))))
base = rows[0]
print("%-12s %8s %10s %9s" % ("kol", "MAPE", "fark", "bias"))
for r in sorted(rows, key=lambda x: x["mape"]):
    print("%-12s %7.3f%% %+9.3f %+9.0f" % (r["id"], r["mape"], r["mape"] - base["mape"], r["bias"]))

with open(os.path.join(OUT, "ledger_l2.jsonl"), "a", encoding="utf-8") as fh:
    fh.write(json.dumps(dict(ts=pd.Timestamp.now().isoformat(timespec="seconds"),
                             eval_days=EVAL, rows=rows), ensure_ascii=False) + "\n")
json.dump(dict(eval_days=EVAL, rows=rows), open(os.path.join(OUT, "summary_l2.json"), "w"),
          indent=1, default=str)
print("[l2] bitti", flush=True)
