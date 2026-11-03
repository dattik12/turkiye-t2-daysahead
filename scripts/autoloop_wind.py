"""OTONOM DENEY DONGUSU — ruzgar.

Protokol (wind_spatial_wf2.py ile ayni, kiyas adil):
  egitim 60 gun, adim 7 gun, degerlendirme 90 gun, hedef = RITM gerceklesmesi.
  Agirliklar her blokta SADECE egitim penceresinden NNLS ile (sizinti yok).

Yuk tarafinda ise yarayan iki kaldirac buraya da aktariliyor:
  - egitim penceresi taramasi (yukte tepe 2000 gun; burada 60 gun cok kisa olabilir)
  - seed ortalamasi (yukte -0.022 pp, ucuz varyans azaltma)

Kullanim: python autoloop_wind.py
"""
import os, sys, json, glob, time
os.chdir(r"C:\projeler\turkiye-t2-daysahead")
sys.path.insert(0, r"C:\projeler\turkiye-t2-daysahead")
import numpy as np, pandas as pd
import lightgbm as lgb
from scipy.optimize import nnls
from src.wind import pull_actual as PA

OUT = r"C:\projeler\autoloop"
os.makedirs(OUT, exist_ok=True)
LEDGER = os.path.join(OUT, "ledger_wind.jsonl")
STEP_D, EVAL_D = 7, 90

# ---------------- veri ----------------
try:
    tgt = PA.pull_generation("2026-03-01", "2026-11-03").rename("y")
    if getattr(tgt.index, "tz", None) is None:
        tgt.index = tgt.index.tz_localize("Europe/Istanbul")
except Exception as ex:
    print("RITM gerceklesme yok, rt-gen:", str(ex)[:100], flush=True)
    g = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob("data/generation/rt_gen_*.parquet"))])
    g.index = pd.DatetimeIndex(g.index)
    tgt = g["wind"].rename("y")
tgt = tgt[~tgt.index.duplicated(keep="last")].sort_index()

W = pd.read_parquet(r"C:\projeler\prov_speed_long.parquet")
W.index = pd.DatetimeIndex(W.index)
if W.index.tz is None:
    W.index = W.index.tz_localize("Europe/Istanbul")
W = W[~W.index.duplicated(keep="last")].sort_index()

j = W.join(tgt, how="inner").dropna(subset=["y"])
j = j[j["y"] > 0]
print(f"eslesen {len(j)} saat | {j.index.min()} -> {j.index.max()}", flush=True)
sp = j[W.columns]


def build_static(frame, sp_stats=True, sp_raw=True, lags=(168, 336)):
    """Agirliktan BAGIMSIZ ozellikler. Blok basina bir kez hesaplanir, cache'lenir."""
    idx = frame.index
    f = pd.DataFrame(index=idx)
    f["hour"] = idx.hour
    f["dow"] = idx.dayofweek
    f["is_weekend"] = (idx.dayofweek >= 5).astype(int)
    y = frame["y"]
    for L in lags:
        f[f"gen_lag{L}"] = y.shift(L)
    f["gen_roll168"] = y.shift(168).rolling(168).mean()
    if sp_stats:
        s = frame[W.columns]
        f["sp_mean"] = s.mean(axis=1)
        f["sp_std"] = s.std(axis=1)
        f["sp_min"] = s.min(axis=1)
        f["sp_max"] = s.max(axis=1)
        f["sp_spread"] = f["sp_max"] - f["sp_min"]
        f["sp_q25"] = s.quantile(0.25, axis=1)
        f["sp_q75"] = s.quantile(0.75, axis=1)
    if sp_raw:
        for c in W.columns:
            f[f"p_{c}"] = frame[c]
    return f


def build_block(static, weighted, y):
    """Agirliga BAGIMLI kolonlar. Her (blok, varyant) icin yeniden hesaplanir."""
    f = static.copy()
    f["w100"] = weighted.reindex(f.index).clip(lower=0)
    f["w100_2"] = f["w100"] ** 2
    f["w100_3"] = f["w100"] ** 3
    err = y - weighted.reindex(y.index)
    f["err_lag168"] = err.shift(168).reindex(f.index)
    f["err_lag336"] = err.shift(336).reindex(f.index)
    return f


def wmape(p, a):
    p = np.asarray(p, dtype=float); a = np.asarray(a, dtype=float)
    return float(np.abs(p - a).sum() / a.sum() * 100)


# ---------------- varyantlar ----------------
BASE = dict(train_d=60, seeds=[42], trees=700, lr=0.05, leaves=31,
            sp_stats=True, sp_raw=True, lags=(168, 336), mono=False)
VARIANTS = [
    ("W00_base", "Uretim benzeri referans: 60 gun, tek seed", {}),
    ("W10_td90", "Pencere 90 gun (yuk dersi: pencere kritik)", dict(train_d=90)),
    ("W11_td120", "Pencere 120 gun", dict(train_d=120)),
    ("W13_td45", "Pencere 45 gun (kisa daha mi iyi)", dict(train_d=45)),
    ("W20_seed3", "3 seed ortalamasi (yukte en ucuz kazanc)", dict(seeds=[42, 7, 99])),
    ("W21_seed5", "5 seed ortalamasi (doyum testi)", dict(seeds=[42, 7, 99, 123, 2024])),
    ("W30_slow_more", "1200 agac + lr 0.03 (daha yavas ogrenme)", dict(trees=1200, lr=0.03)),
    ("W31_deep", "num_leaves 63 (daha derin)", dict(leaves=63)),
    ("W32_shallow", "num_leaves 15 (daha sig)", dict(leaves=15)),
    ("W40_noraw", "39 il hiz kolonu YOK (sadece istatistik)", dict(sp_raw=False)),
    ("W41_longlag", "lag 504 + 672 ekle (3-4 haftalik)", dict(lags=(168, 336, 504, 672))),
    ("W50_mono", "w100 uzerinde monoton artan kisit", dict(mono=True)),
    ("W60_combo", "Pencere 90 + 3 seed (iki kaldırac birlikte)", dict(train_d=90, seeds=[42, 7, 99])),
]

# ---------------- kosu ----------------
days = pd.date_range(j.index.min().normalize(), j.index.max().normalize(), freq="D")
MAXTR = max(v[2].get("train_d", BASE["train_d"]) for v in VARIANTS)
# Tum kollar TAM pencereyle egitilebilsin: degerlendirme, en uzun pencerenin
# doldugu gunden once baslamaz. Aksi halde uzun pencereli kol erken bloklarda
# daha az veriyle egitilir ve kiyas bozulur.
eval_start = max(days.max() - pd.Timedelta(days=EVAL_D),
                 days.min() + pd.Timedelta(days=MAXTR))
print(f"degerlendirme {eval_start.date()} -> {days.max().date()} (en uzun pencere {MAXTR}g)", flush=True)

FEATS_CACHE = {}
sinks = {v[0]: [] for v in VARIANTS}
blocks = 0
d = eval_start.normalize()
t_start = time.time()
while d <= days.max():
    te0, te1 = d, min(d + pd.Timedelta(days=STEP_D - 1), days.max())
    if te0 < eval_start:
        d += pd.Timedelta(days=STEP_D); continue
    mte = (j.index >= te0) & (j.index <= te1 + pd.Timedelta(hours=23))
    if mte.sum() < 100:
        d += pd.Timedelta(days=STEP_D); continue

    # agirlik SADECE o varyantin egitim penceresinden
    for vid, hyp, ov in VARIANTS:
        cfg = {**BASE, **ov}
        tr0 = d - pd.Timedelta(days=cfg["train_d"])
        mtr = (j.index >= tr0) & (j.index < d)
        if mtr.sum() < cfg["train_d"] * 20:          # pencere ~%83 dolmadiysa kolu atla
            continue
        Xs = sp[mtr].fillna(sp[mtr].mean())
        yv = j.loc[mtr, "y"]
        wv, _ = nnls((Xs / yv.max()).to_numpy(), (yv / yv.max()).to_numpy())
        wv = wv / wv.sum() if wv.sum() > 0 else np.full(len(wv), 1 / len(wv))
        weighted = sp.mul(pd.Series(wv, index=W.columns), axis=1).sum(axis=1)

        key = (cfg["sp_stats"], cfg["sp_raw"], cfg["lags"])
        if key not in FEATS_CACHE:
            FEATS_CACHE[key] = build_static(j, cfg["sp_stats"], cfg["sp_raw"], cfg["lags"])
        f = build_block(FEATS_CACHE[key], weighted, j["y"])
        feats = [c for c in f.columns if not f[c].isna().all()]

        p = np.zeros(mte.sum())
        for s in cfg["seeds"]:
            m = lgb.LGBMRegressor(
                n_estimators=cfg["trees"], learning_rate=cfg["lr"], num_leaves=cfg["leaves"],
                min_child_samples=30, subsample=0.85, colsample_bytree=0.85,
                random_state=s, verbose=-1,
                **({"monotone_constraints": [1 if c == "w100" else 0 for c in feats],
                    "monotone_constraints_method": "advanced"} if cfg["mono"] else {}))
            m.fit(f.loc[mtr, feats].fillna(0), j.loc[mtr, "y"])
            p += m.predict(f.loc[mte, feats].fillna(0))
        p /= len(cfg["seeds"])
        sinks[vid].append(pd.DataFrame({"p": np.clip(p, 0, None), "y": j.loc[mte, "y"].values},
                                       index=j.loc[mte].index))
    blocks += 1
    if blocks % 3 == 0:
        print(f"  blok {blocks} ({(time.time()-t_start)/60:.1f} dk)", flush=True)
    d += pd.Timedelta(days=STEP_D)

print(f"\nblok: {blocks}", flush=True)
print("=== RUZGAR SONUC TABLOSU ===")
print("%-14s %8s %8s %9s  %s" % ("varyant", "WMAPE", "fark", "bias", "karar"))
rows = []
for vid, hyp, ov in VARIANTS:
    if not sinks[vid]:
        continue
    d0 = pd.concat(sinks[vid])
    w = wmape(d0["p"].to_numpy(), d0["y"].to_numpy())
    b = float((d0["p"] - d0["y"]).mean())
    rows.append(dict(id=vid, hyp=hyp, wmape=w, bias=b, n=int(len(d0)), cfg=ov))
base = next((r for r in rows if r["id"] == "W00_base"), None)
for r in sorted(rows, key=lambda x: x["wmape"]):
    if base:
        dd = r["wmape"] - base["wmape"]
        karar = "KEEP" if dd < -0.30 else ("esit" if abs(dd) <= 0.30 else "rollback")
        print("%-14s %7.2f%% %+7.2f %+9.0f  %s" % (r["id"], r["wmape"], dd, r["bias"], karar))
    else:
        print("%-14s %7.2f%% %8s %+9.0f" % (r["id"], r["wmape"], "-", r["bias"]))

try:
    r = PA.pull_ritm_forecast(str(eval_start.date()), str(days.max().date()))[["ritm_fc_mw"]]
    if getattr(r.index, "tz", None) is None:
        r.index = r.index.tz_localize("Europe/Istanbul")
    r = r[~r.index.duplicated(keep="last")]
    yy = tgt.reindex(r.index); ok = yy.notna() & (yy > 0)
    print("  RITM                    %7.2f%%" % wmape(r["ritm_fc_mw"][ok].to_numpy(), yy[ok].to_numpy()))
except Exception as ex:
    print("  RITM alinamadi:", str(ex)[:100])

with open(LEDGER, "a", encoding="utf-8") as fh:
    fh.write(json.dumps(dict(ts=pd.Timestamp.now().isoformat(timespec="seconds"),
                             blocks=blocks, rows=rows), ensure_ascii=False) + "\n")
json.dump(dict(blocks=blocks, rows=rows), open(os.path.join(OUT, "summary_wind.json"), "w"),
          indent=1, default=str)
print("\n[wind-loop] bitti ->", LEDGER, flush=True)
