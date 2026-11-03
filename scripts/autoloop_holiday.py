"""OTONOM DENEY DONGUSU — tatil acigi.

Sorun: 720 gunluk protoklde tatil MAPE 4.66, normal 1.49. Gunlerin %8'i ama
ortalamayi +0.285 pp yukari cekiyor. En kotu 5 gunun 5'i de tatil.

Bu turda mekanizma IZOLE ediliyor: tek LightGBM (uretim parametreleri), 720 gun,
aylik yeniden egitim. Tatil agirligi ise yararsa topluluga tasinir.

Kollar:
  H00  agirlik yok (referans)
  H40  tatil satirlari x2
  H41  tatil satirlari x3
  H42  tatil satirlari x5
  H43  tatil komsulugu (bayram oncesi/sonrasi 2 gun) x2 + tatil x3
  H44  tatil satirlari TAMAMEN CIkarildi (tatiller mi bozuyor testi)

Kullanim: python autoloop_holiday.py [eval_gun]
"""
import os, sys, json, time
os.chdir(r"C:\projeler\turkiye-t2-daysahead")
sys.path.insert(0, r"C:\projeler\turkiye-t2-daysahead")
import numpy as np, pandas as pd
import lightgbm as lgb
from src import data as D, morning as MO, models as M, features as F

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

# ---------------- tatil isaretleme ----------------
# DIKKAT 1: takvim kolonlarini DESENLE tarama — "cal_okul_tatil" gibi kolonlar
#   10 yilda 1297 gunu "tatil" yapar.
# DIKKAT 2: F.HOLIDAY_DATES SADECE dini bayramlari icerir (2026'da 7 gun);
#   milli bayramlar (23 Nisan, 19 Mayis, 30 Agustos, 29 Ekim...) orada YOK.
#   Model onlari saatlik takvimdeki cal_milli_gun uzerinden goruyor.
# Dogru tanim: takvimin resmi tatil bayraklari.
cal = MO.load_calendar()
HOL_FLAGS = [c for c in ("cal_bayram_gun", "cal_milli_gun", "cal_yarim_gun") if c in cal.columns]
hflag = cal[HOL_FLAGS].fillna(0).abs().sum(axis=1) > 0
hol_days = pd.DatetimeIndex(sorted(set(hflag[hflag].index.normalize())))
print(f"tatil gunu ({HOL_FLAGS}): {len(hol_days)} | {hol_days.min().date()} -> {hol_days.max().date()}", flush=True)
print("  son 8:", [str(d.date()) for d in hol_days[-8:]], flush=True)

# komsuluk: tatilden 1-2 gun once/sonra
near = set()
for d in hol_days:
    for k in (-2, -1, 1, 2):
        near.add(d + pd.Timedelta(days=k))
near = pd.DatetimeIndex(sorted(near))
near = near[~near.isin(hol_days)]

# kopru gunleri (cal_bridge) — 30 Ekim gibi tek gunluk bosluklar
if "cal_bridge" in cal.columns:
    br = cal["cal_bridge"].fillna(0)
    br_days = pd.DatetimeIndex(sorted(set(br[br > 0].index.normalize())))
else:
    br_days = pd.DatetimeIndex([])
print(f"kopru gunu: {len(br_days)} | son 6: {[str(d.date()) for d in br_days[-6:]]}", flush=True)

WARM = 60
targets = pd.date_range(last_full - pd.Timedelta(days=EVAL + WARM - 1), last_full, freq="D")
EVAL_START = targets[WARM]
print(f"hedef {len(targets)} gun | degerlendirme {EVAL_START.date()} -> {last_full.date()}", flush=True)

LGB = dict(n_estimators=1500, learning_rate=0.05, num_leaves=95, colsample_bytree=0.8,
           subsample=0.8, min_child_samples=50, objective="regression", verbose=-1)


def weight_of(idx, cfg):
    d = pd.DatetimeIndex(idx).normalize()
    w = np.ones(len(idx))
    is_h = d.isin(hol_days)
    is_n = d.isin(near)
    is_b = d.isin(br_days)
    if cfg.get("drop"):
        w[is_h] = 0.0
        return w
    w[is_h] = cfg.get("hol", 1.0)
    w[is_n] = np.maximum(w[is_n], cfg.get("near", 1.0))
    w[is_b] = np.maximum(w[is_b], cfg.get("bridge", 1.0))
    return w


VARIANTS = [
    ("H00_base", "Agirlik yok — referans", dict()),
    ("H40_hol2", "Tatil satirlari x2", dict(hol=2.0)),
    ("H41_hol3", "Tatil satirlari x3", dict(hol=3.0)),
    ("H42_hol5", "Tatil satirlari x5", dict(hol=5.0)),
    ("H43_near", "Tatil x3 + komsuluk x2", dict(hol=3.0, near=2.0)),
    ("H44_drop", "Tatil satirlari tamamen cikarildi", dict(drop=True)),
    ("H45_bridge", "Kopru gunleri x3", dict(bridge=3.0)),
    ("H46_all", "Tatil x2 + komsuluk x1.5 + kopru x3", dict(hol=2.0, near=1.5, bridge=3.0)),
]

sinks = {v[0]: [] for v in VARIANTS}
t0 = time.time()
fit_until, Xtr_static, fits = None, None, {}
for day in targets:
    tgt = pd.date_range(f"{day} 00:00", f"{day} 23:00", freq="h")
    if not fits or tgt[0] >= fit_until:
        dec = day - pd.Timedelta(days=1)
        tr_end = dec + pd.Timedelta(hours=8)          # 09:00'da bilinen son tam saat
        tr = pd.date_range(tr_end - pd.Timedelta(days=2000), tr_end, freq="h")
        X = MO.build_features(y, nat, tr)
        X = X[y.reindex(X.index).notna()]
        X = MO.attach_calendar(X, cal)
        yv = y.reindex(X.index)
        feats = list(X.columns)
        fits = {}
        for vid, hyp, cfg in VARIANTS:
            w = weight_of(X.index, cfg)
            keep = w > 0
            m = lgb.LGBMRegressor(random_state=42, **LGB)
            m.fit(X.loc[keep].to_numpy(), yv[keep].to_numpy(), sample_weight=w[keep])
            fits[vid] = (m, feats)
        print(f"    retrain @{day.date()}: n={len(X)} ({(time.time()-t0)/60:.1f} dk)", flush=True)
        fit_until = tgt[0] + pd.Timedelta(days=RETRAIN)

    Xp = MO.build_features(y, nat, tgt)
    Xp = MO.attach_calendar(Xp, cal)
    a = y.reindex(tgt)
    ok = a.notna() & (a > 0)
    if ok.sum() < 20:
        continue
    for vid, hyp, cfg in VARIANTS:
        m, feats = fits[vid]
        p = np.clip(m.predict(Xp[feats].to_numpy()), 1e3, None) * F.halfday_factor(tgt)
        sinks[vid].append(pd.DataFrame({"p": p[ok], "y": a[ok].to_numpy()}, index=tgt[ok]))

print(f"\n[{time.time()-t0:.0f}s] === TATIL SONUC TABLOSU ===", flush=True)
rows = []
for vid, hyp, cfg in VARIANTS:
    d0 = pd.concat(sinks[vid])
    d0 = d0[pd.DatetimeIndex(d0.index).normalize() >= EVAL_START]
    is_h = pd.DatetimeIndex(d0.index).normalize().isin(hol_days)
    err = np.abs(d0["p"] - d0["y"]) / d0["y"] * 100
    rows.append(dict(id=vid, hyp=hyp, all=float(err.mean()),
                     hol=float(err[is_h].mean()) if is_h.sum() else float("nan"),
                     nor=float(err[~is_h].mean()), n=int(len(d0)), n_hol=int(is_h.sum())))
base = rows[0]
print("%-12s %8s %9s %9s %8s" % ("varyant", "TUMU", "TATIL", "NORMAL", "fark"))
for r in sorted(rows, key=lambda x: x["all"]):
    print("%-12s %7.3f%% %8.3f%% %8.3f%% %+7.3f" % (r["id"], r["all"], r["hol"], r["nor"], r["all"] - base["all"]))
print(f"\ntatil saati: {rows[0]['n_hol']} / {rows[0]['n']}")

json.dump(dict(eval_days=EVAL, hol_days=len(hol_days), rows=rows),
          open(os.path.join(OUT, "summary_holiday.json"), "w"), indent=1, default=str)
with open(os.path.join(OUT, "ledger_holiday.jsonl"), "a", encoding="utf-8") as fh:
    fh.write(json.dumps(dict(ts=pd.Timestamp.now().isoformat(timespec="seconds"),
                             eval_days=EVAL, rows=rows), ensure_ascii=False) + "\n")
print("[holiday-loop] bitti", flush=True)
