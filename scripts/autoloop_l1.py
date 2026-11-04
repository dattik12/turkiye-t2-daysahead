"""L1 — MASTER DOSYA FEATURE'LARI (26 eksik).

Master_Dosya_Variables_v3.xlsx'in "use_in_baseline" dedigi ama ulusal modelde
OLMAYAN feature'lar. Her blok AYRI test edilir, sonra hepsi birlikte.

Kollar:
  A_base       mevcut 83 feature
  B_termal     + sicaklik sekli ve rejim (temp_diff_3h/24h, gunluk min/max/aralik,
                 4 rejim bayragi)
  C_exposure   + CDH/HDH maruziyet (cdh_22, hdh_16, 72h toplam, 14g saat, ardisik gun)
  D_nem        + nem konforu (wet_bulb, humidex, cdh_x_nem, cdh_x_dewpoint,
                 ruzgar_x_cdh, yuksek_s_dusuk_ruzgar)
  E_momentum   + Load_Chg_24h, Load_Chg_168h, Load_Dev_3d
  F_astro      + daylight_fraction_hour
  G_hepsi      hepsi birden

720 gun, aylik yeniden egitim, tek LGBM, sabah protokolu (h+16 lag kurali korunur).
Sizinti: tum yeni feature'lar hava (tahmin) veya >=24s yuk gecmisinden turetilir.

Kullanim: python autoloop_l1.py [eval_gun]
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

T = nat["temperature_2m"]
RH = nat["relative_humidity_2m"]
WS = nat["wind_speed_10m"]


def dewpoint(t, rh):
    """Magnus formulu — dew point (C)."""
    a, b = 17.27, 237.7
    g = (a * t) / (b + t) + np.log(rh.clip(1, 100) / 100.0)
    return (b * g) / (a - g)


def wet_bulb(t, rh):
    """Stull 2011 yaklasik islak termometre."""
    return (t * np.arctan(0.151977 * np.sqrt(rh.clip(1, 100) + 8.313659))
            + np.arctan(t + rh.clip(1, 100)) - np.arctan(rh.clip(1, 100) - 1.676331)
            + 0.00391838 * rh.clip(1, 100) ** 1.5 * np.arctan(0.023101 * rh.clip(1, 100))
            - 4.686035)


def add_termal(idx, base):
    out = pd.DataFrame(index=base.index)
    out["temp_diff_3h"] = T.diff(3).reindex(base.index)
    out["temp_diff_24h"] = T.diff(24).reindex(base.index)
    # tmax/tmin kolonlari zaten saat bazina yayilmis gun degerleri -> dogrudan kullan
    out["daily_temp_range"] = base["tmax"] - base["tmin"]
    out["daily_tmax"] = base["tmax"]
    out["daily_tmin"] = base["tmin"]
    for nm, lo, hi in (("cold", -99, 16), ("comfort", 16, 22), ("cooling", 22, 27), ("hot", 27, 99)):
        out[f"temp_regime_{nm}"] = ((T >= lo) & (T < hi)).astype(np.int8).reindex(base.index)
    return out


def add_exposure(idx, base):
    out = pd.DataFrame(index=base.index)
    cdh = (T - 22).clip(lower=0)
    hdh = (16 - T).clip(lower=0)
    out["cdh_22"] = cdh.reindex(base.index)
    out["hdh_16"] = hdh.reindex(base.index)
    out["cdh_72h_sum"] = cdh.rolling(72, min_periods=36).sum().reindex(base.index)
    out["hdh_72h_sum"] = hdh.rolling(72, min_periods=36).sum().reindex(base.index)
    out["cdh_14d_hours"] = (cdh > 0).rolling(336, min_periods=168).sum().reindex(base.index)
    out["hdh_14d_hours"] = (hdh > 0).rolling(336, min_periods=168).sum().reindex(base.index)
    dc = (cdh > 0).astype(int).groupby(cdh.index.normalize()).max()
    dh = (hdh > 0).astype(int).groupby(hdh.index.normalize()).max()
    def runlen(s):
        out_, run = [], 0
        for v in s.values:
            run = run + 1 if v else 0
            out_.append(run)
        return pd.Series(out_, index=s.index)
    out["consecutive_cdh_days"] = runlen(dc).reindex(out.index.normalize()).values
    out["consecutive_hdh_days"] = runlen(dh).reindex(out.index.normalize()).values
    return out


def add_nem(idx, base):
    out = pd.DataFrame(index=base.index)
    dp = dewpoint(T, RH)
    wb = wet_bulb(T, RH)
    cdh = (T - 22).clip(lower=0)
    out["wet_bulb_temp"] = wb.reindex(base.index)
    # humidex: hissedilen sicaklik benzeri (C)
    e = (RH.clip(1, 100) / 100.0) * 6.105 * np.exp(17.27 * T / (237.7 + T))
    out["humidex_proxy"] = (T + 0.5555 * (e - 10)).reindex(base.index)
    out["cdh_x_humidity"] = (cdh * RH.clip(1, 100) / 100.0).reindex(base.index)
    out["cdh_x_dewpoint"] = (cdh * dp).reindex(base.index)
    out["wind_x_cdh"] = (WS * cdh).reindex(base.index)
    out["high_t_low_wind_flag"] = ((T > 27) & (WS < 2.0)).astype(np.int8).reindex(base.index)
    return out


def add_momentum(idx, base):
    """TODO #3: yon/hiz/anormallik. Hepsi >=24s gecmis -> sizintisiz."""
    out = pd.DataFrame(index=base.index)
    for col, L in (("lag24", 24), ("lag48", 48), ("lag168", 168), ("lag192", 192)):
        if col not in base.columns:
            base[col] = y.shift(L).reindex(base.index)
    out["Load_Chg_24h"] = base["lag24"] - base["lag48"]
    out["Load_Chg_168h"] = base["lag24"] - base["lag192"]
    roll3 = y.shift(24).rolling(72, min_periods=48).mean().reindex(base.index)
    out["Load_Dev_3d"] = base["lag24"] - roll3
    return out


def add_astro(idx, base):
    out = pd.DataFrame(index=base.index)
    yday = base.index.dayofyear.to_numpy()
    hour = base.index.hour.to_numpy() + base.index.minute.to_numpy() / 60.0
    lat = np.deg2rad(39.0)
    decl = np.deg2rad(-23.44 * np.cos(np.deg2rad(360.0 / 365.0 * (yday + 10))))
    h = np.deg2rad((hour - 12.0) * 15.0)
    cosz = np.sin(lat) * np.sin(decl) + np.cos(lat) * np.cos(decl) * np.cos(h)
    out["daylight_fraction_hour"] = np.clip(cosz, 0, None)
    return out


BLOCKS = {"B_termal": add_termal, "C_exposure": add_exposure, "D_nem": add_nem,
          "E_momentum": add_momentum, "F_astro": add_astro}
ARMS = ["A_base"] + list(BLOCKS) + ["G_hepsi"]

LGB = dict(n_estimators=900, learning_rate=0.05, num_leaves=63, colsample_bytree=0.8,
           subsample=0.8, min_child_samples=40, random_state=42, verbose=-1)

t_base = T.groupby(T.index.normalize())
base_w = pd.DataFrame({"tmax": t_base.max(), "tmin": t_base.min()}).reindex(nat.index).ffill()

WARM = 60
targets = pd.date_range(last_full - pd.Timedelta(days=EVAL + WARM - 1), last_full, freq="D")
EVAL_START = targets[WARM]
print(f"degerlendirme {EVAL_START.date()} -> {last_full.date()} | {len(ARMS)} kol", flush=True)

sinks = {a: [] for a in ARMS}
t0 = time.time()
fit_until, fits = None, {}
for day in targets:
    tgt = pd.date_range(f"{day} 00:00", f"{day} 23:00", freq="h")
    if not fits or tgt[0] >= fit_until:
        dec = day - pd.Timedelta(days=1)
        tr_end = dec + pd.Timedelta(hours=8)
        tr = pd.date_range(tr_end - pd.Timedelta(days=2000), tr_end, freq="h")
        X0 = MO.build_features(y, nat, tr)
        X0 = X0[y.reindex(X0.index).notna()]
        X0 = MO.attach_calendar(X0, cal)
        if "tmax" not in X0.columns:
            X0["tmax"] = base_w["tmax"].reindex(X0.index).values
            X0["tmin"] = base_w["tmin"].reindex(X0.index).values
        yv = y.reindex(X0.index)
        fits = {}
        for arm in ARMS:
            X = X0.copy()
            if arm not in ("A_base", "G_hepsi"):
                X = pd.concat([X, BLOCKS[arm](X.index, X)], axis=1)
            elif arm == "G_hepsi":
                for fn in BLOCKS.values():
                    X = pd.concat([X, fn(X.index, X)], axis=1)
            m = lgb.LGBMRegressor(**LGB)
            m.fit(X.to_numpy(), yv.to_numpy())
            fits[arm] = (m, list(X.columns))
        fit_until = tgt[0] + pd.Timedelta(days=RETRAIN)
        print(f"    retrain @{day.date()} feats={len(fits['G_hepsi'][1])} ({(time.time()-t0)/60:.1f} dk)", flush=True)

    a = y.reindex(tgt)
    ok = a.notna() & (a > 0)
    if ok.sum() < 20:
        continue
    Xp0 = MO.build_features(y, nat, tgt)
    Xp0 = MO.attach_calendar(Xp0, cal)
    if "tmax" not in Xp0.columns:
        Xp0["tmax"] = base_w["tmax"].reindex(Xp0.index).values
        Xp0["tmin"] = base_w["tmin"].reindex(Xp0.index).values
    for arm in ARMS:
        m, cols = fits[arm]
        Xp = Xp0.copy()
        if arm not in ("A_base", "G_hepsi"):
            Xp = pd.concat([Xp, BLOCKS[arm](Xp.index, Xp)], axis=1)
        elif arm == "G_hepsi":
            for fn in BLOCKS.values():
                Xp = pd.concat([Xp, fn(Xp.index, Xp)], axis=1)
        p = np.clip(m.predict(Xp[cols].to_numpy()), 1e3, None) * F.halfday_factor(tgt)
        sinks[arm].append(pd.DataFrame({"p": p[ok], "y": a[ok].to_numpy()}, index=tgt[ok]))

print(f"\n[{time.time()-t0:.0f}s] === L1 FEATURE SONUCU ===", flush=True)
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

with open(os.path.join(OUT, "ledger_l1.jsonl"), "a", encoding="utf-8") as fh:
    fh.write(json.dumps(dict(ts=pd.Timestamp.now().isoformat(timespec="seconds"),
                             eval_days=EVAL, rows=rows), ensure_ascii=False) + "\n")
json.dump(dict(eval_days=EVAL, rows=rows), open(os.path.join(OUT, "summary_l1.json"), "w"),
          indent=1, default=str)
print("[l1] bitti", flush=True)
