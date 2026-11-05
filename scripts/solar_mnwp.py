"""T6 — GUNES ML MODELI (EPiAS ulusal `sun` serisi hedefi).

Hedef: kapasite faktoru  y = sun / C_t ,  C_t = geriye donuk 30 gunun p99'u.
Tahmin:  cf_hat * C_cutoff  -> MW.
Cutoff: D 08:00. T+1 = D+1 (lead-1 hava), T+2 = D+2 (lead-2 hava).
Sizinti yok: tum gecmis seriler shift'li, hava previous-runs arsivi.
Gece: solar_elevation <= 0 -> 0.0 MW.
"""
import os, sys
import numpy as np, pandas as pd, lightgbm as lgb

os.chdir(r"C:\projeler\turkiye-t2-daysahead")
sys.path.insert(0, r"C:\projeler\turkiye-t2-daysahead")

RAW = r"data\raw\epias_national_rt_gen_2021_2026.parquet"
LEAD = r"C:\projeler\lead_weather"

# --- MULTI-NWP GUNES: 8 GES ili x 3 model ---
MNW = r"C:\projeler\multinwp\solar"
MODELS = ["ecmwf_ifs025", "gfs_seamless", "icon_seamless"]


def load_multi_solar():
    import glob as _g, re as _re
    per = {}
    for f in _g.glob(os.path.join(MNW, "*.parquet")):
        m = _re.match(r"(.+)_(20\d\dQ\d)\.parquet", os.path.basename(f))
        if m:
            per.setdefault(m.group(1), []).append(pd.read_parquet(f))
    out = {}
    for prov, lst in per.items():
        d = pd.concat(lst, axis=0).sort_index()
        if getattr(d.index, "tz", None) is not None:
            d.index = d.index.tz_localize(None)
        d = d[~d.index.duplicated(keep="last")]
        out[prov] = d
    return out


MNW_S = load_multi_solar()
print(f"multi-NWP gunes: {len(MNW_S)} il")


def mnwp_series(var, idx):
    """3 model ortalamasi ve spread (8 il ortalamasi)."""
    cols = [f"{var}_{m}" for m in MODELS]
    vals, sprs = [], []
    for prov, d in MNW_S.items():
        dd = d.reindex(idx)
        cc = [c for c in cols if c in dd.columns]
        if not cc:
            continue
        vals.append(dd[cc].mean(axis=1))
        sprs.append(dd[cc].std(axis=1))
    if not vals:
        return None, None
    V = pd.DataFrame(vals)
    S = pd.DataFrame(sprs)
    return V.mean(axis=0), S.mean(axis=0)

# GES KURULU GUC AGIRLIGI (MW). Nufus agirligi Marmara'yi sisiriyordu.
POP = {"konya": 26.0, "ankara": 13.0, "kayseri": 11.0, "antalya": 10.0, "izmir": 8.0,
       "adana": 7.0, "gaziantep": 7.0, "bursa": 5.0, "istanbul": 3.0, "trabzon": 1.0}
LAT, LON = 39.0, 35.0
import os as _os
TRAIN_D = int(_os.environ.get("TRAIN_D", "540"))   # tarama: 365=15.81 540=15.44 730=15.44/nMAE114
STEP_D = int(_os.environ.get("STEP_D", "7"))
NLEAVES = int(_os.environ.get("NLEAVES", "15"))
TAG = _os.environ.get("TAG", "")
print(f"AYAR: TRAIN_D={TRAIN_D} STEP_D={STEP_D} NLEAVES={NLEAVES}")


# ---------- gunes geometrisi ----------
def solar_elev(idx: pd.DatetimeIndex, lat=LAT, lon=LON) -> np.ndarray:
    doy = idx.dayofyear.to_numpy()
    hh = idx.hour.to_numpy() + idx.minute.to_numpy() / 60.0
    decl = 23.45 * np.sin(np.deg2rad(360.0 * (284 + doy) / 365.0))
    b = np.deg2rad(360.0 * (doy - 81) / 364.0)
    eot = 9.87 * np.sin(2 * b) - 7.53 * np.cos(b) - 1.5 * np.sin(b)
    solar_time = hh + (4.0 * (lon - 15.0 * 3) + eot) / 60.0     # TR = UTC+3
    ha = np.deg2rad(15.0 * (solar_time - 12.0))
    phi = np.deg2rad(lat); dec = np.deg2rad(decl)
    sin_el = np.sin(phi) * np.sin(dec) + np.cos(phi) * np.cos(dec) * np.cos(ha)
    return np.rad2deg(np.arcsin(np.clip(sin_el, -1, 1)))


def clear_sky_ghi(elev_deg: np.ndarray) -> np.ndarray:
    """Haurwitz clear-sky GHI (W/m2)."""
    z = np.deg2rad(90.0 - np.clip(elev_deg, -90, 90))
    cz = np.cos(z)
    g = np.where(cz > 0.01, 1098.0 * cz * np.exp(-0.059 / np.maximum(cz, 0.01)), 0.0)
    return np.clip(g, 0, None)


# ---------- veri ----------
sun = pd.read_parquet(RAW)["sun"].astype(float)
sun.index = pd.DatetimeIndex(sun.index).tz_localize(None)
sun = sun[~sun.index.duplicated(keep="last")].sort_index()
print(f"hedef: {len(sun)} saat | {sun.index.min().date()} -> {sun.index.max().date()}")

ghi2 = cloud2 = temp2 = wspd2 = rh2 = pr2 = None
ghi = cloud = temp = wspd = rh = pr = None
for nm, w in POP.items():
    p = os.path.join(LEAD, f"{nm}.parquet")
    if not os.path.exists(p):
        continue
    d = pd.read_parquet(p)
    d.index = pd.DatetimeIndex(d.index)
    if d.index.tz is not None:
        d.index = d.index.tz_localize(None)
    ww = w / sum(POP.values())
    for var, acc in (("shortwave_radiation", "ghi"), ("cloud_cover", "cloud"),
                     ("temperature_2m", "temp"), ("wind_speed_10m", "wspd")):
        for sfx in ("day1", "day2"):
            col = f"{var}_previous_{sfx}"
            if col not in d:
                continue
            s = d[col] * ww
            tgt_var = acc if sfx == "day1" else acc + "2"
            cur = locals().get(tgt_var)
            locals()[tgt_var] = s if cur is None else cur.add(s, fill_value=0)
    # locals() atamasi guvenilmez -> acik birikim
    break

# ACIK kurulum (yukaridaki dongu sadece varlik kontrolu icin); gercek birikim:
ghi = cloud = temp = wspd = None
ghi2 = cloud2 = temp2 = wspd2 = None
for nm, w in POP.items():
    p = os.path.join(LEAD, f"{nm}.parquet")
    if not os.path.exists(p):
        continue
    d = pd.read_parquet(p)
    d.index = pd.DatetimeIndex(d.index)
    if d.index.tz is not None:
        d.index = d.index.tz_localize(None)
    ww = w / sum(POP.values())
    pairs = [("shortwave_radiation", "ghi", "ghi2"), ("cloud_cover", "cloud", "cloud2"),
             ("temperature_2m", "temp", "temp2"), ("wind_speed_10m", "wspd", "wspd2"),
             ("relative_humidity_2m", "rh", "rh2"), ("precipitation", "pr", "pr2")]
    for var, a1, a2 in pairs:
        for sfx, tgt in (("day1", a1), ("day2", a2)):
            col = f"{var}_previous_{sfx}"
            if col not in d:
                continue
            s = d[col] * ww
            if tgt == "ghi":
                ghi = s if ghi is None else ghi.add(s, fill_value=0)
            elif tgt == "ghi2":
                ghi2 = s if ghi2 is None else ghi2.add(s, fill_value=0)
            elif tgt == "cloud":
                cloud = s if cloud is None else cloud.add(s, fill_value=0)
            elif tgt == "cloud2":
                cloud2 = s if cloud2 is None else cloud2.add(s, fill_value=0)
            elif tgt == "temp":
                temp = s if temp is None else temp.add(s, fill_value=0)
            elif tgt == "temp2":
                temp2 = s if temp2 is None else temp2.add(s, fill_value=0)
            elif tgt == "wspd":
                wspd = s if wspd is None else wspd.add(s, fill_value=0)
            elif tgt == "wspd2":
                wspd2 = s if wspd2 is None else wspd2.add(s, fill_value=0)
            elif tgt == "rh":
                rh = s if rh is None else rh.add(s, fill_value=0)
            elif tgt == "rh2":
                rh2 = s if rh2 is None else rh2.add(s, fill_value=0)
            elif tgt == "pr":
                pr = s if pr is None else pr.add(s, fill_value=0)
            else:
                pr2 = s if pr2 is None else pr2.add(s, fill_value=0)

# ---------- kapasite zarfi (causal) ----------
# 1 SAATLIK HIZA: EPiAS `sun` damgasi ile Open-Meteo GHI damgasi 1 saat kayik.
# Olculdu (2021-2026, 2160 saat): corr 0.875 -> 0.957, WMAPE 42.4 -> 26.6.
# Hizalama: weather serisini -1 kaydir (tahmin edilen fiziksel saat ile eslesir).
SHIFT_H = -1
for _nm in ("ghi", "ghi2", "cloud", "cloud2", "temp", "temp2", "wspd", "wspd2", "rh", "rh2", "pr", "pr2"):
    _s = globals().get(_nm)
    if _s is not None:
        globals()[_nm] = _s.shift(SHIFT_H)
print(f"hiza duzeltmesi: weather shift {SHIFT_H} saat")

C = sun.rolling(30 * 24, min_periods=15 * 24).quantile(0.99).shift(1)
C = C.clip(lower=50.0)
cf = (sun / C).clip(0, 1.5)
print(f"kapasite zarfi: {C.iloc[-1]:,.0f} MW (son) | {C.min():,.0f} -> {C.max():,.0f}")


GHI_CS30 = ghi.rolling(30 * 24, min_periods=15 * 24).quantile(0.98).shift(1)


def build(decision):
    cut = decision + pd.Timedelta(hours=8)
    if cut not in C.index:
        cut = C.index[C.index <= cut][-1] if (C.index <= cut).any() else None
    if cut is None:
        return None
    Cc = float(C.loc[cut])
    tgt = pd.date_range(decision + pd.Timedelta(days=1), periods=48, freq="h")
    el = solar_elev(tgt)
    cs = clear_sky_ghi(el)
    rows = []
    for i, t in enumerate(tgt):
        hz = 1 if t < decision + pd.Timedelta(days=2) else 2
        # T+2 icin lead-2 hava (previous_day2), T+1 icin lead-1
        g_s = ghi2 if hz == 2 else ghi
        c_s = cloud2 if hz == 2 else cloud
        t_s = temp2 if hz == 2 else temp
        w_s = wspd2 if hz == 2 else wspd
        rh_s = rh2 if hz == 2 else rh
        pr_s = pr2 if hz == 2 else pr
        g_i = float(g_s.get(t, np.nan)) if g_s is not None else np.nan
        cl = float(c_s.get(t, np.nan)) if c_s is not None else np.nan
        tp = float(t_s.get(t, np.nan)) if t_s is not None else np.nan
        ws = float(w_s.get(t, np.nan)) if w_s is not None else np.nan
        r = dict(elev=el[i], clear_sky=cs[i],
                 clear_idx=(g_i / cs[i]) if cs[i] > 20 else np.nan,
                 ghi=g_i, cloud=cl, temp=tp, wspd=ws,
                 hour=t.hour, month=t.month, hz=hz,
                 sin_h=np.sin(2 * np.pi * t.hour / 24), cos_h=np.cos(2 * np.pi * t.hour / 24),
                 sin_y=np.sin(2 * np.pi * t.dayofyear / 365), cos_y=np.cos(2 * np.pi * t.dayofyear / 365))
        for L in (48, 72, 168):
            r[f"cf_lag{L}"] = cf.get(t - pd.Timedelta(hours=L), np.nan)
        g7 = [ghi.get(t - pd.Timedelta(days=k), np.nan) for k in range(1, 8)]
        g7m = float(np.nanmean(g7)) if np.any(~np.isnan(g7)) else np.nan
        r["ghi_7d_same_hour"] = g7m
        r["ghi_ratio_7d"] = (g_i / g7m) if (g7m and g7m > 20) else np.nan
        cs30 = float(GHI_CS30.get(t, np.nan)) if GHI_CS30 is not None else np.nan
        r["csi_30d"] = (g_i / cs30) if (cs30 == cs30 and cs30 > 20 and g_i == g_i) else np.nan
        w7 = [cf.get(t - pd.Timedelta(days=k), np.nan) for k in range(1, 8)]
        r["cf_same_hour_7d"] = float(np.nanmean(w7)) if np.any(~np.isnan(w7)) else np.nan
        # --- FIZIK: panel sicakligi, isi kaybi, clear-sky indeksi, nem ---
        tcell = tp + (g_i / 800.0) * 25.0 if (tp == tp and g_i == g_i) else np.nan
        r["tcell"] = tcell
        r["heat_loss"] = (1.0 - 0.004 * max(0.0, tcell - 25.0)) if tcell == tcell else np.nan
        r["ghi_heat"] = (g_i * r["heat_loss"]) if (g_i == g_i and r["heat_loss"] == r["heat_loss"]) else np.nan
        rhv = float(rh_s.get(t, np.nan)) if rh_s is not None else np.nan
        prv = float(pr_s.get(t, np.nan)) if pr_s is not None else np.nan
        r["rh"] = rhv
        r["precip"] = prv
        r["is_rain"] = 1.0 if (prv == prv and prv > 0.1) else (0.0 if prv == prv else np.nan)
        r["C_cut"] = Cc
        rows.append((t, r))
    X = pd.DataFrame([r for _, r in rows], index=[t for t, _ in rows])
    return X


def wape(p, a):
    m = a.sum() > 0
    return float(np.abs(p - a)[m].sum() / a[m].sum() * 100)


def main():
    decs = pd.date_range(sun.index.min().normalize() + pd.Timedelta(days=35),
                         sun.index.max().normalize() - pd.Timedelta(days=2), freq="D")
    Xs, Ys, Hs, Cs = [], [], [], []
    for d in decs:
        X = build(d)
        if X is None or len(X) < 48:
            continue
        y = sun.reindex(X.index)
        if y.notna().sum() < 24:
            continue
        Xs.append(X); Ys.append((y / X["C_cut"]).clip(0, 1.5).to_numpy())
        Hs.append(X["hz"].to_numpy()); Cs.append(X["C_cut"].to_numpy())
    X = pd.concat(Xs); Y = np.concatenate(Ys); H = np.concatenate(Hs); CC = np.concatenate(Cs)
    FEAT = [c for c in X.columns if c != "C_cut"]
    print(f"ornek: {len(X)} | feature: {len(FEAT)}")

    days = pd.DatetimeIndex(sorted(set(X.index.normalize())))
    ev = days.min() + pd.Timedelta(days=TRAIN_D)
    res = {1: [], 2: []}
    d = ev.normalize()
    while d <= days.max():
        mtr = (X.index < d) & (X.index >= d - pd.Timedelta(days=TRAIN_D))
        mte = (X.index >= d) & (X.index < d + pd.Timedelta(days=STEP_D))
        if mtr.sum() < 1500 or mte.sum() < 100:
            d += pd.Timedelta(days=STEP_D); continue
        preds = []
        for obj, nest in (("l2", 1500), ("l1", 1500), ("huber", 1200)):
            m = lgb.LGBMRegressor(n_estimators=nest, learning_rate=0.03, num_leaves=15,
                                  max_depth=5, min_child_samples=50, subsample=0.85,
                                  colsample_bytree=0.85, reg_alpha=0.5, reg_lambda=1.0,
                                  objective=obj, random_state=42, verbose=-1)
            m.fit(X.loc[mtr, FEAT], Y[mtr], eval_set=[(X.loc[mte, FEAT], Y[mte])],
                  callbacks=[lgb.early_stopping(80, verbose=False)])
            preds.append(m.predict(X.loc[mte, FEAT]))
        cfh = np.clip(np.mean(preds, axis=0), 0, 1.2)
        mw = cfh * CC[mte]
        el = X.loc[mte, "elev"].to_numpy()
        mw = np.where(el <= 0, 0.0, mw)
        for hz in (1, 2):
            s = H[mte] == hz
            if s.sum() > 20:
                res[hz].append(pd.DataFrame({"p": mw[s], "a": sun.reindex(X.index[mte][s]).to_numpy(),
                                             "el": el[s]}))
        d += pd.Timedelta(days=STEP_D)

    print("\n=== GUNES (cf-normalize, cutoff-safe) ===")
    for hz in (1, 2):
        if not res[hz]:
            continue
        D = pd.concat(res[hz]).dropna()
        day = D[D.el > 0]
        nmae = float((D.p - D.a).abs().mean() / CC[:len(D)].mean() * 100) if len(D) else float("nan")
        print(f"  T+{hz}: tum WAPE {wape(D.p.to_numpy(), D.a.to_numpy()):5.2f}% | "
              f"gunduz WAPE {wape(day.p.to_numpy(), day.a.to_numpy()):5.2f}% | "
              f"nMAE {float((D.p-D.a).abs().mean()):.0f} MW | bias {float((D.p-D.a).mean()):+.0f} | n={len(D)}")
    print("\n  KARSILASTIRMA: fizik formulu baseline %26.6 tum gun / 42.4 gunduz")


if __name__ == "__main__":
    main()
