"""GUNES CANLI — egit + 48 saat tahmin (T+1 ve T+2).

Model: kapasite faktoru (sun / rolling_30d_p99), LGBM regulezize, 540 gun.
Hiza: weather shift -1 (olculdu: corr 0.828 vs 0.677).
Canli hava: api.open-meteo.com/v1/forecast (bugun+3 gun), 10 il nufus agirlikli.
Cikti: data/forecast/solar_live.parquet  (datetime, solar_mw, horizon, status)
"""
import os, sys, json
import numpy as np, pandas as pd, lightgbm as lgb, requests

os.chdir(r"C:\projeler\turkiye-t2-daysahead")
sys.path.insert(0, r"C:\projeler\turkiye-t2-daysahead")

RAW = r"data\raw\epias_national_rt_gen_2021_2026.parquet"
LEAD = r"C:\projeler\lead_weather"
OUT = r"data\forecast\solar_live.parquet"
# GES KURULU GUC AGIRLIGI (backtest: nufus %14.43 -> GES %13.01 gunduz WAPE)
POP = {"konya": 26.0, "ankara": 13.0, "kayseri": 11.0, "antalya": 10.0, "izmir": 8.0,
       "adana": 7.0, "gaziantep": 7.0, "bursa": 5.0, "istanbul": 3.0, "trabzon": 1.0}
COORD = {"istanbul": (41.01, 28.98), "ankara": (39.93, 32.86), "izmir": (38.42, 27.14),
         "bursa": (40.19, 29.06), "antalya": (36.90, 30.70), "adana": (37.00, 35.32),
         "konya": (37.87, 32.48), "gaziantep": (37.06, 37.38), "kayseri": (38.73, 35.49),
         "trabzon": (41.00, 39.72)}
LAT, LON = 39.0, 35.0
TRAIN_D = 540
SHIFT_H = -1


def solar_elev(idx, lat=LAT, lon=LON):
    doy = idx.dayofyear.to_numpy()
    hh = idx.hour.to_numpy() + idx.minute.to_numpy() / 60.0
    decl = 23.45 * np.sin(np.deg2rad(360.0 * (284 + doy) / 365.0))
    b = np.deg2rad(360.0 * (doy - 81) / 364.0)
    eot = 9.87 * np.sin(2 * b) - 7.53 * np.cos(b) - 1.5 * np.sin(b)
    solar_time = hh + (4.0 * (lon - 45.0) + eot) / 60.0
    ha = np.deg2rad(15.0 * (solar_time - 12.0))
    phi = np.deg2rad(lat); dec = np.deg2rad(decl)
    sin_el = np.sin(phi) * np.sin(dec) + np.cos(phi) * np.cos(dec) * np.cos(ha)
    return np.rad2deg(np.arcsin(np.clip(sin_el, -1, 1)))


def clear_sky_ghi(el):
    z = np.deg2rad(90.0 - np.clip(el, -90, 90))
    cz = np.cos(z)
    g = np.where(cz > 0.01, 1098.0 * cz * np.exp(-0.059 / np.maximum(cz, 0.01)), 0.0)
    return np.clip(g, 0, None)


def series(prefix, y1, y2):
    tot = 0.0
    acc = None
    for nm, w in POP.items():
        p = os.path.join(LEAD, f"{nm}.parquet")
        if not os.path.exists(p):
            continue
        d = pd.read_parquet(p)
        d.index = pd.DatetimeIndex(d.index)
        if d.index.tz is not None:
            d.index = d.index.tz_localize(None)
        col = prefix
        if col not in d:
            continue
        tot += w
        s = d[col] * w
        acc = s if acc is None else acc.add(s, fill_value=0)
    return (acc / tot).shift(SHIFT_H) if acc is not None else None


def live_weather(decision, days=3):
    """Canli 48-72 saat tahmini, 10 il nufus agirlikli."""
    lat = ",".join(str(COORD[n][0]) for n in POP)
    lon = ",".join(str(COORD[n][1]) for n in POP)
    hv = "shortwave_radiation,cloud_cover,temperature_2m,wind_speed_10m"
    r = requests.get("https://api.open-meteo.com/v1/forecast",
                     params=dict(latitude=lat, longitude=lon, hourly=hv,
                                 forecast_days=max(days, 3),
                                 timezone="Europe/Istanbul"), timeout=120)
    r.raise_for_status()
    blocks = r.json() if isinstance(r.json(), list) else [r.json()]
    wt = 0.0
    accs = {}
    for nm, b in zip(POP, blocks):
        w = POP[nm]; wt += w
        h = pd.DataFrame(b["hourly"])
        h["time"] = pd.to_datetime(h["time"])
        h = h.set_index("time")
        for v in hv.split(","):
            s = h[v] * w
            accs[v] = s if v not in accs else accs[v].add(s, fill_value=0)
    out = pd.DataFrame({v: accs[v] / wt for v in accs})
    return out.shift(SHIFT_H)


def main():
    sun = pd.read_parquet(RAW)["sun"].astype(float)
    sun.index = pd.DatetimeIndex(sun.index).tz_localize(None)
    sun = sun[~sun.index.duplicated(keep="last")].sort_index()

    g1 = series("shortwave_radiation_previous_day1", "previous_day1", None)
    g2 = series("shortwave_radiation_previous_day2", "previous_day2", None)
    c1 = series("cloud_cover_previous_day1", "previous_day1", None)
    t1 = series("temperature_2m_previous_day1", "previous_day1", None)
    w1 = series("wind_speed_10m_previous_day1", "previous_day1", None)

    C = sun.rolling(30 * 24, min_periods=15 * 24).quantile(0.99).shift(1).clip(lower=50.0)
    cf = (sun / C).clip(0, 1.5)
    g7 = sum(g1.shift(24 * k) for k in range(1, 8)) / 7.0

    def feats(idx, gh, cl, tp, ws):
        el = solar_elev(idx)
        cs = clear_sky_ghi(el)
        X = pd.DataFrame(index=idx)
        X["elev"] = el; X["clear_sky"] = cs
        gv = gh.reindex(idx).to_numpy()
        X["clear_idx"] = np.where(cs > 20, gv / np.maximum(cs, 1), np.nan)
        X["ghi"] = gv
        X["cloud"] = cl.reindex(idx).to_numpy()
        X["temp"] = tp.reindex(idx).to_numpy()
        X["wspd"] = ws.reindex(idx).to_numpy()
        X["hour"] = idx.hour; X["month"] = idx.month; X["hz"] = 1
        X["sin_h"] = np.sin(2 * np.pi * idx.hour / 24); X["cos_h"] = np.cos(2 * np.pi * idx.hour / 24)
        X["sin_y"] = np.sin(2 * np.pi * idx.dayofyear / 365); X["cos_y"] = np.cos(2 * np.pi * idx.dayofyear / 365)
        for L in (48, 72, 168):
            X[f"cf_lag{L}"] = [cf.get(t - pd.Timedelta(hours=L), np.nan) for t in idx]
        X["cf_7d"] = [float(np.nanmean([cf.get(t - pd.Timedelta(days=k), np.nan) for k in range(1, 8)]))
                      for t in idx]
        gs = g7.reindex(idx).to_numpy()
        X["ghi_7d"] = gs
        X["ghi_ratio_7d"] = np.where(gs > 20, gv / gs, np.nan)
        return X

    decision = pd.Timestamp.now().normalize()
    cut = decision + pd.Timedelta(hours=8)
    Cc = float(C.asof(cut))
    tr_idx = pd.date_range(cut - pd.Timedelta(days=TRAIN_D), cut - pd.Timedelta(hours=1), freq="h")
    Xtr = feats(tr_idx, g1, c1, t1, w1)
    ytr = (sun.reindex(tr_idx) / Cc).clip(0, 1.5)
    m = ytr.notna() & Xtr.notna().any(axis=1)
    Xtr, ytr = Xtr[m], ytr[m]
    print(f"egitim: {len(Xtr)} | C_cut {Cc:,.0f} MW")

    # ENSEMBLE L2 + L1 + Huber (backtest: 15.44 -> 15.07 gunduz WAPE)
    mdls = []
    for obj in ("l2", "l1", "huber"):
        m = lgb.LGBMRegressor(n_estimators=1500, learning_rate=0.03, num_leaves=15,
                              max_depth=5, min_child_samples=50, subsample=0.85,
                              colsample_bytree=0.85, reg_alpha=0.5, reg_lambda=1.0,
                              objective=obj, random_state=42, verbose=-1)
        m.fit(Xtr, ytr)
        mdls.append(m)

    tgt = pd.date_range(decision + pd.Timedelta(days=1), periods=48, freq="h")
    lw = live_weather(decision, 3)
    print(f"canli hava: {lw.index.min()} -> {lw.index.max()} | GHI tepe {lw.shortwave_radiation.max():.0f}")
    X = feats(tgt, lw["shortwave_radiation"], lw["cloud_cover"],
              lw["temperature_2m"], lw["wind_speed_10m"])
    p = np.clip(np.mean([m.predict(X) for m in mdls], axis=0), 0, 1.2) * Cc
    p = np.where(X["elev"].to_numpy() <= 0, 0.0, p)

    out = pd.DataFrame({"datetime": tgt, "solar_mw": np.clip(p, 0, None),
                        "horizon": ["T+1"] * 24 + ["T+2"] * 24, "status": "ml_v4_ges"})
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    out.to_parquet(OUT, index=False)
    print(f"YAZILDI {OUT} | T+1 tepe {out.solar_mw[:24].max():,.0f} | T+2 tepe {out.solar_mw[24:].max():,.0f}")


if __name__ == "__main__":
    main()
