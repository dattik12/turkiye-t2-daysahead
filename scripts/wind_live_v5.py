"""RUZGAR CANLI v5 — MULTI-NWP (IFS + GFS + ICON).

Backtest: tek NWP %17.78 -> multi-NWP %15.41 (-2.37 puan).
10 RES ili x wind_speed_100m x 3 model. Cikti: data/forecast/wind_live.parquet
status='ml_v5_multinwp'
"""
import os, sys, glob, re
import numpy as np, pandas as pd, lightgbm as lgb, requests

os.chdir(r"C:\projeler\turkiye-t2-daysahead")
sys.path.insert(0, r"C:\projeler\turkiye-t2-daysahead")

MW = r"C:\projeler\multinwp\wind"
RAW = r"data\raw\epias_national_rt_gen_2021_2026.parquet"
OUT = r"data\forecast\wind_live.parquet"
TRAIN_D = 365
MODELS = ["ecmwf_ifs025", "gfs_seamless", "icon_seamless"]

RES = {"Balikesir": (39.65, 27.89), "Izmir": (38.42, 27.14), "Canakkale": (40.15, 26.41),
       "Manisa": (38.62, 27.43), "Hatay": (36.40, 36.35), "Aydin": (37.85, 27.84),
       "Istanbul": (41.01, 28.98), "Kirklareli": (41.74, 27.22),
       "Afyon": (38.75, 30.54), "Kayseri": (38.73, 35.49)}

REG = {"marmara": ["Balikesir", "Canakkale", "Kirklareli", "Istanbul"],
       "ege": ["Izmir", "Manisa", "Aydin", "Afyon"],
       "akdeniz": ["Hatay"], "icanadolu": ["Kayseri"]}


def pc(v):
    ms = np.asarray(v, dtype=float) / 3.6
    q = 1.0 / (1.0 + np.exp(-(ms - 8.5) / 1.8))
    return np.clip(q, 0, 1) * ((ms >= 3) & (ms <= 25))


def load_multi():
    per = {}
    for f in glob.glob(os.path.join(MW, "*.parquet")):
        m = re.match(r"(.+)_(20\d\dQ\d)\.parquet", os.path.basename(f))
        if m:
            per.setdefault(m.group(1), []).append(pd.read_parquet(f))
    out = {}
    for prov, lst in per.items():
        d = pd.concat(lst).sort_index()
        if getattr(d.index, "tz", None) is not None:
            d.index = d.index.tz_localize(None)
        out[prov] = d[~d.index.duplicated(keep="last")]
    return out


def features(M, idx):
    ens, spr = {}, {}
    for prov, d in M.items():
        dd = d.reindex(idx)
        # DIKKAT: model adlari ALT CIKLI ("ecmwf_ifs025") -> split("_")[-1] YANLIS.
        cols = [c for c in dd.columns
                if c.replace("wind_speed_100m_", "") in MODELS]
        if not cols:
            continue
        ens[prov] = dd[cols].mean(axis=1); spr[prov] = dd[cols].std(axis=1)
    E = pd.DataFrame(ens).astype(float); S = pd.DataFrame(spr).astype(float)
    F = pd.DataFrame(index=idx)
    F["ens_mean"] = E.mean(axis=1); F["ens_std_il"] = E.std(axis=1)
    F["ens_min_il"] = E.min(axis=1); F["ens_max_il"] = E.max(axis=1)
    F["nwp_spread"] = S.mean(axis=1); F["nwp_spread_max"] = S.max(axis=1)
    F["pc_ens"] = pc(E).mean(axis=1); F["v3"] = (F.ens_mean / 3.6) ** 3
    for r, cols in REG.items():
        c = [x for x in cols if x in E.columns]
        if c:
            F[f"pc_{r}"] = pc(E[c]).mean(axis=1); F[f"ws_{r}"] = E[c].mean(axis=1)
    nh = pd.DatetimeIndex(idx)
    F["hour"] = nh.hour; F["month"] = nh.month
    F["sin_h"] = np.sin(2*np.pi*nh.hour/24); F["cos_h"] = np.cos(2*np.pi*nh.hour/24)
    F["sin_y"] = np.sin(2*np.pi*nh.dayofyear/365); F["cos_y"] = np.cos(2*np.pi*nh.dayofyear/365)
    F["pc_ramp_1h"] = F["pc_ens"].diff(1); F["pc_ramp_2h"] = F["pc_ens"].diff(2)
    F["spread_ramp"] = F["nwp_spread"].diff(1)
    return F


def live_multi(days=4):
    names = list(RES)
    r = requests.get("https://api.open-meteo.com/v1/forecast",
                     params=dict(latitude=",".join(str(RES[n][0]) for n in names),
                                 longitude=",".join(str(RES[n][1]) for n in names),
                                 hourly="wind_speed_100m", models=",".join(MODELS),
                                 forecast_days=days, timezone="Europe/Istanbul"), timeout=180)
    r.raise_for_status()
    js = r.json(); blocks = js if isinstance(js, list) else [js]
    out = {}
    for n, b in zip(names, blocks):
        h = pd.DataFrame(b["hourly"]); h["time"] = pd.to_datetime(h["time"])
        out[n] = h.set_index("time")
    return out


def main():
    M = load_multi()
    wnd = pd.read_parquet(RAW)["wind"].astype(float)
    wnd.index = pd.DatetimeIndex(wnd.index).tz_localize(None)
    wnd = wnd[~wnd.index.duplicated(keep="last")].sort_index()
    C = wnd.rolling(30*24, min_periods=15*24).quantile(0.99).shift(1).clip(lower=100.0)
    cf = (wnd / C).clip(0, 1.5)

    decision = pd.Timestamp.now().normalize(); cut = decision + pd.Timedelta(hours=8)
    Cc = float(C.asof(cut))
    tr = pd.date_range(cut - pd.Timedelta(days=TRAIN_D), cut - pd.Timedelta(hours=1), freq="h")
    Ftr = features(M, tr)
    for L in (24, 48, 168):
        Ftr[f"cf_lag{L}"] = cf.shift(L).reindex(tr)
    Ftr["cf_7d"] = sum(cf.shift(24*k) for k in range(1, 8)).reindex(tr) / 7.0
    last_gen = float(wnd.asof(cut))
    hh = np.array([(t - cut).total_seconds()/3600.0 for t in tr])
    Ftr["decay"] = np.exp(-np.clip(hh, 0, None)/24.0)
    Ftr["last_gen_mw"] = last_gen * Ftr["decay"].to_numpy()
    ytr = cf.reindex(tr)
    m = ytr.notna() & Ftr["ens_mean"].notna()
    Ftr, ytr = Ftr[m], ytr[m]
    mdl = lgb.LGBMRegressor(n_estimators=1500, learning_rate=0.03, num_leaves=15,
                            max_depth=5, min_child_samples=50, subsample=0.85,
                            colsample_bytree=0.85, reg_alpha=0.5, reg_lambda=1.0,
                            random_state=42, verbose=-1)
    mdl.fit(Ftr, ytr)
    print(f"egitim: {len(Ftr)} | C_cut {Cc:,.0f} | cutoff uretim {last_gen:,.0f} MW")

    tgt = pd.date_range(decision + pd.Timedelta(days=1), periods=48, freq="h")
    Lw = live_multi(4)
    print(f"canli multi-NWP: {len(Lw)} il x 3 model")
    Ft = features(Lw, tgt)
    for Lg in (24, 48, 168):
        Ft[f"cf_lag{Lg}"] = cf.shift(Lg).reindex(tgt)
    Ft["cf_7d"] = sum(cf.shift(24*k) for k in range(1, 8)).reindex(tgt) / 7.0
    h2 = np.array([(t - cut).total_seconds()/3600.0 for t in tgt])
    Ft["decay"] = np.exp(-np.clip(h2, 0, None)/24.0)
    Ft["last_gen_mw"] = last_gen * Ft["decay"].to_numpy()
    Ft = Ft.reindex(columns=mdl.feature_name_)
    p = np.clip(mdl.predict(Ft), 0, 1.3) * Cc
    out = pd.DataFrame({"datetime": tgt, "wind_mw": np.clip(p, 0, None),
                        "horizon": ["T+1"]*24 + ["T+2"]*24, "status": "ml_v5_multinwp"})
    out.to_parquet(OUT, index=False)
    print(f"YAZILDI | T+1 tepe {out.wind_mw[:24].max():,.0f} | T+2 tepe {out.wind_mw[24:].max():,.0f}")


if __name__ == "__main__":
    main()
