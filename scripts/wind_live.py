"""RUZGAR CANLI — egit + 48 saat tahmin (T+1 ve T+2).

Model: wind100 ile ayni (power curve + 6 bolge + cf-normalize), TRAIN_D=365.
Canli hava: api.open-meteo.com/v1/forecast, 39 il batch, wind_speed_100m.
Cikti: data/forecast/wind_live.parquet (datetime, wind_mw, horizon, status)
"""
import os, sys, glob, re
import numpy as np, pandas as pd, lightgbm as lgb, requests

os.chdir(r"C:\projeler\turkiye-t2-daysahead")
sys.path.insert(0, r"C:\projeler\turkiye-t2-daysahead")

DIR = r"C:\projeler\prov_lead100"
RAW = r"data\raw\epias_national_rt_gen_2021_2026.parquet"
OUT = r"data\forecast\wind_live.parquet"
TRAIN_D = 365

REG = {
 "marmara": ["Balikesir","Canakkale","Kirklareli","Tekirdag","Bursa","Yalova","Sakarya","Edirne","Bilecik"],
 "ege": ["Izmir","Manisa","Mugla","Aydin","Afyon","Kutahya","Usak"],
 "akdeniz": ["Hatay","Osmaniye","Mersin","Kahramanmaras","Adana"],
 "icanadolu": ["Sivas","Kayseri","Konya","Karaman","Nigde","Kirikkale","Kirsehir","Tokat","Amasya","Corum"],
 "karadeniz": ["Sinop","Kastamonu","Ordu"],
 "dogu": ["Malatya","Elazig","Gaziantep","Adiyaman","Erzurum"],
}
COORD = {
 "Balikesir": (39.65,27.89), "Canakkale": (40.15,26.41), "Izmir": (38.42,27.14),
 "Manisa": (38.62,27.43), "Kirklareli": (41.74,27.22), "Tekirdag": (40.98,27.51),
 "Bursa": (40.19,29.06), "Yalova": (40.65,29.27), "Sakarya": (40.77,30.40),
 "Edirne": (41.68,26.56), "Bilecik": (40.15,29.98), "Mugla": (37.21,28.36),
 "Aydin": (37.85,27.84), "Afyon": (38.75,30.54), "Kutahya": (39.42,29.98),
 "Usak": (38.68,29.41), "Hatay": (36.40,36.35), "Osmaniye": (37.07,36.25),
 "Mersin": (36.80,34.63), "Kahramanmaras": (37.58,36.93), "Adana": (37.00,35.32),
 "Sivas": (39.75,37.02), "Kayseri": (38.73,35.49), "Konya": (37.87,32.48),
 "Karaman": (37.18,33.22), "Nigde": (37.97,34.68), "Kirikkale": (39.85,33.52),
 "Kirsehir": (39.15,34.16), "Tokat": (40.32,36.55), "Amasya": (40.65,35.83),
 "Sinop": (42.03,35.15), "Corum": (40.55,34.95), "Kastamonu": (41.38,33.78),
 "Ordu": (40.98,37.88), "Malatya": (38.35,38.31), "Elazig": (38.67,39.22),
 "Gaziantep": (37.06,37.38), "Adiyaman": (37.76,38.28), "Erzurum": (39.90,41.27)}


def pc(v):
    p = 1.0 / (1.0 + np.exp(-(v - 8.5) / 1.8))
    return np.clip(p, 0, 1) * ((v >= 3) & (v <= 25))


def load_arch():
    per = {}
    for f in glob.glob(os.path.join(DIR, "*Q*.parquet")):
        m = re.match(r"(.+)_(20\d\dQ\d)\.parquet", os.path.basename(f))
        if not m:
            continue
        d = pd.read_parquet(f)
        col = "wind_speed_100m" if "wind_speed_100m" in d.columns else d.columns[0]
        per.setdefault(m.group(1), []).append(d[col])
    X = pd.DataFrame({p: pd.concat(v).sort_index() for p, v in per.items()})
    return X[~X.index.duplicated(keep="last")].sort_index()


def build_feats(X, idx):
    F = pd.DataFrame(index=idx)
    for r, cols in REG.items():
        c = [x for x in cols if x in X.columns]
        if not c:
            continue
        sub = X[c].reindex(idx)
        F[f"pc_{r}"] = pc(sub).mean(axis=1)
        F[f"ws_{r}"] = sub.mean(axis=1)
        F[f"sd_{r}"] = sub.std(axis=1)
    allv = X.reindex(idx)
    F["ws_mean"] = allv.mean(axis=1)
    F["ws_std"] = allv.std(axis=1)
    F["ws_max"] = allv.max(axis=1)
    F["pc_all"] = pc(allv).mean(axis=1)
    F["v3"] = F.ws_mean ** 3
    nh = pd.DatetimeIndex(idx)
    F["hour"] = nh.hour; F["month"] = nh.month
    F["sin_h"] = np.sin(2 * np.pi * nh.hour / 24)
    F["cos_h"] = np.cos(2 * np.pi * nh.hour / 24)
    F["sin_y"] = np.sin(2 * np.pi * nh.dayofyear / 365)
    F["cos_y"] = np.cos(2 * np.pi * nh.dayofyear / 365)
    return F


def live_100m(days=4):
    names = list(COORD)
    lat = ",".join(str(COORD[n][0]) for n in names)
    lon = ",".join(str(COORD[n][1]) for n in names)
    r = requests.get("https://api.open-meteo.com/v1/forecast",
                     params=dict(latitude=lat, longitude=lon, hourly="wind_speed_100m",
                                 forecast_days=days, timezone="Europe/Istanbul"), timeout=180)
    r.raise_for_status()
    js = r.json()
    blocks = js if isinstance(js, list) else [js]
    out = {}
    for n, b in zip(names, blocks):
        h = pd.DataFrame(b["hourly"])
        h["time"] = pd.to_datetime(h["time"])
        out[n] = h.set_index("time")["wind_speed_100m"]
    return pd.DataFrame(out)


def main():
    X = load_arch()
    wnd = pd.read_parquet(RAW)["wind"].astype(float)
    wnd.index = pd.DatetimeIndex(wnd.index).tz_localize(None)
    wnd = wnd[~wnd.index.duplicated(keep="last")].sort_index()
    C = wnd.rolling(30 * 24, min_periods=15 * 24).quantile(0.99).shift(1).clip(lower=100.0)
    cf = (wnd / C).clip(0, 1.5)

    decision = pd.Timestamp.now().normalize()
    cut = decision + pd.Timedelta(hours=8)
    Cc = float(C.asof(cut))
    tr_idx = pd.date_range(cut - pd.Timedelta(days=TRAIN_D), cut - pd.Timedelta(hours=1), freq="h")
    Ftr = build_feats(X, tr_idx)
    Ftr["cf_lag24"] = cf.shift(24).reindex(tr_idx)
    Ftr["cf_lag48"] = cf.shift(48).reindex(tr_idx)
    Ftr["cf_lag168"] = cf.shift(168).reindex(tr_idx)
    Ftr["cf_7d"] = sum(cf.shift(24 * k) for k in range(1, 8)).reindex(tr_idx) / 7.0
    ytr = cf.reindex(tr_idx)
    m = ytr.notna() & Ftr[["pc_all"]].notna().all(axis=1)
    mdl = lgb.LGBMRegressor(n_estimators=1200, learning_rate=0.03, num_leaves=15,
                            max_depth=5, min_child_samples=50, subsample=0.85,
                            colsample_bytree=0.85, reg_alpha=0.5, reg_lambda=1.0,
                            random_state=42, verbose=-1)
    mdl.fit(Ftr[m], ytr[m])
    print(f"egitim: {int(m.sum())} | C_cut {Cc:,.0f} MW")

    tgt = pd.date_range(decision + pd.Timedelta(days=1), periods=48, freq="h")
    lw = live_100m(4)
    print(f"canli 100m: {lw.index.min()} -> {lw.index.max()} | tepe {lw.max().max():.1f} m/s")
    Ft = build_feats(lw, tgt)
    # canli cf lag: arsivden bilinen son degerler
    Ft["cf_lag24"] = cf.shift(24).reindex(tgt)
    Ft["cf_lag48"] = cf.shift(48).reindex(tgt)
    Ft["cf_lag168"] = cf.shift(168).reindex(tgt)
    Ft["cf_7d"] = sum(cf.shift(24 * k) for k in range(1, 8)).reindex(tgt) / 7.0
    Ft = Ft[mdl.feature_name_]
    p = np.clip(mdl.predict(Ft.fillna(Ft.median())), 0, 1.3) * Cc

    out = pd.DataFrame({"datetime": tgt, "wind_mw": np.clip(p, 0, None),
                        "horizon": ["T+1"] * 24 + ["T+2"] * 24, "status": "ml100m"})
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    out.to_parquet(OUT, index=False)
    print(f"YAZILDI {OUT} | T+1 tepe {out.wind_mw[:24].max():,.0f} | T+2 tepe {out.wind_mw[24:].max():,.0f}")


if __name__ == "__main__":
    main()
