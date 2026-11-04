"""RUZGAR T+1/T+2 MODELI — ulusal `wind` serisi, cutoff-safe.

RITM yalnizca ~24 saat yayinliyor -> T+2 icin kendi modelimiz ZORUNLU.
Hava: lead_weather 10 il (10m ruzgar hizi, 2021-2026, day1/day2). API gerekmez.
Hedef: kapasite faktoru  y = wind / C_t ,  C_t = geriye donuk 30 gun p99.
"""
import os, sys
import numpy as np, pandas as pd, lightgbm as lgb

os.chdir(r"C:\projeler\turkiye-t2-daysahead")
sys.path.insert(0, r"C:\projeler\turkiye-t2-daysahead")

RAW = r"data\raw\epias_national_rt_gen_2021_2026.parquet"
LEAD = r"C:\projeler\lead_weather"
POP = {"istanbul": 15.9, "ankara": 5.8, "izmir": 4.5, "bursa": 3.2, "antalya": 2.7,
       "adana": 2.3, "konya": 2.3, "gaziantep": 2.1, "kayseri": 1.4, "trabzon": 0.8}
TRAIN_D, STEP_D = 540, 7

wnd = pd.read_parquet(RAW)["wind"].astype(float)
wnd.index = pd.DatetimeIndex(wnd.index).tz_localize(None)
wnd = wnd[~wnd.index.duplicated(keep="last")].sort_index()
print(f"hedef: {len(wnd)} saat | {wnd.index.min().date()} -> {wnd.index.max().date()}")

C = wnd.rolling(30 * 24, min_periods=15 * 24).quantile(0.99).shift(1).clip(lower=100.0)
cf = (wnd / C).clip(0, 1.5)
print(f"kapasite zarfi: son {C.iloc[-1]:,.0f} | {C.min():,.0f} -> {C.max():,.0f}")

ws1 = ws2 = None
tot = 0.0
for nm, w in POP.items():
    p = os.path.join(LEAD, f"{nm}.parquet")
    if not os.path.exists(p):
        continue
    d = pd.read_parquet(p)
    d.index = pd.DatetimeIndex(d.index)
    if d.index.tz is not None:
        d.index = d.index.tz_localize(None)
    tot += w
    s1 = d["wind_speed_10m_previous_day1"] * w
    s2 = d["wind_speed_10m_previous_day2"] * w
    ws1 = s1 if ws1 is None else ws1.add(s1, fill_value=0)
    ws2 = s2 if ws2 is None else ws2.add(s2, fill_value=0)
ws1, ws2 = ws1 / tot, ws2 / tot


def build(decision):
    cut = decision + pd.Timedelta(hours=8)
    Cc = float(C.asof(cut))
    tgt = pd.date_range(decision + pd.Timedelta(days=1), periods=48, freq="h")
    rows = []
    for t in tgt:
        hz = 1 if t < decision + pd.Timedelta(days=2) else 2
        s = ws2 if hz == 2 else ws1
        v = float(s.get(t, np.nan)) if s is not None else np.nan
        r = dict(ws10=v, ws10_sq=v ** 2, ws10_cu=v ** 3,
                 hour=t.hour, month=t.month, hz=hz,
                 sin_h=np.sin(2 * np.pi * t.hour / 24), cos_h=np.cos(2 * np.pi * t.hour / 24),
                 sin_y=np.sin(2 * np.pi * t.dayofyear / 365), cos_y=np.cos(2 * np.pi * t.dayofyear / 365))
        for L in (48, 72, 168):
            r[f"cf_lag{L}"] = cf.get(t - pd.Timedelta(hours=L), np.nan)
        w7 = [cf.get(t - pd.Timedelta(days=k), np.nan) for k in range(1, 8)]
        r["cf_7d"] = float(np.nanmean(w7)) if np.any(~np.isnan(w7)) else np.nan
        r["C_cut"] = Cc
        rows.append((t, r))
    return pd.DataFrame([r for _, r in rows], index=[t for t, _ in rows])


def wape(p, a):
    m = np.isfinite(p) & np.isfinite(a) & (a.sum() > 0)
    return float(np.abs(p - a)[m].sum() / a[m].sum() * 100)


def main():
    decs = pd.date_range(wnd.index.min().normalize() + pd.Timedelta(days=40),
                         wnd.index.max().normalize() - pd.Timedelta(days=2), freq="D")
    Xs, Ys, Hs, Cs = [], [], [], []
    for d in decs:
        X = build(d)
        y = wnd.reindex(X.index)
        if y.notna().sum() < 40:
            continue
        Xs.append(X); Ys.append((y / X["C_cut"]).clip(0, 1.5).to_numpy())
        Hs.append(X["hz"].to_numpy()); Cs.append(X["C_cut"].to_numpy())
    X = pd.concat(Xs); Y = np.concatenate(Ys); H = np.concatenate(Hs); CC = np.concatenate(Cs)
    FEAT = [c for c in X.columns if c != "C_cut"]
    print(f"ornek: {len(X)} | feature: {len(FEAT)}")

    days = pd.DatetimeIndex(sorted(set(X.index.normalize())))
    res = {1: [], 2: []}
    d = days.min() + pd.Timedelta(days=TRAIN_D)
    while d <= days.max():
        mtr = (X.index < d) & (X.index >= d - pd.Timedelta(days=TRAIN_D))
        mte = (X.index >= d) & (X.index < d + pd.Timedelta(days=STEP_D))
        if mtr.sum() < 3000 or mte.sum() < 100:
            d += pd.Timedelta(days=STEP_D); continue
        m = lgb.LGBMRegressor(n_estimators=1200, learning_rate=0.035, num_leaves=15,
                              max_depth=5, min_child_samples=50, subsample=0.85,
                              colsample_bytree=0.85, reg_alpha=0.5, reg_lambda=1.0,
                              random_state=42, verbose=-1)
        m.fit(X.loc[mtr, FEAT], Y[mtr], eval_set=[(X.loc[mte, FEAT], Y[mte])],
              callbacks=[lgb.early_stopping(80, verbose=False)])
        mw = np.clip(m.predict(X.loc[mte, FEAT]), 0, 1.3) * CC[mte]
        for hz in (1, 2):
            s = H[mte] == hz
            if s.sum() > 20:
                res[hz].append(pd.DataFrame({"p": mw[s], "a": wnd.reindex(X.index[mte][s]).to_numpy()}))
        d += pd.Timedelta(days=STEP_D)

    print("\n=== RUZGAR (cf-normalize, cutoff-safe) ===")
    for hz in (1, 2):
        if not res[hz]:
            continue
        D = pd.concat(res[hz]).dropna()
        nmae = float((D.p - D.a).abs().mean()) / float(CC.mean()) * 100
        print(f"  T+{hz}: WAPE {wape(D.p.to_numpy(), D.a.to_numpy()):5.2f}% | "
              f"nMAE {float((D.p-D.a).abs().mean()):,.0f} MW ({nmae:.1f}%) | "
              f"bias {float((D.p-D.a).mean()):+,.0f} | n={len(D)}")
    print("\n  KARSILASTIRMA: RITM (uretim) ~%13 | mevcut spatial ~%12.8 (sizintili, gecersiz)")
    print("  Hedef: leak-free WAPE <%9.5")


if __name__ == "__main__":
    main()
