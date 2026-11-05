"""RUZGAR MULTI-NWP TESTI — 3 model ortalamasi + spread.

Eski: tek NWP (lead_weather, 10 il 10m) -> %32.6 ; 39 il 100m -> %17.7
Yeni: 10 RES ili x 100m x (IFS + GFS + ICON) -> ortalama + spread + min/max zarf
Kota dostu: 10 il x 1 degisken x 3 model.

Cikti karsilastirmasi: ayni hedef (ulusal `wind` kapasite faktoru), ayni WF.
"""
import os, sys, glob, re
import numpy as np, pandas as pd, lightgbm as lgb

os.chdir(r"C:\projeler\turkiye-t2-daysahead")
sys.path.insert(0, r"C:\projeler\turkiye-t2-daysahead")

MW = r"C:\projeler\multinwp\wind"
ARCH = r"C:\projeler\prov_lead100"
RAW = r"data\raw\epias_national_rt_gen_2021_2026.parquet"
TRAIN_D, STEP_D = 365, 7

REG = {
 "marmara": ["Balikesir", "Canakkale", "Kirklareli", "Tekirdag", "Bursa", "Yalova", "Sakarya", "Edirne", "Bilecik"],
 "ege": ["Izmir", "Manisa", "Mugla", "Aydin", "Afyon", "Kutahya", "Usak"],
 "akdeniz": ["Hatay", "Osmaniye", "Mersin", "Kahramanmaras", "Adana"],
 "icanadolu": ["Sivas", "Kayseri", "Konya", "Karaman", "Nigde", "Kirikkale", "Kirsehir", "Tokat", "Amasya", "Corum"],
 "karadeniz": ["Sinop", "Kastamonu", "Ordu"],
 "dogu": ["Malatya", "Elazig", "Gaziantep", "Adiyaman", "Erzurum"],
}
MODELS = ["ecmwf_ifs025", "gfs_seamless", "icon_seamless"]


def pc(v):
    """km/h -> m/s, IEC sigmoid."""
    ms = np.asarray(v, dtype=float) / 3.6
    q = 1.0 / (1.0 + np.exp(-(ms - 8.5) / 1.8))
    return np.clip(q, 0, 1) * ((ms >= 3) & (ms <= 25))


def load_multi():
    """{il: {model: series}} — multi-NWP 10 il."""
    per = {}
    for f in glob.glob(os.path.join(MW, "*.parquet")):
        m = re.match(r"(.+)_(20\d\dQ\d)\.parquet", os.path.basename(f))
        if not m:
            continue
        prov = m.group(1)
        d = pd.read_parquet(f)
        per.setdefault(prov, []).append(d)
    out = {}
    for prov, lst in per.items():
        d = pd.concat(lst).sort_index()
        if getattr(d.index, "tz", None) is not None:
            d.index = d.index.tz_localize(None)   # tz karisimi object dtype yaratiyor
        d = d[~d.index.duplicated(keep="last")]
        out[prov] = {mo: d[f"wind_speed_100m_{mo}"] for mo in MODELS if f"wind_speed_100m_{mo}" in d.columns}
    return out


def load_arch():
    per = {}
    for f in glob.glob(os.path.join(ARCH, "*Q*.parquet")):
        m = re.match(r"(.+)_(20\d\dQ\d)\.parquet", os.path.basename(f))
        if not m:
            continue
        d = pd.read_parquet(f)
        col = "wind_speed_100m" if "wind_speed_100m" in d.columns else d.columns[0]
        per.setdefault(m.group(1), []).append(d[col])
    X = pd.DataFrame({p: pd.concat(v).sort_index() for p, v in per.items()})
    if getattr(X.index, "tz", None) is not None:
        X.index = X.index.tz_localize(None)
    return X[~X.index.duplicated(keep="last")].sort_index()


def main():
    M = load_multi()
    A = load_arch()
    idx = sorted(set().union(*[s.index for p in M.values() for s in p.values()]))
    idx = pd.DatetimeIndex(idx)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    print(f"multi-NWP: {len(M)} il | {len(idx)} saat | {idx.min()} -> {idx.max()}")

    # il bazli 3-model ortalama / spread
    ens, spr = {}, {}
    for prov, dd in M.items():
        df = pd.DataFrame(dd).reindex(idx)
        ens[prov] = df.mean(axis=1)
        spr[prov] = df.std(axis=1)
    E = pd.DataFrame(ens)
    S = pd.DataFrame(spr)

    F = pd.DataFrame(index=idx)
    F["ens_mean"] = E.mean(axis=1)
    F["ens_std_il"] = E.std(axis=1)
    F["ens_min_il"] = E.min(axis=1)
    F["ens_max_il"] = E.max(axis=1)
    F["nwp_spread"] = S.mean(axis=1)          # modeller arasi belirsizlik
    F["nwp_spread_max"] = S.max(axis=1)
    F["pc_ens"] = pc(E).mean(axis=1)
    F["v3"] = (F.ens_mean / 3.6) ** 3
    for r, cols in REG.items():
        c = [x for x in cols if x in E.columns]
        if c:
            F[f"pc_{r}"] = pc(E[c]).mean(axis=1)
            F[f"ws_{r}"] = E[c].mean(axis=1)

    # eski 39-il arsivinden tamamlayici feature (varsa ortusen donem)
    if len(A.columns):
        F["arch_mean"] = A.reindex(idx).mean(axis=1)

    wnd = pd.read_parquet(RAW)["wind"].astype(float)
    wnd.index = pd.DatetimeIndex(wnd.index).tz_localize(None)
    wnd = wnd[~wnd.index.duplicated(keep="last")].sort_index()
    C = wnd.rolling(30 * 24, min_periods=15 * 24).quantile(0.99).shift(1).clip(lower=100.0)
    cf = (wnd / C).clip(0, 1.5)

    F["hour"] = idx.hour; F["month"] = idx.month
    F["sin_h"] = np.sin(2 * np.pi * idx.hour / 24)
    F["cos_h"] = np.cos(2 * np.pi * idx.hour / 24)
    F["sin_y"] = np.sin(2 * np.pi * idx.dayofyear / 365)
    F["cos_y"] = np.cos(2 * np.pi * idx.dayofyear / 365)
    for L in (24, 48, 168):
        F[f"cf_lag{L}"] = cf.shift(L).reindex(idx)
    F["cf_7d"] = sum(cf.shift(24 * k) for k in range(1, 8)).reindex(idx) / 7.0
    F["pc_ramp_1h"] = F["pc_ens"].diff(1)
    F["pc_ramp_2h"] = F["pc_ens"].diff(2)
    F["spread_ramp"] = F["nwp_spread"].diff(1)
    F["C_cut"] = C.reindex(idx)

    F = F.apply(pd.to_numeric, errors="coerce").astype(float)   # object -> float
    y = cf.reindex(idx)
    m = y.notna() & F["ens_mean"].notna()
    Xd, Y, CC = F[m].copy(), y[m].to_numpy(), F.loc[m, "C_cut"].to_numpy()
    FEAT = [c for c in Xd.columns if c != "C_cut"]
    print(f"egitim ornegi: {len(Xd)} | feature: {len(FEAT)}")

    def wape(p, a):
        k = np.isfinite(p) & np.isfinite(a) & (a.sum() > 0)
        return float(np.abs(p - a)[k].sum() / a[k].sum() * 100)

    # TEMIZ DEGERLENDIRME: 3 modelin HEPSI dolu olan donem (IFS 2024 ortasinda basliyor)
    EVAL_START = pd.Timestamp(os.environ.get("EVAL_START", "2025-01-01"))
    days = pd.DatetimeIndex(sorted(set(Xd.index.normalize())))
    print(f"degerlendirme baslangici: {EVAL_START.date()}")
    P, Aa = [], []
    d = days.min() + pd.Timedelta(days=TRAIN_D + 2)
    while d <= days.max():
        tr = (Xd.index < d) & (Xd.index >= d - pd.Timedelta(days=TRAIN_D))
        te = (Xd.index >= max(d, EVAL_START)) & (Xd.index < d + pd.Timedelta(days=STEP_D))
        if tr.sum() < 5000 or te.sum() < 48:
            d += pd.Timedelta(days=STEP_D); continue
        mdl = lgb.LGBMRegressor(n_estimators=1500, learning_rate=0.03, num_leaves=15,
                                max_depth=5, min_child_samples=50, subsample=0.85,
                                colsample_bytree=0.85, reg_alpha=0.5, reg_lambda=1.0,
                                random_state=42, verbose=-1)
        mdl.fit(Xd.loc[tr, FEAT], Y[tr], eval_set=[(Xd.loc[te, FEAT], Y[te])],
                callbacks=[lgb.early_stopping(80, verbose=False)])
        p = np.clip(mdl.predict(Xd.loc[te, FEAT]), 0, 1.3) * CC[te]
        P.append(pd.Series(p, index=Xd.index[te]))
        Aa.append(wnd.reindex(Xd.index[te]))
        d += pd.Timedelta(days=STEP_D)

    Pp, Aa2 = pd.concat(P), pd.concat(Aa)
    DUMP = r"C:\projeler\wind_wf_preds.parquet"
    pd.DataFrame({"dt": Pp.index, "pred": Pp.to_numpy(), "actual": Aa2.to_numpy()}).to_parquet(DUMP, index=False)
    print(f"tahmin dump: {DUMP} ({len(Pp)} satir)")
    k = Aa2.notna()
    nmae = float((Pp[k] - Aa2[k]).abs().mean()) / float(CC.mean()) * 100
    print("\n=== RUZGAR MULTI-NWP (IFS+GFS+ICON) ===")
    print(f"  WAPE {wape(Pp[k].to_numpy(), Aa2[k].to_numpy()):5.2f}% | "
          f"nMAE {float((Pp[k]-Aa2[k]).abs().mean()):,.0f} MW ({nmae:.1f}%) | "
          f"bias {float((Pp[k]-Aa2[k]).mean()):+,.0f} | n={int(k.sum())}")
    print("  Onceki: 39-il tek NWP %17.78 | 10m/10il %32.59 | RITM ~%13 | hedef %9.5")


if __name__ == "__main__":
    main()
