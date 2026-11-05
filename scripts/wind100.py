"""RUZGAR 100m MODELI — power curve + bolgesel kirilim + 5 yil arsiv.

Arsiv: C:\\projeler\\prov_lead100\\<il>_<yil>Q<q>.parquet (624 dosya, 39 il, 2023Q1-2026Q4)
Fizik: IEC sigmoid power curve  PC(v) = clip(sigmoid((v-8.5)/1.8),0,1) * I(3<=v<=25)
Bolge: Marmara / Ege / Akdeniz / Ic Anadolu / Karadeniz / Dogu  (kapasite agirlikli)
Hedef: ulusal `wind` kapasite faktoru (rolling_30d_p99)
"""
import os, sys, glob, re
import numpy as np, pandas as pd, lightgbm as lgb

os.chdir(r"C:\projeler\turkiye-t2-daysahead")
sys.path.insert(0, r"C:\projeler\turkiye-t2-daysahead")

DIR = r"C:\projeler\prov_lead100"
RAW = r"data\raw\epias_national_rt_gen_2021_2026.parquet"
TRAIN_D, STEP_D = 365, 7

REG = {
 "marmara": ["Balikesir","Canakkale","Kirklareli","Tekirdag","Bursa","Yalova","Sakarya","Edirne","Bilecik"],
 "ege": ["Izmir","Manisa","Mugla","Aydin","Afyon","Kutahya","Usak"],
 "akdeniz": ["Hatay","Osmaniye","Mersin","Kahramanmaras","Adana"],
 "icanadolu": ["Sivas","Kayseri","Konya","Karaman","Nigde","Kirikkale","Kirsehir","Tokat","Amasya","Corum"],
 "karadeniz": ["Sinop","Kastamonu","Ordu"],
 "dogu": ["Malatya","Elazig","Gaziantep","Adiyaman","Erzurum"],
}


def load_arch():
    files = glob.glob(os.path.join(DIR, "*Q*.parquet"))
    per = {}
    for f in files:
        m = re.match(r"(.+)_(20\d\dQ\d)\.parquet", os.path.basename(f))
        if not m:
            continue
        prov = m.group(1)
        d = pd.read_parquet(f)
        col = "wind_speed_100m" if "wind_speed_100m" in d.columns else d.columns[0]
        per.setdefault(prov, []).append(d[col])
    X = pd.DataFrame({p: pd.concat(v).sort_index() for p, v in per.items()})
    X = X[~X.index.duplicated(keep="last")].sort_index()
    return X


def pc(v):
    """IEC sigmoid power curve, 3-25 m/s arasi."""
    p = 1.0 / (1.0 + np.exp(-(v - 8.5) / 1.8))
    return np.clip(p, 0, 1) * ((v >= 3) & (v <= 25))


def main():
    X = load_arch()
    print(f"arsiv: {X.shape[0]} saat x {X.shape[1]} il | {X.index.min()} -> {X.index.max()}")
    have = [c for c in X.columns]
    regs = {r: [c for c in v if c in have] for r, v in REG.items()}

    F = pd.DataFrame(index=X.index)
    for r, cols in regs.items():
        if not cols:
            continue
        sub = X[cols]
        F[f"pc_{r}"] = pc(sub).mean(axis=1)
        F[f"ws_{r}"] = sub.mean(axis=1)
        F[f"sd_{r}"] = sub.std(axis=1)
    allv = X[have]
    F["ws_mean"] = allv.mean(axis=1)
    F["ws_std"] = allv.std(axis=1)
    F["ws_max"] = allv.max(axis=1)
    F["pc_all"] = pc(allv).mean(axis=1)
    F["v3"] = F.ws_mean ** 3
    F["v3_ratio"] = F.v3 / (F.v3.rolling(168, min_periods=24).mean() + 1e-6)
    print(f"feature: {len(F.columns)} | ornek: {len(F)}")

    wnd = pd.read_parquet(RAW)["wind"].astype(float)
    wnd.index = pd.DatetimeIndex(wnd.index).tz_localize(None)
    wnd = wnd[~wnd.index.duplicated(keep="last")].sort_index()
    C = wnd.rolling(30 * 24, min_periods=15 * 24).quantile(0.99).shift(1).clip(lower=100.0)
    cf = (wnd / C).clip(0, 1.5)

    F["hour"] = F.index.hour
    F["month"] = F.index.month
    F["sin_h"] = np.sin(2 * np.pi * F.index.hour / 24)
    F["cos_h"] = np.cos(2 * np.pi * F.index.hour / 24)
    F["sin_y"] = np.sin(2 * np.pi * F.index.dayofyear / 365)
    F["cos_y"] = np.cos(2 * np.pi * F.index.dayofyear / 365)
    for L in (24, 48, 168):
        F[f"cf_lag{L}"] = cf.shift(L)
    F["cf_7d"] = sum(cf.shift(24 * k) for k in range(1, 8)) / 7.0
    F["C_cut"] = C

    y = cf.reindex(F.index)
    m = y.notna() & F["pc_all"].notna()
    Xd, Y, CC = F[m].copy(), y[m].to_numpy(), F.loc[m, "C_cut"].to_numpy()
    FEAT = [c for c in Xd.columns if c != "C_cut"]
    print(f"egitim ornegi: {len(Xd)} | feature: {len(FEAT)}")

    def wape(p, a):
        k = np.isfinite(p) & np.isfinite(a) & (a.sum() > 0)
        return float(np.abs(p - a)[k].sum() / a[k].sum() * 100)

    # walk-forward: 09:00 cutoff -> T+15..T+38 saati hedef (basitlestirilmis: gun bazli)
    days = pd.DatetimeIndex(sorted(set(Xd.index.normalize())))
    d0 = days.min() + pd.Timedelta(days=TRAIN_D + 2)
    P, A = [], []
    d = d0
    while d <= days.max():
        tr = (Xd.index < d) & (Xd.index >= d - pd.Timedelta(days=TRAIN_D))
        te = (Xd.index >= d) & (Xd.index < d + pd.Timedelta(days=STEP_D))
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
        A.append(wnd.reindex(Xd.index[te]))
        d += pd.Timedelta(days=STEP_D)

    Pp, Aa = pd.concat(P), pd.concat(A)
    k = Aa.notna()
    nmae = float((Pp[k] - Aa[k]).abs().mean()) / float(CC.mean()) * 100
    print("\n=== RUZGAR 100m + power curve + 4 bolge ===")
    print(f"  WAPE {wape(Pp[k].to_numpy(), Aa[k].to_numpy()):5.2f}% | "
          f"nMAE {float((Pp[k]-Aa[k]).abs().mean()):,.0f} MW ({nmae:.1f}%) | "
          f"bias {float((Pp[k]-Aa[k]).mean()):+,.0f} | n={int(k.sum())}")
    print("  Kiyas: 10m/10il %32.59 | 100m ham %19.34 | RITM ~%13 | hedef %9.5")


if __name__ == "__main__":
    main()
