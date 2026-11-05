"""RUZGAR HATA YAPISI — wind_wf_preds.parquet uzerinden (YENI EGITIM YOK).

Boyutlar:
  1. Saat bazli bias/MAE (gece/gunduz)
  2. Havza: bolgesel 100m hiz aktivitesine gore MAE (Marmara/Ege/Akdeniz/Ic Anadolu)
  3. Rejim: NWP spread dusuk/orta/yuksek -> hata kac kat?
  4. Cut-out: 100m hiz > 25 m/s gun/leri
  5. Rampa: 3 saatlik |dPC| buyuk gunler
"""
import os, sys, glob, re
import numpy as np, pandas as pd

os.chdir(r"C:\projeler\turkiye-t2-daysahead")
sys.path.insert(0, r"C:\projeler\turkiye-t2-daysahead")

DUMP = r"C:\projeler\wind_wf_preds.parquet"
MW = r"C:\projeler\multinwp\wind"
MODELS = ["ecmwf_ifs025", "gfs_seamless", "icon_seamless"]
REG = {
 "marmara": ["Balikesir", "Canakkale", "Kirklareli", "Istanbul"],
 "ege": ["Izmir", "Manisa", "Aydin", "Afyon"],
 "akdeniz": ["Hatay"], "icanadolu": ["Kayseri"],
}


def load_multi():
    per = {}
    for f in glob.glob(os.path.join(MW, "*.parquet")):
        m = re.match(r"(.+)_(20\d\dQ\d)\.parquet", os.path.basename(f))
        if m:
            per.setdefault(m.group(1), []).append(pd.read_parquet(f))
    out = {}
    for p, lst in per.items():
        d = pd.concat(lst).sort_index()
        if getattr(d.index, "tz", None) is not None:
            d.index = d.index.tz_localize(None)
        out[p] = d[~d.index.duplicated(keep="last")]
    return out


def main():
    D = pd.read_parquet(DUMP)
    D["dt"] = pd.to_datetime(D["dt"])
    D = D.dropna(subset=["pred", "actual"])
    D["err"] = D.pred - D.actual
    D["ae"] = D.err.abs()
    print(f"dump: {len(D)} saat | {D.dt.min()} -> {D.dt.max()}")
    print(f"genel: MAE {D.ae.mean():,.0f} MW | bias {D.err.mean():+,.0f} MW")

    # 1) SAAT
    D["hr"] = D.dt.dt.hour
    h = D.groupby("hr").agg(mae=("ae", "mean"), bias=("err", "mean"))
    print("\n=== 1. SAAT BAZLI ===")
    for hr in range(24):
        r = h.loc[hr]
        print(f"  {hr:02d}: MAE {r.mae:5,.0f} | bias {r.bias:+6,.0f}")
    gece = D[D.hr.isin([0,1,2,3,4,5,22,23])]
    gunduz = D[D.hr.isin([9,10,11,12,13,14,15,16])]
    print(f"  gece MAE {gece.ae.mean():,.0f} | gunduz MAE {gunduz.ae.mean():,.0f} "
          f"| oran {gece.ae.mean()/max(gunduz.ae.mean(),1):.2f}x")

    # 2) HAVZA
    M = load_multi()
    idx = pd.DatetimeIndex(D.dt)
    print("\n=== 2. HAVZA (bolgesel hiz aktivitesine gore) ===")
    for reg, cols in REG.items():
        c = [x for x in cols if x in M]
        if not c:
            continue
        v = pd.DataFrame({x: M[x].reindex(idx).mean(axis=1) for x in c}).mean(axis=1)
        D[f"ws_{reg}"] = v.to_numpy()
    for reg in REG:
        col = f"ws_{reg}"
        if col not in D or D[col].notna().sum() < 100:
            continue
        q = D[col].quantile([0.33, 0.67])
        lo = D[D[col] <= q.iloc[0]]
        hi = D[D[col] >= q.iloc[1]]
        print(f"  {reg:10s}: dusuk hiz MAE {lo.ae.mean():5,.0f} | yuksek hiz MAE {hi.ae.mean():5,.0f} "
              f"| fark {hi.ae.mean()-lo.ae.mean():+6,.0f} MW")

    # 3) NWP SPREAD REJIMI
    print("\n=== 3. NWP SPREAD REJIMI ===")
    sp = []
    for p in M:
        cc = [f"wind_speed_100m_{m}" for m in MODELS if f"wind_speed_100m_{m}" in M[p].columns]
        if len(cc) >= 2:
            sp.append(M[p].reindex(idx)[cc].std(axis=1))
    if sp:
        S = pd.concat(sp, axis=1).mean(axis=1).to_numpy()
        if len(S) != len(D):
            S = np.resize(S, len(D))
        D["spread"] = S
        q = D["spread"].quantile([0.33, 0.67])
        for nm, sub in (("dusuk", D[D.spread <= q.iloc[0]]),
                        ("orta", D[(D.spread > q.iloc[0]) & (D.spread < q.iloc[1])]),
                        ("yuksek", D[D.spread >= q.iloc[1]])):
            print(f"  {nm:7s}: MAE {sub.ae.mean():5,.0f} MW | n={len(sub)}")
        r = D[D.spread >= q.iloc[1]].ae.mean() / max(D[D.spread <= q.iloc[0]].ae.mean(), 1)
        print(f"  yuksek/dusuk oran: {r:.2f}x")

    # 4) CUT-OUT
    print("\n=== 4. CUT-OUT (>25 m/s) ===")
    mx = []
    for p in M:
        cc = [f"wind_speed_100m_{m}" for m in MODELS if f"wind_speed_100m_{m}" in M[p].columns]
        if cc:
            mx.append(M[p].reindex(idx)[cc].max(axis=1))
    if mx:
        MX = pd.concat(mx, axis=1).max(axis=1).to_numpy() / 3.6
        if len(MX) != len(D):
            MX = np.resize(MX, len(D))
        D["ws_max_ms"] = MX
        co = D[D.ws_max_ms > 25]
        print(f"  cut-out saati: {len(co)} / {len(D)} ({100*len(co)/len(D):.1f}%) | "
              f"MAE {co.ae.mean():,.0f} MW" if len(co) else "  cut-out yok")
        if len(co):
            days = sorted(set(pd.to_datetime(co.dt).dt.date))
            print(f"  gun sayisi: {len(days)} | ornek: {days[:5]}")

    # 5) RAMPA
    print("\n=== 5. 3 SAATLIK RAMPA ===")
    D = D.sort_values("dt").reset_index(drop=True)
    D["ramp3"] = D.actual.diff(3).abs()
    q = D["ramp3"].quantile(0.9)
    big = D[D.ramp3 >= q]
    small = D[D.ramp3 < D.ramp3.quantile(0.5)]
    print(f"  buyuk rampa (>p90 {q:,.0f} MW): MAE {big.ae.mean():,.0f} | n={len(big)}")
    print(f"  kucuk rampa (<p50): MAE {small.ae.mean():,.0f} | n={len(small)}")
    print(f"  oran: {big.ae.mean()/max(small.ae.mean(),1):.2f}x")


if __name__ == "__main__":
    main()
