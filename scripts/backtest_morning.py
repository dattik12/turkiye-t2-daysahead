"""SABAH KOŞUSU backtest'i — lag kuralinin karar-anina gore dogru hali.

Iki kol, tek fark OZELLIK KURALI (model/hava/egitim ayni):
  OLD  : repo'nun bugunki T+2 lag seti (lag72+ sabit)
  NEW  : sabah-sozlesmesi lag kurali (lag L kullanilir ancak L >= h+16)

Yurutus: her ay yeniden egitim (expanding, 1000 gun pencere), hedef = yarin.
SIZINTI: her hedef saat icin yalnizca karar aninda (yarin-1 09:00) bilinen
degerler girer; ayni kural hem egitimde hem tahminde.

Kullanim: python -m scripts.backtest_morning [gun_sonu]
"""
from __future__ import annotations
import os, sys, time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config as C
from src import data as D
from src import models as M
from src import features as F
from src import morning as MO

RETRAIN_EVERY_DAYS = 30


def old_features(y, wnat, tgt, cal):
    """Repo'nun mevcut BASE_H2 lag/ozet kurali (karar-anindan bagimsiz >=72)."""
    if isinstance(wnat, pd.Series):
        wnat = wnat.to_frame()
    wnat = wnat[~wnat.index.duplicated(keep="last")].sort_index()
    y = y[~y.index.duplicated(keep="last")].sort_index()
    X = pd.DataFrame(index=tgt)
    for L in [72, 96, 120, 144, 168, 216, 336]:
        X[f"lag{L}"] = y.reindex(tgt - pd.Timedelta(hours=L)).to_numpy()
    sh = {L: y.reindex(tgt - pd.Timedelta(hours=L)).to_numpy() for L in range(72, 193, 24)}
    X["samehr_6d_72"] = pd.DataFrame(sh, index=tgt).mean(axis=1)
    r24 = y.rolling(24, min_periods=24).mean()
    r24max = y.rolling(24, min_periods=24).max()
    r168 = y.rolling(168, min_periods=168).mean()
    X["roll24_prev72"] = r24.shift(72).reindex(tgt).to_numpy()
    X["roll24max_prev72"] = r24max.shift(72).reindex(tgt).to_numpy()
    X["roll168_prev72"] = r168.shift(72).reindex(tgt).to_numpy()
    g = y.groupby(y.index.normalize())
    dm, dmax, dmin = g.mean(), g.max(), g.min()
    dn = tgt.normalize()
    for off, nm, src in [(2, "day_mean_D2", dm), (3, "day_mean_D2_3", dm),
                         (9, "day_mean_D2_9", dm), (2, "day_max_D2", dmax),
                         (2, "day_min_D2", dmin)]:
        X[nm] = (dn - pd.Timedelta(days=off)).map(src).to_numpy()
    X["day_delta_2"] = X["day_mean_D2"] - X["day_mean_D2_9"]
    for c in ["hour", "weekday", "month", "hour_sin", "hour_cos", "dow_sin", "dow_cos",
              "doy_sin", "doy_cos", "is_holiday", "is_official", "is_arife",
              "is_after_holiday", "is_holiday_tail", "is_bridge", "is_holiday_effect",
              "holiday_day_index", "days_after_holiday", "ramadan_day"]:
        if c in cal:
            X[c] = cal[c].reindex(tgt).to_numpy()
    for c in ["temperature_2m", "apparent_temperature", "relative_humidity_2m",
              "shortwave_radiation", "wind_speed_10m", "precipitation", "cloud_cover"]:
        if c in wnat:
            X[c] = wnat[c].reindex(tgt).to_numpy()
    if "temperature_2m" in wnat:
        t = wnat["temperature_2m"].reindex(tgt)
        X["CDD"] = (t - 26.0).clip(lower=0)
        X["HDD"] = (16.0 - t).clip(lower=0)
        X["temp_diff_1h"] = t.to_numpy() - wnat["temperature_2m"].reindex(
            tgt - pd.Timedelta(hours=1)).to_numpy()
        roll30 = wnat["temperature_2m"].rolling(30 * 24, min_periods=30 * 24).mean().shift(1)
        X["temp_anomaly_30d"] = (t - roll30.reindex(tgt)).to_numpy()
    return X


def main():
    end = sys.argv[1] if len(sys.argv) > 1 else None
    cons = D.load_or_create_consumption()
    nat, _ = D.load_or_create_weather()
    y = cons["rt_cons"].dropna()
    y.index = pd.to_datetime(y.index)
    nat.index = pd.to_datetime(nat.index)
    if end:
        y = y[y.index <= pd.Timestamp(end) + pd.Timedelta(days=1)]
    last_full = y.groupby(y.index.normalize()).size()
    last_full = last_full[last_full == 24].index.max()
    start = pd.Timestamp("2025-01-01")
    print(f"veri: {y.index.min()} -> {y.index.max()} | son tam gun {last_full.date()}")

    # hedefler: her karar gunu D icin D+1
    targets = pd.date_range(start + pd.Timedelta(days=1), last_full, freq="D")
    print(f"hedef gun sayisi: {len(targets)} | retrain her {RETRAIN_EVERY_DAYS} gun")

    rows = []
    t0 = time.time()
    for variant in ["OLD", "NEW"]:
        model = None
        feats = None
        fit_until = None
        for i, day in enumerate(targets):
            dec = day - pd.Timedelta(days=1)          # karar gunu
            tgt = pd.date_range(f"{day} 00:00", f"{day} 23:00", freq="h")
            if model is None or tgt[0] >= fit_until:
                tr_end = dec + pd.Timedelta(hours=8)  # karar ani: 09:00 -> 08:00'a kadar
                tr_start = tr_end - pd.Timedelta(days=C.TRAIN_DAYS)
                tr_idx = pd.date_range(tr_start, tr_end, freq="h")
                if variant == "NEW":
                    Xtr = MO.build_features(y, nat, tr_idx)
                else:
                    Xtr = old_features(y, nat, tr_idx, F.calendar_cols(tr_idx))
                # NOT: dropna YAPILMAZ. lag24 kural geregi h>8 saatlerde NaN;
                # LightGBM eksigi dogal isler. dropna, modeli yalnizca gece
                # saatlerinde (h<=8) egitip gunduze uygulatirdi -> felaket.
                Xtr = Xtr[y.reindex(Xtr.index).notna()]
                if len(Xtr) < 3000:
                    continue
                feats = [c for c in Xtr.columns]
                model, feats = M.fit(Xtr, y.reindex(Xtr.index), feats)
                fit_until = tgt[0] + pd.Timedelta(days=RETRAIN_EVERY_DAYS)
                print(f"  [{variant}] fit @ {tr_end.date()} n={len(Xtr)} "
                      f"({time.time()-t0:.0f}s)", flush=True)
            if variant == "NEW":
                X = MO.build_features(y, nat, tgt)
            else:
                X = old_features(y, nat, tgt, F.calendar_cols(tgt))
            p = pd.Series(np.asarray(M.predict(model, feats, X)), index=tgt)
            a = y.reindex(tgt)
            ok = p.notna() & a.notna()
            for h in range(24):
                m = ok & (tgt.hour == h)
                if m.sum():
                    rows.append((variant, day.date(), h,
                                 float((p[m] - a[m]).abs().mean()),
                                 float(a[m].mean())))
            if (i + 1) % 60 == 0:
                print(f"  [{variant}] {day.date()} ({i+1}/{len(targets)}) {time.time()-t0:.0f}s", flush=True)

    df = pd.DataFrame(rows, columns=["variant", "day", "hour", "ae", "act"])
    df.to_csv(os.path.join(C.RESULTS_DIR, "backtest_morning.csv"), index=False)
    print("\n=== OZET (MAPE, hedef = yarin) ===")
    for v in ["OLD", "NEW"]:
        s = df[df.variant == v]
        mape = (s.groupby(["day", "hour"]).ae.sum() / s.groupby(["day", "hour"]).act.sum()).mean()
        wmape = s.ae.sum() / s.act.sum() * 100
        print(f"{v}: MAPE %{mape:.3f}  WMAPE %{wmape:.3f}  n={len(s)}")
    piv = df.groupby(["variant", "hour"]).apply(
        lambda s: s.ae.sum() / s.act.sum() * 100, include_groups=False).unstack(0)
    piv["delta"] = piv["NEW"] - piv["OLD"]
    print("\nsaat bazli WMAPE:")
    print(piv.round(3).to_string())
    print(f"\ntoplam sure {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
