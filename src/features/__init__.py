"""Feature katmani.
Iki blok:
  A) v4.3 bayram/takvim paketi (elle ayarli, generalize ediyor)
  B) v5.2 surekli "manyak" eklemeler: daylight_fraction, same_hour_median, spatial std,
     wet-bulb, temp anomaly, segment inter.
Ensemble icin en iyi iki model: BASE (A) ve CONT (A+B). Training == inference (leak-free).
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd

from .. import config as C

# ----------------------------------------------------------- takvim/bayram ----
HOLIDAY_DATES = {
    "2024-04-10","2024-04-11","2024-04-12","2024-06-16","2024-06-17","2024-06-18","2024-06-19",
    "2025-03-30","2025-03-31","2025-04-01","2025-06-06","2025-06-07","2025-06-08","2025-06-09",
    "2026-03-19","2026-03-20","2026-03-21","2026-05-27","2026-05-28","2026-05-29","2026-05-30",
    "2027-03-09","2027-03-10","2027-03-11","2027-05-17","2027-05-18","2027-05-19","2027-05-20",
    "2028-02-26","2028-02-27","2028-02-28","2028-05-05","2028-05-06","2028-05-07","2028-05-08",
}
OFFICIAL = {
    "2024-01-01","2024-04-23","2024-05-01","2024-05-19","2024-07-15","2024-08-30","2024-10-29",
    "2025-01-01","2025-04-23","2025-05-01","2025-05-19","2025-07-15","2025-08-30","2025-10-29",
    "2026-01-01","2026-04-23","2026-05-01","2026-05-19","2026-07-15","2026-08-30","2026-10-29",
    "2027-01-01","2027-04-23","2027-05-01","2027-05-19","2027-07-15","2027-08-30","2027-10-29",
    "2028-01-01","2028-04-23","2028-05-01","2028-05-19","2028-07-15","2028-08-30","2028-10-29",
}
RAMADAN = {
    "2024": ("2024-03-11", "2024-04-09"),
    "2025": ("2025-03-01", "2025-03-30"),
    "2026": ("2026-02-18", "2026-03-19"),
    "2027": ("2027-02-08", "2027-03-09"),
    "2028": ("2028-01-29", "2028-02-26"),
}


def calendar_cols(idx: pd.DatetimeIndex) -> pd.DataFrame:
    d = pd.DataFrame(index=idx)
    d["hour"] = idx.hour
    d["weekday"] = idx.dayofweek
    d["month"] = idx.month
    d["dayofyear"] = idx.dayofyear
    d["hour_sin"] = np.sin(2 * np.pi * d["hour"] / 24)
    d["hour_cos"] = np.cos(2 * np.pi * d["hour"] / 24)
    d["dow_sin"] = np.sin(2 * np.pi * d["weekday"] / 7)
    d["dow_cos"] = np.cos(2 * np.pi * d["weekday"] / 7)
    d["doy_sin"] = np.sin(2 * np.pi * d["dayofyear"] / 365.25)
    d["doy_cos"] = np.cos(2 * np.pi * d["dayofyear"] / 365.25)
    ds = pd.Series(idx).dt.date
    rel = pd.to_datetime(list(HOLIDAY_DATES)).date
    off = pd.to_datetime(list(OFFICIAL)).date
    rel_set, off_set = set(rel), set(off)
    arife_set = set((pd.to_datetime(list(HOLIDAY_DATES)) - pd.Timedelta(days=1)).date)
    after_set = set((pd.to_datetime(list(HOLIDAY_DATES)) + pd.Timedelta(days=1)).date)
    # EDA tail: bayram block end +1 and +2 still holiday-like (2026-03-23, 2025-04-03 leak)
    tail_days = getattr(C, "HOLIDAY_TAIL_DAYS", 2)
    tail_set = set()
    for _hd in pd.to_datetime(list(HOLIDAY_DATES)).date:
        for k in range(2, 2+tail_days):
            cand = (pd.Timestamp(_hd) + pd.Timedelta(days=k)).date()
            # if not already holiday/arife/after, mark as tail
            if cand not in rel_set and cand not in arife_set and cand not in after_set:
                tail_set.add(cand)
    # bridge: single workday between holiday block and weekend (EDA 6.1)
    bridge_set = set()
    hol_all = rel_set | off_set | arife_set  # arife is work-effectively holiday
    for _d in pd.date_range("2024-01-01", "2029-01-01", freq="D"):
        dd = _d.date()
        if dd in hol_all or dd.weekday() >= 5:
            continue
        prev = (_d - pd.Timedelta(days=1)).date()
        nxt = (_d + pd.Timedelta(days=1)).date()
        if (prev in hol_all and nxt.weekday() >= 5) or (prev.weekday() >= 5 and nxt in hol_all):
            bridge_set.add(dd)
    d["is_holiday"] = ds.isin(rel_set).astype(np.int8).to_numpy()
    d["is_official"] = ds.isin(off_set).astype(np.int8).to_numpy()
    d["is_arife"] = ds.isin(arife_set).astype(np.int8).to_numpy()
    d["is_after_holiday"] = ds.isin(after_set).astype(np.int8).to_numpy()
    d["is_holiday_tail"] = ds.isin(tail_set).astype(np.int8).to_numpy()
    d["is_bridge"] = ds.isin(bridge_set).astype(np.int8).to_numpy()
    # combined soft signal: any holiday-effect day (bayram/arife/after/tail/bridge/official)
    d["is_holiday_effect"] = ((d["is_holiday"] | d["is_arife"] | d["is_after_holiday"] | d["is_holiday_tail"] | d["is_bridge"] | d["is_official"]) ).astype(np.int8).to_numpy()
    d["prev_week_holiday"] = ds.map(lambda x: (x - pd.Timedelta(days=7)) in (rel_set | off_set)).astype(np.int8).to_numpy()
    d["prev_day_holiday"] = ds.map(lambda x: (x - pd.Timedelta(days=1)) in (rel_set | off_set)).astype(np.int8).to_numpy()
    d["next_day_holiday"] = ds.map(lambda x: (x + pd.Timedelta(days=1)) in (rel_set | off_set)).astype(np.int8).to_numpy()
    sorted_rel = sorted(rel_set)
    def dist(x):
        if x in rel_set:
            return 0
        prev = max([y for y in sorted_rel if y < x], default=None)
        nxt = min([y for y in sorted_rel if y > x], default=None)
        dp = (prev - x).days if prev else -99
        dn = (nxt - x).days if nxt else 99
        return dp if abs(dp) <= abs(dn) else dn
    d["days_to_holiday"] = ds.map(dist).to_numpy()
    d["is_ramadan"] = ds.map(
        lambda x: any(str(x.year) in RAMADAN and RAMADAN[str(x.year)][0] <= x.isoformat() <= RAMADAN[str(x.year)][1]
                      for _ in [0])).astype(np.int8).to_numpy()
    return d


# ------------------------------------------------------------- precompute ----
def precompute(cons: pd.Series) -> pd.DataFrame:
    """Yuk gecmisi lag/rolling/gunluk ozet + robust medyan (vektorize, once hesaplanir)."""
    y = cons
    P = pd.DataFrame(index=y.index)
    for lg in [24, 48, 72, 96, 120, 144, 168, 336]:
        P[f"lag{lg}"] = y.shift(lg)
    cols = {f"sh{h}": y.shift(h) for h in range(24, 192 + 1, 24)}
    same = pd.concat(cols, axis=1)
    P["samehr_7d_48"] = same[["sh48", "sh72", "sh96", "sh120", "sh144", "sh168", "sh192"]].mean(axis=1)
    P["samehr_7d_24"] = same.mean(axis=1)
    P["samehr_median_3d"] = pd.concat([y.shift(h) for h in [24, 48, 72]], axis=1).median(axis=1)
    P["samehr_median_7d"] = pd.concat([y.shift(h) for h in range(24, 168 + 1, 24)], axis=1).median(axis=1)
    r24 = y.rolling(24, min_periods=24).mean()
    r24max = y.rolling(24, min_periods=24).max()
    r168 = y.rolling(168, min_periods=168).mean()
    P["roll24_prev1"] = r24.shift(1); P["roll24max_prev1"] = r24max.shift(1); P["roll168_prev1"] = r168.shift(1)
    P["roll24_prev24"] = r24.shift(24); P["roll24max_prev24"] = r24max.shift(24); P["roll168_prev24"] = r168.shift(24)
    g = y.groupby(y.index.normalize())
    dm, dmax, dmin = g.mean(), g.max(), g.min()

    def attach(src, name, off):
        tgt = y.index.normalize() - pd.Timedelta(days=off)
        P[name] = tgt.map(lambda dte: src.get(dte)).to_numpy()
    for off, nm, sr in [(1, "day_mean_D1", dm), (2, "day_mean_D1_2", dm), (8, "day_mean_D1_8", dm),
                        (1, "day_max_D1", dmax), (1, "day_min_D1", dmin),
                        (2, "day_mean_D2", dm), (3, "day_mean_D2_3", dm), (9, "day_mean_D2_9", dm),
                        (2, "day_max_D2", dmax), (2, "day_min_D2", dmin)]:
        attach(sr, nm, off)
    P["day_delta_1"] = P["day_mean_D1"] - P["day_mean_D1_8"]
    P["day_delta_2"] = P["day_mean_D2"] - P["day_mean_D2_9"]
    return P


def lep_rel_feature(lp: pd.Series, S: pd.Series, idx) -> np.ndarray:
    """Scale-free LEP orani (H1): lep(gun(t), ayni saat) / samehr_7d_24(t-48s).
    Payda her zaman tamamlanmis gunlere bakar -> nedensel ve NaN'siz.
    Is 17:00 TR'de kostugu icin lep(gun(t)) kararda yayimlanmis olur.
    Eksik LEP saatleri notr 1.0 ile dolar (plan=normal varsayimi)."""
    idx = pd.DatetimeIndex(idx)
    num = lp.reindex(idx).to_numpy()
    den = S.reindex(idx - pd.Timedelta(hours=48)).to_numpy()
    rel = num / den
    return np.where(np.isfinite(rel), rel, 1.0)


BASE_H1 = ["lag24", "lag48", "lag72", "lag168", "lag336", "samehr_7d_24",
           "roll24_prev1", "roll24max_prev1", "roll168_prev1",
           "day_mean_D1", "day_max_D1", "day_min_D1", "day_mean_D1_2", "day_mean_D1_8", "day_delta_1"]
BASE_H2 = ["lag48", "lag72", "lag96", "lag120", "lag144", "lag168", "lag336", "samehr_7d_48",
           "roll24_prev24", "roll24max_prev24", "roll168_prev24",
           "day_mean_D2", "day_max_D2", "day_min_D2", "day_mean_D2_3", "day_mean_D2_9", "day_delta_2"]
CONTRA = ["samehr_median_3d", "samehr_median_7d"]  # v5.2'ye eklenen surekli (yuk-side) feature'lar


# --------------------------------------------------------- hava manyaklar ----
def daylight_fraction_series(start: str, end: str, cities: list = None) -> pd.Series:
    """Gunes ufkun ustunde orani; sehirler pop-agirlikli (basit NOAA yaklasimi)."""
    cities = cities or C.CITIES
    idx = pd.date_range(start, end, freq="h")
    yday = idx.dayofyear.to_numpy()
    hour = idx.hour.to_numpy() + idx.minute.to_numpy() / 60.0
    frac = np.zeros(len(idx))
    total = 0.0
    for c in cities:
        decl = np.deg2rad(-23.44 * np.cos(np.deg2rad(360.0 / 365.0 * (yday + 10))))
        latr = np.deg2rad(c["lat"])
        H = np.deg2rad((hour - 12.0) * 15.0)
        cosz = np.sin(latr) * np.sin(decl) + np.cos(latr) * np.cos(decl) * np.cos(H)
        frac += (cosz > 0) * c["pop"]
        total += c["pop"]
    return pd.Series(frac / total, index=idx, name="daylight_fraction")


def weather_dynamics(wnat: pd.DataFrame) -> pd.DataFrame:
    """Surekli hava dinamikleri (hedef saat lisansinda veri)."""
    w = wnat
    temp = w["temperature_2m"]
    W = pd.DataFrame(index=w.index)
    W["temp_diff_1h"] = temp - temp.shift(1)
    W["temp_diff_3h"] = temp - temp.shift(3)
    dm = temp.groupby(w.index.normalize()).max()
    dn = temp.groupby(w.index.normalize()).min()
    # her saat kendi gununun (max-min) degerini alir; NaN kalmaz (v5.2 ile parite)
    W["daily_temp_range"] = (dm - dn).reindex(w.index.normalize()).to_numpy()
    roll30 = temp.rolling(30 * 24, min_periods=30 * 24).mean().shift(1)
    W["temp_anomaly_30d"] = (temp - roll30).fillna(0)
    T = temp.to_numpy()
    RH = w["relative_humidity_2m"].to_numpy()
    W["wet_bulb"] = T * np.arctan(0.151977 * np.sqrt(RH + 8.313659)) + np.arctan(T + RH) \
        - np.arctan(RH - 1.676331) + 0.00391838 * RH ** 1.5 * np.arctan(0.023101 * RH) - 4.686035
    W["humidex_proxy"] = T + 5 / 9 * (7e-5 * (RH / 100 * 6.105 * np.exp(17.27 * T / (237.7 + T))) * 100 - 10)
    W["first_heat_wave"] = ((temp > 30) & (temp.shift(1) > 28)).fillna(0).astype(np.int8)
    return W


def spatial_city_frame(cities_meta, start: str, end: str):
    """(daylight icin) sehir meta listesi -> (lats, lons, weights) uyumlu doner."""
    return cities_meta


def make_row(P: pd.DataFrame, wnat: pd.DataFrame, wdyn: pd.DataFrame, dayfrac: pd.Series,
             cw: pd.DataFrame, seg_urban: pd.Series, seg_ind: pd.Series,
             idx: pd.DatetimeIndex, horizon: int, d1_pred: pd.Series | None = None,
             cont: bool = True) -> pd.DataFrame:
    """Karma row builder. cont=True -> BASE+CONT ticks: v5.2; cont=False -> BASE only: v4.3.
    Dikkat: training ve inference icin BIREBIR ayni fonksiyon kullanilir -> leak yok."""
    X = calendar_cols(idx)
    pv = P.reindex(idx)
    base = BASE_H1 if horizon == 1 else BASE_H2
    for c in (base + CONTRA if cont else base):
        X[c] = pv[c].to_numpy() if c in pv.columns else np.nan
    X["daylight_fraction"] = dayfrac.reindex(idx).to_numpy()
    wn = wnat.reindex(idx)
    for v in C.OM_VARS + ["HDD", "CDD"]:
        if v in wn.columns:
            X[v] = wn[v].to_numpy()
    X["temp2"] = X["temperature_2m"] ** 2
    h = X["hour"]
    X["temp_hour"] = X["temperature_2m"] * np.cos(2 * np.pi * h / 24)
    if cont:   # MANYAK feature'lar yalnizca deneysel (cont=True) — regulasyon gerekir, uretim kullanmaz
        wd = wdyn.reindex(idx)
        for c in wd.columns:
            X[c] = wd[c].to_numpy()
        cwi = cw.reindex(idx)
        X["temp_spread_maxmin"] = (cwi.max(axis=1) - cwi.min(axis=1)).to_numpy()
        X["temp_std_10c"] = cwi.std(axis=1).to_numpy()
        X["urban_temp"] = seg_urban.reindex(idx).to_numpy()
        X["industrial_temp"] = seg_ind.reindex(idx).to_numpy()
        X["commercial_CDD_x_workhour"] = X["CDD"] * ((h >= 9) & (h <= 18)).astype(int)
        X["industrial_CDD_x_workday"] = X["CDD"] * (X["weekday"] < 5).astype(int)
    if horizon == 2 and d1_pred is not None:
        X["pred_d1"] = pd.Series(idx, index=idx).map(lambda t: d1_pred.get(t - pd.Timedelta(hours=24))).to_numpy()
    return X
