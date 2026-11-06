"""SABAH KOŞUSU — yarimin 24 saatlik tahmini (tek urun sozlesmesi).

Sozlesme:
  KOSU      : her gun 09:00 TR
  KARAR ANI : bugun 09:00 (son tamamlanmis saat 08:00)
  URETIM    : yarinin 24 saati (D+1 00:00 .. 23:00)
  LAG KURALI: hedef saat h icin bilinen en taze saat = h + 16 saat oncei
              -> lag L ancak L >= h + 16 ise kullanilabilir
  HAVA      : IFS 00z
  LEP       : sabah yayimlanmamis olur -> devre disi

Eski T+2 modeli lag'i hedefe gore >=72'ye sabitliyordu; bu, h<=32 olan her
saatte lag48'i (ve h<=8'de lag24'u) bosuna atiyordu. Bu modul lag'i KARAR
ANINA gore dogru sekilde hesaplar.
"""
from __future__ import annotations
from pathlib import Path
import os
import numpy as np
import pandas as pd


def _use_exposure() -> bool:
    """CDH/HDH maruziyet bloklari (L1 deneyi). Env ile acilir: USE_EXPOSURE=1.
    720 gun A/B: -0.055 pp (tek LGBM). Toplulukla dogrulanana kadar varsayilan KAPALI."""
    return os.environ.get("USE_EXPOSURE", "0") == "1"


def _runlen(s: pd.Series) -> pd.Series:
    """Ardisik 1'lerin uzunlugu (gun bazli seri)."""
    out, run = [], 0
    for v in s.to_numpy():
        run = run + 1 if v else 0
        out.append(run)
    return pd.Series(out, index=s.index)

from . import config as C

DECISION_HOUR = 9          # kosu saati (TR)
LAST_KNOWN_HOUR = DECISION_HOUR - 1   # = 8: karar aninda bilinen son TAM saat
KNOWN_BASE_H = DECISION_HOUR + 7      # = 16: yalnizca T+1 icin minimum lag (geriye uyum)
ALWAYS_LAGS = [48, 72, 96, 120, 144, 168, 336]   # T+1'de her h icin guvenli
FRESH_LAGS = [24]                                # sadece h <= 8 icin guvenli


def known_cutoff(target_idx: pd.DatetimeIndex,
                 decision_date=None) -> pd.DatetimeIndex:
    """Her hedef saat icin karar aninda bilinen EN SON tam saat.

    Cutoff HEDEF TARIHTEN DEGIL, karar gununden turetilir: D 08:00.
    decision_date verilmezse T+1 varsayilir (hedef gunun bir onceki gunu) —
    bu geriye uyum icindir; T+2 cagiranlar ACIKCA gecmelidir.
    """
    idx = pd.DatetimeIndex(target_idx)
    if decision_date is None:
        decision_date = idx.normalize().min() - pd.Timedelta(days=1)
    d = pd.Timestamp(decision_date).normalize() + pd.Timedelta(hours=LAST_KNOWN_HOUR)
    return pd.DatetimeIndex([d] * len(idx), name=idx.name)


def _lag_known(L: int, h: np.ndarray, horizons=None) -> np.ndarray:
    """lag L, hedef saat h icin karar aninda biliniyor mu?

    horizons = hedef gun - karar gunu (T+1 -> 1, T+2 -> 2).
    Esik = horizons*24 + h - LAST_KNOWN_HOUR  (T+1 icin h+16'ya esit).
    """
    if horizons is None:
        horizons = np.ones(len(h))
    return L >= (np.asarray(horizons, dtype=float) * 24 + h - LAST_KNOWN_HOUR)


def build_features(y: pd.Series, wnat: pd.DataFrame, target_idx: pd.DatetimeIndex,
                   cal: pd.DataFrame | None = None, decision_date=None) -> pd.DataFrame:
    """Hedef saatlere gore ozellik tablosu. Kaynak verideki her deger
    karar aninda (bugun 09:00) bilinen tek bir saate baglanir."""
    target_idx = pd.DatetimeIndex(target_idx)
    if isinstance(wnat, pd.Series):
        wnat = wnat.to_frame()
    wnat = wnat[~wnat.index.duplicated(keep="last")].sort_index()
    y = y[~y.index.duplicated(keep="last")].sort_index()
    h = target_idx.hour.to_numpy()
    if decision_date is None:
        decision_date = target_idx.normalize().min() - pd.Timedelta(days=1)
    _D0 = pd.Timestamp(decision_date).normalize()
    H = ((target_idx.normalize() - _D0).days).to_numpy().astype(float)
    anchor = known_cutoff(target_idx, decision_date)   # karar gunu 08:00, ufuktan BAGIMSIZ
    F = pd.DataFrame(index=target_idx)

    # --- taze lag'lar (hedefin kendi saat-eslisi, karar-anina gore guvenli) ---
    for L in ALWAYS_LAGS:
        v = y.reindex(target_idx - pd.Timedelta(hours=L)).to_numpy()
        F[f"lag{L}"] = np.where(_lag_known(L, h, H), v, np.nan)
    for L in FRESH_LAGS:
        v = y.reindex(target_idx - pd.Timedelta(hours=L)).to_numpy()
        F[f"lag{L}"] = np.where(_lag_known(L, h, H), v, np.nan)

    # --- ayni-saat ortalamasi: yalnizca BILINEN gunler ---
    shifts = list(range(24, 193, 24))
    M = pd.DataFrame({f"s{L}": y.reindex(target_idx - pd.Timedelta(hours=L)).to_numpy()
                      for L in shifts}, index=target_idx)
    known = pd.DataFrame({f"s{L}": _lag_known(L, h, H) for L in shifts}, index=target_idx)
    Mv = M.where(known)
    F["samehr_known_mean"] = Mv.mean(axis=1)
    F["samehr_known_median"] = Mv.median(axis=1)
    F["samehr_known_n"] = known.sum(axis=1).astype(np.int8)

    # --- karar anina bagli rolling ozetler (bilinen pencere) ---
    r24 = y.rolling(24, min_periods=24).mean()
    r24max = y.rolling(24, min_periods=24).max()
    r168 = y.rolling(168, min_periods=168).mean()
    r9 = y.rolling(9, min_periods=9).mean()
    F["roll24_at_anchor"] = r24.reindex(anchor).to_numpy()
    F["roll24max_at_anchor"] = r24max.reindex(anchor).to_numpy()
    F["roll168_at_anchor"] = r168.reindex(anchor).to_numpy()
    F["roll9_today"] = r9.reindex(anchor).to_numpy()   # bugun 00-08 ortalamasi

    # --- gunluk ozetler (dun / onceden) ---
    g = y.groupby(y.index.normalize())
    dm, dmax, dmin = g.mean(), g.max(), g.min()
    dnorm = anchor.normalize()
    for off, nm, src in [(1, "day_mean_D1", dm), (1, "day_max_D1", dmax),
                         (1, "day_min_D1", dmin), (2, "day_mean_D2", dm),
                         (8, "day_mean_D1_8", dm)]:
        F[nm] = (dnorm - pd.Timedelta(days=off)).map(src).to_numpy()
    F["day_delta"] = F["day_mean_D1"] - F["day_mean_D1_8"]
    F["day_mean_today_partial"] = F["roll9_today"]

    # --- takvim ---
    if cal is None:
        from .features import calendar_cols
        cal = calendar_cols(target_idx)
    for c in ["hour", "weekday", "month", "hour_sin", "hour_cos", "dow_sin", "dow_cos",
              "doy_sin", "doy_cos", "is_holiday", "is_official", "is_arife",
              "is_after_holiday", "is_holiday_tail", "is_bridge", "is_holiday_effect",
              "holiday_day_index", "days_after_holiday", "ramadan_day"]:
        if c in cal:
            F[c] = cal[c].reindex(target_idx).to_numpy()

    # --- hava (hedef saatin IFS tahmini) ---
    for c in ["temperature_2m", "apparent_temperature", "relative_humidity_2m",
              "shortwave_radiation", "wind_speed_10m", "precipitation", "cloud_cover"]:
        if c in wnat:
            F[c] = wnat[c].reindex(target_idx).to_numpy()
    if "temperature_2m" in wnat:
        t = wnat["temperature_2m"].reindex(target_idx)
        F["CDD"] = (t - 26.0).clip(lower=0)
        F["HDD"] = (16.0 - t).clip(lower=0)
        F["temp_diff_1h"] = t.to_numpy() - wnat["temperature_2m"].reindex(
            target_idx - pd.Timedelta(hours=1)).to_numpy()
        roll30 = wnat["temperature_2m"].rolling(30 * 24, min_periods=30 * 24).mean().shift(1)
        F["temp_anomaly_30d"] = (t - roll30.reindex(target_idx)).to_numpy()

        # --- CDH/HDH maruziyet bloklari (master dosya plani; L1 deneyi) ---
        if _use_exposure():
            T = wnat["temperature_2m"]
            cdh = (T - 22.0).clip(lower=0)
            hdh = (16.0 - T).clip(lower=0)
            F["cdh_22"] = cdh.reindex(target_idx).to_numpy()
            F["hdh_16"] = hdh.reindex(target_idx).to_numpy()
            F["cdh_72h_sum"] = cdh.rolling(72, min_periods=36).sum().reindex(target_idx).to_numpy()
            F["hdh_72h_sum"] = hdh.rolling(72, min_periods=36).sum().reindex(target_idx).to_numpy()
            F["cdh_14d_hours"] = (cdh > 0).rolling(336, min_periods=168).sum().reindex(target_idx).to_numpy()
            F["hdh_14d_hours"] = (hdh > 0).rolling(336, min_periods=168).sum().reindex(target_idx).to_numpy()
            dc = (cdh > 0).astype(int).groupby(cdh.index.normalize()).max()
            dh = (hdh > 0).astype(int).groupby(hdh.index.normalize()).max()
            dday = pd.DatetimeIndex(pd.DatetimeIndex(target_idx).normalize())
            F["consecutive_cdh_days"] = _runlen(dc).reindex(dday).to_numpy()
            F["consecutive_hdh_days"] = _runlen(dh).reindex(dday).to_numpy()
    return F


def feature_cols(F: pd.DataFrame) -> list:
    return [c for c in F.columns if c != "target"]


# ---------------------------------------------------------------------------
# Saatlik olay takvimi (Excel: turkiye_ozel_gun_saatlik_talep_feature_takvimi)
# Ilk yil (asagi-akim) bayram/arife/kandil/okul pencerelerini saat bazinda verir.
# Gelecekte bilinen bilgidir -> sizinti yok.
# ---------------------------------------------------------------------------
CAL_PATH = Path(__file__).resolve().parents[1] / "data" / "calendar" / "hourly_events.parquet"
_cal_cache = None


def load_calendar(path=None):
    """Saatlik olay bayraklarini yukle (yoksa None)."""
    global _cal_cache
    p = Path(path) if path is not None else CAL_PATH
    if _cal_cache is not None and path is None:
        return _cal_cache
    if not p.exists():
        return None
    c = pd.read_parquet(p)
    c.index = pd.DatetimeIndex(c.index)
    c = c[~c.index.duplicated(keep="last")].sort_index()
    c = c.astype("float32")
    if path is None:
        _cal_cache = c
    return c


def attach_calendar(F: pd.DataFrame, cal=None) -> pd.DataFrame:
    """Hedef zaman damgasina gore saatlik takvim bayraklarini ekle."""
    if cal is None:
        cal = load_calendar()
    if cal is None:
        return F
    idx = pd.DatetimeIndex(F.index)
    add = cal.reindex(idx)
    add.index = F.index
    add = add.fillna(0.0).astype("float32")
    return pd.concat([F, add], axis=1)
