"""EPiAS RITM RES uretim + tahmin cekimi (wind-forecast endpoint'i).

Dogrulanmis sema (2026-08 probe): 15-dak cozumluk; `generation` = izlenebilen
RES'lerin gerceklesmesi, `forecast` = RITM tahmini, quarter1-4 = kantil bantlari.
Cikti: saatlik (ortalama) MW serisi.
"""
from __future__ import annotations
import pandas as pd


def _hourly(df: pd.DataFrame, col: str) -> pd.Series:
    idx = pd.DatetimeIndex(pd.to_datetime(df["time"] if "time" in df else df["date"]))
    if idx.tz is not None:
        idx = idx.tz_convert("Etc/GMT-3").tz_localize(None)
    s = pd.Series(pd.to_numeric(df[col], errors="coerce").values, index=idx)
    return s.sort_index().resample("h").mean()


def pull_generation(start: str, end: str) -> pd.Series:
    """Saatlik RES gerceklesmesi (MW). Egitim hedefi."""
    from eptr2.calls.renewables import get_wind_forecast
    df = get_wind_forecast(start, end)
    s = _hourly(df, "generation")
    s.name = "wind_gen_mw"
    return s


def pull_ritm_forecast(start: str, end: str) -> pd.DataFrame:
    """RITM tahmin + kantil bantlari (saatlik). Benchmark / yedek girdi."""
    from eptr2.calls.renewables import get_wind_forecast
    df = get_wind_forecast(start, end)
    out = pd.DataFrame({"ritm_fc_mw": _hourly(df, "forecast")})
    for q in ["quarter1", "quarter2", "quarter3", "quarter4"]:
        if q in df:
            out[f"ritm_{q}_mw"] = _hourly(df, q)
    return out
