"""RES meteorolojisi: 7 hub icin IFS 100m ruzgar (tarihce + tahmin).

Birim: OpenMeteo km/h verir, m/s'ye cevrilir. Tarihce gunluk dilimlerle cekilir
(historical-forecast-api); tahmin tek cagri (forecast API, ecmwf_ifs).
"""
from __future__ import annotations
import pandas as pd

from .. import config as C
from .. import data as D
from .weights import load_matrix

WIND_VARS = ["wind_speed_100m", "wind_speed_10m", "wind_direction_100m"]
KMH_TO_MS = 1.0 / 3.6


def _hub_pull(lat: float, lon: float, s: str, e: str, url: str) -> pd.DataFrame:
    h = D.om_fetch(url, lat, lon, s, e, vars=WIND_VARS)
    return pd.DataFrame({v: h[v] for v in WIND_VARS}, index=pd.to_datetime(h["time"]))


def history_wind(s: str, e: str, matrix=None) -> pd.DataFrame:
    """Saatlik hub ruzgari (m/s), long format: dt/city/w100/w10/wdir."""
    matrix = load_matrix() if matrix is None else matrix
    out = []
    for _, r in matrix.iterrows():
        h = _hub_pull(r["lat"], r["lon"], s, e, C.OM_HISTO_URL).sort_index()
        h = h[~h.index.duplicated(keep="last")]
        h["city"] = r["province"]
        h = h.rename(columns={"wind_speed_100m": "w100", "wind_speed_10m": "w10",
                              "wind_direction_100m": "wdir"})
        h[["w100", "w10"]] = h[["w100", "w10"]] * KMH_TO_MS
        h["dt"] = h.index
        out.append(h.reset_index(drop=True))
    return pd.concat(out, ignore_index=True)


def hub_speed_table(wind_long: pd.DataFrame) -> pd.DataFrame:
    """Long -> (saat x il) w100 tablosu (m/s)."""
    return wind_long.pivot_table(index="dt", columns="city", values="w100")


def forecast_wind(s: str, e: str, matrix=None) -> pd.DataFrame:
    """Hedef gunler icin hub IFS tahmini, long format (history_wind ile ayni sema)."""
    matrix = load_matrix() if matrix is None else matrix
    out = []
    for _, r in matrix.iterrows():
        h = _hub_pull(r["lat"], r["lon"], s, e, C.OM_FC_URL).sort_index()
        h = h[~h.index.duplicated(keep="last")]
        h["city"] = r["province"]
        h = h.rename(columns={"wind_speed_100m": "w100", "wind_speed_10m": "w10",
                              "wind_direction_100m": "wdir"})
        h[["w100", "w10"]] = h[["w100", "w10"]] * KMH_TO_MS
        h["dt"] = h.index
        out.append(h.reset_index(drop=True))
    return pd.concat(out, ignore_index=True)
