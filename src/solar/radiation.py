"""GES v1 — ulusal saatlik kisa-dalga radyasyon serisi (OpenMeteo, ecmwf_ifs).

Mevcut kanitli altyapiyi yeniden kullanir (src.data.pull_weather_national /
forecast_weather: retry + 8-sehir baraji dahil). Cekilen tum degiskenler
gelir; bu modul yalnizca `shortwave_radiation` (W/m2) kolonunu alir.

Agirlik notu (v1): 10 sehir NUFUS agirligi kullanilir (tuketim hattiyla ayni).
GES kurulu-guc cografyasi agirligina v2'de gecilecek (Akdeniz/GAP agirlikli).
"""
from __future__ import annotations
import pandas as pd

from .. import data as D
from .. import config as C

RAD_VAR = "shortwave_radiation"


def history_radiation(s: str, e: str) -> pd.Series:
    """[s,e] tarihce radyasyon (historical-forecast API) -> saatlik ulusal W/m2.

    Hizalama (14 Eki 2026): Open-Meteo radyasyonu "onceki saatin ortalamasi"
    etiketlidir; hedef saat t icin [t, t+1) ortalamasina shift(-1) ile hizalanir.
    Cagiran taraf pencereyi 1 gun fazla cekmeli (kenar saati NaN olur)."""
    df = D.pull_weather_national(s, e, url=C.OM_HISTO_URL)
    out = df[RAD_VAR].copy().sort_index().shift(-1)
    out.name = "sw_rad_wm2"
    return out


def forecast_radiation(s: str, e: str) -> pd.Series:
    """[s,e] CANLI radyasyon tahmini (forecast API, ecmwf_ifs) -> saatlik ulusal W/m2.

    Hizalama (14 Eki 2026): Open-Meteo radyasyonu "onceki saatin ortalamasi"
    etiketlidir; hedef saat t icin [t, t+1) ortalamasina shift(-1) ile hizalanir.
    Cagiran taraf pencereyi 1 gun fazla cekmeli (kenar saati NaN olur)."""
    df = D.forecast_weather(s, e)
    out = df[RAD_VAR].copy().sort_index().shift(-1)
    out.name = "sw_rad_wm2"
    return out


def forecast_temp(s: str, e: str) -> pd.Series:
    """[s,e] CANLI sicaklik tahmini (forecast API, ecmwf_ifs) -> saatlik ulusal C."""
    df = D.forecast_weather(s, e)
    out = df["temperature_2m"].copy()
    out.name = "temp_c"
    return out.sort_index()
