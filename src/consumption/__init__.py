"""Tuketim hatti referans paketi (v4.3 uretim).

Mevcut moduller TASINMADI (import kirilmamasi icin); bu paket yalnizca
PTF mimarisi icindeki kanonik adresi saglar:
    src.consumption  ->  src.{data, features, forecast, models, scoring} + scripts.daily_run

Uretim hatti aynen: Engine(cons, nat, cities).forecast(decision) -> forecast_results.csv
"""
from .. import config, data, features, forecast, scoring

try:
    from .. import models
except Exception:  # agir bagimliliklar yoksa (saf iskelet importu) sorun degil
    models = None

__all__ = ["config", "data", "features", "forecast", "models", "scoring"]
