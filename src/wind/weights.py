"""RES agirlik matrisi yukleyici + agirlikli ruzgar hizi yardimcilari.

Matris: data/wind/WIND_FARMS.csv — TUREB Ocak 2026 bolge MW capali; il agirliklari
v1-gecici (bolge ici esit paylasim, coverage=0.8026 uzerinden renormalize).
Backfill'de TUREB il tablosuyla sabitlenecek (TODO).
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd

from .. import config as C


def load_matrix(csv_path: str | None = None) -> pd.DataFrame:
    p = csv_path or C.WIND_FARMS_CSV
    df = pd.read_csv(p)
    assert abs(df["w"].sum() - 1.0) < 1e-3, f"agirliklar toplami 1 degil: {df['w'].sum()}"
    return df


def weighted_speed(wind_by_city: pd.DataFrame, matrix: pd.DataFrame) -> pd.Series:
    """Sehir x saat ruzgar hizi tablosu -> agirlikli ulusal seri."""
    w = matrix.set_index("province")["w"]
    cols = [c for c in wind_by_city.columns if c in w.index]
    if not cols:
        raise KeyError("agirlik matrisindeki iller veri tablosunda yok")
    v = wind_by_city[cols].mul(w[cols]).sum(axis=1) / w[cols].sum()
    v.name = "wind_ms"
    return v


def power_curve_features(v: pd.Series) -> pd.DataFrame:
    """Kubik guc egrisi feature'lari (v1 seti; model asamasinda kullanilacak)."""
    v = v.clip(lower=0)
    return pd.DataFrame({
        "wind_ms": v,
        "wind_v2": v ** 2,
        "wind_v3": v ** 3,
        "wind_rated_clip": v.clip(upper=13),          # rated ~12-14 m/s
        "wind_cutout_flag": (v >= 25).astype(int),    # cut-out 25 m/s
        "wind_low_flag": (v < 3.5).astype(int),       # cut-in alti
    }, index=v.index)
