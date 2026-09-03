"""GES v1 guc modeli: radyasyon -> MW (lineer, PR + kurulu guc + sicaklik derate)."""
from __future__ import annotations
import numpy as np
import pandas as pd

TEMP_COEFF = -0.004   # kristal-Si guc sicaklik katsayisi (literatur ortalamasi)
NOCT_K = 0.03         # hucre sicakligi yaklasimi: Tcell = Tamb + NOCT_K * GHI


def cell_temp(amb_c: pd.Series, ghi: pd.Series) -> pd.Series:
    return amb_c + NOCT_K * ghi.clip(lower=0)


def derate_factor(amb_c: pd.Series, ghi: pd.Series) -> pd.Series:
    return (1.0 + TEMP_COEFF * (cell_temp(amb_c, ghi) - 25.0)).clip(lower=0.5, upper=1.05)


def daylight_mask(idx: pd.DatetimeIndex, cities=None) -> pd.Series:
    """Zenit-acili gunduz maskesi (nufus agirlikli cogunluk). Gece == False."""
    from .. import config as C
    cities = cities or C.CITIES
    yday = idx.dayofyear.to_numpy()
    hour = idx.hour.to_numpy() + idx.minute.to_numpy() / 60.0
    votes = np.zeros(len(idx))
    total = 0.0
    for c in cities:
        decl = np.deg2rad(-23.44 * np.cos(np.deg2rad(360.0 / 365.0 * (yday + 10))))
        latr = np.deg2rad(c["lat"])
        h = np.deg2rad((hour - 12.0) * 15.0)
        cosz = np.sin(latr) * np.sin(decl) + np.cos(latr) * np.cos(decl) * np.cos(h)
        votes += (cosz > 0) * c.get("pop", 1.0)
        total += c.get("pop", 1.0)
    return pd.Series(votes / total > 0.5, index=idx)


def solar_from_radiation(rad_wm2: pd.Series, capacity_mw: float, pr: float = 0.80,
                         temp_c: pd.Series | None = None) -> pd.Series:
    """Saatlik radyasyon (W/m2) -> GES guc tahmini (MW). temp_c verilirse sicaklik derate uygular."""
    if capacity_mw is None or capacity_mw <= 0:
        raise ValueError("capacity_mw tanimli olmali (config.SOLAR_CAPACITY_MW; TEIAS aylik istatistik).")
    out = rad_wm2.clip(lower=0) / 1000.0 * float(capacity_mw) * float(pr)
    if temp_c is not None:
        out = out * derate_factor(temp_c.reindex(out.index).ffill().fillna(20.0), rad_wm2)
    out.name = "solar_pred_mw"
    return out


def fit_scale(rad_history: pd.Series, solar_actual_mw: pd.Series) -> dict:
    """Backfill asamasinda kalibre edilecek (su an iskelet)."""
    raise NotImplementedError(
        "fit_scale backfill asamasinda implemente edilecek: "
        "tarihce radyasyon x GES gerceklesmesiyle PR/scale fit edilecek."
    )
