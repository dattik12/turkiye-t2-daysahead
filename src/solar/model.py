"""GES v1 guc modeli: radyasyon -> MW (lineer, PR + kurulu guc).

    solar_mw = max(rad_wm2, 0) / 1000 * capacity_mw * pr

Gece saatleri dogal olarak ~0 uretir (radyasyon ~0). Dagitik/lisanssiz GES'in
sebeke cekisini dusurdugu gercegi model disi degil: rt_cons zaten NET cekis
oldugu icin residual_load = cons_pred - solar_pred PTF'ye giren net yuktur.

Kalibrasyon: `fit_scale` backfill asamasinda tarihce radyasyon x GES
gerceklesmesiyle fit edilecek; o zamana kadar config'teki sabit PR kullanilir.
"""
from __future__ import annotations
import pandas as pd


def solar_from_radiation(rad_wm2: pd.Series, capacity_mw: float, pr: float = 0.80) -> pd.Series:
    """Saatlik radyasyon (W/m2) -> GES guc tahmini (MW)."""
    if capacity_mw is None or capacity_mw <= 0:
        raise ValueError("capacity_mw tanimli olmali (config.SOLAR_CAPACITY_MW; TEIAS aylik istatistik).")
    out = rad_wm2.clip(lower=0) / 1000.0 * float(capacity_mw) * float(pr)
    out.name = "solar_pred_mw"
    return out


def fit_scale(rad_history: pd.Series, solar_actual_mw: pd.Series) -> dict:
    """Backfill asamasinda kalibre edilecek (su an iskelet)."""
    raise NotImplementedError(
        "fit_scale backfill asamasinda implemente edilecek: "
        "tarihce radyasyon x GES gerceklesmesiyle PR/scale fit edilecek."
    )
