"""GES v1 guc modeli: radyasyon -> MW (lineer, PR + kurulu guc + sicaklik derate)."""
from __future__ import annotations
import numpy as np
import pandas as pd

TEMP_COEFF = -0.004   # kristal-Si guc sicaklik katsayisi (literatur ortalamasi)
NOCT_K = 0.03         # hucre sicakligi yaklasimi: Tcell = Tamb + NOCT_K * GHI
# AC/inverter tavan: DC/AC orani ~1.2-1.3 -> AC cikis kurulu gucun ~%85'inde doyar.
# Gerekce: yillik CF %18.24 (gercek %17-19, seviye DOGRU) ama aylik tepe %90 (fizik disi).
SOLAR_AC_CLIP = 0.85


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
    # --- AC KIRPMA (inverter siniri) ---
    # Olculdu: model yillik kapasite faktoru %18.24 (Turkiye PV gercegi %17-19) -> SEVIYE DOGRU.
    # Ama aylik tepeler fizik disi: Haziran %90, Mayis %87.6, Temmuz %88.4 (kurulu gucun).
    # Gercek santralde DC/AC orani ~1.2-1.3 -> AC cikis DC kapasitesinin ~%78-85'inde doyar.
    # Bu yuzden 0.85 x kapasite tavan uygulanir. Yillik enerji etkisi kucuk (<%1).
    out = out.clip(upper=SOLAR_AC_CLIP * float(capacity_mw))
    out.name = "solar_pred_mw"
    return out


def fit_scale(rad_history: pd.Series, solar_actual_mw: pd.Series,
              capacity_mw: float | None = None,
              min_ghi: float = 200.0) -> dict:
    """Radyasyon -> MW kalibrasyonu. fit_scale() ARTIK BOS DEGIL.

    Iki sey fit edilir:
      1) shape_hour : 24 saatlik ORAN duzeltmesi (gercek/tahmin, gun toplami korunur)
      2) pr_implied : lisansli filo icin zorunlu PR (kapasite bilinirse)

    NEDEN SEKIL: olculdu ki model ogleni fazla, sabah/aksami az uretiyor
    (oran 06:00'da 0.41, 12:00'de 0.21). Sebep tek-nokta GHI egrisinin sivri olmasi;
    gercek filo cografi yayili oldugu icin yayvan.

    NEDEN SEVIYE FIT EDILEMIYOR: elimizdeki gerceklesme yalnizca lisansli santraller
    (kurulu gucun ~%15'i) ve lisansli kapasite bilinmiyor. Bu yuzden national PR
    dogrulanamaz; pr_implied yalnizca capacity_mw verilirse hesaplanir ve
    "lisansli filo icin" etiketiyle dondurulur.

    Donus: dict(shape_hour, pr_implied, ratio_median, n, note)
    """
    j = pd.concat({"a": solar_actual_mw, "g": rad_history}, axis=1).dropna()
    j = j[j["g"] > min_ghi]
    if len(j) < 200:
        raise ValueError(f"fit_scale: yetersiz eslesme ({len(j)} saat, min 200)")

    pred = j["g"] / 1000.0 * 1.0          # birim kapasite, PR=1 -> MW/MW
    ratio = (j["a"] / pred).replace([np.inf, -np.inf], np.nan).dropna()
    by_h = ratio.groupby(ratio.index.hour).median()
    daily = float(ratio.median())

    shape = {}
    for h in range(24):
        if h in by_h.index and np.isfinite(by_h[h]) and daily > 0:
            shape[h] = round(float(by_h[h] / daily), 4)

    pr_implied = None
    if capacity_mw and capacity_mw > 0:
        pr_implied = round(float(j["a"].sum() / (j["g"].sum() / 1000.0 * capacity_mw)), 4)

    return dict(
        shape_hour=shape,
        pr_implied=pr_implied,
        ratio_median=round(daily, 1),
        n=int(len(j)),
        note=("shape_hour: 24 saatlik oran duzeltmesi (1.0 = degisiklik yok), gun toplami korunur. "
              "pr_implied yalnizca capacity_mw verildiginde ve LISANSLI filo icin gecerlidir; "
              "ulusal seviye dogrulanamaz (gerceklesme kapsami ~%15)."),
    )
