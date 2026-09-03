"""PTF input feature store: 11 kolonluk data contract (v1).

Sema (her satir 1 saatlik uzlastirma donemi; karar gunu basina 48 satir):
    datetime (tz-aware Europe/Istanbul, PK) | horizon (T+1/T+2)
    consumption_pred_mw | solar_pred_mw | wind_pred_mw       (float32, MW)
    residual_load_mw (= cons - solar - wind)                  (float32)
    renewable_generation_mw (= solar + wind)                  (float32)
    renewable_penetration (= ren / cons, [0,1))               (float32)
    solar_status (ok | zero_night | unconfigured)             (category)
    wind_status (ritm | lgbm_challenger | fallback)           (category)
    is_peak_hour (08:00<=saat<20:00 -> 1)                     (uint8)

Kaynak onceligi RES: RITM (holdout galibi) -> LGBM challenger -> fallback(NaN).
Invariant'lar export ONCESI validate() ile kontrol edilir; ihlalde yazim YOK
(durust fail — daily run'in PTF adimi continue-on-error oldugu icin tuketim
hatti etkilenmez).

Depolama: data/forecast/ptf_features/archive/ptf_features_YYYY-MM-DD.parquet
(+csv) + latest.parquet/csv kopyalari (downstream dogrudan latest okur).
"""
from __future__ import annotations
import os
import shutil
import numpy as np
import pandas as pd

from .. import config as C
from .. import scoring as S
from ..solar import radiation as R
from ..solar import model as SM
from ..wind import met as WM
from ..wind import model as WMODEL
from ..wind import pull_actual as WPA
from ..wind.weights import load_matrix, weighted_speed

TZ = "Europe/Istanbul"
PEAK_START, PEAK_END = 8, 20  # 08:00 <= saat < 20:00

COLS = ["datetime", "horizon", "consumption_pred_mw", "solar_pred_mw",
        "wind_pred_mw", "residual_load_mw", "renewable_generation_mw",
        "renewable_penetration", "solar_status", "wind_status", "is_peak_hour"]
FLOAT_COLS = ["consumption_pred_mw", "solar_pred_mw", "wind_pred_mw",
              "residual_load_mw", "renewable_generation_mw",
              "renewable_penetration"]


def _rad_series(decision: pd.Timestamp) -> pd.Series:
    d1 = (decision + pd.Timedelta(days=1)).date().isoformat()
    d2 = (decision + pd.Timedelta(days=2)).date().isoformat()
    return R.forecast_radiation(d1, d2)


def _temp_series(decision: pd.Timestamp) -> pd.Series:
    d1 = (decision + pd.Timedelta(days=1)).date().isoformat()
    d2 = (decision + pd.Timedelta(days=2)).date().isoformat()
    return R.forecast_temp(d1, d2)


def _wind_series(decision: pd.Timestamp) -> tuple[pd.Series, str]:
    """RES oncelik zinciri: RITM -> LGBM challenger -> fallback(NaN)."""
    d1 = (decision + pd.Timedelta(days=1)).date().isoformat()
    d2 = (decision + pd.Timedelta(days=2)).date().isoformat()
    idx = pd.date_range(decision + pd.Timedelta(days=1), periods=48, freq="h")
    try:
        r = WPA.pull_ritm_forecast(d1, d2)["ritm_fc_mw"].reindex(idx)
        if r.notna().sum() >= 40:
            return r, "ritm"
        print("PTF uyari: RITM kapsama yetersiz, challenger deneniyor")
    except Exception as ex:
        print(f"PTF uyari: RITM alinamadi ({str(ex)[:80]}), challenger deneniyor")
    try:
        booster = WMODEL.load(C.WIND_MODEL_TXT)
        if booster is not None:
            wl = WM.forecast_wind(d1, d2)
            spd = WM.hub_speed_table(wl)
            w10 = wl.pivot_table(index="dt", columns="city", values="w10")
            m = load_matrix()
            f = WMODEL.build_features(weighted_speed(spd, m), None)
            f = WMODEL.attach_shear(f, weighted_speed(spd, m),
                                    weighted_speed(w10, m))
            # NOT: challenger ritm_fc ile egitildi; sunumda NaN->0 skew'u vardir.
            return WMODEL.predict(booster, f).reindex(idx), "lgbm_challenger"
    except Exception as ex:
        print(f"PTF uyari: challenger basarisiz ({str(ex)[:80]})")
    return pd.Series(np.nan, index=idx), "fallback"


def build(master_df: pd.DataFrame, rad: pd.Series, decision: pd.Timestamp,
          capacity_mw: float | None, pr: float,
          wind: pd.Series | None, wind_status: str,
          temp: pd.Series | None = None) -> pd.DataFrame:
    key = decision.strftime("%Y-%m-%d")
    fc = master_df[master_df["decision_date"] == key].copy()
    if fc.empty:
        raise ValueError(f"master'da {key} karar gunu yok")
    fc["datetime"] = pd.to_datetime(fc["dt"]).dt.tz_localize(TZ)
    fc["horizon"] = fc["horizon"].map({1: "T+1", 2: "T+2"})
    fc = fc.sort_values("datetime").reset_index(drop=True)

    naive = fc["datetime"].dt.tz_localize(None)
    fc["consumption_pred_mw"] = fc["pred_mw"].astype("float32")
    fc["sw_rad_tmp"] = rad.reindex(naive).ffill().fillna(0).values
    if capacity_mw:
        if temp is not None:
            temp_aligned = pd.Series(temp.reindex(naive).ffill().fillna(20.0).values,
                                    index=fc.index)
        else:
            temp_aligned = None
        fc["solar_pred_mw"] = SM.solar_from_radiation(
            pd.Series(fc["sw_rad_tmp"].values, index=fc.index),
            capacity_mw, pr, temp_c=temp_aligned).astype("float32").values
        fc["solar_status"] = "ok"
    else:
        fc["solar_pred_mw"] = np.nan
        fc["solar_status"] = "unconfigured"

    # Gece GES sifirlamasi (v2: zenit maskesi ^ kontrat saat kurali): tam 0.0 + zero_night.
    hrs = fc["datetime"].dt.hour
    zen_night = ~SM.daylight_mask(pd.DatetimeIndex(naive)).to_numpy()
    night = zen_night | (hrs < 5).to_numpy() | (hrs > 20).to_numpy()
    fc.loc[night, "solar_pred_mw"] = 0.0
    fc.loc[night & (fc["solar_status"] == "ok"), "solar_status"] = "zero_night"

    if wind is not None:
        fc["wind_pred_mw"] = wind.reindex(naive).astype("float32").values
    else:
        fc["wind_pred_mw"] = np.nan
    fc["wind_status"] = wind_status

    fc["renewable_generation_mw"] = (
        fc["solar_pred_mw"].fillna(0) + fc["wind_pred_mw"].fillna(0)
    ).astype("float32")
    fc["residual_load_mw"] = (
        fc["consumption_pred_mw"] - fc["renewable_generation_mw"]).astype("float32")
    fc["renewable_penetration"] = (
        fc["renewable_generation_mw"] / fc["consumption_pred_mw"]).astype("float32")
    fc["is_peak_hour"] = ((fc["datetime"].dt.hour >= PEAK_START) &
                          (fc["datetime"].dt.hour < PEAK_END)).astype("uint8")
    fc["horizon"] = fc["horizon"].astype("category")
    fc["solar_status"] = fc["solar_status"].astype("category")
    fc["wind_status"] = fc["wind_status"].astype("category")
    return fc[COLS]


def validate(df: pd.DataFrame) -> None:
    """Export oncesi invariant'lar; ihlalde ValueError (yazim yapilmaz)."""
    if len(df) != 48:
        raise ValueError(f"48 satir beklenir, {len(df)} geldi")
    if df["datetime"].duplicated().any():
        raise ValueError("mukerrer datetime satiri var")
    if (df["datetime"].diff().dropna() != pd.Timedelta(hours=1)).any():
        raise ValueError("datetime ekseninde bosluk var")
    tol = 1e-2  # MW: float32 kuantizasyon payi (fiziksel anlami yok)
    con = (df["consumption_pred_mw"] - df["solar_pred_mw"].fillna(0)
           - df["wind_pred_mw"].fillna(0) - df["residual_load_mw"]).abs().max()
    if con > tol:
        raise ValueError(f"toplamsallik ihlali (max sapma {con})")
    if (df["residual_load_mw"] <= 0).any():
        raise ValueError("negatif/sifir residual_load var")
    night = (df["datetime"].dt.hour < 5) | (df["datetime"].dt.hour > 20)
    if ((df.loc[night, "solar_pred_mw"] != 0.0).any()
            or (df.loc[night & (df["solar_status"] == "ok"), "solar_status"]
                != "zero_night").any()):
        raise ValueError("gece GES sifirlama ihlali")
    pen = df["renewable_penetration"]
    if ((pen < 0.0).any() or (pen >= 1.0).any()):
        raise ValueError("penetrasyon [0,1) disinda")


def build_for_decision(decision_date: str | None = None) -> str | None:
    master = S.load_master()
    if master.empty:
        print("PTF: master forecast bos — once daily_run kosmalidir.")
        return None
    if decision_date is None:
        decision_date = pd.to_datetime(master["decision_date"]).max().strftime("%Y-%m-%d")
    decision = pd.Timestamp(decision_date)
    try:
        rad = _rad_series(decision)
    except Exception as ex:
        print(f"PTF: radyasyon yok ({str(ex)[:80]}) — cikti uretilemedi.")
        return None
    wind, wind_status = _wind_series(decision)
    try:
        temp = _temp_series(decision)
    except Exception as ex:
        print(f"PTF uyari: sicaklik yok ({str(ex)[:60]}), derate atlandi")
        temp = None
    out = build(master, rad, decision, C.SOLAR_CAPACITY_MW, C.SOLAR_PR,
                None if wind_status == "fallback" else wind, wind_status, temp=temp)
    validate(out)  # ihlalde raise -> yazim YOK
    arch = os.path.join(C.PTF_FEATURES_DIR, "archive")
    os.makedirs(arch, exist_ok=True)
    base = os.path.join(arch, f"ptf_features_{decision_date}")
    out.to_parquet(base + ".parquet", index=False)
    out.to_csv(base + ".csv", index=False)
    shutil.copy(base + ".parquet", os.path.join(C.PTF_FEATURES_DIR, "latest.parquet"))
    shutil.copy(base + ".csv", os.path.join(C.PTF_FEATURES_DIR, "latest.csv"))
    print(f"PTF: {len(out)} satir -> {base}.parquet (+csv) + latest | "
          f"wind={wind_status} | solar ok={int((out['solar_status'] == 'ok').sum())}/48")
    return base + ".parquet"
