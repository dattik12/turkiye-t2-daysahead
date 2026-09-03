"""PTF input feature store ureticisi: tuketim + GES + RES + net yuk.

Residual kontrat (v1):
    residual_load_mw = cons_pred_mw - solar_pred_mw - wind_pred_mw
Eksik bilesen 0 sayilir; hangi bilesenin eksik oldugu solar_status /
wind_status kolonlarinda acikca yazilir (sessiz tamamlama yok).

Kaynaklar:
  - Tuketim: forecast_results.csv master (mevcut uretim hatti, degismedi)
  - GES: src.solar (radyasyon + PR modeli)
  - RES v1: EPiAS RITM tahmini (holdout'ta LGBM %10.14'e karsi %9.57 ile onde).
    LGBM challenger olarak models/wind_lgbm.txt'de durur (train_wind.py).
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd

from .. import config as C
from .. import scoring as S
from ..solar import radiation as R
from ..solar import model as SM
from ..wind import pull_actual as WPA

COLS = ["dt", "horizon", "decision_date", "cons_pred_mw", "load_plan_mw",
        "sw_rad_wm2", "solar_pred_mw", "solar_status",
        "wind_pred_mw", "wind_status", "residual_load_mw"]


def _rad_series(decision: pd.Timestamp) -> tuple[pd.Series, str]:
    d1 = (decision + pd.Timedelta(days=1)).date().isoformat()
    d2 = (decision + pd.Timedelta(days=2)).date().isoformat()
    try:
        return R.forecast_radiation(d1, d2), "ok"
    except Exception as ex:
        print(f"PTF uyari: radyasyon alinamadi ({str(ex)[:80]})")
        idx = pd.date_range(decision + pd.Timedelta(days=1), periods=48, freq="h")
        return pd.Series(0.0, index=idx, name="sw_rad_wm2"), "radiation_failed"


def _wind_series(decision: pd.Timestamp) -> tuple[pd.Series, str]:
    """v1: RITM tahminini dene (holdout galibi); yoksa NaN + wind_missing."""
    d1 = (decision + pd.Timedelta(days=1)).date().isoformat()
    d2 = (decision + pd.Timedelta(days=2)).date().isoformat()
    try:
        r = WPA.pull_ritm_forecast(d1, d2)["ritm_fc_mw"]
        idx = pd.date_range(decision + pd.Timedelta(days=1), periods=48, freq="h")
        r = r.reindex(idx)
        if r.notna().sum() >= 40:
            return r, "ritm"
    except Exception as ex:
        print(f"PTF uyari: RITM alinamadi ({str(ex)[:80]})")
    idx = pd.date_range(decision + pd.Timedelta(days=1), periods=48, freq="h")
    return pd.Series(np.nan, index=idx, name="wind_pred_mw"), "wind_missing"


def build(master_df: pd.DataFrame, rad: pd.Series, decision: pd.Timestamp,
          capacity_mw: float | None, pr: float,
          wind: pd.Series | None, wind_status: str) -> pd.DataFrame:
    key = decision.strftime("%Y-%m-%d")
    fc = master_df[master_df["decision_date"] == key].copy()
    if fc.empty:
        raise ValueError(f"master'da {key} karar gunu yok")
    fc["dt"] = pd.to_datetime(fc["dt"])
    fc = fc.sort_values("dt").reset_index(drop=True)
    fc = fc.rename(columns={"pred_mw": "cons_pred_mw"})

    fc["sw_rad_wm2"] = rad.reindex(fc["dt"]).ffill().fillna(0).values
    if capacity_mw:
        fc["solar_pred_mw"] = SM.solar_from_radiation(
            fc["sw_rad_wm2"], capacity_mw, pr).values
        fc["solar_status"] = "ok"
    else:
        fc["solar_pred_mw"] = np.nan
        fc["solar_status"] = "unconfigured"

    if wind is not None:
        fc["wind_pred_mw"] = wind.reindex(fc["dt"]).values
    else:
        fc["wind_pred_mw"] = np.nan
    fc["wind_status"] = wind_status

    fc["residual_load_mw"] = (fc["cons_pred_mw"]
                              - fc["solar_pred_mw"].fillna(0)
                              - fc["wind_pred_mw"].fillna(0))
    fc["decision_date"] = key
    return fc[["dt", "horizon", "decision_date", "cons_pred_mw", "load_plan_mw",
               "sw_rad_wm2", "solar_pred_mw", "solar_status",
               "wind_pred_mw", "wind_status", "residual_load_mw"]]


def build_for_decision(decision_date: str | None = None) -> str | None:
    master = S.load_master()
    if master.empty:
        print("PTF: master forecast bos — once daily_run kosmalidir.");
        return None
    if decision_date is None:
        decision_date = pd.to_datetime(master["decision_date"]).max().strftime("%Y-%m-%d")
    decision = pd.Timestamp(decision_date)
    rad, rad_status = _rad_series(decision)
    if rad_status != "ok":
        print("PTF: radyasyon yok — iskelet cikti uretilemedi.");
        return None
    wind, wind_status = _wind_series(decision)
    out = build(master, rad, decision, C.SOLAR_CAPACITY_MW, C.SOLAR_PR,
                None if wind_status == "wind_missing" else wind, wind_status)
    os.makedirs(C.PTF_FEATURES_DIR, exist_ok=True)
    base = os.path.join(C.PTF_FEATURES_DIR, f"{decision_date}_ptf")
    out.to_parquet(base + ".parquet", index=False)
    out.to_csv(base + ".csv", index=False)
    n_ok = int(out["wind_pred_mw"].notna().sum())
    print(f"PTF: {len(out)} satir -> {base}.parquet (+csv) | "
          f"wind={wind_status} ({n_ok}/{len(out)}) | solar={out['solar_status'].iloc[0]}")
    return base + ".parquet"
