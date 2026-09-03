"""PTF input feature store ureticisi: tuketim + GES + net yuk tekilleştirme.

Girdi:
  - Tuketim tahmini: data/results/forecast_results.csv (master; ayni karar
    gununun D+1/D+2 satirlari). Mevcut uretim hattina DOKUNMAZ, sadece okur.
  - Radyasyon: hedef gunler icin CANLI OpenMeteo forecast (src.solar.radiation).
  - GES gucu: src.solar.model (kapasite config'te yoksa NaN + status).

Cikti (her karar gunu bir dosya):
  data/forecast/ptf_features/<decision>_ptf.parquet  (+ .csv)
  Kolonlar: dt | horizon | decision_date | cons_pred_mw | load_plan_mw |
            sw_rad_wm2 | solar_pred_mw | residual_load_mw | solar_status

solar_status: 'ok' | 'unconfigured' (SOLAR_CAPACITY_MW yok; residual=gross,
PTF tarafi bu satirlari bilir) | 'radiation_failed'.
"""
from __future__ import annotations
import os
import pandas as pd

from .. import config as C
from .. import scoring as S

COLS = ["dt", "horizon", "decision_date", "cons_pred_mw", "load_plan_mw",
        "sw_rad_wm2", "solar_pred_mw", "residual_load_mw", "solar_status"]


def build(master_df: pd.DataFrame, rad: pd.Series | None,
          decision: str, capacity_mw: float | None = None,
          pr: float | None = None) -> pd.DataFrame:
    """Saf birlestirme (IO yok; test edilebilir)."""
    from ..solar import model as SM
    pr = C.SOLAR_PR if pr is None else pr
    m = master_df[master_df["decision_date"] == decision].copy()
    if m.empty:
        raise ValueError(f"master'da decision_date={decision} satiri yok")
    m["dt"] = pd.to_datetime(m["dt"])
    out = pd.DataFrame({
        "dt": m["dt"].to_numpy(),
        "horizon": m["horizon"].to_numpy(),
        "decision_date": decision,
        "cons_pred_mw": pd.to_numeric(m["pred_mw"], errors="coerce").to_numpy(),
        "load_plan_mw": pd.to_numeric(m["load_plan_mw"], errors="coerce").to_numpy(),
    }).sort_values("dt").reset_index(drop=True)
    if rad is not None and len(rad):
        r = rad.copy()
        r.index = pd.to_datetime(r.index)
        out["sw_rad_wm2"] = r.reindex(out["dt"]).to_numpy()
    else:
        out["sw_rad_wm2"] = float("nan")
    if capacity_mw:
        out["solar_pred_mw"] = SM.solar_from_radiation(
            out["sw_rad_wm2"].fillna(0.0), capacity_mw, pr).to_numpy()
        out["solar_status"] = "ok"
    else:
        out["solar_pred_mw"] = float("nan")
        out["solar_status"] = "unconfigured" if out["sw_rad_wm2"].notna().any() else "radiation_failed"
    out["residual_load_mw"] = out["cons_pred_mw"] - out["solar_pred_mw"].fillna(0.0)
    return out[COLS]


def build_for_decision(decision: str | None = None) -> str | None:
    """IO sarici: master + canli radyasyon -> dosya yaz. Donus: yazilan parquet yolu."""
    from ..solar import radiation as SR
    master = S.load_master()
    if master.empty:
        print("PTF: master bos — once daily pipeline kosmali.")
        return None
    if decision is None:
        decision = pd.to_datetime(master["decision_date"]).max().strftime("%Y-%m-%d")
    sub = master[master["decision_date"] == decision]
    if sub.empty:
        print(f"PTF: {decision} icin tahmin yok.")
        return None
    dt = pd.to_datetime(sub["dt"])
    try:
        rad = SR.forecast_radiation(dt.min().date().isoformat(), dt.max().date().isoformat())
    except Exception as ex:
        print(f"PTF UYARI: radyasyon forecast alinamadi ({str(ex)[:80]}); NaN ile devam")
        rad = None
    out = build(master, rad, decision, capacity_mw=C.SOLAR_CAPACITY_MW)
    os.makedirs(C.PTF_FEATURES_DIR, exist_ok=True)
    base = os.path.join(C.PTF_FEATURES_DIR, f"{decision}_ptf")
    out.to_parquet(base + ".parquet", index=False)
    out.to_csv(base + ".csv", index=False)
    n_ok = int((out["solar_status"] == "ok").sum())
    print(f"PTF: {len(out)} satir -> {base}.parquet (+csv) | solar_ok={n_ok}/{len(out)} "
          f"| status={out['solar_status'].iloc[0]}")
    return base + ".parquet"
