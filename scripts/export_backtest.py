"""Backtest sonuclarini FTP modeli icin temiz/dos dostu formatta disa aktarir.
Saat basina TEK satir: zaman (TR), gerceklesen, T+1 tahmin, T+2 tahmin, TEIAS plan.
Ayrica gunluk ozet (gunluk MAPE, T+1/T+2/TEIAS).
Cikti: data/exports/backtest_t1_t2_2025_2026.csv  ve  backtest_daily_summary_2025_2026.csv
Kullanim:  python -m scripts.export_backtest
"""
from __future__ import annotations
import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config as C

SRC = os.path.join(C.RESULTS_DIR, "backtest_sim_2025_2026.csv")
OUT_DIR = os.path.join(C.DATA, "exports")


def main():
    if not os.path.exists(SRC):
        print(f"Once backtest kosun: python -m scripts.backtest_sim  ({SRC} yok)")
        return
    df = pd.read_csv(SRC, parse_dates=["dt"])
    df["dt"] = pd.to_datetime(df["dt"])

    # ---- saat basina tek satir: T+1 / T+2 / TEIAS / gerceklesen ----
    h1 = df[df["horizon"] == 1].set_index("dt")
    h2 = df[df["horizon"] == 2].set_index("dt")
    out = pd.DataFrame(index=h1.index)
    out["tarih"] = h1.index.strftime("%Y-%m-%d")
    out["saat"] = h1.index.strftime("%H:%M")
    out["gerceklesen_mw"] = h1["actual_mw"].round(2)
    out["t1_forecast_mw"] = h1["pred_mw"].round(2)
    out["t2_forecast_mw"] = h2["pred_mw"].round(2)
    out["teias_plan_mw"] = np.where(h1["load_plan_mw"].notna(), h1["load_plan_mw"], np.nan)
    out["t1_karar_gunu"] = h1["decision_date"].astype(str)
    out["t2_karar_gunu"] = h2["decision_date"].astype(str)
    out["t1_mape_pct"] = ((out["t1_forecast_mw"] - out["gerceklesen_mw"]).abs()
                          / out["gerceklesen_mw"] * 100).round(3)
    out["t2_mape_pct"] = ((out["t2_forecast_mw"] - out["gerceklesen_mw"]).abs()
                          / out["gerceklesen_mw"] * 100).round(3)
    out.reset_index(inplace=True)
    out.rename(columns={"dt": "zaman_utc3"}, inplace=True)

    os.makedirs(OUT_DIR, exist_ok=True)
    p1 = os.path.join(OUT_DIR, "backtest_t1_t2_2025_2026.csv")
    out.to_csv(p1, index=False)

    # ---- gunluk ozet ----
    d = df.copy()
    d["tarih"] = d["dt"].dt.date.astype(str)
    piv = d.pivot_table(index="tarih", columns="horizon", values="pred_mw", aggfunc="first")
    act = d.groupby("tarih")["actual_mw"].first()
    lp = d[d["horizon"] == 1].groupby("tarih")["load_plan_mw"].first()
    daily = pd.DataFrame({
        "tarih": piv.index,
        "gerceklesen_gunluk_ort_mw": act.round(2),
        "t1_tahmin_gunluk_ort_mw": piv[1].round(2),
        "t2_tahmin_gunluk_ort_mw": piv[2].round(2),
        "teias_gunluk_ort_mw": lp.round(2),
    })
    daily["t1_mape_pct"] = ((daily["t1_tahmin_gunluk_ort_mw"] - daily["gerceklesen_gunluk_ort_mw"]).abs()
                            / daily["gerceklesen_gunluk_ort_mw"] * 100).round(3)
    daily["t2_mape_pct"] = ((daily["t2_tahmin_gunluk_ort_mw"] - daily["gerceklesen_gunluk_ort_mw"]).abs()
                            / daily["gerceklesen_gunluk_ort_mw"] * 100).round(3)
    p2 = os.path.join(OUT_DIR, "backtest_daily_summary_2025_2026.csv")
    daily.to_csv(p2, index=False)

    print(f"-> {p1}  ({len(out)} saatlik satir, {out['tarih'].nunique()} gun)")
    print(f"-> {p2}  ({len(daily)} gunluk satir)")
    print("\nGenel MAPE (saatlik):")
    print(f"  T+1: {out['t1_mape_pct'].mean():.2f}%  |  T+2: {out['t2_mape_pct'].mean():.2f}%")
    lp2 = out["teias_plan_mw"].notna()
    print(f"  TEIAS (T+1): {((out.loc[lp2,'teias_plan_mw']-out.loc[lp2,'gerceklesen_mw']).abs()/out.loc[lp2,'gerceklesen_mw']*100).mean():.2f}%")


if __name__ == "__main__":
    main()
