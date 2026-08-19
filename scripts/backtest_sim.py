"""Simule backtest: 2025-01-01 -> bugun (haftalik retrain'li konservatif simulasyon).
Cikti: data/results/backtest_sim_2025_2026.csv (saatlik) + terminal ozet (gunluk MAPE, bayram-disi,
TEIAS kiyas). Training yalnizca karar gunune kadar olan veriyi kullanir -> leak-free.
Kullanim:  python -m scripts.backtest_sim
"""
from __future__ import annotations
import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config as C
from src import data as D
from src.forecast import Engine

START = "2025-01-01"
RETRAIN_EVERY = 7


def main():
    cons = D.load_or_create_consumption()
    nat, cities = D.load_or_create_weather()
    if cons.empty:
        print("Once bootstrap calistirin:  python -m scripts.bootstrap")
        return

    engine = Engine(cons, nat, cities)
    decision = D.last_full_day(cons)
    if decision is None:
        print("Son tamamlanmis gun yok")
        return

    start = pd.Timestamp(START)
    last_dec = decision - pd.Timedelta(days=2)   # D+2 target veride olsun
    if last_dec <= start:
        print("Baslangic tarihi son karar gununden sonra — startup gecmisi kisa.")
        last_dec = start + pd.Timedelta(days=1)

    decisions = list(pd.date_range(start, last_dec, freq="D"))
    print(f"Simulasyon: {decisions[0].date()} -> {decisions[-1].date()}  ({len(decisions)} karar gunu)")
    print(f"Haftalik retrain ({RETRAIN_EVERY} gun). Model: BASE+CONT ensemble.")

    cache = None
    rows = []
    for i, dec in enumerate(decisions):
        if i % RETRAIN_EVERY == 0:
            cache = None
        out, cache = engine.forecast(dec, models_cache=cache, return_models=True)
        act = cons["rt_cons"].reindex(out["dt"])
        lp = out["load_plan_mw"] if "load_plan_mw" in out else np.nan
        for _, r in out.iterrows():
            rows.append((r["decision_date"].date(), r["dt"], int(r["horizon"]),
                         r["pred_mw"], r["load_plan_mw"], act.get(r["dt"], np.nan)))
        if (i + 1) % 30 == 0:
            print(f"  {dec.date()} ({i+1}/{len(decisions)})", flush=True)

    df = pd.DataFrame(rows, columns=["decision_date", "dt", "horizon", "pred_mw", "load_plan_mw", "actual_mw"])
    df["dt"] = pd.to_datetime(df["dt"])
    df = df.dropna(subset=["actual_mw"])
    df["mape"] = (df["pred_mw"] - df["actual_mw"]).abs() / df["actual_mw"] * 100
    os.makedirs(C.RESULTS_DIR, exist_ok=True)
    df.to_csv(C.BACKTEST_CSV, index=False)

    for hz in [1, 2]:
        s = df[df["horizon"] == hz]
        print(f"T+{hz}  MAPE %{s['mape'].mean():.2f}  MAE {np.mean((s['pred_mw']-s['actual_mw']).abs()):,.0f} MW  (n={len(s)})")
    d2 = df[df["horizon"] == 2]
    # bayram penceresi disi (2026 kurban 24-31 Mayis + tum yillarin bayram gunleri yaklasik)
    hol = set()
    for y in ["2025", "2026"]:
        hol |= {f"{y}-03-28", f"{y}-03-29", f"{y}-03-30", f"{y}-03-31", f"{y}-04-01"}
        hol |= {f"{y}-06-05", f"{y}-06-06", f"{y}-06-07", f"{y}-06-08", f"{y}-06-09"}
        hol |= {f"{y}-05-25", f"{y}-05-26", f"{y}-05-27", f"{y}-05-28", f"{y}-05-29", f"{y}-05-30"}
    ins = d2["dt"].dt.date.astype(str).isin(hol)
    print(f"D+2 bayram-disi: %{d2[~ins]['mape'].mean():.2f} | bayram-hicret: %{d2[ins]['mape'].mean():.2f} (n={ins.sum()})")
    ts = df[df["horizon"] == 1]
    b = ts[["dt", "actual_mw", "load_plan_mw"]].dropna()
    if len(b):
        lp_err = np.mean((b["load_plan_mw"] - b["actual_mw"]).abs() / b["actual_mw"]) * 100
        print(f"TEIAS load_plan (D+1, ayni pencere): %{lp_err:.2f}")
    # gunluk ozet
    daily = df.groupby(["dt"].map(pd.Timestamp.normalize), as_index=False) if False else df.groupby(df["dt"].dt.normalize())
    print(f"\n-> {C.BACKTEST_CSV}  ({len(df)} saatlik satir)")


if __name__ == "__main__":
    main()
