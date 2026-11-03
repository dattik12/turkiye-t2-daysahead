"""Sabah sozlesmesi CANLI skor tahtasi.

morning_history.csv'deki tahminleri gerceklesenle ve TEIAS'in o gun icin
yayimladigi yuk planiyla karsilastirir; defteri yerinde gunceller.

Kural:
  - Sadece hedef gunu TAMAMEN gecmis satirlar degerlendirilir.
  - TEIAS kiyasi: ayni hedef gun icin load_plan (gun-ici plan).
  - Sabah kosusu 09:00'da karar verir; TEIAS'in o gunku plani ~17:00'de
    yayimlanir. Yani TEIAS bir gun daha fazla bilgiye sahiptir — bu
    kiyas bizim aleyhimize, o yuzden kazanmak anlamli.

Kullanim: python -m scripts.morning_scoreboard
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

LEDGER = ROOT / "data" / "results" / "morning_history.csv"


def main() -> int:
    if not LEDGER.exists():
        print("[skor] defter yok:", LEDGER)
        return 0
    led = pd.read_csv(LEDGER)
    if led.empty:
        print("[skor] defter bos")
        return 0

    for c in ("dt", "decision_date", "target_date"):
        led[c] = pd.to_datetime(led[c])
    led["target_date"] = led["target_date"].dt.normalize()

    from src import data as D
    cons = D.load_or_create_consumption()
    act = cons["rt_cons"].dropna(); act.index = pd.to_datetime(act.index)
    plan = cons["load_plan"].dropna(); plan.index = pd.to_datetime(plan.index)

    # --- gerceklesen ve TEIAS planini doldur
    need_act = led["actual_mw"].isna()
    if need_act.any():
        led.loc[need_act, "actual_mw"] = act.reindex(led.loc[need_act, "dt"]).values
    if "teias_mw" not in led.columns:
        led["teias_mw"] = np.nan
    need_t = led["teias_mw"].isna()
    if need_t.any():
        led.loc[need_t, "teias_mw"] = plan.reindex(led.loc[need_t, "dt"]).values

    ok = led["pred_mw"].notna() & led["actual_mw"].notna() & (led["actual_mw"] > 0)
    led.loc[ok, "mape_hour"] = (
        (led.loc[ok, "pred_mw"] - led.loc[ok, "actual_mw"]).abs() / led.loc[ok, "actual_mw"] * 100)
    led["teias_mape_hour"] = np.where(
        led["teias_mw"].notna() & ok,
        (led["teias_mw"] - led["actual_mw"]).abs() / led["actual_mw"] * 100, np.nan)

    led.to_csv(LEDGER, index=False)

    # --- gun bazli ozet
    g = (led[ok].groupby("target_date")
         .agg(biz=("mape_hour", "mean"),
              teias=("teias_mape_hour", "mean"),
              n=("mape_hour", "size")))
    g = g[g["n"] >= 20]
    if g.empty:
        print(f"[skor] henuz degerlendirilecek tam gun yok "
              f"(defterde {led['target_date'].nunique()} hedef gun, hepsi gelecekte)")
        pend = led.loc[~ok, "target_date"].nunique()
        print(f"[skor] bekleyen hedef gun: {pend}")
        return 0

    g["fark"] = g["biz"] - g["teias"]
    print(f"=== SABAH SOZLESMESI CANLI SKOR ({len(g)} hedef gun) ===")
    print(f"  donem: {g.index.min().date()} -> {g.index.max().date()}")
    print(f"  bizim   : {g['biz'].mean():.4f}%")
    tt = g["teias"].dropna()
    if len(tt):
        print(f"  TEIAS   : {tt.mean():.4f}%  ({len(tt)} gun ortak)")
        gg = g.dropna(subset=["teias"])
        print(f"  fark    : {gg['fark'].mean():+.4f} pp | kazanan {(gg['fark'] < 0).sum()}/{len(gg)}"
              f" ({(gg['fark'] < 0).mean() * 100:.0f}%)")
    for lbl, sub in [("son 7", g.tail(7)), ("son 30", g.tail(30))]:
        if len(sub) >= 2:
            print(f"  {lbl:8s}: bizim {sub['biz'].mean():.4f}% | TEIAS {sub['teias'].mean():.4f}%"
                  if sub["teias"].notna().any() else f"  {lbl:8s}: bizim {sub['biz'].mean():.4f}%")
    print("\n  gunluk:")
    print(g.round(3).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
