"""Gunluk 03:00 isi:
  1) EPIAS'tan son gunun tamamlanan verisini cek, veri setini guncelle
  2) OpenMeteo hava gecmisini tazele
  3) Karar gunu = son TAM veri gunu; T+1 ve T+2 tahmini uret (ensemble)
  4) Gecmis tahminleri gerceklesenle eslestir -> MAPE log guncelle
  5) Ozet yazdir (CSV + terminal)
GitHub Actions crona 03:00 Turkiye = 00:00 UTC.
"""
from __future__ import annotations
import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config as C
from src import data as D
from src import scoring as S
from src.forecast import Engine


def main():
    today = pd.Timestamp.now().normalize()
    cons = D.load_or_create_consumption()
    if cons.empty:
        print("Ilk calistirma: tam gecmis cekiliyor...")
        cons = D.append_consumption(cons, "2016-01-01",
                                    (today + pd.Timedelta(days=1)).date().isoformat())
    else:
        last = cons.index.max()
        if last.normalize() < today - pd.Timedelta(days=1):
            cons = D.append_consumption(cons,
                                        (last.normalize() + pd.Timedelta(days=1)).date().isoformat(),
                                        (today + pd.Timedelta(days=1)).date().isoformat())

    nat, cities = D.load_or_create_weather()

    decision = D.last_full_day(cons)
    if decision is None:
        print("Tamamlanmis gun yok — bekleyin (rt-cons gun bazinda 24 saat dolmali).")
        return

    engine = Engine(cons, nat, cities)
    fc = engine.forecast(decision)

    master = S.load_master()
    master = S.append_forecast(fc.assign(decision_date=decision.date().isoformat()))
    master = S.reconcile(master, cons)
    hist = S.update_mape_history(master)

    print("=" * 62)
    print(f"Karar gunu (D):        {decision.date()}")
    print(f"Tahmin edilen gunler:   D+1 { (decision+pd.Timedelta(days=1)).date() } | D+2 { (decision+pd.Timedelta(days=2)).date() }")
    print(f"Forecast master satir:  {len(master)}")
    if not hist.empty:
        recent = hist.sort_values(["target_date"]).tail(8)
        print("\nSon MAPE kayitlari (target_date | horizon | mape% | mae_mw | teias%):")
        print(recent[["target_date", "horizon", "mape_pct", "mae_mw", "teias_lp_mape_pct"]].to_string(index=False))
        avg = hist.groupby("horizon")["mape_pct"].mean()
        print("\nBirikimli ortalama MAPE:")
        print(avg.round(3).to_string())
    # hedef gunun ozeti
    d2 = fc[fc["horizon"] == 2]
    if len(d2):
        print(f"\nD+2 ({d2['dt'].dt.date.iloc[0]}) tahmini: tepe {d2['pred_mw'].max():,.0f} MW @{d2.loc[d2['pred_mw'].idxmax(),'dt'].hour:02d}:00 | ort {d2['pred_mw'].mean():,.0f} MW | dip {d2['pred_mw'].min():,.0f} MW")
    print(f"\nDosyalar:\n  forecasts: {C.FORECAST_MASTER}\n  mape:      {C.MAPE_HISTORY}")


if __name__ == "__main__":
    main()
