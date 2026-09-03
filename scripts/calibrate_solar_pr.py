"""GES v1 kalibrasyonu: TEIAS aylik uretim capasiyla PR fit etme.

Yontem (aylik capali, durust sinirlariyla):
  E_model(ay) = sum(rad_saati)/1000 * CAP_MW * PR   (histfc_national, OpenMeteo IFS)
  PR* = E_TEIAS(ay) / (sum(rad)/1000 * CAP_MW)

Girdi capasi: TEIAS aylik istatistik (TWh). Varsayilan: Temmuz 2026 = 5.37 TWh
(gunes; TEIAS, GENSED aktarimi). Kapasite: SOLAR_CAPACITY_MW (ETKB Tem'26).

Cikti: onerilen PR + CF kontrolu. PR'yi config'e ELLE yazar (goz karari, loglu).
Saatlik sekil dogrulamasi lisansli-orneklemle backfill asamasina birakildi
(EPiAS ulusal toplam GES endpoint'i yok; lisansli ~%10 kapsar).
"""
from __future__ import annotations
import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
from src import config as C
from src import data as D


def monthly_pr(year: int, month: int, teias_twh: float, capacity_mw: float) -> dict:
    nat = D.load_or_create_weather()[0]
    nat.index = pd.to_datetime(nat.index)
    m = nat[(nat.index.year == year) & (nat.index.month == month)]
    if "shortwave_radiation" not in m:
        raise KeyError("histfc_national'da shortwave_radiation yok")
    full_sun_h = float(m["shortwave_radiation"].clip(lower=0).sum() / 1000.0)
    e_teias_mwh = teias_twh * 1e6
    pr = e_teias_mwh / (full_sun_h * capacity_mw)
    cf = e_teias_mwh / (capacity_mw * len(m))
    return dict(year=year, month=month, saat=len(m), full_sun_h=round(full_sun_h, 1),
                teias_gwh=round(e_teias_mwh / 1000.0, 1), pr=round(pr, 4), cf=round(cf, 4))


def main():
    # kullanim: python -m scripts.calibrate_solar_pr [YYYY-MM] [TEIAS_TWh]
    a1 = sys.argv[1] if len(sys.argv) > 1 else "2026-07"
    twh = float(sys.argv[2]) if len(sys.argv) > 2 else 5.37
    y, mo = map(int, a1.split("-"))
    r = monthly_pr(y, mo, twh, C.SOLAR_CAPACITY_MW or float("nan"))
    print(f"Ay: {r['year']}-{r['month']:02d} | saat={r['saat']} | tam-gunes-esdeger={r['full_sun_h']}h")
    print(f"TEIAS gunes uretimi: {r['teias_gwh']} GWh | kapasite={C.SOLAR_CAPACITY_MW} MW")
    print(f"--> PR* = {r['pr']} | aylik CF = {r['cf']*100:.1f}%")
    if not (0.5 <= r["pr"] <= 1.0):
        print("UYARI: PR fiziksel aralik disinda — capa veya kapasiteyi kontrol et (commit YOK).")
    elif not (0.10 <= r["cf"] <= 0.35):
        print("UYARI: CF supheli — capayi kontrol et.")
    else:
        print(f"OK: config'e SOLAR_PR = {r['pr']} yazilabilir.")


if __name__ == "__main__":
    main()
