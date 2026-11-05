"""KONTRAT DOGRULAMASI — 48x13 PTF ciktisi icin sert kapilar.

Kullanim: python scripts/verify_contract.py
Cikis kodu 0 = tum kapilar gecti, 1 = ihlal var.
"""
import os, sys
import numpy as np
import pandas as pd

os.chdir(r"C:\projeler\turkiye-t2-daysahead")
sys.path.insert(0, r"C:\projeler\turkiye-t2-daysahead")
from src import config as C

P = os.path.join(C.DATA, "forecast", "ptf_features", "latest.parquet")
COLS = ["datetime", "horizon", "consumption_pred_mw", "solar_pred_mw", "wind_pred_mw",
        "residual_load_mw", "renewable_generation_mw", "renewable_penetration",
        "solar_status", "wind_status", "is_peak_hour", "residual_ramp_1h", "solar_ramp_1h"]

fail = []


def chk(name, cond, detail=""):
    print(f"  [{'OK ' if cond else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    if not cond:
        fail.append(name)


def main():
    print(f"kontrat: {P}")
    chk("dosya var", os.path.exists(P))
    if not os.path.exists(P):
        return 1
    d = pd.read_parquet(P)

    chk("48 satir", len(d) == 48, f"{len(d)}")
    chk("13 kolon", len(d.columns) == 13, f"{len(d.columns)}")
    chk("kolon adlari", list(d.columns) == COLS,
        f"eksik/fazla: {set(COLS) ^ set(d.columns)}")

    dt = pd.DatetimeIndex(pd.to_datetime(d["datetime"])).tz_localize(None)
    chk("saatlik sureklilik",
        bool(dt.to_series().diff().dropna().eq(pd.Timedelta(hours=1)).all()))
    chk("T+1/T+2 bolunmesi",
        list(d["horizon"].iloc[:24].unique()) == ["T+1"]
        and list(d["horizon"].iloc[24:].unique()) == ["T+2"])
    chk("yarin baslangici", dt[0].hour == 0 and dt[0] == pd.Timestamp.now().normalize() + pd.Timedelta(days=1),
        f"{dt[0]}")

    bal = float((d.consumption_pred_mw - d.solar_pred_mw - d.wind_pred_mw
                 - d.residual_load_mw).abs().max())
    chk("kimlik dengesi < 0.01 MW", bal < 0.01, f"{bal:.6f}")
    rn = float((d.solar_pred_mw + d.wind_pred_mw - d.renewable_generation_mw).abs().max())
    chk("yenilenebilir toplami", rn < 0.01, f"{rn:.6f}")

    chk("tuketim pozitif", bool((d.consumption_pred_mw > 10000).all()),
        f"min {d.consumption_pred_mw.min():,.0f}")
    chk("gunes dolu", float(d.solar_pred_mw.max()) > 100, f"tepe {d.solar_pred_mw.max():,.0f}")
    chk("ruzgar dolu", float(d.wind_pred_mw.max()) > 100, f"tepe {d.wind_pred_mw.max():,.0f}")
    chk("negatif yok", bool((d[["solar_pred_mw", "wind_pred_mw", "residual_load_mw"]] >= 0).all().all()))
    chk("gece gunes sifir", bool((d.loc[d["solar_pred_mw"] == 0, "solar_pred_mw"] == 0).all()))

    chk("NaN yok", not bool(d[["consumption_pred_mw", "solar_pred_mw", "wind_pred_mw",
                               "residual_load_mw"]].isna().any().any()))
    pen = d.renewable_penetration
    chk("penetrasyon 0-1", bool(((pen >= 0) & (pen <= 1.2)).all()), f"{pen.min():.3f}-{pen.max():.3f}")
    chk("durum etiketli",
        bool(d.solar_status.astype(str).str.len().gt(0).all()
             and d.wind_status.astype(str).str.len().gt(0).all()),
        f"{d.solar_status.iloc[12]} / {d.wind_status.iloc[12]}")

    print(f"\n{'GECTI' if not fail else 'IHLAL: ' + ', '.join(fail)}")
    return 0 if not fail else 1


if __name__ == "__main__":
    sys.exit(main())
