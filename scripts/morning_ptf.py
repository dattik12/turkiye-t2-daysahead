"""SABAH → PTF KONTRATI (48 satir). Refactor YOK, mevcut moduller cagrilir.

T+1 = yarin 24 saat, T+2 = ertesi gun 24 saat.
Tuketim: sabah toplulugu (ensemble_morning) iki gun icin ayri egitilir.
Gunes/Ruzgar: ptf_store yardimcilari (ayni kaynak, ayni kontrat).
Cikti: data/forecast/ptf_features/latest.parquet + .csv  (48 x 13)

Kullanim: python -m scripts.morning_ptf
"""
from __future__ import annotations
import os, sys
import numpy as np
import pandas as pd

os.chdir(r"C:\projeler\turkiye-t2-daysahead")
sys.path.insert(0, r"C:\projeler\turkiye-t2-daysahead")

from src import config as C
from src import data as D
from src import morning as MO
from src import ensemble_morning as EN
from src.features import ptf_store as PS
from src.solar import model as SM

OUT_DIR = os.path.join(C.DATA, "forecast", "ptf_features")


def build_consumption(decision: pd.Timestamp, days: tuple[int, ...] = (1, 2)):
    """Her ufuk icin ayri egitim + tahmin. Donus: (48,) array + tgt index."""
    cons = D.load_or_create_consumption()
    y = cons["rt_cons"].dropna()
    y.index = pd.to_datetime(y.index)
    if getattr(y.index, "tz", None) is not None:
        y.index = y.index.tz_localize(None)

    nat, _ = D.load_or_create_weather()
    nat.index = pd.to_datetime(nat.index)
    if getattr(nat.index, "tz", None) is not None:
        nat.index = nat.index.tz_localize(None)

    out = {}
    for off in days:
        tgt = pd.date_range(decision + pd.Timedelta(days=off), periods=24, freq="h")
        # hava: lead-N tahmini
        try:
            fc = D.forecast_weather(tgt[0].date().isoformat(),
                                    tgt[-1].date().isoformat())
            fc.index = pd.to_datetime(fc.index)
            if getattr(fc.index, "tz", None) is not None:
                fc.index = fc.index.tz_localize(None)
            nat = nat.reindex(nat.index.union(fc.index))
            for c in fc.columns:
                if c in nat:
                    nat.loc[fc.index, c] = fc[c].to_numpy()
        except Exception as ex:
            print(f"[ptf48] UYARI hava {off}: {str(ex)[:70]}")

        tr_end = decision + pd.Timedelta(hours=MO.DECISION_HOUR - 1)
        tr_start = tr_end - pd.Timedelta(days=EN.TRAIN_DAYS)
        tr_idx = pd.date_range(tr_start, tr_end, freq="h")
        Xtr = MO.build_features(y, nat, tr_idx)
        Xtr = Xtr[y.reindex(Xtr.index).notna()]
        Xtr = MO.attach_calendar(Xtr)
        models = EN.fit(Xtr, y.reindex(Xtr.index))

        X = MO.attach_calendar(MO.build_features(y, nat, tgt))
        p = EN.predict(models, X, tgt)
        p.index = tgt
        from src import features as F
        p = p * F.halfday_factor(tgt)
        out[off] = p
        print(f"[ptf48] T+{off} egitim n={len(Xtr)} | ort {p.mean():,.0f} MW")
    return out


def main():
    decision = pd.Timestamp.now().normalize()
    print(f"[ptf48] karar {decision.date()}")

    cons = build_consumption(decision, (1, 2))
    idx = cons[1].index.append(cons[2].index)
    consumption = pd.concat([cons[1], cons[2]])

    # gunes / ruzgar (ptf_store ile ayni kaynak)
    rad = PS._rad_series(decision)
    tmp = PS._temp_series(decision)
    wind, wstat = PS._wind_series(decision, min_cov=24)
    for s in (rad, tmp, wind):
        if getattr(s.index, "tz", None) is not None:
            s.index = s.index.tz_localize(None)
    solar = SM.solar_from_radiation(rad.reindex(idx), C.SOLAR_CAPACITY_MW,
                                    C.SOLAR_PR, temp_c=tmp.reindex(idx))
    wind = wind.reindex(idx)

    df = pd.DataFrame({
        "datetime": idx,
        "horizon": ["T+1"] * 24 + ["T+2"] * 24,
        "consumption_pred_mw": consumption.to_numpy(dtype="float32"),
        "solar_pred_mw": np.clip(solar.fillna(0).to_numpy(), 0, None).astype("float32"),
        "wind_pred_mw": np.clip(wind.fillna(0).to_numpy(), 0, None).astype("float32"),
    })
    df["renewable_generation_mw"] = (df.solar_pred_mw + df.wind_pred_mw).astype("float32")
    df["residual_load_mw"] = (df.consumption_pred_mw - df.renewable_generation_mw).astype("float32")
    df["renewable_penetration"] = (df.renewable_generation_mw / df.consumption_pred_mw).astype("float32")
    df["residual_ramp_1h"] = df.residual_load_mw.diff().fillna(0).astype("float32")
    df["solar_ramp_1h"] = df.solar_pred_mw.diff().fillna(0).astype("float32")
    df["is_peak_hour"] = df.datetime.dt.hour.between(8, 19)
    df["solar_status"] = np.where(df.solar_pred_mw > 0, "ok", "zero_night")
    df["wind_status"] = wstat
    cols = ["datetime", "horizon", "consumption_pred_mw", "solar_pred_mw", "wind_pred_mw",
            "residual_load_mw", "renewable_generation_mw", "renewable_penetration",
            "solar_status", "wind_status", "is_peak_hour", "residual_ramp_1h", "solar_ramp_1h"]
    df = df[cols]

    # DOGRULAMA
    ident = float((df.consumption_pred_mw - df.solar_pred_mw - df.wind_pred_mw
                   - df.residual_load_mw).abs().max())
    assert len(df) == 48, f"48 satir bekleniyor, {len(df)} geldi"
    assert len(cols) == 13, "13 kolon bekleniyor"
    assert ident < 0.01, f"denge hatasi {ident}"
    assert df.datetime.diff().dropna().dt.total_seconds().eq(3600).all(), "saatlik bosluk"

    os.makedirs(OUT_DIR, exist_ok=True)
    df.to_parquet(os.path.join(OUT_DIR, "latest.parquet"), index=False)
    df.to_csv(os.path.join(OUT_DIR, "latest.csv"), index=False)
    print(f"[ptf48] YAZILDI {len(df)}x{len(cols)} | denge {ident:.6f} MW")
    print(f"[ptf48] tuketim ort {df.consumption_pred_mw.mean():,.0f} | "
          f"gunes tepe {df.solar_pred_mw.max():,.0f} | ruzgar tepe {df.wind_pred_mw.max():,.0f}")


if __name__ == "__main__":
    main()
