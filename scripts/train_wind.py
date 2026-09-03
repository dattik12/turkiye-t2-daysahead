"""RES modeli egitimi: RITM gerceklesmesi + hub IFS ruzgari + RITM tahmini.

Kullanim:
    python -m scripts.train_wind            # son 60 gun
    python -m scripts.train_wind 90         # son 90 gun

Cikti: models/wind_lgbm.txt + terminalde train forearasyonu (son 7 gun holdout).
RITM `forecast` kolonu gecmis tarihlerde arsiv tahmin ise model residual ogrenir;
degilse (generation'a esitse) kolon dusurulup saf meteorolojik model egitilir
(durust fallback — loga yazilir).
"""
from __future__ import annotations
import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
from src import config as C
from src.wind import met, model as M, pull_actual as PA
from src.wind.weights import load_matrix, weighted_speed


def main():
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    end = (pd.Timestamp.now().normalize() - pd.Timedelta(days=2)).date().isoformat()
    start = (pd.Timestamp(end) - pd.Timedelta(days=days - 1)).date().isoformat()
    print(f"Egitim penceresi: {start} .. {end} ({days} gun)")

    gen = PA.pull_generation(start, end).rename("wind_gen_mw")
    fc = PA.pull_ritm_forecast(start, end)["ritm_fc_mw"]
    same = np.isclose(fc.reindex(gen.index).ffill(), gen, rtol=1e-6).mean()
    use_ritm = same < 0.99
    print(f"RITM forecast generation'a esit orani: {same:.3f} -> "
          f"{'residual ogrenme (ritm_fc girdi)' if use_ritm else 'saf meteorolojik model'}")

    wl = met.history_wind(start, end)
    spd = met.hub_speed_table(wl)
    w10 = wl.pivot_table(index="dt", columns="city", values="w10")
    matrix = load_matrix()
    v = weighted_speed(spd, matrix)
    v10 = weighted_speed(w10, matrix)

    f = M.build_features(v, fc if use_ritm else None)
    f = M.attach_shear(f, v, v10)
    df = f.join(gen, how="inner").dropna(subset=["wind_gen_mw"])
    print(f"Egitim satiri: {len(df)}")

    cut = df.index.max() - pd.Timedelta(days=7)
    tr, te = df[df.index <= cut], df[df.index > cut]
    b = M.train(tr if use_ritm else tr.assign(ritm_fc=0))
    pred = M.predict(b, te)
    mape = (abs(pred - te["wind_gen_mw"]) / te["wind_gen_mw"]).mean() * 100
    mae = abs(pred - te["wind_gen_mw"]).mean()
    rmape = (abs(te["ritm_fc"] - te["wind_gen_mw"]) / te["wind_gen_mw"]).mean() * 100 \
        if use_ritm else float("nan")
    print(f"HOLDOUT son 7 gun: model MAPE %{mape:.2f} MAE {mae:.0f} MW"
          + (f" | RITM MAPE %{rmape:.2f}" if use_ritm else ""))
    M.save(b, C.WIND_MODEL_TXT)
    print(f"Model yazildi: {C.WIND_MODEL_TXT}")


if __name__ == "__main__":
    main()
