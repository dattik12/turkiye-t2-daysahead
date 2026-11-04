"""GES KALIBRASYON TABLOSU — saatlik sekil duzeltmesi.

fit_scale() ile lisansli GES gerceklesmesinden 24 saatlik oran duzeltmesi cikarir.
Cikti: data/calibration/solar_hour_shape.json

Kullanim: python -m scripts.build_solar_cal
"""
from __future__ import annotations
import glob
import json
from pathlib import Path

import pandas as pd

from src import config as C
from src.solar import model as SM
from src.solar import radiation as R

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "calibration" / "solar_hour_shape.json"


def main() -> None:
    files = sorted(glob.glob(str(ROOT / "data" / "generation" / "rt_gen_*.parquet")))
    if not files:
        raise SystemExit("rt_gen dosyasi yok")
    g = pd.concat([pd.read_parquet(f) for f in files])
    g.index = pd.DatetimeIndex(g.index)
    if g.index.tz is None:
        g.index = g.index.tz_localize("Europe/Istanbul")
    g = g[~g.index.duplicated(keep="last")].sort_index()
    act = g["sun"].dropna()

    s = act.index.min().date().isoformat()
    e = (act.index.max() + pd.Timedelta(days=1)).date().isoformat()
    rad = R.history_radiation(s, e)
    if (act.index.tz is None) != (rad.index.tz is None):
        rad.index = rad.index.tz_localize("Europe/Istanbul") if rad.index.tz is None \
            else rad.index.tz_convert("Europe/Istanbul")
    if act.index.tz is None:
        act.index = act.index.tz_localize("Europe/Istanbul")

    fit = SM.fit_scale(rad, act, capacity_mw=None)
    payload = dict(kind="solar_hour_shape", source="rt-gen sun (lisansli GES)",
                   window=[s, str(act.index.max().date())], **fit)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=1, ensure_ascii=False), encoding="utf-8")

    print(f"yazildi: {OUT}")
    print(f"  saat: {fit['n']} | medyan oran {fit['ratio_median']}")
    print("  sekil (1.0 = degisiklik yok):")
    print("  " + " ".join(f"{h}:{v:.2f}" for h, v in sorted(fit["shape_hour"].items())))


if __name__ == "__main__":
    main()
