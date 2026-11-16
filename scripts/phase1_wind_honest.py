"""P1.3 — HONEST BASELINE (ruzgar)

KURAL: TEK degisiklik = hava kaynagi.
  wind_mnwp.load_multi() monkey-patch edilir; arsiv yerine
  data/weather/honest/wind/<Il>_<Q>.parquet icindeki
  wind_speed_100m_previous_day{HZ}_{model} kolonlari okunur.
  SHIFT_H, lag maskeleri, egitim penceresi, feature seti — HEPSI AYNI.

Kullanim:
  python scripts/phase1_wind_honest.py 1    # T+1 (previous_day1)
  python scripts/phase1_wind_honest.py 2    # T+2 (previous_day2)
"""
import glob
import os
import re
import sys

import pandas as pd

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)
sys.path.insert(0, r"C:\projeler	urkiye-t2-daysahead\scripts")
import wind_mnwp as W  # noqa: E402

HONEST = r"C:\projeler\turkiye-t2-daysahead\data\weather\honest\wind"
MODELS = ["ecmwf_ifs025", "gfs_seamless", "icon_seamless"]
HZ = int(sys.argv[1]) if len(sys.argv) > 1 else 1
os.environ["HZ"] = str(HZ)      # wind_mnwp lag maskesi bunu okur
VAR = f"wind_speed_100m_previous_day{HZ}"


def load_multi_honest():
    """wind_mnwp.load_multi ile ayni sozlesme, hava kaynagi honest arsiv."""
    per = {}
    for f in glob.glob(os.path.join(HONEST, "*.parquet")):
        base = os.path.basename(f)
        if base.startswith("_"):
            continue
        m = re.match(r"(.+)_(20\d\dQ\d)\.parquet", base)
        if not m:
            continue
        per.setdefault(m.group(1), []).append(f)
    out = {}
    for prov, files in per.items():
        d = pd.concat([pd.read_parquet(f) for f in sorted(files)]).sort_index()
        if getattr(d.index, "tz", None) is not None:
            d.index = d.index.tz_localize(None)
        d = d[~d.index.duplicated(keep="last")]
        got = {}
        for mo in MODELS:
            c = f"{VAR}_{mo}"
            if c in d.columns and d[c].notna().any():
                got[mo] = d[c]
        if got:
            out[prov] = got
    return out


if __name__ == "__main__":
    print(f"=== HONEST BASELINE | HZ={HZ} | kaynak: {VAR} ===", flush=True)
    orig = W.load_multi
    W.load_multi = load_multi_honest
    try:
        n = len(load_multi_honest())
        print(f"honest arsivden okunan il: {n}", flush=True)
        W.main()
    finally:
        W.load_multi = orig
