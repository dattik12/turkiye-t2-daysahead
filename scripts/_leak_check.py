"""Leak denetimi: statik liste taramasi + dinamik H2 degismezlik testi."""
from __future__ import annotations
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def static_check() -> None:
    from src import features as F
    h2lags = [c for c in F.BASE_H2 if re.fullmatch(r"lag\d+", c)]
    assert "lag24" not in F.BASE_H2 and "lag48" not in F.BASE_H2, "H2'de taze lag var!"
    small = [c for c in h2lags if int(c[3:]) < 72]
    assert not small, f"H2'de 72s-alti lag: {small}"
    print(f"statik OK: H2 lag min = {min(int(c[3:]) for c in h2lags)}s (ufuk-guvenli)")


def dynamic_check() -> None:
    import warnings
    warnings.filterwarnings("ignore")
    import pandas as pd
    from src import data as D
    from src.forecast import Engine
    cons = D.load_or_create_consumption()
    nat, cities = D.load_or_create_weather()
    dec = D.last_full_day(cons) - pd.Timedelta(days=5)
    e1 = Engine(cons, nat, cities)
    a, _ = e1.forecast(dec, return_models=True)
    cons2 = cons.copy()
    cutoff = (dec + pd.Timedelta(days=1)).normalize()  # D+1 00:00 sonrasi karar aninda BILINMEZ
    cons2.loc[cons2.index >= cutoff, "rt_cons"] = float("nan")
    e2 = Engine(cons2, nat, cities)
    b, _ = e2.forecast(dec, return_models=True)
    h2a = a[a["horizon"] == 2].reset_index(drop=True)
    h2b = b[b["horizon"] == 2].reset_index(drop=True)
    h1a = a[a["horizon"] == 1].reset_index(drop=True)
    h1b = b[b["horizon"] == 1].reset_index(drop=True)
    same2 = bool((h2a["pred_mw"] == h2b["pred_mw"]).all())
    same1 = bool((h1a["pred_mw"] == h1b["pred_mw"]).all())
    print(f"dinamik H1 degismezlik: {'OK' if same1 else 'LEAK VAR!'}")
    print(f"dinamik H2 degismezlik: {'OK' if same2 else 'LEAK VAR!'}")
    assert same1 and same2, "Tahmin karar-sonrasi tuketime bagimli!"


def main() -> None:
    static_check()
    dynamic_check()
    print("LEAK CHECK: temiz")


if __name__ == "__main__":
    main()
