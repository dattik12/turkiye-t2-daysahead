"""PTF sanity check: latest.parquet butunluk + tazelik kapisi.

Kullanim (CI):
    python -m scripts.check_ptf_latest              # data/forecast/ptf_features/latest.parquet
    python -m scripts.check_ptf_latest --path X --max-age-days 1

Kontroller:
  1) dosya var + okunabiliyor,
  2) ptf_store.validate() invariant'lari (48 satir, gap/dup, toplamsallik,
     residual>0, gece sifir, penetrasyon),
  3) tazelik: dosya ici max karar gunu <= max-age-days eski (varsayilan 2 —
     retry/gecikme payi; otesi STALE).
Cikis kodu 0 = saglikli, 1 = bozuk/stale (CI kirmizi).
"""
from __future__ import annotations
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
from src import config as C
from src.features import ptf_store as P


def check(path: str, max_age_days: int) -> list[str]:
    problems: list[str] = []
    if not os.path.exists(path):
        return [f"dosya yok: {path}"]
    try:
        df = pd.read_parquet(path)
    except Exception as ex:
        return [f"parquet okunamadi: {str(ex)[:100]}"]
    if list(df.columns) != P.COLS:
        problems.append(f"sema uyusmazligi: {list(df.columns)}")
        return problems
    try:
        P.validate(df)
    except ValueError as ex:
        problems.append(f"invariant ihlali: {ex}")
    try:
        newest = pd.to_datetime(df["datetime"]).max().tz_localize(None).normalize()
        age = (pd.Timestamp.now().normalize() - newest).days - 2  # D+2 ufku payi
        if age > max_age_days:
            problems.append(f"STALE: en yeni hedef gun {newest.date()} ({age} gun geride)")
    except Exception as ex:
        problems.append(f"tazelik olculemedi: {str(ex)[:80]}")
    return problems


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default=os.path.join(C.PTF_FEATURES_DIR, "latest.parquet"))
    ap.add_argument("--max-age-days", type=int, default=2)
    a = ap.parse_args()
    problems = check(a.path, a.max_age_days)
    if problems:
        print("PTF_CHECK FAILED:")
        for p in problems:
            print(f"  - {p}")
        sys.exit(1)
    print(f"PTF_CHECK OK: {a.path}")


if __name__ == "__main__":
    main()
