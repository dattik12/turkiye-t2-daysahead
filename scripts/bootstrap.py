"""Ilk kurulum: EPIAS tam gecmis (2016-01-01 -> bugun) + OpenMeteo hava gecmisi ceker.
Kullanim:  python -m scripts.bootstrap   (onkosul: .env'de EPTR_USERNAME/PASSWORD, invazif cekim)
Not: GitHub Actions'ta ilk calistirmada otomatik yapilir; yerelde de calistirilabilir.
"""
from __future__ import annotations
import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config as C
from src import data as D

START = "2016-01-01"


def main():
    today = pd.Timestamp.now().normalize()
    end = (today + pd.Timedelta(days=1)).date().isoformat()
    cons = D.load_or_create_consumption()
    if not cons.empty and cons.index.max().date().isoformat() >= (today - pd.Timedelta(days=1)).date().isoformat():
        print("Veri seti zaten guncel. Sadece hava tazelenir.")
    else:
        cons = D.append_consumption(cons, START, end)
    # rt_cons yalnizca tamamlanmis gunler (hedef kolon)
    cons = cons[~cons.index.duplicated(keep="last")].sort_index()
    # hava
    nat, cities = D.load_or_create_weather()
    print(f"Tuketim: {cons.index.min()} .. {cons.index.max()} ({len(cons)} saat)")
    print(f"rt_cons eksik: {cons['rt_cons'].isna().sum() if 'rt_cons' in cons else 'YOK'}")
    print(f"Hava (national): {nat.index.min()} .. {nat.index.max()}")
    print(f"Hava (cities): {cities.index.min()} .. {cities.index.max()}")


if __name__ == "__main__":
    main()
