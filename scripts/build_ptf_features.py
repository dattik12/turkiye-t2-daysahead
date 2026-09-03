"""PTF input feature uretimi: gunluk tuketim tahminini GES/net-yuk ile birlestirir.

Kullanim:
    python -m scripts.build_ptf_features            # son karar gunu
    python -m scripts.build_ptf_features 2026-09-02 # belirli karar gunu

Mevcut tuketim hattini CALISTIRMAZ, sadece forecast_results.csv'yi okur.
SOLAR_CAPACITY_MW tanimsizsa solar NaN + status='unconfigured' yazar (exit 0;
daily run'i bloklamaz). Kalibrasyon/backfill sonraya birakildi.
"""
from __future__ import annotations
import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.features import ptf_store as P


def main():
    decision = sys.argv[1] if len(sys.argv) > 1 else None
    path = P.build_for_decision(decision)
    if path is None:
        print("PTF: uretilecek tahmin bulunamadi (daily pipeline once kosmali).")


if __name__ == "__main__":
    main()
