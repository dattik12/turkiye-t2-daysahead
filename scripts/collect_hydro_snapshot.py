"""Baraj/hidro gunluk snapshot arsivi (EPİAŞ Şeffaflık, eptr2).

Neden: baraj servisleri tarih parametresi kabul etmez — her cagri yalnizca
BUGUNUN anlik goruntusunu doner (17:00 TSİ itibariyla nihai). Bugun
kaydedilmezse o gun kalici olarak kayiptir; gecmis API'den geri cekilemez.

Cikti: data/hydro/raw/hydro_YYYY-MM-DD_{slot}.json — cagri basina ham yanit.
Slot: PTF_RUN_SLOT env (workflow cron slotu) yoksa TR saatinden turetilir.
K1 (sabah) goruntusu gecici, K2 (aksam) final olabilir -> ikisi ayri saklanir.

Kullanim: python -m scripts.collect_hydro_snapshot   (daily.yml: continue-on-error)
"""
from __future__ import annotations
import json, os, sys, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
from src import config as C

HYDRO_DIR = os.path.join(C.DATA, "hydro", "raw")
HYDRO_CALLS = [
    "dams-active-fullness",        # aktif doluluk (%)
    "dams-active-volume",          # aktif hacim
    "dams-water-energy-provision", # suyun enerji karsiligi (GWh)
    "dams-daily-volume",           # gunluk hacim
    "dams-daily-level",            # gunluk kot
]


def _slot() -> str:
    slot = (os.environ.get("PTF_RUN_SLOT") or "").strip().lower()
    if slot in ("morning", "evening"):
        return slot
    return "morning" if pd.Timestamp.now(tz="Europe/Istanbul").hour < 12 else "evening"


def main() -> None:
    from eptr2 import EPTR2
    eptr = EPTR2(use_dotenv=True, recycle_tgt=True, dotenv_path=".env", tgt_path=".")
    tr_now = pd.Timestamp.now(tz="Europe/Istanbul")
    out = {
        "captured_at": tr_now.isoformat(),
        "slot": _slot(),
        "source": "EPIAS Seffaflik (eptr2)",
        "note": "date parametresi yok; bugunun snapshot'i, 17:00 TSİ sonrasi nihai",
        "responses": {},
    }
    ok = 0
    for call in HYDRO_CALLS:
        try:
            df = eptr.call(call)
            out["responses"][call] = df.to_dict(orient="records")
            ok += 1
            print(f"  [{call}] {len(df)} satir")
        except Exception as ex:  # tek cagri hatasi tum arsivi dusurmesin
            out["responses"][call] = {"error": str(ex)[:300]}
            print(f"  [{call}] HATA: {str(ex)[:120]}")
        time.sleep(1.0)  # 429 korumasina kibar davran
    if ok == 0:
        raise RuntimeError("hidro snapshot: hicbir cagri basarili olmadi")
    os.makedirs(HYDRO_DIR, exist_ok=True)
    path = os.path.join(HYDRO_DIR, f"hydro_{tr_now.date().isoformat()}_{out['slot']}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"HYDRO: {ok}/{len(HYDRO_CALLS)} cagri -> {path}")


if __name__ == "__main__":
    main()
