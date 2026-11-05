"""KONTRAT ARSIVI + FEATURE SURUM DAMGASI.

Her 09:00 kosusundan sonra calisir:
  - latest.parquet -> data/forecast/ptf_features/archive/ptf_YYYY-MM-DD.parquet
  - ayni tarihe sidecar JSON: model surumleri + feature hash + ozet

Amac: PTF ablation (sisik gunes vs duzeltilmis gunes) icin gunluk, degismez
feature seti izi. 30 gun birikince gerceklesen PTF ile karsilastirilabilir.

Kullanim: python scripts/archive_contract.py
"""
import os, sys, json, hashlib
import numpy as np
import pandas as pd

os.chdir(r"C:\projeler\turkiye-t2-daysahead")
sys.path.insert(0, r"C:\projeler\turkiye-t2-daysahead")
from src import config as C

SRC = os.path.join(C.DATA, "forecast", "ptf_features", "latest.parquet")
ARC = os.path.join(C.DATA, "forecast", "ptf_features", "archive")


def main():
    if not os.path.exists(SRC):
        print("HATA: latest.parquet yok")
        return 1
    os.makedirs(ARC, exist_ok=True)
    d = pd.read_parquet(SRC)
    d["datetime"] = pd.to_datetime(d["datetime"])

    day = d["datetime"].min().date().isoformat()          # hedef gun (T+1)
    dst = os.path.join(ARC, f"ptf_{day}.parquet")
    d.to_parquet(dst, index=False)

    cols = [c for c in d.columns if c not in ("datetime",)]
    h = hashlib.sha256(d[cols].to_numpy(dtype="float64", na_value=np.nan)
                       .tobytes()).hexdigest()[:16] if False else \
        hashlib.sha256(pd.util.hash_pandas_object(d[cols], index=False).values.tobytes()).hexdigest()[:16]

    meta = {
        "target_day": day,
        "archived_at": pd.Timestamp.now().isoformat(timespec="seconds"),
        "rows": int(len(d)),
        "cols": int(len(d.columns)),
        "contract_hash": h,
        "solar_status": str(d["solar_status"].iloc[12]),
        "wind_status": str(d["wind_status"].iloc[12]),
        "consumption_mean_mw": round(float(d["consumption_pred_mw"].mean()), 1),
        "solar_peak_mw": round(float(d["solar_pred_mw"].max()), 1),
        "wind_peak_mw": round(float(d["wind_pred_mw"].max()), 1),
        "balance_max_mw": round(float((d.consumption_pred_mw - d.solar_pred_mw
                                       - d.wind_pred_mw - d.residual_load_mw).abs().max()), 6),
    }
    with open(os.path.join(ARC, f"ptf_{day}.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1, ensure_ascii=False)
    print(f"arsivlendi: {dst}")
    print(json.dumps(meta, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
