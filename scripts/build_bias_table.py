"""SAAT BAZLI BIAS TABLOSU — yuk tahmini icin kalibrasyon.

Kaynak: expert_preds.parquet (720 gun walk-forward, out-of-sample, 18720 saat).
Yontem: her saat icin ortalama hata (tahmin - gercek).
Uygulama: yeni tahminden bu tabloyu cikar.

Dogrulama (yuruyen, sizintisiz):
  ham 1.604% | saat_bias 1.512% (-0.093) | bias -323 -> -55

UYARI: tablo backtest kosullarindan geliyor (analiz havasi). Canlida hava tahmini
kullanildigi icin bias bir miktar farkli olabilir. Bu yuzden morning_run HEM
duzeltilmis HEM ham tahmini deftere yazar; canli skor birikince karsilastirilir.

Kullanim: python -m scripts.build_bias_table
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = Path(r"C:\projeler\autoloop\expert_preds.parquet")
OUT = ROOT / "data" / "calibration" / "load_hour_bias.json"
EXPERTS = ["lgbm", "cat", "mlp"]
W = np.array([0.10, 0.20, 0.70])
MIN_SAMPLES = 60    # 90 gunluk pencerede saat basina ~90 ornek var


def main() -> None:
    import sys
    fit_days = int(sys.argv[1]) if len(sys.argv) > 1 else 90   # sadece son N gun
    if not SRC.exists():
        raise SystemExit(f"kaynak yok: {SRC} (autoloop LOOP_ROUND=5 kosulmali)")
    P = pd.read_parquet(SRC)
    P["dt"] = pd.to_datetime(P["dt"])
    P = P.sort_values("dt")
    end = P["dt"].max()
    start = end - pd.Timedelta(days=fit_days)
    P = P[P["dt"] > start]
    P["p"] = P[EXPERTS].to_numpy() @ W
    P["err"] = P["p"] - P["y"]
    P["h"] = P["dt"].dt.hour

    # HOLDOUT: son %25'i tabloya KATILMAZ; gercek dis-ornek kazanci raporlar.
    cut = P["dt"].quantile(0.75)
    fit, hold = P[P["dt"] <= cut], P[P["dt"] > cut]
    g_fit = fit.groupby("h")["err"].mean()
    corr = hold["h"].map(g_fit).fillna(0.0)
    raw = float((hold["p"] - hold["y"]).abs().div(hold["y"]).mean() * 100)
    adj = float(((hold["p"] - corr) - hold["y"]).abs().div(hold["y"]).mean() * 100)

    g = P.groupby("h")["err"].agg(["mean", "std", "count"])
    table = {}
    for h, r in g.iterrows():
        if int(r["count"]) < MIN_SAMPLES:
            continue
        table[int(h)] = round(float(r["mean"]), 1)

    payload = {
        "kind": "load_hour_bias",
        "source": str(SRC),
        "fit_days": fit_days,
        "fit_window": [str(P["dt"].min().date()), str(end.date())],
        "n_hours": int(len(P)),
        "weights": {"lgbm": 0.10, "cat": 0.20, "mlp": 0.70},
        "bias_mw_by_hour": table,
        "overall_bias_mw": round(float(P["err"].mean()), 1),
        "holdout": {"raw_mape": round(raw, 3), "corrected_mape": round(adj, 3),
                    "gain_pp": round(adj - raw, 3)},
        "note": ("tahminden CIKARILIR (err = pred - actual; negatifse tahmin dusuk). "
                 "holdout: tablo verinin ilk %75'inden fit, son %25'inde olculdu = GERCEK "
                 "dis-ornek kazanc. Ilan edilen kazanc bu degeri kullanmali."),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    print(f"yazildi: {OUT}")
    print(f"  {len(table)} saat | ort bias {payload['overall_bias_mw']:+.1f} MW")
    print("  " + " ".join(f"{h}:{v:+.0f}" for h, v in sorted(table.items())))


if __name__ == "__main__":
    main()
