"""FAZ 5b — T+2 YUK RETRAIN (per-horizon anchored egitim).

Fark: egitim satirlari da T+HZ semantigiyle insa edilir (karar ani = hedef - HZ gun,
08:00). Boylece lag72+ guvenli set kullanilir; T+2 serve ile egitim ayni rejimde.

HZ=1 -> kontrol (eski T+1 protokolu, ~%1.67 bekleniyor)
HZ=2 -> yeni T+2 modeli (baseline %4.88)
"""
import os, sys, importlib.util
import numpy as np, pandas as pd

os.chdir(r"C:\projeler\turkiye-t2-daysahead")
sys.path.insert(0, r"C:\projeler\turkiye-t2-daysahead")

from src import data as D
from src import morning as MO
from src import ensemble_morning as EN
from src import features as F

HZ = int(os.environ.get("HZ", "2"))
EVAL_DAYS = int(os.environ.get("EVAL_DAYS", "365"))
RETRAIN = int(os.environ.get("RETRAIN_DAYS", "30"))

spec = importlib.util.spec_from_file_location("la2", os.path.join("scripts", "load_audit2.py"))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def wape(p, a):
    k = np.isfinite(p) & np.isfinite(a)
    s = a[k].sum()
    return float(np.abs(p - a)[k].sum() / s * 100) if s > 0 else float("nan")


def main():
    cons = D.load_or_create_consumption()
    y = cons["rt_cons"].dropna()
    y.index = pd.to_datetime(y.index)
    if getattr(y.index, "tz", None) is not None:
        y.index = y.index.tz_localize(None)

    NAT = m.build_nat(HZ)

    def feats(idx, dec):
        return MO.attach_calendar(MO.build_features(y, NAT, idx, decision_date=dec))

    end = y.index.max().normalize()
    dec_days = pd.date_range(end - pd.Timedelta(days=EVAL_DAYS),
                             end - pd.Timedelta(days=HZ), freq="D")
    print(f"HZ={HZ} | degerlendirme {dec_days.min().date()} -> {dec_days.max().date()} "
          f"({len(dec_days)} gun)")

    recs, models, last_fit = [], None, None
    for i, dec in enumerate(dec_days):
        if models is None or (dec - last_fit).days >= RETRAIN:
            tr_end = dec + pd.Timedelta(hours=MO.DECISION_HOUR - 1)
            tr_idx = pd.date_range(tr_end - pd.Timedelta(days=EN.TRAIN_DAYS), tr_end, freq="h")
            tr_dec = tr_idx.normalize() - pd.Timedelta(days=HZ)      # per-satir karar ani
            Xtr = feats(tr_idx, tr_dec)
            Xtr = Xtr[y.reindex(Xtr.index).notna()]
            models = EN.fit(Xtr, y.reindex(Xtr.index))
            last_fit = dec
            print(f"  [{i+1}/{len(dec_days)}] {dec.date()} egitim n={len(Xtr)} | "
                  f"lag72 NaN {int(Xtr['lag72'].isna().sum()) if 'lag72' in Xtr else -1}",
                  flush=True)
        tgt = pd.date_range(dec + pd.Timedelta(days=HZ), periods=24, freq="h")
        X = feats(tgt, dec)
        p = EN.predict(models, X, tgt) * F.halfday_factor(tgt)
        a = y.reindex(tgt)
        for t, pv, av in zip(tgt, p.to_numpy(), a.to_numpy()):
            recs.append(dict(dt=t, p=pv, a=av))

    R = pd.DataFrame(recs).dropna(subset=["p", "a"])
    R.to_parquet(rf"C:\projeler\load_t{HZ}_preds.parquet", index=False)

    print(f"\n=== YUK T+{HZ} (per-horizon anchored egitim) ===")
    print(f"  WAPE {wape(R.p.to_numpy(), R.a.to_numpy()):5.2f}% | "
          f"nMAE {float((R.p-R.a).abs().mean()):,.0f} MW | "
          f"bias {float((R.p-R.a).mean()):+,.0f} MW | n={len(R)}")

    ts = pd.to_datetime(R.dt)
    mid = ts.min() + (ts.max() - ts.min()) / 2
    for nm, sub in (("1. yari", R[ts <= mid]), ("2. yari", R[ts > mid])):
        print(f"  {nm}: WAPE {wape(sub.p.to_numpy(), sub.a.to_numpy()):5.2f}% | "
              f"bias {float((sub.p-sub.a).mean()):+,.0f} MW | n={len(sub)}")

    R["hr"] = pd.to_datetime(R.dt).dt.hour
    print("\n  Saat bazli:")
    for hr, g in R.groupby("hr"):
        print(f"    {hr:02d}: WAPE {wape(g.p.to_numpy(), g.a.to_numpy()):5.2f}% | "
              f"bias {float((g.p-g.a).mean()):+6.0f} MW")


if __name__ == "__main__":
    main()
