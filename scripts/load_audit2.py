"""YUK OUT-OF-TIME DENETIMI v2 — lead_weather day1/day2 (SIZINTISIZ).

ONCEKI SURUM GECERSIZDI: D.forecast_weather() canli API (OM_FC_URL) kullaniyordu
ve nat tek seriydi (lead ayrimi yok).

BU SURUM:
  - T+1 hedefi icin lead_weather `_previous_day1` serileri
  - T+2 hedefi icin `_previous_day2` serileri
  - Ikisi de ARŞIV TAHMINI (historical-forecast-api), ayni vintage
  - Hicbir gun "gerceklesen hava" kullanilmaz

Kontrat: cutoff D 08:00, hedef D+1 / D+2, 24 saat.
"""
import os, sys
import numpy as np, pandas as pd

os.chdir(r"C:\projeler\turkiye-t2-daysahead")
sys.path.insert(0, r"C:\projeler\turkiye-t2-daysahead")

from src import config as C
from src import data as D
from src import morning as MO
from src import ensemble_morning as EN
from src import features as F

LEAD = r"C:\projeler\lead_weather"
EVAL_DAYS = int(os.environ.get("EVAL_DAYS", "180"))
RETRAIN = int(os.environ.get("RETRAIN_DAYS", "30"))
POP = {"istanbul": 15.9, "ankara": 5.8, "izmir": 4.5, "bursa": 3.2, "antalya": 2.7,
       "adana": 2.3, "konya": 2.3, "gaziantep": 2.1, "kayseri": 1.4, "trabzon": 0.8}
VARS = ["temperature_2m", "apparent_temperature", "relative_humidity_2m",
        "shortwave_radiation", "wind_speed_10m", "precipitation", "cloud_cover"]


def build_nat(day: int):
    """Ulusal nufus-agirlikli hava, SADECE archived lead-N tahmini."""
    acc = {}
    tot = 0.0
    for nm, w in POP.items():
        p = os.path.join(LEAD, f"{nm}.parquet")
        if not os.path.exists(p):
            continue
        d = pd.read_parquet(p)
        d.index = pd.DatetimeIndex(d.index)
        if d.index.tz is not None:
            d.index = d.index.tz_localize(None)
        tot += w
        for v in VARS:
            col = f"{v}_previous_day{day}"
            if col not in d:
                continue
            s = d[col] * w
            acc[v] = s if v not in acc else acc[v].add(s, fill_value=0)
    out = pd.DataFrame({v: acc[v] / tot for v in acc})
    print(f"nat(day{day}): {out.shape} | {out.index.min()} -> {out.index.max()}")
    return out


def wape(p, a):
    m = np.isfinite(p) & np.isfinite(a)
    s = a[m].sum()
    return float(np.abs(p - a)[m].sum() / s * 100) if s > 0 else float("nan")


def main():
    cons = D.load_or_create_consumption()
    y = cons["rt_cons"].dropna()
    y.index = pd.to_datetime(y.index)
    if getattr(y.index, "tz", None) is not None:
        y.index = y.index.tz_localize(None)

    N1, N2 = build_nat(1), build_nat(2)
    both = pd.concat([N1, N2]).sort_index()
    both = both[~both.index.duplicated(keep="last")]

    end = y.index.max().normalize()
    days = pd.date_range(end - pd.Timedelta(days=EVAL_DAYS), end - pd.Timedelta(days=2), freq="D")
    print(f"degerlendirme: {days.min().date()} -> {days.max().date()} ({len(days)} gun)")

    recs, models, last_fit = [], None, None
    for i, dec in enumerate(days):
        cut = dec + pd.Timedelta(hours=MO.DECISION_HOUR - 1)
        if models is None or (dec - last_fit).days >= RETRAIN:
            # EGITIM: lead-1 arsiv tahmini (cutoff'a kadar)
            tr_idx = pd.date_range(cut - pd.Timedelta(days=EN.TRAIN_DAYS), cut, freq="h")
            Xtr = MO.attach_calendar(MO.build_features(y, both, tr_idx))
            Xtr = Xtr[y.reindex(Xtr.index).notna()]
            models = EN.fit(Xtr, y.reindex(Xtr.index))
            last_fit = dec
        for off, NAT in ((1, N1), (2, N2)):
            tgt = pd.date_range(dec + pd.Timedelta(days=off), periods=24, freq="h")
            X = MO.attach_calendar(MO.build_features(y, NAT, tgt))
            p = EN.predict(models, X, tgt) * F.halfday_factor(tgt)
            a = y.reindex(tgt)
            for t, pv, av in zip(tgt, p.to_numpy(), a.to_numpy()):
                recs.append(dict(dt=t, hz=off, p=pv, a=av))
        if (i + 1) % 30 == 0:
            print(f"  {i+1}/{len(days)} | {dec.date()}", flush=True)

    R = pd.DataFrame(recs).dropna(subset=["p", "a"])
    R.to_parquet(r"C:\projeler\load_audit2.parquet", index=False)
    print(f"\nornek: {len(R)}")

    print("\n=== YUK OUT-OF-TIME (lead_weather, sizintisiz) ===")
    for hz in (1, 2):
        S = R[R.hz == hz]
        print(f"  T+{hz}: WAPE {wape(S.p.to_numpy(), S.a.to_numpy()):5.2f}% | "
              f"nMAE {float((S.p-S.a).abs().mean()):,.0f} MW | "
              f"bias {float((S.p-S.a).mean()):+,.0f} MW | n={len(S)}")

    S = R[R.hz == 1].copy()
    S["hr"] = pd.to_datetime(S.dt).dt.hour
    print("\n  Saat bazli (T+1):")
    for hr, g in S.groupby("hr"):
        print(f"    {hr:02d}: WAPE {wape(g.p.to_numpy(), g.a.to_numpy()):5.2f}% | "
              f"bias {float((g.p-g.a).mean()):+6.0f} MW")

    # yarim raporu
    ts = pd.to_datetime(S.dt)
    mid = ts.min() + (ts.max() - ts.min()) / 2
    for nm, sub in (("1. yari", S[ts <= mid]), ("2. yari", S[ts > mid])):
        print(f"  {nm}: WAPE {wape(sub.p.to_numpy(), sub.a.to_numpy()):5.2f}% | n={len(sub)}")


if __name__ == "__main__":
    main()
