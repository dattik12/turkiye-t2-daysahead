"""P1.3-A — HONEST ARSIV VALIDASYONU

Il x model x degisken x ay bazinda: satir sayisi, NaN %, duplicate timestamp.
day1 ve day2 AYRI raporlanir.
Cikti: data/weather/honest/_validation_{kind}.csv + konsol ozeti
"""
import glob
import os
import sys

import pandas as pd

ROOT = r"C:\projeler\turkiye-t2-daysahead\data\weather\honest"


def load_kind(kind):
    files = sorted(glob.glob(os.path.join(ROOT, kind, "*.parquet")))
    files = [f for f in files if not os.path.basename(f).startswith("_")]
    print(f"[{kind}] parquet dosyasi: {len(files)}")
    if not files:
        return None
    frames = []
    for f in files:
        d = pd.read_parquet(f)
        d.index = pd.DatetimeIndex(d.index)
        if d.index.tz is not None:
            d.index = d.index.tz_localize(None)
        d["_prov"] = os.path.basename(f).split("_")[0]
        frames.append(d)
    m = pd.concat(frames).sort_index()
    print(f"[{kind}] toplam satir {len(m)} | {m.index.min()} .. {m.index.max()}")
    # DIKKAT: tum iller tek cercevede -> index tekrari NORMAL (ayni saat her ilde var).
    # Gercek duplicate kontrolu IL BASINA yapilir.
    per_prov_dup = {p: int(g.index.duplicated().sum()) for p, g in m.groupby("_prov")}
    bad_dup = {p: n for p, n in per_prov_dup.items() if n > 0}
    print(f"[{kind}] il basina duplicate: {bad_dup if bad_dup else 'YOK'} "
          f"(iller arasi ortak saat normal, hata degil)")
    return m


def report(kind, m):
    rows = []
    # kolon adi: {var}_previous_day{1,2}_{model}
    for c in m.columns:
        if c == "_prov" or "_previous_day" not in c:
            continue
        model = None
        for _m in ("ecmwf_ifs025", "gfs_seamless", "icon_seamless"):
            if c.endswith("_" + _m):
                model = _m
                head = c[: -len(_m) - 1]
                break
        if model is None:
            continue
        var, day = head.split("_previous_")
        for prov, g in m.groupby("_prov"):
            s = g[c]
            for mo, h in s.groupby(s.index.to_period("M")):
                rows.append(dict(kind=kind, prov=prov, var=var, day=day, model=model,
                                 month=str(mo), n=int(len(h)), nonnull=int(h.notna().sum()),
                                 null_pct=round(100 * h.isna().mean(), 2),
                                 dup=int(h.index.duplicated().sum())))
    r = pd.DataFrame(rows)
    out = os.path.join(ROOT, f"_validation_{kind}.csv")
    r.to_csv(out, index=False)
    print(f"[{kind}] validasyon -> {out} ({len(r)} satir)")

    print(f"\n--- {kind} · degisken x gun (tum il/ay) ---")
    piv = (r.groupby(["var", "day"])
             .apply(lambda x: pd.Series({
                 "ort_doluluk%": round(100 * x.nonnull.sum() / x.n.sum(), 1),
                 "min_doluluk%": round(100 * (x.nonnull / x.n).min(), 1),
                 "bos_ay_sayisi": int((x.nonnull == 0).sum()),
                 "dup": int(x.dup.sum())}), include_groups=False))
    print(piv.to_string())

    print(f"\n--- {kind} · model bazinda doluluk ---")
    pm = (r.groupby("model")
            .apply(lambda x: pd.Series({
                "doluluk%": round(100 * x.nonnull.sum() / x.n.sum(), 1),
                "bos_ay": int((x.nonnull == 0).sum())}), include_groups=False))
    print(pm.to_string())

    bad = r[(r.null_pct > 5) & (r.nonnull > 0)]
    if len(bad):
        print(f"\n[!] %5'ten fazla NaN iceren {len(bad)} parca (ilk 8):")
        print(bad.head(8)[["prov", "var", "day", "model", "month", "null_pct"]].to_string(index=False))
    else:
        print("\n[OK] %5'ten fazla NaN iceren parca yok")


if __name__ == "__main__":
    kinds = sys.argv[1:] or ["wind", "solar"]
    for k in kinds:
        m = load_kind(k)
        if m is not None:
            report(k, m)
        print()
