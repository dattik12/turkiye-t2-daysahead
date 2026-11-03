"""AGIRLIK SEMALARI — rejime gore degisen uzman agirliklari.

Girdi: expert_preds.parquet (autoloop_load.py LOOP_ROUND=5 ciktisi)
  kolonlar: dt, y, lgbm, cat, mlp

Kural: her 30 gunluk blok icin agirliklar SADECE onceki pencerelerin
GERCEKLESMIS tahminlerinden kestirilir (out-of-sample, sizinti yok).

Semalar:
  sabit        uretimdeki .10/.20/.70
  global       tek agirlik seti, tum gecmis
  mevsim       4 mevsim
  hafta_sonu   hafta ici / sonu
  mevsim_x_ws  mevsim x hafta sonu
  tatil        resmi tatil / normal
  ay           12 ay

Kullanim: python weight_schemes.py [pencere_gun]
"""
import os, sys, itertools, json
os.chdir(r"C:\projeler\turkiye-t2-daysahead")
sys.path.insert(0, r"C:\projeler\turkiye-t2-daysahead")
import numpy as np, pandas as pd
from src import morning as MO

OUT = r"C:\projeler\autoloop"
WIN = int(sys.argv[1]) if len(sys.argv) > 1 else 180
BLOCK = 30
EXPERTS = ["lgbm", "cat", "mlp"]
STEP = 0.05

P = pd.read_parquet(os.path.join(OUT, "expert_preds.parquet"))
P["dt"] = pd.to_datetime(P["dt"])
P = P.sort_values("dt").reset_index(drop=True)
print(f"uzman tahmini: {P.shape} | {P.dt.min()} -> {P.dt.max()}", flush=True)

# --- takvim ---
cal = MO.load_calendar()
HOL_FLAGS = [c for c in ("cal_bayram_gun", "cal_milli_gun", "cal_yarim_gun") if c in cal.columns]
hf = cal[HOL_FLAGS].fillna(0).abs().sum(axis=1) > 0
hol_days = pd.DatetimeIndex(sorted(set(hf[hf].index.normalize())))
d = P["dt"].dt.normalize()
P["is_hol"] = d.isin(hol_days)
P["is_ws"] = P["dt"].dt.dayofweek >= 5
P["month"] = P["dt"].dt.month
P["season"] = ((P["month"] % 12) // 3).map({0: "kis", 1: "ilkbahar", 2: "yaz", 3: "sonbahar"})
print(f"tatil saati: {int(P.is_hol.sum())} / {len(P)}", flush=True)

GRID = [w for w in itertools.product(np.arange(0, 1.0001, STEP), repeat=3)
        if abs(sum(w) - 1) < 1e-9]
GRID = np.array(GRID)
print(f"izgara: {len(GRID)} agirlik kombinasyonu", flush=True)


def best_w(sub):
    """Verilen alt kumede MAPE'yi en kucuk yapan agirlik."""
    if len(sub) < 200:
        return None
    A = sub[EXPERTS].to_numpy()
    y = sub["y"].to_numpy()
    # MAPE = mean(|sum(w_i p_i) - y| / y)
    inv = 1.0 / y
    err = np.abs(A @ GRID.T - y[:, None]) * inv[:, None]
    return GRID[int(err.mean(axis=0).argmin())]


def evaluate(regime_col=None):
    """Yuruyen pencere: her blok icin onceki WIN gunun agirliklari."""
    blocks = pd.date_range(P.dt.min().normalize(), P.dt.max().normalize(), freq=f"{BLOCK}D")
    out = []
    for b in blocks[1:]:
        past = P[(P.dt >= b - pd.Timedelta(days=WIN)) & (P.dt < b)]
        cur = P[(P.dt >= b) & (P.dt < b + pd.Timedelta(days=BLOCK))]
        if len(past) < 500 or len(cur) < 100:
            continue
        gw = best_w(past) if regime_col is None else None
        if gw is None and regime_col is None:
            continue
        gm = best_w(past)
        if regime_col is None:
            w = pd.Series([tuple(gw)] * len(cur), index=cur.index)
        else:
            tbl = {}
            for k, g in past.groupby(regime_col):
                wv2 = best_w(g)
                tbl[k] = wv2 if wv2 is not None else gm
            w = cur[regime_col].map(tbl)
            w = w.where(w.notna(), pd.Series([gm] * len(cur), index=cur.index))
        pr = np.array([np.dot(np.asarray(ww, float), r) for ww, r in
                       zip(w, cur[EXPERTS].to_numpy())])
        out.append(pd.DataFrame({"p": pr, "y": cur["y"].to_numpy(), "dt": cur["dt"],
                                 "is_hol": cur["is_hol"].to_numpy()}))
    if not out:
        return None
    return pd.concat(out)


SCHEMES = [("sabit .10/.20/.70", None, True), ("global", None, False),
           ("mevsim", "season", False), ("hafta_sonu", "is_ws", False),
           ("mevsim x hafta_sonu", None, False), ("tatil", "is_hol", False),
           ("ay", "month", False)]

print(f"\npencere {WIN} gun | blok {BLOCK} gun", flush=True)
print("%-22s %8s %9s %9s %8s" % ("sema", "MAPE", "tatil", "normal", "n"))
res = {}
for nm, col, fixed in SCHEMES:
    if fixed:
        pr = P[EXPERTS].to_numpy() @ np.array([0.10, 0.20, 0.70])
        e = pd.DataFrame({"p": pr, "y": P.y.to_numpy(), "dt": P.dt, "is_hol": P.is_hol})
    elif nm == "mevsim x hafta_sonu":
        P["sw"] = P.season.astype(str) + "_" + P.is_ws.astype(str)
        e = evaluate("sw")
    else:
        e = evaluate(col)
    if e is None:
        print("%-22s  (yetersiz veri)" % nm)
        continue
    err = np.abs(e.p - e.y) / e.y * 100
    res[nm] = dict(all=float(err.mean()), hol=float(err[e.is_hol].mean()),
                   nor=float(err[~e.is_hol].mean()), n=int(len(e)))
    if nm == "global":
        # ADIL KIYAS: sabit agirligi da AYNI satirlarda olc (walk-forward ilk bloku atiyor)
        idx = pd.DatetimeIndex(e.dt)
        same = P.set_index("dt").reindex(idx)
        pf = same[EXPERTS].to_numpy() @ np.array([0.10, 0.20, 0.70])
        ef = np.abs(pf - same.y.to_numpy()) / same.y.to_numpy() * 100
        res["sabit (ayni satirlar)"] = dict(all=float(ef.mean()),
                                           hol=float(ef[same.is_hol.to_numpy()].mean()),
                                           nor=float(ef[~same.is_hol.to_numpy()].mean()),
                                           n=int(len(ef)))
        print("%-22s %7.3f%% %8.3f%% %8.3f%% %8d" % (
            "sabit (ayni satirlar)", res["sabit (ayni satirlar)"]["all"],
            res["sabit (ayni satirlar)"]["hol"], res["sabit (ayni satirlar)"]["nor"],
            res["sabit (ayni satirlar)"]["n"]))

base = res.get("sabit .10/.20/.70")
if base:
    print("\nfark (sabit'e gore):")
    for k, v in sorted(res.items(), key=lambda x: x[1]["all"]):
        print("  %-22s %+7.3f" % (k, v["all"] - base["all"]))

json.dump(dict(window=WIN, results=res), open(os.path.join(OUT, "summary_weights.json"), "w"),
          indent=1, default=str)
print("\n[weights] bitti", flush=True)
