"""TEIAS PLANI 4. UZMAN — rejime gore karisim.

Gerçek: TEIAS tatilde bizi geciyor (5.48 vs 5.65), normalde biz geciyoruz (1.86 vs 2.68).
Plan 09:00'da elde (onceki gun ~17:00 yayimlaniyor) -> adil kullanim.

Yeniden egitim YOK: topluluk tahminleri expert_preds.parquet'te, plan load_plan'da.
Kiyas: topluluk tek / TEIAS tek / sabit karisim / rejime gore karisim.
Agirliklar her blok icin SADECE gecmisteki 180 gunden kestirilir.

Kullanim: python plan_blend.py [pencere_gun]
"""
import os, sys, json, itertools
os.chdir(r"C:\projeler\turkiye-t2-daysahead")
sys.path.insert(0, r"C:\projeler\turkiye-t2-daysahead")
import numpy as np, pandas as pd
from src import data as D, morning as MO

OUT = r"C:\projeler\autoloop"
WIN = int(sys.argv[1]) if len(sys.argv) > 1 else 180
BLOCK = 30
W_ENS = np.array([0.10, 0.20, 0.70])
EXPERTS = ["lgbm", "cat", "mlp"]

cons = D.load_or_create_consumption()
y = cons["rt_cons"].dropna(); y.index = pd.to_datetime(y.index)
plan = cons["load_plan"].dropna(); plan.index = pd.to_datetime(plan.index)

P = pd.read_parquet(os.path.join(OUT, "expert_preds.parquet"))
P["dt"] = pd.to_datetime(P["dt"])
P["ens"] = P[EXPERTS].to_numpy() @ W_ENS
P["plan"] = plan.reindex(P["dt"]).values
P = P.dropna(subset=["plan"]).sort_values("dt").reset_index(drop=True)
print(f"kiyas: {len(P)} saat | {P.dt.min()} -> {P.dt.max()}", flush=True)

cal = MO.load_calendar()
HOL = [c for c in ("cal_bayram_gun", "cal_milli_gun", "cal_yarim_gun") if c in cal.columns]
hf = cal[HOL].fillna(0).abs().sum(axis=1) > 0
hol_days = pd.DatetimeIndex(sorted(set(hf[hf].index.normalize())))
P["is_hol"] = P.dt.dt.normalize().isin(hol_days)
P["is_ws"] = P.dt.dt.dayofweek >= 5
P["month"] = P.dt.dt.month
P["season"] = ((P["month"] % 12) // 3).map({0: "kis", 1: "ilkbahar", 2: "yaz", 3: "sonbahar"})
P["hs"] = P.is_hol.astype(str) + "_" + P.season.astype(str)
print(f"tatil saati: {int(P.is_hol.sum())} / {len(P)}", flush=True)

GRID = np.arange(0, 1.0001, 0.05)


def mape_of(p, y_):
    return float((np.abs(p - y_) / y_).mean() * 100)


def best_w(sub):
    """Topluluk ile plan arasindaki en iyi agirlik (w = toplulugun payi)."""
    if len(sub) < 200:
        return None
    e, pl, yy = sub.ens.to_numpy(), sub.plan.to_numpy(), sub.y.to_numpy()
    errs = [(np.abs(w * e + (1 - w) * pl - yy) / yy).mean() for w in GRID]
    return float(GRID[int(np.argmin(errs))])


def walk(regime=None):
    blocks = pd.date_range(P.dt.min().normalize(), P.dt.max().normalize(), freq=f"{BLOCK}D")
    out = []
    for b in blocks[1:]:
        past = P[(P.dt >= b - pd.Timedelta(days=WIN)) & (P.dt < b)]
        cur = P[(P.dt >= b) & (P.dt < b + pd.Timedelta(days=BLOCK))]
        if len(past) < 500 or len(cur) < 100:
            continue
        gw = best_w(past)
        if gw is None:
            continue
        if regime is None:
            w = np.full(len(cur), gw)
        else:
            tbl = {}
            for k, g in past.groupby(regime):
                bw = best_w(g)
                tbl[k] = bw if bw is not None else gw
            w = cur[regime].map(tbl).fillna(gw).to_numpy(dtype=float)
        p = w * cur.ens.to_numpy() + (1 - w) * cur.plan.to_numpy()
        out.append(pd.DataFrame({"p": p, "y": cur.y.to_numpy(), "dt": cur.dt,
                                 "is_hol": cur.is_hol.to_numpy()}))
    return pd.concat(out) if out else None


print(f"\npencere {WIN} gun | blok {BLOCK}", flush=True)
rows = []


def rep(nm, d0):
    if d0 is None:
        print(f"  {nm}: yetersiz veri"); return
    err = np.abs(d0.p - d0.y) / d0.y * 100
    h = d0.is_hol.to_numpy()
    rows.append(dict(id=nm, all=float(err.mean()), hol=float(err[h].mean()),
                     nor=float(err[~h].mean()), n=int(len(d0))))


E = pd.DataFrame({"p": P.ens.to_numpy(), "y": P.y.to_numpy(), "dt": P.dt, "is_hol": P.is_hol})
rep("topluluk (mevcut)", E)
T = pd.DataFrame({"p": P.plan.to_numpy(), "y": P.y.to_numpy(), "dt": P.dt, "is_hol": P.is_hol})
rep("TEIAS plani", T)
rep("sabit karisim 0.5", pd.DataFrame({"p": 0.5 * P.ens + 0.5 * P.plan, "y": P.y,
                                      "dt": P.dt, "is_hol": P.is_hol}))
for nm, r in (("global", None), ("tatil", "is_hol"), ("hafta_sonu", "is_ws"),
              ("mevsim", "season"), ("tatil_x_mevsim", "hs")):
    res = walk(r)
    rep(f"karisim {nm}", res)
    if nm == "global" and res is not None:
        # ADIL KIYAS: toplulugu de AYNI satirlarda olc
        idx = pd.DatetimeIndex(res.dt)
        same = P.set_index("dt").reindex(idx)
        e2 = np.abs(same.ens.to_numpy() - same.y.to_numpy()) / same.y.to_numpy() * 100
        h2 = same.is_hol.to_numpy()
        rows.append(dict(id="topluluk (ayni satirlar)", all=float(e2.mean()),
                         hol=float(e2[h2].mean()), nor=float(e2[~h2].mean()), n=int(len(e2))))

base = rows[0]["all"]
print("\n%-22s %8s %9s %9s %8s" % ("kol", "MAPE", "tatil", "normal", "n"), flush=True)
for r in sorted(rows, key=lambda x: x["all"]):
    print("%-22s %7.3f%% %8.3f%% %8.3f%% %8d%+8.3f" % (
        r["id"], r["all"], r["hol"], r["nor"], r["n"], r["all"] - base), flush=True)

json.dump(dict(window=WIN, rows=rows), open(os.path.join(OUT, "summary_plan.json"), "w"),
          indent=1, default=str)
print("\n[plan] bitti", flush=True)
