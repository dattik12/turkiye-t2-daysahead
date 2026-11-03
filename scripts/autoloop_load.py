"""OTONOM DENEY DONGUSU — sabah sozlesmesi (yuk).

Her varyant = bir hipotez. Ayni walk-forward protokolu, ayni veri, ayni sizinti kurali.
Sonuc JSONL defterine yazilir; incumbent'i gecen varyant KEEP isaretlenir.

Kullanim: python autoloop_load.py <eval_gun> <retrain_gun>
"""
import os, sys, json, time, hashlib
import numpy as np, pandas as pd
sys.path.insert(0, r"C:\projeler\turkiye-t2-daysahead")
os.chdir(r"C:\projeler\turkiye-t2-daysahead")
from src import data as D, morning as MO, models as M, config as C, features as F

EVAL = int(sys.argv[1]) if len(sys.argv) > 1 else 180
RETRAIN = int(sys.argv[2]) if len(sys.argv) > 2 else 30
OUT = r"C:\projeler\autoloop"
os.makedirs(OUT, exist_ok=True)
LEDGER = os.path.join(OUT, "ledger.jsonl")

cons = D.load_or_create_consumption()
nat, _ = D.load_or_create_weather(); nat.index = pd.to_datetime(nat.index)
nat = nat[~nat.index.duplicated(keep="last")].sort_index()
y = cons["rt_cons"].dropna(); y.index = pd.to_datetime(y.index)
fd = y.groupby(y.index.normalize()).size()
last_full = fd[fd == 24].index.max()
cal = MO.load_calendar()

WARM = 60
targets = pd.date_range(last_full - pd.Timedelta(days=EVAL + WARM - 1), last_full, freq="D")
EVAL_START = targets[WARM]
print(f"[loop] hedef {len(targets)} gun | degerlendirme {EVAL_START.date()} -> {last_full.date()}", flush=True)


# ---------------- uzmanlar ----------------
def fit_lgbm(X, yv, p=None):
    p = p or {}
    m, f = M.fit(X, yv, list(X.columns))
    return ("lgbm", m, f)


def fit_cat(X, yv, p=None):
    from catboost import CatBoostRegressor
    p = p or {}
    kw = dict(iterations=p.get("iters", 1500), learning_rate=p.get("lr", 0.05),
              depth=p.get("depth", 8), l2_leaf_reg=p.get("l2", 3.0))
    m = CatBoostRegressor(loss_function="RMSE", random_seed=p.get("seed", 42),
                          verbose=0, thread_count=-1, **kw)
    m.fit(X.values, yv.values)
    return ("cat", m, list(X.columns))


def fit_xgb(X, yv, p=None):
    import xgboost as xgb
    p = p or {}
    m = xgb.XGBRegressor(n_estimators=p.get("iters", 1200), learning_rate=p.get("lr", 0.05),
                         max_depth=p.get("depth", 8), subsample=0.8, colsample_bytree=0.8,
                         min_child_weight=20, reg_lambda=1.0, tree_method="hist",
                         n_jobs=-1, random_state=p.get("seed", 42))
    m.fit(X.values, yv.values)
    return ("xgb", m, list(X.columns))


def _mlp_core(X, yv, hid=(384, 192, 96), seed=7, epochs=90, drop=0.10, lr=3e-3):
    import torch, torch.nn as nn
    torch.manual_seed(seed); np.random.seed(seed)
    idx = X.index
    hh = idx.hour.values.astype(np.int64); dd = idx.dayofweek.values.astype(np.int64)
    mm = (idx.month.values - 1).astype(np.int64)
    Xv = X.values.astype(np.float64)
    mu = np.nanmean(Xv, 0); mu = np.where(np.isfinite(mu), mu, 0.0)
    sd = np.nanstd(Xv, 0); sd = np.where(np.isfinite(sd) & (sd > 1e-9), sd, 1.0)
    Xs = np.nan_to_num((Xv - mu) / sd, nan=0.0, posinf=0.0, neginf=0.0)
    rec = pd.DataFrame({"v": yv.values}, index=idx)
    rec["d"] = idx.dayofweek; rec["h"] = idx.hour
    rec = rec[rec.index >= idx.max() - pd.Timedelta(weeks=12)]
    prof = rec.groupby(["d", "h"])["v"].mean(); glob = float(yv.mean())
    base = np.array([prof.get((a, b), glob) for a, b in zip(dd, hh)])
    r = yv.values - base; rm, rs = float(r.mean()), float(r.std() + 1e-9)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    Xt = torch.tensor(Xs, dtype=torch.float32, device=dev)
    Ht = torch.tensor(hh, dtype=torch.long, device=dev)
    Dt = torch.tensor(dd, dtype=torch.long, device=dev)
    Mt = torch.tensor(mm, dtype=torch.long, device=dev)
    yt = torch.tensor((r - rm) / rs, dtype=torch.float32, device=dev).view(-1, 1)
    a, b, c = hid

    class Net(nn.Module):
        def __init__(s, nf):
            super().__init__()
            s.eh = nn.Embedding(24, 8); s.ed = nn.Embedding(7, 4); s.em = nn.Embedding(12, 3)
            s.f = nn.Sequential(
                nn.Linear(nf + 15, a), nn.SiLU(), nn.Dropout(drop),
                nn.Linear(a, b), nn.SiLU(), nn.Dropout(drop),
                nn.Linear(b, c), nn.SiLU(), nn.Linear(c, 1))
        def forward(s, x, h, d, m):
            return s.f(torch.cat([x, s.eh(h), s.ed(d), s.em(m)], 1))

    net = Net(len(X.columns)).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=3e-4)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs)
    lossf = nn.HuberLoss(delta=1.0)
    n = len(yt); bs = 1024
    g = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        net.train()
        perm = torch.randperm(n, generator=g).to(dev)
        for i in range(0, n, bs):
            j = perm[i:i + bs]
            opt.zero_grad(); lossf(net(Xt[j], Ht[j], Dt[j], Mt[j]), yt[j]).backward(); opt.step()
        sch.step()
    net.eval()
    return (net, mu, sd, prof, glob, rm, rs, dev, list(X.columns))


def _mlp_pred(obj, X, tgt):
    import torch
    net, mu, sd, prof, glob, rm, rs, dev, cols = obj
    Xs = np.nan_to_num((X[cols].values.astype(np.float64) - mu) / sd, nan=0.0, posinf=0.0, neginf=0.0)
    hh = tgt.hour.values.astype(np.int64); dd = tgt.dayofweek.values.astype(np.int64)
    mm = (tgt.month.values - 1).astype(np.int64)
    base = np.array([prof.get((a, b), glob) for a, b in zip(dd, hh)])
    with torch.no_grad():
        p = net(torch.tensor(Xs, dtype=torch.float32, device=dev),
                torch.tensor(hh, dtype=torch.long, device=dev),
                torch.tensor(dd, dtype=torch.long, device=dev),
                torch.tensor(mm, dtype=torch.long, device=dev)).cpu().numpy().ravel()
    return base + p * rs + rm


def fit_mlp(X, yv, p=None):
    p = p or {}
    seeds = p.get("seeds", [7])
    return ("mlp", [_mlp_core(X, yv, hid=p.get("hid", (384, 192, 96)), seed=s,
                              epochs=p.get("epochs", 90), drop=p.get("drop", 0.10)) for s in seeds])


FIT = dict(lgbm=fit_lgbm, cat=fit_cat, xgb=fit_xgb, mlp=fit_mlp)


def pred(kind, obj, X, tgt):
    if kind == "lgbm":
        _, m, f = obj
        return np.asarray(M.predict(m, f, X), dtype=float)
    if kind in ("cat", "xgb"):
        _, m, _ = obj
        return np.asarray(m.predict(X.values), dtype=float)
    if kind == "mlp":
        _, nets = obj
        return np.mean([_mlp_pred(o, X, tgt) for o in nets], axis=0)
    raise ValueError(kind)


# ---------------- ek feature ureticileri ----------------
def add_extra_lags(X, yy, lags):
    out = X.copy()
    for L in lags:
        out[f"xlag{L}"] = yy.reindex(out.index - pd.Timedelta(hours=L)).values
    return out


def add_roll(X, yy, hours):
    out = X.copy()
    for H in hours:
        out[f"xroll{H}"] = yy.shift(24).rolling(H, min_periods=H // 2).mean().reindex(out.index).values
    return out


# ---------------- varyantlar (hipotezler) ----------------
ROUND = os.environ.get("LOOP_ROUND", "1")

ROUND1 = [
    dict(id="H00_incumbent", hyp="Mevcut: lgbm .15 / cat .25 / mlp .60 (referans)",
         experts=["lgbm", "cat", "mlp"], w=[.15, .25, .60]),
    dict(id="H01_xgb", hyp="XGBoost 4. uzman olarak eklenirse cesitlilik artar",
         experts=["lgbm", "cat", "xgb", "mlp"], w=[.12, .18, .18, .52]),
    dict(id="H02_mlp_heavy", hyp="MLP agirligi .70'e cikarsa (en iyi tek uzman) kazanir",
         experts=["lgbm", "cat", "mlp"], w=[.10, .20, .70]),
    dict(id="H03_mlp_seed3", hyp="MLP 3 seed ortalamasi varyansi dusurur",
         experts=["lgbm", "cat", "mlp"], w=[.15, .25, .60],
         params=dict(mlp=dict(seeds=[7, 17, 27]))),
    dict(id="H04_extra_lags", hyp="Uzun lag'ler (240/336/504) haftalik donguyu yakalar",
         experts=["lgbm", "cat", "mlp"], w=[.15, .25, .60], lags=[240, 336, 504]),
    dict(id="H05_train_long", hyp="Egitim penceresi 1500 gun buyume egilimini ogrenir",
         experts=["lgbm", "cat", "mlp"], w=[.15, .25, .60], train_days=1500),
    dict(id="H06_mlp_big", hyp="Daha buyuk MLP (512-256-128) kapasiteyi artirir",
         experts=["lgbm", "cat", "mlp"], w=[.15, .25, .60],
         params=dict(mlp=dict(hid=(512, 256, 128)))),
    dict(id="H07_cat_heavy", hyp="CatBoost agirligi .35 (en iyi agac) daha iyi",
         experts=["lgbm", "cat", "mlp"], w=[.05, .35, .60]),
    dict(id="H08_rolls", hyp="24/72/168 saatlik yuvarlanan ortalama seviye hafizasi katar",
         experts=["lgbm", "cat", "mlp"], w=[.15, .25, .60], rolls=[24, 72, 168]),
]

# Round 1 kazananlari: H03 (seed3) ve H05 (1500 gun). Simdi bunlari birlestir.
ROUND2 = [
    dict(id="H00_incumbent", hyp="Mevcut: lgbm .15 / cat .25 / mlp .60 (referans)",
         experts=["lgbm", "cat", "mlp"], w=[.15, .25, .60]),
    dict(id="H10_seed3_tl1500", hyp="Iki kazanan birlikte: seed3 + 1500 gun (toplanir mi?)",
         experts=["lgbm", "cat", "mlp"], w=[.15, .25, .60], train_days=1500,
         params=dict(mlp=dict(seeds=[7, 17, 27]))),
    dict(id="H11_s3tl_xgb", hyp="seed3 + 1500g + XGBoost 4. uzman (dortlu)",
         experts=["lgbm", "cat", "xgb", "mlp"], w=[.12, .18, .18, .52], train_days=1500,
         params=dict(mlp=dict(seeds=[7, 17, 27]))),
    dict(id="H12_seed5_tl1500", hyp="5 seed daha da iyi mi, yoksa doyum noktasi mi",
         experts=["lgbm", "cat", "mlp"], w=[.15, .25, .60], train_days=1500,
         params=dict(mlp=dict(seeds=[7, 17, 27, 37, 47]))),
    dict(id="H13_seed3_tl2000", hyp="2000 gun daha fazla gecmis daha iyi mi",
         experts=["lgbm", "cat", "mlp"], w=[.15, .25, .60], train_days=2000,
         params=dict(mlp=dict(seeds=[7, 17, 27]))),
    dict(id="H14_s3tl_bigmlp", hyp="seed3 + 1500g + buyuk MLP",
         experts=["lgbm", "cat", "mlp"], w=[.15, .25, .60], train_days=1500,
         params=dict(mlp=dict(seeds=[7, 17, 27], hid=(512, 256, 128)))),
    dict(id="H15_s3tl_mlpw", hyp="seed3 + 1500g + MLP agirligi .70",
         experts=["lgbm", "cat", "mlp"], w=[.10, .20, .70], train_days=1500,
         params=dict(mlp=dict(seeds=[7, 17, 27]))),
    dict(id="H16_s3tl_ep120", hyp="seed3 + 1500g + 120 epoch (daha uzun MLP egitimi)",
         experts=["lgbm", "cat", "mlp"], w=[.15, .25, .60], train_days=1500,
         params=dict(mlp=dict(seeds=[7, 17, 27], epochs=120))),
]

VARIANTS = ROUND2 if ROUND == "2" else ROUND1

# ---------------- kosu ----------------
def run_variant(V):
    t0 = time.time()
    models_ = {}
    fit_until = None
    rows = []
    td = V.get("train_days", C.TRAIN_DAYS)
    for day in targets:
        tgt = pd.date_range(f"{day} 00:00", f"{day} 23:00", freq="h")
        if not models_ or tgt[0] >= fit_until:
            dec = day - pd.Timedelta(days=1)
            tr_end = dec + pd.Timedelta(hours=8)
            tr = pd.date_range(tr_end - pd.Timedelta(days=td), tr_end, freq="h")
            X = MO.build_features(y, nat, tr)
            X = X[y.reindex(X.index).notna()]
            X = MO.attach_calendar(X, cal)
            X = add_extra_lags(X, y, V.get("lags", []))
            X = add_roll(X, y, V.get("rolls", []))
            yv = y.reindex(X.index)
            models_ = {}
            params = V.get("params", {})
            for nm in V["experts"]:
                models_[nm] = FIT[nm](X, yv, params.get(nm))
            fit_until = tgt[0] + pd.Timedelta(days=RETRAIN)
        Xp = MO.build_features(y, nat, tgt)
        Xp = MO.attach_calendar(Xp, cal)
        Xp = add_extra_lags(Xp, y, V.get("lags", []))
        Xp = add_roll(Xp, y, V.get("rolls", []))
        wsum = sum(V["w"])
        agg = np.zeros(len(tgt))
        for nm, w in zip(V["experts"], V["w"]):
            agg += w / wsum * pred(nm, models_[nm], Xp, tgt)
        agg = np.clip(agg, 1e3, None) * F.halfday_factor(tgt)
        a = y.reindex(tgt)
        ok = a.notna() & (a > 0)
        if ok.sum() < 20:
            continue
        rows.append({"day": day.date(),
                     "mape": float(np.mean(np.abs(agg[ok] - a[ok]) / a[ok]) * 100),
                     "bias": float(np.mean(agg[ok] - a[ok]))})
    d = pd.DataFrame(rows)
    d = d[pd.to_datetime(d["day"]) >= EVAL_START]
    return dict(mape=float(d.mape.mean()), bias=float(d.bias.mean()), n=int(len(d)),
                secs=round(time.time() - t0, 1))


print(f"[loop] {len(VARIANTS)} varyant", flush=True)
results = []
for i, V in enumerate(VARIANTS):
    try:
        r = run_variant(V)
    except Exception as ex:
        r = dict(mape=float("nan"), bias=float("nan"), n=0, secs=0, error=f"{type(ex).__name__}: {str(ex)[:200]}")
    rec = dict(ts=pd.Timestamp.now().isoformat(timespec="seconds"), id=V["id"], hyp=V["hyp"],
               eval_days=EVAL, retrain=RETRAIN, **r)
    results.append(rec)
    with open(LEDGER, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print("[%d/%d] %-16s MAPE %.4f%% bias %+.0f n=%d (%.0fs)" % (
        i + 1, len(VARIANTS), V["id"], r["mape"], r["bias"], r["n"], r["secs"]), flush=True)

base = next((r for r in results if r["id"] == "H00_incumbent"), None)
print("\n=== SONUC TABLOSU ===", flush=True)
print("%-16s %9s %10s %8s %s" % ("varyant", "MAPE", "fark", "bias", "karar"))
for r in sorted(results, key=lambda x: (np.isnan(x["mape"]), x["mape"])):
    if base and not np.isnan(r["mape"]):
        d = r["mape"] - base["mape"]
        karar = "KEEP" if d < -0.02 else ("esit" if abs(d) <= 0.02 else "rollback")
        print("%-16s %8.4f%% %+9.4f %+8.0f %s" % (r["id"], r["mape"], d, r["bias"], karar))
    else:
        print("%-16s %9s %10s %8s %s" % (r["id"], r["mape"], "-", "-", "HATA"))

json.dump({"eval_days": EVAL, "retrain": RETRAIN, "base": base, "all": results},
          open(os.path.join(OUT, f"summary_{EVAL}.json"), "w"), indent=1, default=str)
print("\n[loop] bitti ->", LEDGER, flush=True)
