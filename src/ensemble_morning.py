"""SABAH SOZLESMESI — heterojen uzman toplulugu.

Uzmanlar (hepsi ayni feature matrisini gorur):
  lgbm : LightGBM      agirlik 0.15
  cat  : CatBoost      agirlik 0.25
  mlp2 : sinir agi     agirlik 0.60   <- profil taban + residual head + hour/dow/month embedding

Sabit agirliklar 720 gun walk-forward uzerinde secildi (0.15/0.25/0.60 -> 1.6592%,
mlp2 tek basina 1.7430% ayni pencerede). Uyarlanabilir walk-forward agirlik daha
kotu cikti (1.6719%), o yuzden sabit tutuluyor.

Rollback: ENSEMBLE=off ortam degiskeni ile yalniz LightGBM'e donulur.
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd

from . import models as M

WEIGHTS = {"lgbm": 0.10, "cat": 0.20, "mlp2": 0.70}
EXPERTS = tuple(WEIGHTS)
# MLP seed listesi. Tek seed egitim varyansi yuksek; 3 seed ortalamasi 180 gunluk
# protoklde -0.022 pp, 720 gunluk terfi dogrulamasinda toplulukla birlikte -0.043 pp.
# 5 seed ayni sonucu verdi (doyum) -> 3'te birakildi.
MLP_SEEDS = [7, 17, 27]
# Egitim penceresi (gun). Pencere taramasi (180 gun, ref 1.6584):
#   1000g 1.6584 | 1500g 1.5931 | 2000g 1.5707 | 2500g 1.6257
#   3000g 1.7125 | 4500g 1.9126  -> net tepe 2000.
# 720 gunluk terfi dogrulamasi: 1.6281 -> 1.5851 (-0.043).
TRAIN_DAYS = 2000


def enabled() -> bool:
    return os.environ.get("ENSEMBLE", "on").lower() not in ("off", "0", "false", "no")


# --------------------------------------------------------------------------
# uzman egitimi
# --------------------------------------------------------------------------
def _fit_lgbm(X, yv):
    m, f = M.fit(X, yv, list(X.columns))
    return {"kind": "lgbm", "model": m, "feats": f}


def _fit_cat(X, yv):
    from catboost import CatBoostRegressor
    m = CatBoostRegressor(iterations=1500, learning_rate=0.05, depth=8, l2_leaf_reg=3.0,
                          loss_function="RMSE", random_seed=42, verbose=0, thread_count=-1)
    m.fit(X.values, yv.values)
    return {"kind": "cat", "model": m}


def _fit_mlp(X, yv, seed: int = 7):
    import torch
    import torch.nn as nn

    torch.manual_seed(seed); np.random.seed(seed)
    idx = X.index
    hh = idx.hour.values.astype(np.int64)
    dd = idx.dayofweek.values.astype(np.int64)
    mm = (idx.month.values - 1).astype(np.int64)

    Xv = X.values.astype(np.float64)
    mu = np.nanmean(Xv, 0); mu = np.where(np.isfinite(mu), mu, 0.0)
    sd = np.nanstd(Xv, 0); sd = np.where(np.isfinite(sd) & (sd > 1e-9), sd, 1.0)
    Xs = np.nan_to_num((Xv - mu) / sd, nan=0.0, posinf=0.0, neginf=0.0)

    # profil taban: (hafta gunu, saat) ortalamasi — son 12 hafta, sadece egitim verisi
    rec = pd.DataFrame({"v": yv.values}, index=idx)
    rec["d"] = idx.dayofweek; rec["h"] = idx.hour
    rec = rec[rec.index >= idx.max() - pd.Timedelta(weeks=12)]
    prof = rec.groupby(["d", "h"])["v"].mean()
    glob = float(yv.mean())
    base = np.array([prof.get((a, b), glob) for a, b in zip(dd, hh)])
    r = yv.values - base
    rm, rs = float(r.mean()), float(r.std() + 1e-9)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    Xt = torch.tensor(Xs, dtype=torch.float32, device=dev)
    Ht = torch.tensor(hh, dtype=torch.long, device=dev)
    Dt = torch.tensor(dd, dtype=torch.long, device=dev)
    Mt = torch.tensor(mm, dtype=torch.long, device=dev)
    yt = torch.tensor((r - rm) / rs, dtype=torch.float32, device=dev).view(-1, 1)

    class Net(nn.Module):
        def __init__(s, nf):
            super().__init__()
            s.eh = nn.Embedding(24, 8); s.ed = nn.Embedding(7, 4); s.em = nn.Embedding(12, 3)
            s.b1 = nn.Sequential(nn.Linear(nf + 15, 384), nn.SiLU(), nn.Dropout(0.10))
            s.b2 = nn.Sequential(nn.Linear(384, 192), nn.SiLU(), nn.Dropout(0.10))
            s.b3 = nn.Sequential(nn.Linear(192, 96), nn.SiLU())
            s.out = nn.Linear(96, 1)

        def forward(s, x, h, d, m):
            z = torch.cat([x, s.eh(h), s.ed(d), s.em(m)], 1)
            return s.out(s.b3(s.b2(s.b1(z))))

    net = Net(len(X.columns)).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=3e-3, weight_decay=3e-4)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-3, total_steps=90)
    lossf = nn.HuberLoss(delta=1.0)
    n = len(yt); bs = 1024
    g = torch.Generator().manual_seed(7)
    for _ in range(90):
        net.train()
        perm = torch.randperm(n, generator=g).to(dev)
        for i in range(0, n, bs):
            j = perm[i:i + bs]
            opt.zero_grad()
            lossf(net(Xt[j], Ht[j], Dt[j], Mt[j]), yt[j]).backward()
            opt.step()
        sch.step()
    net.eval()
    return {"kind": "mlp2", "net": net, "mu": mu, "sd": sd, "prof": prof,
            "glob": glob, "rm": rm, "rs": rs, "dev": dev}


def _fit_mlp_multi(X, yv):
    """MLP_SEEDS uzunluguna gore tek ya da cok seed'li egitim."""
    parts = [_fit_mlp(X, yv, seed=s) for s in MLP_SEEDS]
    if len(parts) == 1:
        return parts[0]
    return {"kind": "mlp2", "parts": parts}


_FITTERS = {"lgbm": _fit_lgbm, "cat": _fit_cat, "mlp2": _fit_mlp_multi}


# --------------------------------------------------------------------------
# tahmin
# --------------------------------------------------------------------------
def _pred_lgbm(e, X):
    return np.asarray(M.predict(e["model"], e["feats"], X))


def _pred_cat(e, X):
    return e["model"].predict(X.values)


def _pred_mlp(e, X, tgt):
    import torch
    if "parts" in e:                       # cok-seed: bagimsiz aglarin ortalamasi
        return np.mean([_pred_mlp(p, X, tgt) for p in e["parts"]], axis=0)
    Xs = np.nan_to_num((X.values.astype(np.float64) - e["mu"]) / e["sd"],
                       nan=0.0, posinf=0.0, neginf=0.0)
    hh = tgt.hour.values.astype(np.int64)
    dd = tgt.dayofweek.values.astype(np.int64)
    mm = (tgt.month.values - 1).astype(np.int64)
    base = np.array([e["prof"].get((a, b), e["glob"]) for a, b in zip(dd, hh)])
    dev = e["dev"]
    with torch.no_grad():
        p = e["net"](torch.tensor(Xs, dtype=torch.float32, device=dev),
                     torch.tensor(hh, dtype=torch.long, device=dev),
                     torch.tensor(dd, dtype=torch.long, device=dev),
                     torch.tensor(mm, dtype=torch.long, device=dev)).cpu().numpy().ravel()
    return base + p * e["rs"] + e["rm"]


# --------------------------------------------------------------------------
# topluluk
# --------------------------------------------------------------------------
def fit(X, yv, experts=None) -> dict:
    """Uzmanlari egit. Bir uzman patlarsa digerleriyle devam eder."""
    names = tuple(experts) if experts else (EXPERTS if enabled() else ("lgbm",))
    out = {}
    for nm in names:
        try:
            out[nm] = _FITTERS[nm](X, yv)
        except Exception as ex:                       # noqa: BLE001
            print(f"[ensemble] {nm} egitilemedi, atlandi: {str(ex)[:120]}")
    if not out:
        raise RuntimeError("hicbir uzman egitilemedi")
    return out


def predict(models: dict, X, tgt: pd.DatetimeIndex) -> pd.Series:
    """Agirlikli topluluk tahmini. Eksik uzmanin agirligi mevcutlara dagitilir."""
    tgt = pd.DatetimeIndex(tgt)
    vals = {}
    for nm, e in models.items():
        if nm == "lgbm":
            vals[nm] = np.asarray(_pred_lgbm(e, X), dtype=np.float64)
        elif nm == "cat":
            vals[nm] = np.asarray(_pred_cat(e, X), dtype=np.float64)
        elif nm == "mlp2":
            vals[nm] = np.asarray(_pred_mlp(e, X, tgt), dtype=np.float64)
    if not vals:
        raise RuntimeError("tahmin ureten uzman yok")

    tot = sum(WEIGHTS[nm] for nm in vals) or 1.0
    print("[ensemble] " + " | ".join(f"{nm} w={WEIGHTS[nm] / tot:.2f}" for nm in vals))
    num = np.zeros(len(tgt), dtype=np.float64)
    den = np.zeros(len(tgt), dtype=np.float64)
    for nm, v in vals.items():
        w = WEIGHTS[nm] / tot
        ok = np.isfinite(v)
        num += np.where(ok, v * w, 0.0)
        den += np.where(ok, w, 0.0)
    out = np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan)
    return pd.Series(out, index=tgt, name="pred_mw")
