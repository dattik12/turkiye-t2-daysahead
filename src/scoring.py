"""Skorlama: tahmin x gerceklesen eslesme + MAPE log + master forecast csv."""
from __future__ import annotations
import os
import numpy as np
import pandas as pd

from . import config as C


COLS = ["decision_date", "dt", "horizon", "pred_mw", "load_plan_mw", "actual_mw", "mape_hour", "mae_mw"]


def load_master() -> pd.DataFrame:
    if os.path.exists(C.FORECAST_MASTER):
        df = pd.read_csv(C.FORECAST_MASTER, parse_dates=["dt"], dtype={"decision_date": str})
    else:
        df = pd.DataFrame(columns=COLS)
    return df


def store_master(df: pd.DataFrame) -> None:
    os.makedirs(C.RESULTS_DIR, exist_ok=True)
    df.to_csv(C.FORECAST_MASTER, index=False)


def append_forecast(engine_out: pd.DataFrame) -> pd.DataFrame:
    """Yeni tahmin satirlarini master'a YAZAR (upsert).
    Ayni (decision_date, dt, horizon) tekrar gelirse ESKİ satir silinip yenisi yazilir.
    Farkli karar gunlarinin ayni hedef gunu (D+1 vs D+2) ayri satir olarak korunur."""
    master = load_master()
    # decision_date hijyeni: her zaman 'YYYY-MM-DD' string (upsert anahtari tutarli olsun)
    master["decision_date"] = pd.to_datetime(master["decision_date"]).dt.strftime("%Y-%m-%d")
    new = engine_out[["decision_date", "dt", "horizon", "pred_mw", "load_plan_mw"]].copy()
    new["decision_date"] = pd.to_datetime(new["decision_date"]).dt.strftime("%Y-%m-%d")
    new["dt"] = pd.to_datetime(new["dt"])
    key = ["decision_date", "dt", "horizon"]
    if not master.empty:
        old_key = master.set_index(key).index
        new_key = new.set_index(key).index
        master = master[~old_key.isin(new_key)]   # eski ayni-key satirlari cikar (upsert)
    new["actual_mw"] = np.nan
    new["mape_hour"] = np.nan
    new["mae_mw"] = np.nan
    out = pd.concat([master, new[COLS]], ignore_index=True)
    out["dt"] = pd.to_datetime(out["dt"])
    out = out.drop_duplicates(subset=key, keep="last").sort_values("dt")
    store_master(out)
    return out


def reconcile(master: pd.DataFrame, cons: pd.DataFrame) -> pd.DataFrame:
    """Gerceklesen (rt_cons) olan gunler için actual/mape/mae doldur."""
    y = cons["rt_cons"]
    master = master.copy()
    master["dt"] = pd.to_datetime(master["dt"])
    day_full = y.dropna().groupby(y.dropna().index.normalize()).size()
    full_days = set(day_full[day_full == 24].index)
    mask = master["dt"].dt.normalize().isin(full_days) & master["actual_mw"].isna()
    actual = master["dt"].map(y)
    master.loc[mask, "actual_mw"] = actual[mask]
    m = master["actual_mw"].notna() & master["pred_mw"].notna()
    master.loc[m, "mape_hour"] = (master.loc[m, "pred_mw"] - master.loc[m, "actual_mw"]).abs() / master.loc[m, "actual_mw"] * 100
    master.loc[m, "mae_mw"] = (master.loc[m, "pred_mw"] - master.loc[m, "actual_mw"]).abs()
    store_master(master)
    return master


def update_mape_history(master: pd.DataFrame) -> pd.DataFrame:
    """target_date x horizon bazinda MAPE ozeti -> mape_history.csv (append/upsert)."""
    rows = []
    for (td, hz), g in master[master["actual_mw"].notna()].groupby([master["dt"].dt.normalize(), "horizon"]):
        n = len(g)
        if n < 24:
            continue
        mape = g["mape_hour"].mean()
        mae = g["mae_mw"].mean()
        lp = g["load_plan_mw"].dropna()
        lp_mape = ((lp - g.loc[lp.index, "actual_mw"]).abs() / g.loc[lp.index, "actual_mw"] * 100).mean() if len(lp) else np.nan
        rows.append(dict(target_date=str(td.date()), horizon=int(hz), n=n,
                         mape_pct=round(mape, 3), mae_mw=round(mae, 1),
                         teias_lp_mape_pct=(round(lp_mape, 3) if pd.notna(lp_mape) else ""),
                         updated_at=pd.Timestamp.now().isoformat(timespec="seconds")))
    if not rows:
        return pd.DataFrame()
    new = pd.DataFrame(rows)
    path = C.MAPE_HISTORY
    if os.path.exists(path):
        old = pd.read_csv(path, dtype={"target_date": str})
        old = old[~old["target_date"].isin(new["target_date"].astype(str)) | ~old["horizon"].isin(new["horizon"])] if False else old
        # upsert: (target_date,horizon) etiketiyle eskiyi düs, yeniyi ekle
        old = old.drop_duplicates(subset=["target_date", "horizon"], keep="last")
        merged = pd.concat([old[~old.set_index(["target_date", "horizon"]).index.isin(
            new.set_index(["target_date", "horizon"]).index)], new], ignore_index=True)
    else:
        merged = new
    merged = merged.sort_values(["target_date", "horizon"]).drop_duplicates(["target_date", "horizon"], keep="last")
    merged.to_csv(path, index=False)
    return merged
