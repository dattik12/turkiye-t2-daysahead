"""RITM vs bizim RES tahmini — ayni gerceklesme uzerinde.

Gerceklesme: EPIAS rt-gen-bulk 'wind' kolonu (ulusal saatlik RES uretimi, MW)
Tahminler : (a) RITM wind-forecast endpoint, (b) bizim PTF archive wind_pred_mw

Ikisi de ayni gerceklesmeye karsi olculur -> temiz kiyas.
"""
import os, sys, glob
os.chdir(r"C:\projeler\turkiye-t2-daysahead")
sys.path.insert(0, r"C:\projeler\turkiye-t2-daysahead")
import numpy as np, pandas as pd
from src.wind import pull_actual as PA

START, END = "2026-08-04", "2026-11-02"

# --- gerceklesme
act = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob("data/generation/rt_gen_*.parquet"))])
act.index = pd.DatetimeIndex(act.index)
act = act[~act.index.duplicated(keep="last")].sort_index()
act = act[["wind"]].rename(columns={"wind": "actual"})
print(f"gerceklesme: {act.index.min()} -> {act.index.max()} ({len(act)} saat)")

# --- RITM
try:
    ritm = PA.pull_ritm_forecast(START, END)
    print(f"RITM: {len(ritm)} satir | kolonlar {list(ritm.columns)}")
    if getattr(ritm.index, "tz", None) is None:
        ritm.index = ritm.index.tz_localize("Europe/Istanbul")
    ritm = ritm[~ritm.index.duplicated(keep="last")].sort_index()
    ritm = ritm[["ritm_fc_mw"]]
except Exception as ex:
    print("RITM HATASI:", str(ex)[:300]); ritm = None

# --- bizim
snaps = []
for f in sorted(glob.glob("data/forecast/ptf_features/archive/*.parquet")):
    try:
        d = pd.read_parquet(f)
    except Exception:
        continue
    if "wind_pred_mw" not in d.columns:
        continue
    d = d.copy(); d["_file"] = os.path.basename(f); snaps.append(d)
fc = pd.concat(snaps, ignore_index=True)
fc["datetime"] = pd.to_datetime(fc["datetime"])
fc["target_day"] = fc["datetime"].dt.normalize()
best = (fc[fc["horizon"].astype(str) == "T+1"].sort_values("_file")
        .groupby(["target_day", "datetime"]).last().reset_index())
best = best[["datetime", "wind_pred_mw"]].set_index("datetime")
if getattr(best.index, "tz", None) is None:
    best.index = best.index.tz_localize("Europe/Istanbul")
print(f"bizim: {len(best)} saat")

j = act.join(best, how="inner")
if ritm is not None:
    j = j.join(ritm, how="inner")
print(f"eslesen: {len(j)} saat | {j.index.min()} -> {j.index.max()}")
j = j.dropna(subset=["actual"])
j = j[j["actual"] > 0]

def metrik(col):
    e = j[col] - j["actual"]
    wmape = e.abs().sum() / j["actual"].sum() * 100
    mape = (e.abs() / j["actual"]).mean() * 100
    bias = e.mean()
    corr = j[col].corr(j["actual"])
    return dict(WMAPE=wmape, MAPE=mape, bias=bias, bias_pct=bias / j["actual"].mean() * 100, corr=corr)

print(f"\n=== RES: ayni gerceklesme, {len(j)} saat ===")
print("%-12s %8s %8s %10s %10s" % ("", "WMAPE", "MAPE", "bias", "korelasyon"))
res = {}
for col, lbl in [("wind_pred_mw", "BIZIM"), ("ritm_fc_mw", "RITM")]:
    if col in j.columns:
        m = metrik(col); res[lbl] = m
        print("%-12s %7.2f%% %7.2f%% %+9.0f MW %10.4f" % (lbl, m["WMAPE"], m["MAPE"], m["bias"], m["corr"]))
        print("%-12s %8s %8s %+9.0f%%" % ("", "", "", m["bias_pct"]))

if "BIZIM" in res and "RITM" in res:
    print("\n  fark (bizim - RITM): WMAPE %+.2f pp | MAPE %+.2f pp" % (
        res["BIZIM"]["WMAPE"] - res["RITM"]["WMAPE"], res["BIZIM"]["MAPE"] - res["RITM"]["MAPE"]))
    j["day"] = j.index.normalize()
    g = j.groupby("day").apply(
        lambda d: pd.Series({
            "biz": (d["wind_pred_mw"] - d["actual"]).abs().sum() / d["actual"].sum() * 100,
            "ritm": (d["ritm_fc_mw"] - d["actual"]).abs().sum() / d["actual"].sum() * 100,
        }), include_groups=False)
    g["fark"] = g["biz"] - g["ritm"]
    print("  gunluk: bizim %.2f%% | RITM %.2f%% | kazandigimiz gun %d/%d (%.0f%%)" % (
        g["biz"].mean(), g["ritm"].mean(), (g["fark"] < 0).sum(), len(g), (g["fark"] < 0).mean() * 100))
    g.to_csv(r"C:\projeler\wind_vs_ritm_daily.csv")
    j.to_csv(r"C:\projeler\wind_vs_ritm_hourly.csv")
    print("\n  aylik:")
    g2 = g.copy(); g2["ay"] = pd.to_datetime(g2.index).to_period("M")
    print(g2.groupby("ay")[["biz", "ritm"]].mean().round(2).to_string())
