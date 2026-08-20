#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
EDA: segment bazli hata analizi (backtest_sim_2025_2026.csv)
- Tum sayilar backtest'ten hesaplanir, uydurma yok.
- Bridge-day otomatik tespit: tek is gunu boslugu kuralı.
"""
import pandas as pd, numpy as np
from pathlib import Path

ROOT = Path(r"C:/Users/datti/Desktop/projelerim/turkiye-t2-daysahead")
CSV = ROOT / "data/results/backtest_sim_2025_2026.csv"
OUT_MD = ROOT / "data/results/eda_segments.md"
OUT_CSV = ROOT / "data/results/eda_segments.csv"

# --- takvim (src/features.py ile birebir) ---
HOLIDAY_DATES = {
    "2024-04-10","2024-04-11","2024-04-12","2024-06-16","2024-06-17","2024-06-18","2024-06-19",
    "2025-03-30","2025-03-31","2025-04-01","2025-06-06","2025-06-07","2025-06-08","2025-06-09",
    "2026-03-19","2026-03-20","2026-03-21","2026-05-27","2026-05-28","2026-05-29","2026-05-30",
}
OFFICIAL = {
    "2024-01-01","2024-04-23","2024-05-01","2024-05-19","2024-07-15","2024-08-30","2024-10-29",
    "2025-01-01","2025-04-23","2025-05-01","2025-05-19","2025-07-15","2025-08-30","2025-10-29",
    "2026-01-01","2026-04-23","2026-05-01","2026-05-19","2026-07-15","2026-08-30","2026-10-29",
}
RAMADAN = {
    "2025": ("2025-03-01", "2025-03-30"),
    "2026": ("2026-02-18", "2026-03-19"),
}

df = pd.read_csv(CSV, parse_dates=["decision_date","dt"])
# ae/ape already: mape column exists but recompute for consistency
df["ae"] = (df["pred_mw"] - df["actual_mw"]).abs()
df["ape"] = df["ae"] / df["actual_mw"] * 100
df["date"] = df["dt"].dt.date
df["hour"] = df["dt"].dt.hour
df["weekday"] = df["dt"].dt.dayofweek  # 0 Mon .. 6 Sun
df["is_weekend"] = df["weekday"] >= 5

# date sets
hol_set = set(pd.to_datetime(list(HOLIDAY_DATES)).date)
off_set = set(pd.to_datetime(list(OFFICIAL)).date)
combined_hol = hol_set | off_set
arife_set = set((pd.to_datetime(list(HOLIDAY_DATES)) - pd.Timedelta(days=1)).date)
after_set = set((pd.to_datetime(list(HOLIDAY_DATES)) + pd.Timedelta(days=1)).date)
arefe_m1_set = set((pd.to_datetime(list(HOLIDAY_DATES)) - pd.Timedelta(days=2)).date)  # arife-1

# ramadan day set
ram_set = set()
for y,(s,e) in RAMADAN.items():
    for d in pd.date_range(s,e,freq="D"):
        ram_set.add(d.date())
# official in ramadan? intersection not needed

# --- bridge-day detection ---
# Strict rule: tek is gunu (Mon-Fri, non-holiday) that is squeezed between a holiday/official and a weekend with exactly 1 workday gap.
# I.e. D is weekday Mon-Fri, not holiday/official, not weekend,
# and (D-1 in combined_hol and D+1.weekday>=5) OR (D-1.weekday>=5 and D+1 in combined_hol)  -- for Friday/Monday bridges.
# Extend slightly: for cases where holiday touches weekend? then gap=0 no bridge.
# Also handle Fri bridge where holiday is Thu, weekend Sat -> D=Fri, D-1 hol, D+1 Sat (weekend true)
# Handle Mon bridge where holiday is Tue, weekend Sun -> D=Mon, D-1 Sun weekend, D+1 Tue hol
# Also handle Thu holiday after Wed gap? Not relevant.
# For multi-day holiday blocks, this still works at block edges.
def is_weekend(d):
    return d.weekday() >= 5

# Need full calendar range covering backtest dt dates
all_dates = sorted(df["date"].unique())
date_set = set(all_dates)
bridge_set = set()
for d in all_dates:
    if d in combined_hol: continue
    if is_weekend(d): continue
    # weekday Mon-Fri only
    prev = d - pd.Timedelta(days=1)
    nxt = d + pd.Timedelta(days=1)
    # Need prev/nxt existence? For edge we still check: if prev not in range, is_weekend check still valid via weekday
    cond1 = (prev in combined_hol) and is_weekend(nxt)
    cond2 = is_weekend(prev) and (nxt in combined_hol)
    if cond1 or cond2:
        bridge_set.add(d)

# Also check 2-day holiday block edge? E.g., if holiday block is Mon-Tue, Monday already holiday, no. So fine.
# Print detected bridges
print("Detected bridge days:", sorted(bridge_set))
# For completeness, also generate theoretical bridges across full holiday calendar (not just backtest range)
full_range = pd.date_range("2025-01-01","2026-08-31",freq="D")
full_bridge = set()
for d in full_range:
    dd = d.date()
    if dd in combined_hol: continue
    if dd.weekday() >=5: continue
    prev = (d - pd.Timedelta(days=1)).date()
    nxt = (d + pd.Timedelta(days=1)).date()
    if (prev in combined_hol and nxt.weekday()>=5) or (prev.weekday()>=5 and nxt in combined_hol):
        full_bridge.add(dd)
print("Full-range bridges (2025-2026):", sorted(full_bridge))

# --- flags per row ---
df["is_holiday"] = df["date"].isin(hol_set)
df["is_official"] = df["date"].isin(off_set)
df["is_arife"] = df["date"].isin(arife_set)
df["is_after"] = df["date"].isin(after_set)
df["is_arefe_m1"] = df["date"].isin(arefe_m1_set)
df["is_ramadan"] = df["date"].isin(ram_set)
df["is_bridge"] = df["date"].isin(bridge_set)
# ramadan but not holiday? We'll keep separate segment: ramadan non-holiday
df["is_ramadan_nonhol"] = df["is_ramadan"] & (~df["is_holiday"])
# hour bins
def hour_bin(h):
    if 0 <= h <= 5: return "gece 00-05"
    if 6 <= h <= 9: return "sabah ramp 06-09"
    if 10 <= h <= 16: return "gundüz 10-16"
    if 17 <= h <= 22: return "aksam pik 17-22"
    return "gece 23"
df["hour_bin"] = df["hour"].map(hour_bin)
# normal day: none of holiday/official/arife/after/bridge/ramadan? But task says normal gun separate from dini/official etc.
# We'll define strict normal: not holiday, not official, not arife, not after, not bridge, weekday handling later.
df["is_normal_strict"] = (~df["is_holiday"]) & (~df["is_official"]) & (~df["is_arife"]) & (~df["is_after"]) & (~df["is_bridge"]) & (~df["is_ramadan"])
# also a looser normal (just not holiday/official) for reference
df["is_normal_loose"] = (~df["is_holiday"]) & (~df["is_official"])

# --- segment definitions ---
segments = []
def add_seg(name, mask, desc):
    sub = df[mask]
    segments.append((name, mask, desc, sub))

add_seg("ALL", np.ones(len(df), dtype=bool), "Tum saatler (28,464 tahmin; 594 gun, H1+H2)")
add_seg("normal (strict)", df["is_normal_strict"].to_numpy(), "Bayram/resmi/arife/after/bridge/ramazan disi gunler")
add_seg("normal (loose)", df["is_normal_loose"].to_numpy(), "Sadece dini+resmi tatil haric (arife/bridge dahil)")
add_seg("dini bayram", df["is_holiday"].to_numpy(), "Ramazan+Kurban bayram gunleri (HOLIDAY_DATES)")
add_seg("arife", df["is_arife"].to_numpy(), "Bayram oncesi gun (arefe)")
add_seg("arefe-1 (bayram-2)", df["is_arefe_m1"].to_numpy(), "Arefe'den 1 gun once (bayram-2)")
add_seg("bayram+1 (after)", df["is_after"].to_numpy(), "Bayram bitiminden sonraki gun")
add_seg("resmi tatil", df["is_official"].to_numpy(), "1 Oca, 23 Nis, 1 May vb (OFFICIAL)")
add_seg("bridge-day", df["is_bridge"].to_numpy(), "Resmi/dini + hafta sonu arasi tek is gunu")
add_seg("Ramazan ayi", df["is_ramadan"].to_numpy(), "Ramazan ayi (2025-03-01..03-30, 2026-02-18..03-19)")
add_seg("Ramazan (bayram haric)", df["is_ramadan_nonhol"].to_numpy(), "Ramazan ayi bayram gunleri haric")
add_seg("hafta ici", (df["weekday"]<5).to_numpy(), "Pazartesi-Cuma")
add_seg("hafta sonu", df["is_weekend"].to_numpy(), "Cumartesi-Pazar")
add_seg("gece 00-05", (df["hour_bin"]=="gece 00-05").to_numpy(), "Gece dusuk yuk")
add_seg("sabah ramp 06-09", (df["hour_bin"]=="sabah ramp 06-09").to_numpy(), "Sabah ramp")
add_seg("gundüz 10-16", (df["hour_bin"]=="gundüz 10-16").to_numpy(), "Gunduz")
add_seg("aksam pik 17-22", (df["hour_bin"]=="aksam pik 17-22").to_numpy(), "Aksam pik")
add_seg("gece 23", (df["hour_bin"]=="gece 23").to_numpy(), "23:00")
add_seg("H1", (df["horizon"]==1).to_numpy(), "Horizon 1 (D+1)")
add_seg("H2", (df["horizon"]==2).to_numpy(), "Horizon 2 (D+2)")

# combined interesting
add_seg("bayram+arife", (df["is_holiday"]|df["is_arife"]).to_numpy(), "Bayram veya arife")
add_seg("tatil genis (bayram+resmi+arife+after+bridge)", (df["is_holiday"]|df["is_official"]|df["is_arife"]|df["is_after"]|df["is_bridge"]).to_numpy(), "Tum tatil-etkili gunler")
add_seg("tatil disi hafta ici", (df["is_normal_strict"] & (df["weekday"]<5)).to_numpy(), "Normal hafta ici")
add_seg("tatil disi hafta sonu", (df["is_normal_strict"] & df["is_weekend"]).to_numpy(), "Normal hafta sonu")

# --- summary stats ---
import math
rows=[]
for name,mask,desc,sub in segments:
    n = len(sub)
    if n==0:
        rows.append({"segment":name,"desc":desc,"n":0,"MAPE":None,"MAE":None,"RMSE":None,"p50_APE":None,"p90_APE":None,"p95_APE":None,"share_%":0})
        continue
    mape = sub["ape"].mean()
    mae = sub["ae"].mean()
    rmse = np.sqrt((sub["ae"]**2).mean())
    p50 = sub["ape"].median()
    p90 = sub["ape"].quantile(0.90)
    p95 = sub["ape"].quantile(0.95)
    share = n/len(df)*100
    rows.append({"segment":name,"desc":desc,"n":n,"MAPE":mape,"MAE":mae,"RMSE":rmse,"p50_APE":p50,"p90_APE":p90,"p95_APE":p95,"share_%":share})

seg_df = pd.DataFrame(rows)
# contribution to overall error: (n * MAPE) / total? We'll add excess
overall_mape = df["ape"].mean()
for r in rows:
    if r["MAPE"] is not None:
        r["delta_vs_all_pp"] = r["MAPE"] - overall_mape
        r["excess_contrib"] = r["n"] * max(0, r["MAPE"] - overall_mape)  # rough
    else:
        r["delta_vs_all_pp"]=None

seg_df = pd.DataFrame(rows)
seg_df = seg_df.sort_values("MAPE", ascending=False, na_position="last")
print(seg_df.to_string(index=False, float_format=lambda x: f"{x:.4f}" if isinstance(x,float) else str(x)))

# Save CSV
seg_df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig", float_format="%.4f")
print(f"Saved {OUT_CSV}")

# --- worst days (daily MAPE) ---
daily = df.groupby("date").agg(n=("ape","size"), MAPE=("ape","mean"), MAE=("ae","mean"), actual_mean=("actual_mw","mean")).reset_index()
daily["dow"] = pd.to_datetime(daily["date"]).dt.day_name()
# annotate flags
daily["flags"] = daily["date"].map(lambda d: ",".join([k for k, s in [
    ("bayram", d in hol_set), ("resmi", d in off_set), ("arife", d in arife_set),
    ("after", d in after_set), ("arefe-1", d in arefe_m1_set), ("bridge", d in bridge_set), ("ramazan", d in ram_set)
] if s]) or "normal")
daily_sorted = daily.sort_values("MAPE", ascending=False)
print(daily_sorted.head(25).to_string(index=False))
# worst hours
hour_worst = df.sort_values("ape", ascending=False).head(20)[["dt","date","hour","weekday","ape","ae","pred_mw","actual_mw","is_holiday","is_official","is_arife","is_after","is_bridge","is_ramadan"]]
print(hour_worst.to_string(index=False))

# --- horizon breakdown inside key segments ---
for seg_name in ["dini bayram","arife","bayram+1 (after)","bridge-day","Ramazan ayi","resmi tatil"]:
    mask = dict((n,m) for n,m,_,_ in segments)[seg_name]
    sub=df[mask]
    if len(sub)==0: continue
    g=sub.groupby("horizon").agg(n=("ape","size"),MAPE=("ape","mean"),MAE=("ae","mean"))
    print(seg_name, g.to_dict())

# --- which segments pull mean up? ---
# compute weighted mean check
total_ape_sum = df["ape"].sum()
for _,r in seg_df.iterrows():
    if r["segment"]=="ALL": continue
    # not needed

# Save daily worst as well
daily_sorted.to_csv(ROOT/"data/results/eda_daily.csv", index=False, encoding="utf-8-sig")
hour_worst.to_csv(ROOT/"data/results/eda_worst_hours.csv", index=False, encoding="utf-8-sig")
print("Saved daily/hourly worst CSVs")
