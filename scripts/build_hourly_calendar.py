"""Excel takvimini saatlik feature matrisine cevir.

Kaynak: turkiye_ozel_gun_saatlik_talep_feature_takvimi_2018_2027_v2.xlsx / Event_Windows
Kural (README): start_datetime dahil, end_datetime_exclusive haric.

Cikti: data/calendar/hourly_events.parquet  (saat indeksli, ~30 anlamli bayrak)
"""
import os, sys, numpy as np, pandas as pd
XLS = r"C:\Users\datti\OneDrive - MRC\İletişim sitesi - _Economic_Dispatch_Modeling\Dagıtılan Enerji Tahmini Internal Projesi\99_Library\turkiye_ozel_gun_saatlik_talep_feature_takvimi_2018_2027_v2.xlsx"
OUT = sys.argv[1] if len(sys.argv) > 1 else r"C:\projeler\turkiye-t2-daysahead\data\calendar\hourly_events.parquet"

ev = pd.read_excel(XLS, "Event_Windows")
ev["s"] = pd.to_datetime(ev["start_datetime"])
ev["e"] = pd.to_datetime(ev["end_datetime_exclusive"])
ev["dur"] = pd.to_numeric(ev["duration_hours"], errors="coerce")
ev = ev[ev["dur"].notna() & (ev["dur"] > 0)].copy()

# --- etiket -> anlamli blok eslemesi -----------------------------------------
RULES = [
    # (cikis_kolonu, eslesme onekleri)
    ("cal_ramazan_sahur",   ["ramazan_sahur", "imsak"]),
    ("cal_ramazan_iftar",   ["ramazan_iftar"]),
    ("cal_ramazan_teravih", ["ramazan_teravih", "teravih"]),
    ("cal_ramazan_gunduz",  ["ramazan_oruc"]),
    ("cal_kandil_gun",      ["kandili_gun", "kandil_gun"]),
    ("cal_kandil_aksam",    ["kandili_aksam", "kandil_aksam", "kadir_gecesi"]),
    ("cal_bayram_arefe",    ["bayram_arefe", "arefe_hazirlik"]),
    ("cal_bayram_namaz",    ["bayram_namazi", "namazi_sabah"]),
    ("cal_bayram_ziyaret",  ["ziyaret"]),
    ("cal_bayram_kesim",    ["kesim", "et_isleme"]),
    ("cal_bayram_donus",    ["donus_yolculugu", "donus"]),
    ("cal_milli_toren",     ["sabah_toren", "toren"]),
    ("cal_milli_aksam",     ["aksam_etkinlik"]),
    ("cal_yarim_gun",       ["yarim_gun", "arefesi_yarim"]),
    ("cal_okul_acik",       ["okul_acik_donem", "okul_open"]),
    ("cal_okul_sabah",      ["sabah_07_10", "servis", "okul_acilis"]),
    ("cal_okul_cikis",      ["cikis_15_18", "ogle_cikis", "cikis"]),
    ("cal_okul_karne",      ["karne"]),
    ("cal_okul_tatil",      ["yariyil", "yaz_tatili", "sömestr", "somestr", "ara_tatil"]),
    ("cal_okul_ilkgun",     ["ilk_gun", "ilkgun"]),
    ("cal_idari_izin",      ["idari_izin"]),
    ("cal_yilbasi_gece",    ["yilbasi_gecesi", "yilbasi_1_ocak_gec"]),
    ("cal_hicri",           ["hicri", "asure"]),
    ("cal_uc_aylar",        ["uc_aylar", "regaib"]),
    ("cal_milli_gun",       ["milli_bayram", "yilbasi_resmi", "resmi_tatil"]),
    ("cal_bridge",          ["bridge"]),
    ("cal_secim",           ["secim"]),
    ("cal_uyum_haftasi",    ["uyum_haftasi"]),
    ("cal_ramazan_ayi",     ["ramazan_ayi"]),
    ("cal_bayram_gun",      ["bayram_1_gun", "bayram_2_gun", "bayram_3_gun", "bayram_4_gun"]),
]

def match(lbl):
    l = str(lbl).lower()
    hits = []
    for col, prefixes in RULES:
        if any(p in l for p in prefixes):
            hits.append(col)
    return hits

ev["cols"] = ev["feature_label"].map(match)
print("eslesmeyen etiketler:")
un = ev[ev["cols"].map(len) == 0]["feature_label"].unique()
print("  ", len(un), list(un)[:30])

start = ev["s"].min().normalize()
end = max(ev["e"].max().ceil("h"), pd.Timestamp("2028-01-01"))
grid = pd.date_range(start, end, freq="h")
print(f"grid: {grid[0]} -> {grid[-1]} ({len(grid)} saat)")

mat = pd.DataFrame(0, index=grid, columns=[c for c, _ in RULES], dtype=np.int8)

# her olay penceresini saatlik grid'e yay
gset = grid
for _, r in ev.iterrows():
    if not r["cols"]:
        continue
    hrs = pd.date_range(r["s"].floor("h"), (r["e"] - pd.Timedelta(seconds=1)).floor("h"), freq="h")
    hrs = hrs.intersection(gset)
    if len(hrs) == 0:
        continue
    for c in r["cols"]:
        mat.loc[hrs, c] = 1

# okul acikken sabah/cikis pencereleri disinda kalan saatler icin is_open yeterli
mat["cal_school_day"] = mat["cal_okul_acik"].values
out = mat.copy()
out.index.name = "datetime"
os.makedirs(os.path.dirname(OUT), exist_ok=True)
out.to_parquet(OUT)
print(f"\nyazildi: {OUT}  {out.shape}")
print("\nkolon doluluk (2018-2028):")
for c in out.columns:
    n = int(out[c].sum())
    if n:
        print(f"  {c:24s} {n:7d} saat  ({n/len(out)*100:5.1f}%)")
print("\n2018-2028 toplam saat:", len(out))
