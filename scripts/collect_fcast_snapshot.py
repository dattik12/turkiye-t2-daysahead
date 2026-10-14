"""IFS tahmin snapshot toplayici: lead-time hizali egitim verisi biriktirir.

Problem: egitim satirlari hedef-saat gerceklesen havasini gorur, sunumda IFS
tahmini gelir (train/serve skew). Cozum: her gun D+1..D+4 IFS tahminini
(run tarihi + kosu slotu etiketli) arsivle; ilerde egitim bu anlik-goruntulerden beslenir.

v2 (2026-10-14): sabah/aksam kosulari ayni gunun FARKLI IFS cycle'larini gorur
(00z/06z). Dedupe anahtarina slot eklendi -> sabah vintage'ini aksam kosusu
artik EZEMEZ. run_ts/run_slot/ifs_cycle_est kolonlari eklendi; yazim oncesi
sehir/hub sayisi ve NaN baraji (eksik cekim arsive girmez). Ruzgar hub'lari
ayri dosyaya yazilir (RES hatti train/serve skew'i icin).

Kullanim: python -m scripts.collect_fcast_snapshot
Cikti: data/weather/fcst_snapshots.parquet (10 sehir) + fcst_snapshots_hubs.parquet (7 hub)."""
from __future__ import annotations
import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
from src import config as C
from src import data as D

SNAP_PATH = os.path.join(C.WEATHER_DIR, "fcst_snapshots.parquet")
SNAP_PATH_HUBS = os.path.join(C.WEATHER_DIR, "fcst_snapshots_hubs.parquet")
HUB_VARS = ["wind_speed_100m", "wind_speed_10m", "wind_direction_100m"]


def _slot() -> str:
    slot = (os.environ.get("PTF_RUN_SLOT") or "").strip().lower()
    if slot in ("morning", "evening"):
        return slot
    return "morning" if pd.Timestamp.now(tz="Europe/Istanbul").hour < 12 else "evening"


def _stamp() -> dict:
    """Kosu ustverisi: gercek zaman + cron slotu + tahmini IFS cycle (sabah=00z, aksam=06z)."""
    tr_now = pd.Timestamp.now(tz="Europe/Istanbul")
    slot = _slot()
    return {
        "run_date": tr_now.date().isoformat(),
        "run_slot": slot,
        "run_ts": tr_now.isoformat(),
        "ifs_cycle_est": "00z" if slot == "morning" else "06z",
    }


def _append_snap(df: pd.DataFrame, path: str, key: list[str]) -> pd.DataFrame:
    if os.path.exists(path):
        old = pd.read_parquet(path)
        df = pd.concat([old, df], ignore_index=True)
    return df.drop_duplicates(subset=key, keep="last")


def main() -> None:
    st = _stamp()
    run = pd.Timestamp.now(tz="Europe/Istanbul").normalize()
    s = (run + pd.Timedelta(days=1)).date().isoformat()
    e = (run + pd.Timedelta(days=4)).date().isoformat()

    # --- 1) sehirler (tuketim hatti: OM_VARS'in tamami) ---
    frames = []
    for c in C.CITIES:
        h = D.om_fetch(C.OM_FC_URL, c["lat"], c["lon"], s, e, vars=C.OM_VARS)
        f = pd.DataFrame({v: h[v] for v in C.OM_VARS}, index=pd.to_datetime(h["time"]))
        f["city"] = c["city"]
        frames.append(f)
    df = pd.concat(frames)
    if df["city"].nunique() < 8:  # eksik cekim arsive yazilmaz
        raise RuntimeError(f"snapshot: yalniz {df['city'].nunique()} sehir geldi (<8) — yazim iptal")
    nan_rate = float(df["temperature_2m"].isna().mean())
    if nan_rate > 0.02:
        raise RuntimeError(f"snapshot: temperature_2m NaN orani %{nan_rate*100:.1f} (>%2) — yazim iptal")
    for k, v in st.items():
        df[k] = v
    df = df.reset_index().rename(columns={"index": "target_dt"})
    df = _append_snap(df, SNAP_PATH, ["run_date", "run_slot", "city", "target_dt"])
    df["run_slot"] = df["run_slot"].fillna("legacy")  # v1 satirlari slot'suz kalir
    df.to_parquet(SNAP_PATH, index=False)
    print(f"SNAPSHOT {st['run_date']} [{st['run_slot']}]: {len(df)} satir -> {SNAP_PATH}")

    # --- 2) ruzgar hub'lari (RES hatti: hub train/serve skew'ini kapatir) ---
    farms = pd.read_csv(C.WIND_FARMS_CSV)
    hub_frames = []
    for _, r in farms.iterrows():
        h = D.om_fetch(C.OM_FC_URL, float(r["lat"]), float(r["lon"]), s, e, vars=HUB_VARS)
        f = pd.DataFrame({v: h[v] for v in HUB_VARS}, index=pd.to_datetime(h["time"]))
        f["province"] = r["province"]
        hub_frames.append(f)
    hb = pd.concat(hub_frames)
    if hb["province"].nunique() < 6:
        raise RuntimeError(f"snapshot(hub): yalniz {hb['province'].nunique()} hub geldi (<6) — yazim iptal")
    for k, v in st.items():
        hb[k] = v
    hb = hb.reset_index().rename(columns={"index": "target_dt"})
    hb = _append_snap(hb, SNAP_PATH_HUBS, ["run_date", "run_slot", "province", "target_dt"])
    hb["run_slot"] = hb["run_slot"].fillna("legacy")
    hb.to_parquet(SNAP_PATH_HUBS, index=False)
    print(f"SNAPSHOT HUB {st['run_date']} [{st['run_slot']}]: {len(hb)} satir -> {SNAP_PATH_HUBS}")


if __name__ == "__main__":
    main()
