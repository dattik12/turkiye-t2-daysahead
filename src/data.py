"""Veri katmani: EPIAS (eptr2) cekim + OpenMeteo hava + veri seti guncelleme."""
from __future__ import annotations
import os, json, time, urllib.request
import numpy as np
import pandas as pd

from . import config as C

# ---------------------------------------------------------------- EPIAS ----
def _eptr():
    from eptr2 import EPTR2
    return EPTR2(use_dotenv=True, recycle_tgt=True, dotenv_path=".env", tgt_path=".")


def pull_epias(start: str, end: str, series: str = "rt-cons", retries: int = 5) -> pd.DataFrame:
    """EPIAS saatlik seri cek; dondurulen df: index=naive TR saati, value kolonu."""
    from eptr2 import EPTR2
    cfg = C.EPIAS_SERIES[series]
    for i in range(retries):
        try:
            eptr = EPTR2(use_dotenv=True, recycle_tgt=True, dotenv_path=".env", tgt_path=".")
            df = eptr.call(series, start_date=start, end_date=end)
            idx = pd.to_datetime(df["date"], errors="coerce")
            vals = pd.to_numeric(df[cfg["vcol"]], errors="coerce")
            out = pd.DataFrame({cfg["col"]: vals.to_numpy()}, index=idx)
            if out.index.tz is not None:
                out.index = out.index.tz_convert("Etc/GMT-3").tz_localize(None)
            return out[~out.index.duplicated(keep="last")].sort_index()
        except Exception as ex:
            wait = 10 * (2 ** i)
            print(f"  [epias {series} {start}..{end}] hata {str(ex)[:80]} — {wait}s")
            time.sleep(wait)
    raise RuntimeError(f"epias {series} {start}..{end} cekilemedi")


def load_or_create_consumption() -> pd.DataFrame:
    """Gerceklesen+loadplan veri setini yukle (index daima naive TR saati); yoksa bos doner."""
    if os.path.exists(C.CONSUMPTION_PARQUET):
        df = pd.read_parquet(C.CONSUMPTION_PARQUET)
        if df.index.tz is not None:
            df = df.tz_convert("Etc/GMT-3").tz_localize(None)
        return df
    return pd.DataFrame(index=pd.DatetimeIndex([], freq="h"))


def append_consumption(cons: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """[start,end] arasini rt-cons + load-plan ile cekip cons'a ekle, kaydet."""
    start_ep = start.replace("-", "-")
    for series in ["rt-cons", "load-plan"]:
        fr = pull_epias(start, end, series)
        if fr.empty:
            continue
        col = C.EPIAS_SERIES[series]["col"]
        cons[col] = cons[col].astype("float64") if col in cons else np.nan
        cons = cons.combine_first(fr) if fr.shape[0] else cons
        cons.loc[fr.index, col] = fr[col]
    # tam grid + takvim -- leak guard: kispi gelecek gunu trim et
    if not cons.index.is_monotonic_increasing:
        cons = cons.sort_index()
    if "rt_cons" in cons.columns:
        y_tmp = cons["rt_cons"].dropna()
        if len(y_tmp):
            days_tmp = y_tmp.groupby(y_tmp.index.normalize()).size()
            full_tmp = days_tmp[days_tmp == 24]
            if len(full_tmp):
                last_full_tmp = full_tmp.index.max()
                # keep future load_plan rows (for forecast comparison), but blank incomplete rt_cons
                mask_future = cons.index.normalize() > last_full_tmp.normalize()
                cons.loc[mask_future, "rt_cons"] = float("nan")
                # drop only wholly empty trailing rows (both rt_cons and load_plan NaN)
                cons = cons[~(cons["rt_cons"].isna() & cons["load_plan"].isna())].copy() if "load_plan" in cons.columns else cons.copy()
    os.makedirs(C.DATASET_DIR, exist_ok=True)
    cons.to_parquet(C.CONSUMPTION_PARQUET)
    return cons[~cons.index.duplicated(keep="last")].sort_index()


def last_full_day(cons: pd.DataFrame) -> pd.Timestamp | None:
    """rt_cons 24 saat tamamlanmis son gun ('naive TR midnight')."""
    if "rt_cons" not in cons or cons["rt_cons"].isna().all():
        return None
    y = cons["rt_cons"].dropna()
    days = y.groupby(y.index.normalize()).size()
    full = days[days == 24]
    if full.empty:
        return None
    return full.index.max()


# ------------------------------------------------------------ OpenMeteo ----
def om_fetch(url: str, lat: float, lon: float, s: str, e: str, model: str = C.OM_MODEL) -> dict:
    params = (f"latitude={lat}&longitude={lon}"
              f"&start_date={s}&end_date={e}"
              f"&hourly=" + ",".join(C.OM_VARS) +
              f"&models={model}&timezone=Europe/Istanbul&cell_selection=nearest")
    req = urllib.request.Request(f"{url}?{params}", headers={"User-Agent": "Mozilla/5.0"})
    for i in range(6):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                d = json.load(r)
            assert "hourly" in d, str(d)[:150]
            return d["hourly"]
        except Exception as ex:
            wait = 8 * (2 ** i) + (i * 3)   # +jitter; rate-limit icin daha uzun geri cekilme
            print(f"  [om {url[:30]}..{s} {lat:.1f}] hata {str(ex)[:60]} — {wait}s")
            time.sleep(wait)
    raise RuntimeError("om_fetch basarisiz")


def pull_weather_national(s: str, e: str, url: str = C.OM_HISTO_URL) -> pd.DataFrame:
    """Nufus agirlikli ulusal hava (10 sehir), deterministik agirliklar, 90 gunluk chunk."""
    start = pd.Timestamp(s)
    end = pd.Timestamp(e)
    frames = []
    chunk = 90
    win_start = start
    while win_start <= end:
        win_end = min(win_start + pd.Timedelta(days=chunk - 1), end)
        cs, ce = win_start.strftime("%Y-%m-%d"), win_end.strftime("%Y-%m-%d")
        per_city = []
        for c in C.CITIES:
            try:
                h = om_fetch(url, c["lat"], c["lon"], cs, ce)
                f = pd.DataFrame({v: h[v] for v in C.OM_VARS}, index=pd.to_datetime(h["time"]))
                f["city"] = c["city"]; f["pop"] = c["pop"]
                per_city.append(f)
            except Exception as ex:
                print(f"  [nat {c['city']} {cs}] atlandi: {str(ex)[:60]}")
        if len(per_city) < 8:
            raise RuntimeError(f"pull_weather_national yetersiz sehir: {len(per_city)}")
        df = pd.concat(per_city)
        nat = {}
        for v in C.OM_VARS:
            p = df.pivot_table(index=df.index, columns="city", values=v)
            pop = df.groupby("city")["pop"].first()
            nat[v] = p.mul(pop).sum(axis=1) / pop.sum()
        out = pd.DataFrame(nat)
        out["HDD"] = (18 - out["temperature_2m"]).clip(lower=0)
        out["CDD"] = (out["temperature_2m"] - 22).clip(lower=0)
        frames.append(out)
        win_start = win_end + pd.Timedelta(days=1)
    big = pd.concat(frames)
    return big[~big.index.duplicated(keep="last")].sort_index()


def refresh_weather(nat: pd.DataFrame | None) -> pd.DataFrame:
    """Son hava tarihinden bugune kadar ulusal havayi tazele; hic yoksa 2021-01-01'den cek."""
    if nat is None or len(nat) == 0:
        start = "2021-01-01"
        nat = pull_weather_national(start, "2021-01-02").iloc[0:0]  # bos iskelet
    last = nat.index.max()
    today = pd.Timestamp.now().normalize() - pd.Timedelta(days=2)  # ~1-2 gun gecikme
    if last is None or last.normalize() >= today:
        return nat
    new = pull_weather_national((last + pd.Timedelta(days=1)).date().isoformat(),
                                (today + pd.Timedelta(days=1)).date().isoformat())
    out = pd.concat([nat, new])
    out = out[~out.index.duplicated(keep="last")].sort_index()
    os.makedirs(C.WEATHER_DIR, exist_ok=True)
    out.to_parquet(C.WEATHER_NATIONAL_PARQUET)
    return out


def forecast_weather(s: str, e: str) -> pd.DataFrame:
    """Hedef gunler icin CANLI OpenMeteo forecast (ecmwf_ifs) -> ulusal agirlikli."""
    return pull_weather_national(s, e, url=C.OM_FC_URL)


def hist_cities(s: str, e: str) -> pd.DataFrame:
    """Tarihi sehir bazli hava (historical-forecast API, ayni model ailesi) -> uzun frame.
    Tek sehir hatasi digerlerini kaybettirmez (try/except + en az 8 sehir zorunlu)."""
    frames = []
    for c in C.CITIES:
        try:
            h = om_fetch(C.OM_HISTO_URL, c["lat"], c["lon"], s, e)
            f = pd.DataFrame({v: h[v] for v in C.OM_VARS}, index=pd.to_datetime(h["time"]))
            f["city"] = c["city"]; f["lat"] = c["lat"]; f["lon"] = c["lon"]; f["pop"] = c["pop"]
            frames.append(f)
            time.sleep(0.5)  # rate-limit tamponu
        except Exception as ex:
            print(f"  [hist_cities {c['city']}] atlandi: {str(ex)[:60]}")
    if len(frames) < 8:
        raise RuntimeError(f"hist_cities yetersiz sehir: {len(frames)}")
    return pd.concat(frames)


def forecast_cities(s: str, e: str) -> pd.DataFrame:
    """Hedef gunler icin sehir bazli CANLI forecast (uzun frame)."""
    frames = []
    for c in C.CITIES:
        try:
            h = om_fetch(C.OM_FC_URL, c["lat"], c["lon"], s, e)
            f = pd.DataFrame({v: h[v] for v in C.OM_VARS}, index=pd.to_datetime(h["time"]))
            f["city"] = c["city"]; f["lat"] = c["lat"]; f["lon"] = c["lon"]; f["pop"] = c["pop"]
            frames.append(f)
            time.sleep(0.6)  # rate-limit tamponu
        except Exception as ex:
            print(f"  [forecast_cities {c['city']}] atlandi: {str(ex)[:60]}")
    if len(frames) < 8:
        raise RuntimeError(f"forecast_cities yetersiz sehir: {len(frames)}")
    return pd.concat(frames)


def _fix_dt_index(df: pd.DataFrame) -> pd.DataFrame:
    """Parquet'ta 'dt' kolon olarak kaldiysa index'e tasi (naive TR)."""
    if df is not None and not df.empty and "dt" in df.columns and not isinstance(df.index, pd.DatetimeIndex):
        df = df.copy()
        df.index = pd.to_datetime(df["dt"])
        df = df.drop(columns=["dt"])
        if df.index.tz is not None:
            df.index = df.index.tz_convert("Etc/GMT-3").tz_localize(None)
    return df


def load_or_create_weather() -> tuple:
    """(nat, cities) hava durumunu yukle; hic yoksa 2021-01-01'den basla ve refresh et."""
    nat = None
    cities = None
    if os.path.exists(C.WEATHER_NATIONAL_PARQUET):
        nat = _fix_dt_index(pd.read_parquet(C.WEATHER_NATIONAL_PARQUET))
    if os.path.exists(C.WEATHER_CITIES_PARQUET):
        cities = _fix_dt_index(pd.read_parquet(C.WEATHER_CITIES_PARQUET))
    if nat is None:
        nat = pull_weather_national("2021-01-01", "2021-01-02").iloc[0:0]
        cities = forecast_cities("2021-01-01", "2021-01-02").iloc[0:0]
    # national refresh
    nat = refresh_weather(nat)
    # cities refresh (tarihi model ailesi: histfc) — kismi/hatali yazma saglam dosyayi ezmesin
    last_c = cities.index.max() if len(cities) else pd.Timestamp("2021-01-01")
    today = pd.Timestamp.now().normalize() - pd.Timedelta(days=2)
    if last_c.normalize() < today:
        try:
            time.sleep(0.6)  # sehir istekleri arasi: rate-limit tamponu
            new = hist_cities((last_c + pd.Timedelta(days=1)).date().isoformat(),
                              (today + pd.Timedelta(days=1)).date().isoformat())
            cand = pd.concat([cities, new])
            cand = cand[~cand.index.duplicated(keep="last")].sort_index()
            if cand["city"].nunique() >= 8 and new["city"].nunique() >= 8:
                cities = cand
                cities.to_parquet(C.WEATHER_CITIES_PARQUET)
            else:
                # PARTIAL/HATALI: diskteki saglam dosyayi koru, bellegi de BOZMA
                print(f"  [weather] cities refresh atlandi (sehir sayisi yetersiz: {cand['city'].nunique()})")
        except Exception as ex:
            print(f"  [weather] cities refresh atlandi: {str(ex)[:80]}")
    return nat, cities
