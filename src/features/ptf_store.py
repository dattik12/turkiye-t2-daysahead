"""PTF input feature store: 11 kolonluk data contract (v1).

Sema (her satir 1 saatlik uzlastirma donemi; karar gunu basina 48 satir):
    datetime (tz-aware Europe/Istanbul, PK) | horizon (T+1/T+2)
    consumption_pred_mw | solar_pred_mw | wind_pred_mw       (float32, MW)
    residual_load_mw (= cons - solar - wind)                  (float32)
    renewable_generation_mw (= solar + wind)                  (float32)
    renewable_penetration (= ren / cons, [0,1))               (float32)
    solar_status (ok | zero_night | unconfigured)             (category)
    wind_status (ritm | lgbm_challenger | fallback)           (category)
    is_peak_hour (08:00<=saat<20:00 -> 1)                     (uint8)

Kaynak onceligi RES: RITM (holdout galibi) -> LGBM challenger -> fallback(NaN).
Invariant'lar export ONCESI validate() ile kontrol edilir; ihlalde yazim YOK
(durust fail — daily run'in PTF adimi continue-on-error oldugu icin tuketim
hatti etkilenmez).

Depolama: data/forecast/ptf_features/archive/ptf_features_YYYY-MM-DD.parquet
(+csv) + latest.parquet/csv kopyalari (downstream dogrudan latest okur).
"""
from __future__ import annotations
import os
import shutil
import numpy as np
import pandas as pd

from .. import config as C
from .. import scoring as S
from ..solar import radiation as R
from ..solar import model as SM
from ..wind import met as WM
from ..wind import model as WMODEL
from ..wind import pull_actual as WPA
from ..wind.weights import load_matrix, weighted_speed

TZ = "Europe/Istanbul"
PEAK_START, PEAK_END = 8, 20  # 08:00 <= saat < 20:00

COLS = ["datetime", "horizon", "consumption_pred_mw", "solar_pred_mw",
        "wind_pred_mw", "residual_load_mw", "renewable_generation_mw",
        "renewable_penetration", "solar_status", "wind_status", "is_peak_hour",
        "residual_ramp_1h", "solar_ramp_1h"]
FLOAT_COLS = ["consumption_pred_mw", "solar_pred_mw", "wind_pred_mw",
              "residual_load_mw", "renewable_generation_mw",
              "renewable_penetration", "residual_ramp_1h", "solar_ramp_1h"]


def _rad_series(decision: pd.Timestamp) -> pd.Series:
    d1 = (decision + pd.Timedelta(days=1)).date().isoformat()
    d2 = (decision + pd.Timedelta(days=2)).date().isoformat()
    return R.forecast_radiation(d1, d2)


def _temp_series(decision: pd.Timestamp) -> pd.Series:
    d1 = (decision + pd.Timedelta(days=1)).date().isoformat()
    d2 = (decision + pd.Timedelta(days=2)).date().isoformat()
    return R.forecast_temp(d1, d2)


def _wind_series(decision: pd.Timestamp) -> tuple[pd.Series, str]:
    """RES: birincil BLEND (RITM + dual-model duzeltme) -> saf RITM -> fallback(NaN).
    NOT: dual model residual ogrenir (taban=RITM); RITM yokken tek basina kosmaz."""
    d1 = (decision + pd.Timedelta(days=1)).date().isoformat()
    d2 = (decision + pd.Timedelta(days=2)).date().isoformat()
    idx = pd.date_range(decision + pd.Timedelta(days=1), periods=48, freq="h")
    try:
        r = WPA.pull_ritm_forecast(d1, d2)["ritm_fc_mw"].reindex(idx)
        if r.notna().sum() < 40:
            print("PTF uyari: RITM kapsama yetersiz")
            return pd.Series(np.nan, index=idx), "fallback"
    except Exception as ex:
        print(f"PTF uyari: RITM alinamadi ({str(ex)[:80]})")
        return pd.Series(np.nan, index=idx), "fallback"
    try:
        return _blended(idx, d1, d2, r)
    except Exception as ex:
        print(f"PTF uyari: blend basarisiz ({str(ex)[:80]}), saf RITM")
        return r, "ritm"


def _blended(idx: pd.DatetimeIndex, d1: str, d2: str,
             ritm: pd.Series) -> tuple[pd.Series, str]:
    """Dual-model residual duzeltme + blend (meta sidecar: bias + blend_w)."""
    b1 = WMODEL.load(C.WIND_MODEL_D1)
    b2 = WMODEL.load(C.WIND_MODEL_D2)
    if b1 is None or b2 is None:
        raise RuntimeError("dual model dosyasi yok (train_wind calistir)")
    meta = WMODEL.load_meta(C.WIND_MODEL_D1)
    w = float(meta.get("blend_w", 0.0))
    bins = meta.get("blend_bins", {}) or {}
    bias = {int(k): float(v) for k, v in meta.get("bias", {}).items()}
    hs = (idx[0] - pd.Timedelta(days=16)).date().isoformat()
    he = (idx[0] - pd.Timedelta(hours=1)).date().isoformat()
    gen = WPA.pull_generation(hs, he)
    rfc = WPA.pull_ritm_forecast(hs, he)["ritm_fc_mw"]
    m = load_matrix()
    wl = WM.forecast_wind(d1, d2)
    spd = WM.hub_speed_table(wl)
    v = weighted_speed(spd, m)
    w10 = wl.pivot_table(index="dt", columns="city", values="w10")
    v10 = weighted_speed(w10, m)
    wd = wl.pivot_table(index="dt", columns="city", values="wdir")
    ww = m.set_index("province")["w"]
    common = [c for c in wd.columns if c in ww.index]
    rr = np.deg2rad(wd[common].fillna(0.0))
    vdir = pd.Series(np.rad2deg(np.arctan2(
        (np.sin(rr) * ww[common].values).sum(axis=1),
        (np.cos(rr) * ww[common].values).sum(axis=1))) % 360.0, index=wd.index)
    try:
        wg = WM.forecast_wind(d1, d2, model="gfs_global", suffix="_gfs")
        vg = weighted_speed(wg.pivot_table(index="dt", columns="city",
                                           values="w100_gfs"), m)
    except Exception:
        vg = None
    f = WMODEL.build_features(v.reindex(idx).ffill(), ritm.reindex(idx))
    f = WMODEL.attach_shear(f, v, v10)
    f = WMODEL.attach_direction(f, vdir)
    if vg is not None:
        f["gfs_w100"] = vg.reindex(f.index).values
        f["nwp_spread"] = (v - vg).abs().reindex(f.index).values
    err = (gen - rfc.reindex(gen.index)).rename("err")
    for col, lag in [("gen_lag24", 24), ("gen_lag48", 48), ("gen_lag168", 168),
                     ("gen_lag336", 336)]:
        f[col] = gen.reindex(f.index - pd.Timedelta(hours=lag)).values
    f["gen_roll24"] = gen.rolling(24, min_periods=24).mean().shift(24).reindex(f.index).values
    f["gen_roll168"] = gen.rolling(168, min_periods=168).mean().reindex(f.index).values
    for col, lag in [("err_lag24", 24), ("err_lag48", 48), ("err_lag168", 168)]:
        f[col] = err.reindex(f.index - pd.Timedelta(hours=lag)).values
    f1, f2 = f.iloc[:24].copy(), f.iloc[24:].copy()
    f2[WMODEL.FRESH_COLS] = np.nan
    p1 = WMODEL.predict(b1, f1) + f1.index.hour.map(lambda h: bias.get(h, 0.0)).values
    p2 = WMODEL.predict(b2, f2)
    r = ritm.reindex(idx)
    thr = bins.get("thr", [])
    wb = bins.get("ws", {})
    if thr and len(thr) == 2:
        bb = pd.cut(f["nwp_spread"].fillna(0), [-1e-9] + thr + [1e9], labels=[0, 1, 2])
        wv = np.array([wb.get(str(int(x)), w) if pd.notna(x) else w for x in bb])
    else:
        wv = np.full(len(idx), w)
    out = wv * pd.concat([p1, p2]).reindex(idx).values + (1 - wv) * r.values
    return pd.Series(out, index=idx), "blend"


def build(master_df: pd.DataFrame, rad: pd.Series, decision: pd.Timestamp,
          capacity_mw: float | None, pr: float,
          wind: pd.Series | None, wind_status: str,
          temp: pd.Series | None = None) -> pd.DataFrame:
    key = decision.strftime("%Y-%m-%d")
    fc = master_df[master_df["decision_date"] == key].copy()
    if fc.empty:
        raise ValueError(f"master'da {key} karar gunu yok")
    fc["datetime"] = pd.to_datetime(fc["dt"]).dt.tz_localize(TZ)
    fc["horizon"] = fc["horizon"].map({1: "T+1", 2: "T+2"})
    fc = fc.sort_values("datetime").reset_index(drop=True)

    naive = fc["datetime"].dt.tz_localize(None)
    fc["consumption_pred_mw"] = fc["pred_mw"].astype("float32")
    fc["sw_rad_tmp"] = rad.reindex(naive).ffill().fillna(0).values
    if capacity_mw:
        if temp is not None:
            temp_aligned = pd.Series(temp.reindex(naive).ffill().fillna(20.0).values,
                                    index=fc.index)
        else:
            temp_aligned = None
        fc["solar_pred_mw"] = SM.solar_from_radiation(
            pd.Series(fc["sw_rad_tmp"].values, index=fc.index),
            capacity_mw, pr, temp_c=temp_aligned).astype("float32").values
        fc["solar_status"] = "ok"
    else:
        fc["solar_pred_mw"] = np.nan
        fc["solar_status"] = "unconfigured"

    # Gece GES sifirlamasi (v2: zenit maskesi ^ kontrat saat kurali): tam 0.0 + zero_night.
    hrs = fc["datetime"].dt.hour
    zen_night = ~SM.daylight_mask(pd.DatetimeIndex(naive)).to_numpy()
    night = zen_night | (hrs < 5).to_numpy() | (hrs > 20).to_numpy()
    fc.loc[night, "solar_pred_mw"] = 0.0
    fc.loc[night & (fc["solar_status"] == "ok"), "solar_status"] = "zero_night"

    if wind is not None:
        fc["wind_pred_mw"] = wind.reindex(naive).astype("float32").values
    else:
        fc["wind_pred_mw"] = np.nan
    fc["wind_status"] = wind_status

    fc["renewable_generation_mw"] = (
        fc["solar_pred_mw"].fillna(0) + fc["wind_pred_mw"].fillna(0)
    ).astype("float32")
    fc["residual_load_mw"] = (
        fc["consumption_pred_mw"] - fc["renewable_generation_mw"]).astype("float32")
    fc["renewable_penetration"] = (
        fc["renewable_generation_mw"] / fc["consumption_pred_mw"]).astype("float32")
    fc["is_peak_hour"] = ((fc["datetime"].dt.hour >= PEAK_START) &
                          (fc["datetime"].dt.hour < PEAK_END)).astype("uint8")
    # Hizli-kazanim rampa kolonlari (blok-ici turev; ilk saat 0).
    fc["residual_ramp_1h"] = fc["residual_load_mw"].diff().fillna(0).astype("float32")
    fc["solar_ramp_1h"] = fc["solar_pred_mw"].fillna(0).diff().fillna(0).astype("float32")
    fc["horizon"] = fc["horizon"].astype("category")
    fc["solar_status"] = fc["solar_status"].astype("category")
    fc["wind_status"] = fc["wind_status"].astype("category")
    return fc[COLS]


def validate(df: pd.DataFrame) -> None:
    """Export oncesi invariant'lar; ihlalde ValueError (yazim yapilmaz)."""
    if len(df) != 48:
        raise ValueError(f"48 satir beklenir, {len(df)} geldi")
    if df["datetime"].duplicated().any():
        raise ValueError("mukerrer datetime satiri var")
    if (df["datetime"].diff().dropna() != pd.Timedelta(hours=1)).any():
        raise ValueError("datetime ekseninde bosluk var")
    tol = 1e-2  # MW: float32 kuantizasyon payi (fiziksel anlami yok)
    con = (df["consumption_pred_mw"] - df["solar_pred_mw"].fillna(0)
           - df["wind_pred_mw"].fillna(0) - df["residual_load_mw"]).abs().max()
    if con > tol:
        raise ValueError(f"toplamsallik ihlali (max sapma {con})")
    if (df["residual_load_mw"] <= 0).any():
        raise ValueError("negatif/sifir residual_load var")
    night = (df["datetime"].dt.hour < 5) | (df["datetime"].dt.hour > 20)
    if ((df.loc[night, "solar_pred_mw"] != 0.0).any()
            or (df.loc[night & (df["solar_status"] == "ok"), "solar_status"]
                != "zero_night").any()):
        raise ValueError("gece GES sifirlama ihlali")
    pen = df["renewable_penetration"]
    if ((pen < 0.0).any() or (pen >= 1.0).any()):
        raise ValueError("penetrasyon [0,1) disinda")


def build_for_decision(decision_date: str | None = None) -> str | None:
    master = S.load_master()
    if master.empty:
        print("PTF: master forecast bos — once daily_run kosmalidir.")
        return None
    if decision_date is None:
        decision_date = pd.to_datetime(master["decision_date"]).max().strftime("%Y-%m-%d")
    decision = pd.Timestamp(decision_date)
    try:
        rad = _rad_series(decision)
    except Exception as ex:
        print(f"PTF: radyasyon yok ({str(ex)[:80]}) — cikti uretilemedi.")
        return None
    wind, wind_status = _wind_series(decision)
    try:
        temp = _temp_series(decision)
    except Exception as ex:
        print(f"PTF uyari: sicaklik yok ({str(ex)[:60]}), derate atlandi")
        temp = None
    out = build(master, rad, decision, C.SOLAR_CAPACITY_MW, C.SOLAR_PR,
                None if wind_status == "fallback" else wind, wind_status, temp=temp)
    validate(out)  # ihlalde raise -> yazim YOK
    arch = os.path.join(C.PTF_FEATURES_DIR, "archive")
    os.makedirs(arch, exist_ok=True)
    # Kosu etiketi (sabah/aksam): ayni karar gununde iki arsiv cakismasin.
    tr_now = pd.Timestamp.now(tz=TZ)
    run = "morning" if tr_now.hour < 12 else "evening"
    base = os.path.join(arch, f"ptf_features_{decision_date}_{run}")
    out.to_parquet(base + ".parquet", index=False)
    out.to_csv(base + ".csv", index=False)
    shutil.copy(base + ".parquet", os.path.join(C.PTF_FEATURES_DIR, "latest.parquet"))
    shutil.copy(base + ".csv", os.path.join(C.PTF_FEATURES_DIR, "latest.csv"))
    print(f"PTF: {len(out)} satir -> {base}.parquet (+csv) + latest | "
          f"wind={wind_status} | solar ok={int((out['solar_status'] == 'ok').sum())}/48 | run={run}")
    return base + ".parquet"
