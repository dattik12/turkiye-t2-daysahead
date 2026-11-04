"""SABAH KOŞUSU — uretim kosucusu.

Her gun 09:00 TR:
  1) EPİAŞ'tan son veriyi cek (dun tam + bugun 00-08)
  2) IFS 00z hava tahminini cek (yarin icin)
  3) YARININ 24 saatlik tahminini uret
  4) data/forecast/morning/yarin_YYYY-MM-DD.parquet + latest_yarın.parquet yaz
  5) mape_history'ye eklenmek uzere master'a isle

Kullanim: python -m scripts.morning_run            # bugun
          python -m scripts.morning_run 2026-11-02 # belirli karar gunu
"""
from __future__ import annotations
import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config as C
from src import data as D
from src import models as M
from src import morning as MO

OUT_DIR = os.path.join(C.DATA, "forecast", "morning")


def _now_tr() -> pd.Timestamp:
    return pd.Timestamp.now(tz=C.TZ).tz_localize(None)


def run(decision: pd.Timestamp | None = None) -> pd.DataFrame:
    now = _now_tr()
    if decision is None:
        decision = now.normalize()
    else:
        decision = pd.Timestamp(decision).normalize()
    target = decision + pd.Timedelta(days=1)
    tgt = pd.date_range(f"{target} 00:00", f"{target} 23:00", freq="h")

    # --- 1) veri ---
    cons = D.load_or_create_consumption()
    if cons.empty:
        raise RuntimeError("consumption bos — bootstrap eksik")
    # Tazelik icin once EPİAŞ (varsa .env), olmazsa GitHub'daki taze kopya.
    # NOT: eptr2 creds yoksa exception DEGIL bos DataFrame donduruyor -> sessiz
    # gecikmeyi ancak veri ucunu kontrol ederek anlarız.
    y_last = cons["rt_cons"].dropna().index.max() if "rt_cons" in cons else pd.NaT
    if pd.notna(y_last):
        want = pd.Timestamp.now(tz=C.TZ).tz_localize(None).normalize()  # bugun
        try:
            start = (pd.Timestamp(y_last).normalize()
                     + pd.Timedelta(days=1)).date().isoformat()
            end = (want + pd.Timedelta(days=1)).date().isoformat()
            cons = D.append_consumption(cons, start, end)
            # append_consumption, TAM OLMAYAN gunu (bugunu) leak guard ile siler.
            # Sabah sozlesmesi tam da bugunun 00-08 saatlerine ihtiyac duyar ->
            # guard'i atlayan ayri cekimle geri ekle. Guvenlik: MO.build_features
            # her hedef saat icin yalnizca (h+16) saat oncesini okur.
            try:
                today = D.pull_today_partial(0)
                if len(today):
                    col = "rt_cons"
                    cons = cons.combine_first(today)
                    cons.loc[today.index, col] = today[col]
                    print(f"[morning] bugunun kismi verisi eklendi: "
                          f"{today.index.min()} -> {today.index.max()} ({len(today)} saat)")
            except Exception as exp:
                print(f"[morning] bugunun kismi verisi alinamadi ({str(exp)[:60]})")
            got = cons["rt_cons"].dropna().index.max()
            if pd.notna(got) and got.normalize() >= want - pd.Timedelta(days=1):
                print(f"[morning] EPİAŞ cekimi OK: uctur {got}")
            else:
                print(f"[morning] EPİAŞ cekimi veri getirmedi (uctur {got}) — creds yok mu?")
                raise RuntimeError("epias bos")
        except Exception:
            print("[morning] GitHub'daki taze kopya denenecek")
            try:
                import urllib.request, io
                url = ("https://raw.githubusercontent.com/dattik12/turkiye-t2-daysahead/"
                       "main/data/dataset/tr_consumption_hourly.parquet")
                gh = pd.read_parquet(io.BytesIO(urllib.request.urlopen(url, timeout=60).read()))
                if gh.index.tz is not None:
                    gh = gh.tz_convert("Etc/GMT-3").tz_localize(None)
                gh_last = gh["rt_cons"].dropna().index.max()
                cur = cons["rt_cons"].dropna().index.max()
                if pd.notna(gh_last) and (pd.isna(cur) or gh_last > cur):
                    cons = gh.combine_first(cons)
                    print(f"[morning] GitHub kopyasi alindi, uctur: {gh_last}")
                else:
                    print(f"[morning] GitHub kopyasi da geride ({gh_last}) — yerelle devam")
            except Exception as ex2:
                print(f"[morning] GitHub denemesi de olmadi ({str(ex2)[:60]})")
    y = cons["rt_cons"].dropna()
    y.index = pd.to_datetime(y.index)
    print(f"[morning] karar {decision.date()} {decision.hour:02d}:00 | hedef {target.date()}")
    full = y.groupby(y.index.normalize()).size()
    last_full = full[full == 24].index.max()
    print(f"[morning] veri ucu: {y.index.max()} | son tam gun: {last_full.date()}")
    # --- TAZELIK KAPISI: sozlesme bugun 00-08 saatlerinin gorulmesini ister.
    # Aksi halde "taze lag" avantajı kaybolur ve sessizce eski kararla kosariz.
    need = decision + pd.Timedelta(hours=MO.DECISION_HOUR - 1)   # bugun 08:00
    if y.index.max() < need:
        gap_h = (need - y.index.max()).total_seconds() / 3600
        print(f"[morning] !!! UYARI: bugun 08:00 verisi YOK (veri ucu {y.index.max()}, "
              f"{gap_h:.0f} saat geride). EPİAŞ çekimi çalışmıyor olabilir (.env / ağ). "
              f"Taze-lag avantajı bu koşuda yok.")

    # --- 2) hava: yarin icin IFS tahmini ---
    nat, _ = D.load_or_create_weather()
    nat.index = pd.to_datetime(nat.index)
    try:
        fc = D.forecast_weather(target.date().isoformat(),
                                (target + pd.Timedelta(days=1)).date().isoformat())
        fc.index = pd.to_datetime(fc.index)
        # pandas 3.0: nat.loc[fc.index] henuz olmayan satirlara yazamaz ->
        # once indeks birlestir, sonra degerleri yerlestir.
        nat = nat.reindex(nat.index.union(fc.index))
        for c in fc.columns:
            if c in nat:
                nat.loc[fc.index, c] = fc[c].to_numpy()
        print(f"[morning] IFS tahmin eklendi: {fc.index.min()} -> {fc.index.max()}")
    except Exception as ex:
        print(f"[morning] UYARI: IFS tahmin alinamadi ({str(ex)[:90]}) — hava NaN kalir")

    from src import ensemble_morning as EN
    # --- 3) model: son EN.TRAIN_DAYS gun, karar aniye kadar veriyle ---
    tr_end = decision + pd.Timedelta(hours=MO.DECISION_HOUR - 1)   # 08:00
    tr_start = tr_end - pd.Timedelta(days=EN.TRAIN_DAYS)
    tr_idx = pd.date_range(tr_start, tr_end, freq="h")
    # NOT: dropna() YOK. lag24 kural geregi h>8'de NaN; dropna modeli yalnizca
    # gece saatlerinde egitir (n~9000) ve gunduz tahminleri coker.
    Xtr = MO.build_features(y, nat, tr_idx)
    Xtr = Xtr[y.reindex(Xtr.index).notna()]
    Xtr = MO.attach_calendar(Xtr)          # saatlik bayram/arife/kandil/okul pencereleri
    if len(Xtr) < 3000:
        raise RuntimeError(f"egitim satiri yetersiz: {len(Xtr)}")
    feats = MO.feature_cols(Xtr)
    from src import ensemble_morning as EN
    models = EN.fit(Xtr, y.reindex(Xtr.index))
    print(f"[morning] model fit: n={len(Xtr)} feats={len(feats)} uzman={list(models)}")

    X = MO.build_features(y, nat, tgt)
    X = MO.attach_calendar(X)
    p = EN.predict(models, X, tgt)
    p.index = tgt
    # 28 Ekim yarim-gun carpani (deterministik takvim olayi; bkz. features.halfday_factor)
    from src import features as _F
    p = p * _F.halfday_factor(tgt)

    # --- SAAT BAZLI BIAS DUZELTMESI (720 gun walk-forward: -0.093 pp) ---
    # Uc uzman da 24 saatin hepsinde eksik tahmin ediyor (ort -297 MW, aksak zirvesi
    # -484 MW). Tablo data/calibration/load_hour_bias.json'da; scripts/build_bias_table
    # ile uretilir. BACKTEST KOSULLARINDAN gelir (analiz havasi) -> canlida hafif farkli
    # olabilir; bu yuzden deftere HEM ham HEM duzeltilmis yazilir, canli skorla kiyaslanir.
    p_raw = p.copy()
    _bias_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                              "data", "calibration", "load_hour_bias.json")
    if os.environ.get("LOAD_BIAS_FIX", "1") == "1" and os.path.exists(_bias_path):
        try:
            import json as _json
            with open(_bias_path, encoding="utf-8") as _fh:
                _tbl = _json.load(_fh)["bias_mw_by_hour"]
            _corr = pd.Series([_tbl.get(str(h), 0.0) for h in tgt.hour], index=tgt, dtype=float)
            # SINIR: saatlik sapma zamanla kayiyor (iki yari korelasyonu 0.508). 90 gunluk
            # pencere bazen uc deger uretiyor (sabah -946 MW). Asiri duzeltmeyi engellemek
            # icin |duzeltme| <= 400 MW ile kirpiliyor (uzun donem ortalamasina yakin).
            _corr = _corr.clip(-400, 400)
            p = p - _corr
            print(f"[morning] bias duzeltmesi: ort {_corr.mean():+.0f} MW "
                  f"(ham ort {p_raw.mean():,.0f} -> {p.mean():,.0f})")
        except Exception as ex:
            print(f"[morning] UYARI: bias tablosu okunamadi ({str(ex)[:60]}) — ham tahmin")
    else:
        print("[morning] bias duzeltmesi KAPALI")
    out = pd.DataFrame({"datetime": tgt, "pred_mw": p.values})
    out["pred_raw_mw"] = p_raw.values          # bias duzeltmesi oncesi (canli kiyas icin)

    # --- GUNES + RUZGAR: sabah urunu TAM fiziksel denge olmali ---
    # GUNES KAPALI (2026-11-04): talep ve ruzgar yeterli; gunes ayri gorev.
    # Gerekce: EPiAS'in yayinladigi 'sun' serisi yalniz lisansli santralleri kapsiyor
    # (tepe 4.248 MW; ulusal ~9 GW) ve hava verisiyle 1 saat kaymis. Model denemesi
    # 26.6% WMAPE verdi (90 gun veri). AYRI TASK olarak ele alinacak.
    # SOLAR_ENABLED=1 ile geri acilir.
    SOLAR_ON = os.environ.get("SOLAR_ENABLED", "0") == "1"
    try:
        from src.features import ptf_store as _PS
        _idx = pd.DatetimeIndex(tgt)
        _naive = _idx.tz_localize(None) if _idx.tz is not None else _idx
        _wind, _wst = _PS._wind_series(decision, min_cov=24)   # sabah: yalniz yarin gerekli
        if (_wind.index.tz is not None) != (_naive.tz is not None):
            _wind.index = _wind.index.tz_localize(None) if _wind.index.tz is not None else _wind.index.tz_localize("Europe/Istanbul")
        _wind = _wind.reindex(_naive)
        out["wind_mw"] = _wind.to_numpy().clip(min=0)
        if SOLAR_ON:
            from src.solar import model as _SM
            _rad = _PS._rad_series(decision)
            _tmp = _PS._temp_series(decision)
            for _s in (_rad, _tmp):
                if (_s.index.tz is not None) != (_naive.tz is not None):
                    _s.index = _s.index.tz_localize(None) if _s.index.tz is not None else _s.index.tz_localize("Europe/Istanbul")
            _sol = _SM.solar_from_radiation(_rad.reindex(_naive), C.SOLAR_CAPACITY_MW,
                                            C.SOLAR_PR, temp_c=_tmp.reindex(_naive))
            out["solar_mw"] = _sol.to_numpy().clip(min=0)
            out["residual_mw"] = out["pred_mw"].to_numpy() - out["solar_mw"].to_numpy() - out["wind_mw"].to_numpy()
        else:
            out["solar_mw"] = np.nan
            out["residual_mw"] = out["pred_mw"].to_numpy() - out["wind_mw"].to_numpy()
        print(f"[morning] ruzgar tepe {out['wind_mw'].max():,.0f} | durum {_wst} | "
              f"gunes {'ACIK' if SOLAR_ON else 'KAPALI'}")
    except Exception as ex:
        print(f"[morning] UYARI: ruzgar eklenemedi ({str(ex)[:90]})")
        out["solar_mw"] = np.nan
        out["wind_mw"] = np.nan
        out["residual_mw"] = np.nan
    out["actual_mw"] = y.reindex(tgt).values

    # --- 4) yazim ---
    # KOSU DAMGASI: ayni hedef gun icin 09:00 vintage'i AKSAм kosusu tarafindan
    # eziliyordu (ML denetimi). Artik her kosu arsivlenir; skor 09:00 kosusundan alinir.
    _now = pd.Timestamp.now()
    run_slot = "0900" if _now.hour < 12 else "aksam"
    os.makedirs(OUT_DIR, exist_ok=True)
    day_path = os.path.join(OUT_DIR, f"yarin_{target.date().isoformat()}.parquet")
    out.to_parquet(day_path, index=False)
    latest = os.path.join(OUT_DIR, "latest_yarin.parquet")
    out.to_parquet(latest, index=False)
    try:
        _arch = os.path.join(OUT_DIR, "archive")
        os.makedirs(_arch, exist_ok=True)
        out.to_parquet(os.path.join(
            _arch, f"yarin_{target.date().isoformat()}_{run_slot}_{_now:%Y%m%d_%H%M}.parquet"), index=False)
    except Exception as ex:
        print(f"[morning] UYARI: arsiv yazilamadi ({str(ex)[:60]})")
    print(f"[morning] yazildi: {day_path} (slot={run_slot} {_now:%H:%M})")
    print(f"[morning] tepe {out.pred_mw.max():,.0f} MW @{out.loc[out.pred_mw.idxmax(),'datetime'].hour:02d}:00"
          f" | ort {out.pred_mw.mean():,.0f} | dip {out.pred_mw.min():,.0f}")

    # --- 5) kendi skor defterine yaz ---
    # ONEMLI: ortak master'a (forecast_results.csv) YAZILMAZ. Orada horizon=1
    # "karar gunu + 1" demek; sabah sozlesmesinde ise "yarin" demek. Ayni anahtara
    # iki farkli anlam yazinca PTF builder 48 satir beklerken 24 buluyordu.
    from src import scoring as S
    mpath = os.path.join(C.RESULTS_DIR, "morning_history.csv")
    new = out[["datetime", "pred_mw", "pred_raw_mw"]].copy()
    new = new.rename(columns={"datetime": "dt", "pred_mw": "pred_mw"})
    new["decision_date"] = decision.date().isoformat()
    new["target_date"] = target.date().isoformat()
    new["run_slot"] = run_slot
    new["run_ts"] = _now.isoformat()
    new["actual_mw"] = np.nan
    new["mape_hour"] = np.nan
    hist = pd.read_csv(mpath) if os.path.exists(mpath) else pd.DataFrame()
    if not hist.empty:
        # ESKI DAVRANIS: ayni hedef gunun TUM satirlari silinip son kosu yaziliyordu ->
        # 09:00 vintage'i aksam kosusuyla eziliyor, canli skor 09:00'u hic puanlamiyordu.
        # YENI: (target_date, run_slot) anahtari; iki slot yan yana yasar.
        if "run_slot" not in hist.columns:
            hist["run_slot"] = "0900"
        _dup = (hist["target_date"] == new["target_date"].iloc[0]) & (hist["run_slot"] == run_slot)
        hist = hist[~_dup]
        hist["dt"] = pd.to_datetime(hist["dt"])
        new["dt"] = pd.to_datetime(new["dt"])
        hist = pd.concat([hist, new], ignore_index=True)
    else:
        hist = new
    hist.to_csv(mpath, index=False)
    print(f"[morning] skor defteri: {mpath} ({len(hist)} satir)")
    return out


if __name__ == "__main__":
    dec = sys.argv[1] if len(sys.argv) > 1 else None
    run(pd.Timestamp(dec) if dec else None)
