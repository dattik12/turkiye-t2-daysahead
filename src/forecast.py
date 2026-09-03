"""Tahmin motoru: veri seti + hava -> T+1/T+2 saatlik tahmin (ensemble, leak-free)."""
from __future__ import annotations
import os
import numpy as np
import pandas as pd

from . import config as C
from . import data as D
from . import features as F
from . import models as M
# Lazy: models_ensemble/stacking sadece ensemble (len>1) gerektiginde yuklenir,
# boylece Actions runner'i yalnizca lightgbm ile calisabilir (xgboost/catboost zorunlu degil).


def to_local(df):
    if df.index.tz is not None:
        df = df.tz_convert("Etc/GMT-3").tz_localize(None)
    return df


class Engine:
    def __init__(self, cons: pd.DataFrame, nat: pd.DataFrame, cities: pd.DataFrame):
        self.cons = to_local(cons)
        self.nat = to_local(nat)
        self.cities = cities  # long: dt, city, lat, lon, pop, vars
        self.P = F.precompute(self.cons["rt_cons"])
        self.dayfrac_def = F.daylight_fraction_series(
            str(self.cons.index.min().date()), str(self.cons.index.max().date() + pd.Timedelta(days=30)))

    # ---------------- hava tablolari (tarihi + hedef gun forecast birlestirilir) ----
    @staticmethod
    def _pivot_temp(fr: pd.DataFrame) -> pd.DataFrame:
        """Sehir bazli uzun frame -> (time x city) temperature pivot. 'dt' kolon ya da index olabilir."""
        if "dt" in fr.columns:
            return fr.pivot_table(index="dt", columns="city", values="temperature_2m")
        return fr.pivot_table(index=fr.index, columns="city", values="temperature_2m")

    def weather_tables(self, target_end):
        """Tarihi hava + hedef gunlere kadar forecast'i tek 'extend' frame olarak birlestir."""
        nat = self.nat.copy()
        cw = self._pivot_temp(self.cities)
        last_nat = nat.index.max()
        if target_end > last_nat:
            s = (last_nat + pd.Timedelta(days=1)).date().isoformat()
            e = target_end.date().isoformat()
            # ulusal forecast basarisiz olursa RUN FAIL olur (degrade tahmin sessiz commit olmaz)
            nat_fc = D.forecast_weather(s, e)
            nat = pd.concat([nat, nat_fc])
            nat = nat[~nat.index.duplicated(keep="last")].sort_index()
            try:
                cites_fc = D.forecast_cities(s, e)
                cw2 = self._pivot_temp(cites_fc)
                cw = pd.concat([cw, cw2])
                cw = cw[~cw.index.duplicated(keep="last")].sort_index()
            except Exception as ex:
                print(f"  UYARI: sehir forecast atlandi ({str(ex)[:60]}); ulusal ile devam")
        # gerekli satirlari doldur
        span = pd.date_range(nat.index.min(), max(nat.index.max(), target_end), freq="h")
        nat = nat.reindex(span).ffill()
        cw = cw.reindex(span).ffill()
        wdyn = F.weather_dynamics(nat)
        pop = self.cities.groupby("city")["pop"].first()

        def seg(cols):
            p = cw[cols]
            pw = pop[cols]
            return p.mul(pw).sum(axis=1) / pw.sum()
        seg_urban = seg(["istanbul", "ankara", "izmir", "bursa", "antalya"])
        seg_ind = seg(["adana", "gaziantep", "konya", "izmir", "bursa"])
        return nat, wdyn, cw, seg_urban, seg_ind

    # ------------------------------------------------------------------ tahmin ----
    def forecast(self, decision_date: pd.Timestamp, models_cache: tuple | None = None,
                 return_models: bool = False):
        """Karar gunu D (= son TAM veri gunu). D+1 ve D+2 icin saatlik tahmin doner.
        models_cache: (m1, m2) onceden egitilmisse egitimi atlar (backtest hizi icin).
        df: dt, pred_mw, horizon, load_plan_mw (TEIAS varsa)."""
        decision_date = pd.Timestamp(decision_date).normalize()
        d1 = decision_date + pd.Timedelta(days=1)
        d2 = decision_date + pd.Timedelta(days=2)
        target_end = d2 + pd.Timedelta(hours=23)

        # v4.4: termal-yuk agregatlari (CDD/HDD_next48) hedef saatlerin 48s sonrasini
        # ister -> hava cercevesini +2 gun genislet (IFS 16-gun ufku icinde, ucuz).
        nat, wdyn, cw, seg_urban, seg_ind = self.weather_tables(target_end + pd.Timedelta(days=2))
        dayfrac = self.dayfrac_def.reindex(pd.date_range(self.P.index.min(), target_end, freq="h"))

        train_cut = decision_date - pd.Timedelta(days=C.TRAIN_DAYS)
        train_idx = pd.date_range(train_cut, decision_date - pd.Timedelta(hours=1), freq="h")
        days1 = pd.date_range(f"{d1} 00:00", f"{d1} 23:00", freq="h")
        days2 = pd.date_range(f"{d2} 00:00", f"{d2} 23:00", freq="h")
        # --- FIX: extend P to target so future lag/roll lookup works (was NaN -> tree fallback) ---
        y_ext = self.cons["rt_cons"].reindex(pd.date_range(self.cons.index.min(), target_end, freq="h"))
        P = F.precompute(y_ext)
        # ffill forward-fill for roll/day aggregates that are NaN for future hours (decision-time known)
        for c in [cc for cc in P.columns if cc.startswith("roll") or cc.startswith("day_") or cc.startswith("samehr")]:
            P[c] = P[c].ffill()
        # keep self.P for back-compat but use extended P for all train/predict
        Peff = P

        use_multi = len(getattr(C, "ENSEMBLE_MODELS", ["lgbm"])) > 1
        if use_multi:
            from . import models_ensemble as ME
            from . import stacking as ST
            TE = ME
        else:
            TE = M
        # --- LEP (TEIAS plan) ozelligi: yalnizca tek-model yolu + H1.
        # Is 17:00 TR'de kostugu icin hedef gunun plani yayimlanmistir; sabah kosulursa
        # degerler NaN doner ve notr 1.0'a duser (graceful degrade).
        use_lep = getattr(C, "USE_LEP_FEATURE", False) and "load_plan" in self.cons and not use_multi

        def lep_extra(idx, hz):
            if not use_lep or hz != 1:
                return None
            return {"lep_rel": F.lep_rel_feature(self.cons["load_plan"], Peff["samehr_7d_24"], idx)}

        # --- v4.4 rezidu duzeltme: son 90 gun rezidulerinden (saat, gun-tipi) hucre bias'i.
        # Karar gununden turetilir; egitim + cache (backtest) dallarinda aynen calisir.
        def _resid(models, hz, base, day_idx, d1models=None):
            if not getattr(C, "V44_RESID", True):
                return base
            tr_idx = self.cons["rt_cons"].dropna().index
            tr_idx = tr_idx[(tr_idx < decision_date.normalize())
                            & (tr_idx >= decision_date - pd.Timedelta(days=90))]
            if len(tr_idx) == 0:
                return base
            d1p = None
            if hz == 2 and d1models is not None:  # H2 egitim satirlari icin D+1 beslemesi
                t24 = tr_idx - pd.Timedelta(hours=24)
                d1p = pd.Series(np.asarray(M.predict_pair(
                    d1models, Peff, nat, wdyn, dayfrac, cw, seg_urban, seg_ind,
                    t24, 1, extra_cols=lep_extra(t24, 1))), index=tr_idx)
            tr = M.predict_pair(models, Peff, nat, wdyn, dayfrac, cw, seg_urban, seg_ind,
                                tr_idx, hz, d1_pred=d1p,
                                extra_cols=lep_extra(tr_idx, hz) if hz == 1 else None)
            y = self.cons["rt_cons"].reindex(tr_idx)
            r = (y - pd.Series(np.asarray(tr), index=tr_idx)).dropna()
            if r.empty:
                return base
            return np.asarray(base) + M.residual_adjust(r, r.index, day_idx)

        if models_cache is None:
            if use_multi:
                m1 = TE.train_engine_multi(Peff, nat, wdyn, dayfrac, cw, seg_urban, seg_ind, self.cons["rt_cons"], train_idx, 1)
                p1_dict = TE.predict_multi(m1, Peff, nat, wdyn, dayfrac, cw, seg_urban, seg_ind, days1, 1)
                p1 = np.mean(list(p1_dict.values()), axis=0) if len(p1_dict) > 1 else list(p1_dict.values())[0]
                # D+1 tahmini for D+2 training: use simple avg per hour
                d1_tr_vals = TE.predict_multi(m1, Peff, nat, wdyn, dayfrac, cw, seg_urban, seg_ind, train_idx - pd.Timedelta(hours=24), 1)
                d1_tr = pd.Series(np.mean(list(d1_tr_vals.values()), axis=0) if len(d1_tr_vals)>1 else list(d1_tr_vals.values())[0], index=train_idx)
                m2 = TE.train_engine_multi(Peff, nat, wdyn, dayfrac, cw, seg_urban, seg_ind, self.cons["rt_cons"], train_idx, 2, d1_pred=d1_tr)
            else:
                m1 = M.train_engine(Peff, nat, wdyn, dayfrac, cw, seg_urban, seg_ind, self.cons["rt_cons"], train_idx, 1,
                                    extra_cols=lep_extra(train_idx, 1))
                p1 = M.predict_pair(m1, Peff, nat, wdyn, dayfrac, cw, seg_urban, seg_ind, days1, 1,
                                    extra_cols=lep_extra(days1, 1))
                t24 = train_idx - pd.Timedelta(hours=24)
                d1_tr = pd.Series(M.predict_pair(m1, Peff, nat, wdyn, dayfrac, cw, seg_urban, seg_ind, t24, 1,
                                                 extra_cols=lep_extra(t24, 1)), index=train_idx)
                m2 = M.train_engine(Peff, nat, wdyn, dayfrac, cw, seg_urban, seg_ind, self.cons["rt_cons"], train_idx, 2, d1_pred=d1_tr)
        else:
            m1, m2, _ = models_cache
            if use_multi and isinstance(m1, dict) and "lgbm" in m1:
                p1_dict = TE.predict_multi(m1, Peff, nat, wdyn, dayfrac, cw, seg_urban, seg_ind, days1, 1)
                p1 = np.mean(list(p1_dict.values()), axis=0) if len(p1_dict) > 1 else list(p1_dict.values())[0]
            else:
                p1 = M.predict_pair(m1, Peff, nat, wdyn, dayfrac, cw, seg_urban, seg_ind, days1, 1,
                                    extra_cols=lep_extra(days1, 1))

        d1_feed = pd.Series(p1, index=days1)
        p1 = _resid(m1, 1, p1, days1)
        if use_multi and isinstance(m2, dict) and "lgbm" in m2:
            p2_dict = ME.predict_multi(m2, Peff, nat, wdyn, dayfrac, cw, seg_urban, seg_ind, days2, 2, d1_pred=d1_feed)
            # optional stacking if weights exist
            w = ST.load_weights()
            if w is not None:
                p2 = ST.apply_stacking(p2_dict, w, 2)
            else:
                p2 = np.mean(list(p2_dict.values()), axis=0) if len(p2_dict) > 1 else list(p2_dict.values())[0]
        else:
            p2 = M.predict_pair(m2, Peff, nat, wdyn, dayfrac, cw, seg_urban, seg_ind, days2, 2, d1_pred=d1_feed)
            p2 = _resid(m2, 2, p2, days2, d1models=m1)

        out = pd.DataFrame({"dt": days1.append(days2),
                            "pred_mw": np.concatenate([p1, p2]),
                            "horizon": [1] * 24 + [2] * 24})
        if "load_plan" in self.cons:
            out["load_plan_mw"] = out["dt"].map(self.cons["load_plan"])
        out["decision_date"] = decision_date
        out["target_date"] = out["dt"].dt.normalize()
        if return_models:
            return out, (m1, m2, d1_feed)
        return out
