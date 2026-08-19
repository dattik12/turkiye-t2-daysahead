"""Tahmin motoru: veri seti + hava -> T+1/T+2 saatlik tahmin (ensemble, leak-free)."""
from __future__ import annotations
import os
import numpy as np
import pandas as pd

from . import config as C
from . import data as D
from . import features as F
from . import models as M


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
            try:
                nat_fc = D.forecast_weather(s, e)
                nat = pd.concat([nat, nat_fc])
                nat = nat[~nat.index.duplicated(keep="last")].sort_index()
                cites_fc = D.forecast_cities(s, e)
                cw2 = self._pivot_temp(cites_fc)
                cw = pd.concat([cw, cw2])
                cw = cw[~cw.index.duplicated(keep="last")].sort_index()
            except Exception as ex:
                print(f"  forecast hava uzatma atlandi: {str(ex)[:80]}")
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

        nat, wdyn, cw, seg_urban, seg_ind = self.weather_tables(target_end)
        dayfrac = self.dayfrac_def.reindex(pd.date_range(self.P.index.min(), target_end, freq="h"))

        train_cut = decision_date - pd.Timedelta(days=C.TRAIN_DAYS)
        train_idx = pd.date_range(train_cut, decision_date - pd.Timedelta(hours=1), freq="h")
        days1 = pd.date_range(f"{d1} 00:00", f"{d1} 23:00", freq="h")
        days2 = pd.date_range(f"{d2} 00:00", f"{d2} 23:00", freq="h")

        if models_cache is None:
            m1 = M.train_engine(self.P, nat, wdyn, dayfrac, cw, seg_urban, seg_ind,
                                self.cons["rt_cons"], train_idx, 1)
            p1 = M.predict_pair(m1, self.P, nat, wdyn, dayfrac, cw, seg_urban, seg_ind, days1, 1)
            # D+1 tahmini (t-24h) -> D+2 EGITIMINE feature
            d1_tr = pd.Series(M.predict_pair(m1, self.P, nat, wdyn, dayfrac, cw, seg_urban, seg_ind,
                                             train_idx - pd.Timedelta(hours=24), 1), index=train_idx)
            m2 = M.train_engine(self.P, nat, wdyn, dayfrac, cw, seg_urban, seg_ind,
                                self.cons["rt_cons"], train_idx, 2, d1_pred=d1_tr)
        else:
            m1, m2, _ = models_cache
            p1 = M.predict_pair(m1, self.P, nat, wdyn, dayfrac, cw, seg_urban, seg_ind, days1, 1)

        # D+2 INFERENCE feed: bugunku D+1 tahmini (karar aninda bilinir) — training ile tutarli
        d1_feed = pd.Series(p1, index=days1)
        p2 = M.predict_pair(m2, self.P, nat, wdyn, dayfrac, cw, seg_urban, seg_ind,
                            days2, 2, d1_pred=d1_feed)

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
