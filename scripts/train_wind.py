"""RES v2 egitimi: residual-reframe + yon sektoru; 3-yollu holdout, kazanan kaydedilir.

Kullanim:
    python -m scripts.train_wind            # son 60 gun
    python -m scripts.train_wind 90         # son 90 gun

Adaylar: residual / raw+monotonik / RITM (benchmark).
Kazanan models/wind_lgbm.txt + .mode sidecar'a yazilir.
"""
from __future__ import annotations
import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
from src import config as C
from src.wind import met, model as M, pull_actual as PA
from src.wind.weights import load_matrix, weighted_speed


def weighted_direction(wdir_table: pd.DataFrame, matrix: pd.DataFrame) -> pd.Series:
    w = matrix.set_index("province")["w"]
    common = [c for c in wdir_table.columns if c in w.index]
    rad = np.deg2rad(wdir_table[common].fillna(0.0))
    s = (np.sin(rad) * w[common].values).sum(axis=1)
    c = (np.cos(rad) * w[common].values).sum(axis=1)
    return pd.Series(np.rad2deg(np.arctan2(s, c)) % 360.0, index=wdir_table.index)


def mape(a: pd.Series, b: pd.Series) -> float:
    return float((abs(a - b) / b).mean() * 100)


def main() -> None:
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    end = (pd.Timestamp.now().normalize() - pd.Timedelta(days=2)).date().isoformat()
    start = (pd.Timestamp(end) - pd.Timedelta(days=days - 1)).date().isoformat()
    print(f"Egitim penceresi: {start} .. {end} ({days} gun)")

    gen = PA.pull_generation(start, end).rename("wind_gen_mw")
    fc = PA.pull_ritm_forecast(start, end)["ritm_fc_mw"]

    wl = met.history_wind(start, end)
    spd = met.hub_speed_table(wl)
    w10 = wl.pivot_table(index="dt", columns="city", values="w10")
    wdir = wl.pivot_table(index="dt", columns="city", values="wdir")
    matrix = load_matrix()
    v = weighted_speed(spd, matrix)
    v10 = weighted_speed(w10, matrix)
    vdir = weighted_direction(wdir, matrix)
    try:  # ikinci NWP (GFS): yayilim + çapraz kontrol girdisi
        wg = met.history_wind(start, end, matrix, model="gfs_global", suffix="_gfs")
        gtab = wg.pivot_table(index="dt", columns="city", values="w100_gfs")
        vg = weighted_speed(gtab, matrix)
    except Exception as ex:
        print(f"GFS alinamadi ({str(ex)[:60]}), IFS-tekli devam")
        vg = None

    f = M.build_features(v, fc)
    f = M.attach_shear(f, v, v10)
    f = M.attach_direction(f, vdir)
    if vg is not None:
        f["gfs_w100"] = vg.reindex(f.index).values
        f["nwp_spread"] = (v - vg).abs().reindex(f.index).values
    # Ufuk-guvenli persistence: D+1 taze lag'leri de doldur (D+2'de maskelenecek)
    back24 = f.index - pd.Timedelta(hours=24)
    back48 = f.index - pd.Timedelta(hours=48)
    back168 = f.index - pd.Timedelta(hours=168)
    back336 = f.index - pd.Timedelta(hours=336)
    f["gen_lag24"] = gen.reindex(back24).values
    f["gen_lag48"] = gen.reindex(back48).values
    f["gen_lag168"] = gen.reindex(back168).values
    f["gen_lag336"] = gen.reindex(back336).values
    f["gen_roll24"] = gen.rolling(24, min_periods=24).mean().shift(24).reindex(f.index).values
    f["gen_roll168"] = gen.rolling(168, min_periods=168).mean().reindex(f.index).values
    err = (gen - fc.reindex(gen.index)).rename("err")
    f["err_lag24"] = err.reindex(back24).values
    f["err_lag48"] = err.reindex(back48).values
    f["err_lag168"] = err.reindex(back168).values
    df = f.join(gen, how="inner").dropna(subset=["wind_gen_mw"]).rename(
        columns={"wind_gen_mw": "target_gen"})
    print(f"Egitim satiri: {len(df)}")

    tmax = df.index.max()
    te = df[df.index > tmax - pd.Timedelta(days=7)]          # final test: son 7 gun
    tu = df[(df.index > tmax - pd.Timedelta(days=21)) & (df.index <= tmax - pd.Timedelta(days=7))]
    tr = df[df.index <= tmax - pd.Timedelta(days=21)]
    tr_m2 = tr.copy()
    tr_m2[M.FRESH_COLS] = np.nan                             # D+2 maskesi (sunumdaki dogal NaN)
    b1 = M.train(tr, target="residual", feats=M.FEATS)       # D+1: taze lag'li
    b2 = M.train(tr_m2, target="residual", feats=M.SAFE_FEATS)  # D+2: guvenli set
    p1_tu = M.predict(b1, tu)
    r_tu = (tu["target_gen"] - p1_tu).groupby(tu.index.hour).mean()
    n_tu = tu.groupby(tu.index.hour).size()
    bias = (r_tu * n_tu / (n_tu + 200)).to_dict()
    p1_tu_b = p1_tu + tu.index.hour.map(bias).fillna(0).values
    # Adaptif blend: NWP yayilim dilimine gore agirlik (belirsiz gunlerde RITM'e yaslan)
    sp = tu["nwp_spread"].fillna(0)
    try:
        bins = pd.qcut(sp, 3, labels=[0, 1, 2], duplicates="drop")
    except ValueError:
        bins = pd.Series(1, index=tu.index)
    edges = [-1e-9, 1e-9, 1e9] if bins.nunique() < 2 else None
    ws, thr = {}, []
    if edges is None:
        qs = sp.quantile([1 / 3, 2 / 3]).tolist()
        thr = [float(qs[0]), float(qs[1])]
    best_w, best_s = 0.0, float("inf")
    for w in [i / 20 for i in range(21)]:
        s = mape(w * p1_tu_b + (1 - w) * tu["ritm_fc"], tu["target_gen"])
        if s < best_s:
            best_w, best_s = w, s
    if thr:
        for b in [0, 1, 2]:
            m = (pd.cut(sp, [-1e-9] + thr + [1e9], labels=[0, 1, 2]) == b)
            bw, bs = best_w, float("inf")
            if m.sum() > 48:
                for w in [i / 20 for i in range(21)]:
                    s = mape(w * p1_tu_b[m] + (1 - w) * tu["ritm_fc"][m], tu["target_gen"][m])
                    if s < bs:
                        bw, bs = w, s
            ws[b] = bw
    else:
        ws = {0: best_w, 1: best_w, 2: best_w}
    p1_te = M.predict(b1, te) + te.index.hour.map(bias).fillna(0).values
    te_m2 = te.copy()
    te_m2[M.FRESH_COLS] = np.nan
    p2_te = M.predict(b2, te_m2)
    def binw(s: pd.Series) -> np.ndarray:
        if not thr:
            return np.full(len(s), best_w)
        b = pd.cut(s.fillna(0), [-1e-9] + thr + [1e9], labels=[0, 1, 2])
        return np.array([ws.get(int(x), best_w) if pd.notna(x) else best_w for x in b])
    w1_te = binw(te["nwp_spread"])
    w2_te = binw(te_m2["nwp_spread"])
    b1_te = w1_te * p1_te + (1 - w1_te) * te["ritm_fc"].values
    b2_te = w2_te * p2_te + (1 - w2_te) * te["ritm_fc"].values
    s1, s2 = mape(b1_te, te["target_gen"]), mape(b2_te, te["target_gen"])
    s_ritm = mape(te["ritm_fc"], te["target_gen"])
    print(f"TEST D+1-proxy: model-blend %{s1:.2f} | D+2-proxy: %{s2:.2f} | "
          f"blok-ort %{(s1 + s2) / 2:.2f} | RITM %{s_ritm:.2f} (w={best_w:.2f}, bins={ws})")
    if (s1 + s2) / 2 < s_ritm:
        p1 = os.path.join("models", "wind_lgbm_d1.txt")
        p2 = os.path.join("models", "wind_lgbm_d2.txt")
        M.save(b1, p1, meta={"blend_w": best_w, "blend_bins": {"thr": thr, "ws": ws},
                             "bias": {str(k): float(v) for k, v in bias.items()}})
        M.save(b2, p2, meta={"blend_w": best_w, "blend_bins": {"thr": thr, "ws": ws},
                             "bias": {}})
        print("Kazanan: dual-model -> models/wind_lgbm_d1|d2.txt")
    else:
        print("Kazanan: RITM — model dosyalari guncellenmedi.")


if __name__ == "__main__":
    main()
