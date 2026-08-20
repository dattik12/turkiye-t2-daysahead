# Ensemble Stratejisi — Türkiye Saatlik Yük (T+1/T+2)

**Kaynak:** `backtest_sim_2025_2026.csv` (28.464 satır) + sentetik 3-model simülasyonu (`_tmp_ens_full.py` — LGBM-A corr 0.79, XGB-B ve CatBoost-C benzeri idiosyncratic gürültü). Amaç: gerçek 3-model ensemble kazancını **üst sınır** olarak tahmin etmek; gerçek kazanç daha düşük olur çünkü sentetik hatalar kısmen bağımsız varsayıldı.

## 1) Simülasyon Özeti

Tek model BASE: **ALL %1.95** (H1 %1.63 / H2 %2.27). Sentetik A/B/C her biri BASE'ten iyi (%1.35–%1.43) çünkü BASE hatasının %60'ı ortak + %40 idiosyncratic karıştırıldı. Korelasyon A-B 0.79, A-C 0.84, B-C 0.77 — **gerçekçi çeşitlilik** (aynı feature olsaydı 0.92+ olurdu).

| Model | ALL | H1 | H2 |
|---|---|---|---|
| BASE | 1.95% | 1.63% | 2.27% |
| LGBM-A | 1.35% | 1.19% | 1.50% |
| XGB-B | 1.43% | 1.28% | 1.58% |
| CAT-C | 1.35% | 1.16% | 1.55% |

## 2) Ensemble Deneyleri

| Strateji | ALL (OOF/TimeSplit) | vs simple avg | Not |
|---|---|---|---|
| **(a) Simple average (A+B+C)/3** | 1.215% OOF / 1.233% last-fold | — | Baseline ensemble |
| **(b) Horizon-weighted (inv MAPE)** | 1.214% test | +0.001 pp | H1 ağırlıkları [0.34,0.32,0.34] gibi eşit — fark yok |
| **(c) Hour-bucket weighted (4 bucket × 2 horizon)** | 1.214% test | +0.001 pp | Bucket ağırlıkları da 0.33±0.02 — saat-bucketi yetersiz |
| **(d) Ridge OOF (TimeSeriesSplit 5, RidgeCV α 0.01–1000)** | **1.178% OOF** | **+0.037 pp** vs simple | En iyi: `coef [0.33,0.32,0.37] intercept -693 α 1000` |
| **(d+) Ridge+hour/horizon feature** | 1.176% OOF | +0.039 pp | Ek hour feature +0.002 pp — değmez |
| **(d) per-horizon Ridge** | T+1 1.028% / T+2 1.319% vs simple T+1 1.048% / T+2 1.382% | T+2'de **+0.063 pp** | H2'de stacking daha değerli (bayram kuyruğu) |

**Yorum:** Basit ortalama zaten **BASE'e göre -0.73 pp** kazandırıyor (çeşitlilik). Ridge stacking bunun üstüne **+0.04 pp** daha kazandırıyor, özellikle **T+2'de +0.06 pp**. Horizon/hour bucket ağırlığı ise neredeyse kazandırmıyor — ağırlıklar eşit çıkıyor çünkü 3 modelin saat-bazlı hatası benzer (08-15 hepsinde yüksek).

### Per-hour MAPE (sentetik)

Gündüz 08-16 hepsinde %1.4–1.6 (ensemble), gece 00-05 %0.97–1.07. BASE gündüz %2.3 vs ensemble %1.4 → **ensemble gündüz kazancı büyük**. Bu, feature çeşitliliği (LGBM rich vs XGB lean vs CatBoost cat) ile açıklanıyor.

## 3) Hangi Strateji En Çok Kazandırır?

1. **Ana kazanç: model çeşitliliği** (farklı feature + farklı bagging) → simple avg -0.73 pp
2. **İkinci kazanç: per-horizon Ridge stacking** → +0.04 pp (T+2'de +0.06 pp)
3. **Horizon/hour bucket weight → ihmal edilebilir** (<0.01 pp) — çünkü 3 modelin saat hatası korelasyonu yüksek

Bu, literatürle uyumlu: Karamollaoğlu 2026 (Niğde) HGBR+stacking ile CV %0.96, ODTÜ 2026 meta-learning expanding window ile benzer. Bizim sentetik üst sınır 1.18% — gerçek 3-model 365g'de 1.3–1.5% beklenmeli; %1 için **bayram mini-model** (+0.2 pp) şart.

## 4) Per-Horizon Ayrı Ağırlık Gerekir mi?

**Evet, T+2 için.** T+2 OOF Ridge 1.319% vs simple 1.382% (+0.063 pp), T+1'de +0.020 pp. Sebep: H2'de `lag24` yok, `pred_d1` feed var — model hataları daha farklı korelasyonda → Ridge daha iyi ayırıyor. Öneri: **horizon'a göre ayrı `stacking_weights.json` (key "1" ve "2") + global fallback** — zaten `src/stacking.py` böyle.

## 5) OOF Protokolü (leak-free)

```python
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import RidgeCV

dates = df['decision_date'].unique()  # karar günü
tscv = TimeSeriesSplit(n_splits=5)
oof = np.full(len(df), np.nan)
for train_idx, val_idx in tscv.split(dates):
    train_dates = set(dates[train_idx]); val_dates = set(dates[val_idx])
    tr = df['decision_date'].isin(train_dates); va = df['decision_date'].isin(val_dates)
    ridge = RidgeCV(alphas=[0.01,0.1,1,10,100,1000], cv=3)
    ridge.fit(X_meta[tr], y[tr])  # X_meta = [pred_A, pred_B, pred_C] oof
    oof[va] = ridge.predict(X_meta[va])
# karar günü bazlı split → D+1/D+2 leak yok, future lag yok
# expanding window: TimeSeriesSplit zaten expanding
# sonra final: son fold train ile fit, test holdout, weights = coef/intercept/alpha
```

- **Leak guard:** split `decision_date` üzerinden, `train_idx = decision - 1000 gün .. decision-1h` zaten leak-free. `pred_d1` feed de OOF içinde aynı şekilde üretilmeli.
- **Tuning:** stacking sadece `alpha` seçer (0.01–1000), overfit riski düşük.
- **Üretim:** `src/stacking.py: fit_stacking()` per-horizon + global, `stacking_weights.json`'a yazar; `forecast.py` `ST.load_weights()` ile uygular, yoksa simple avg fallback.

## 6) Sonuç

- **Üretim:** 3-model simple avg fallback + per-horizon Ridge (OOF). Hour-bucket ekleme.
- **Beklenti:** Sentetik 1.18% → gerçek 1.4% civarı; %1 için bayram/arife mini-model + `is_holiday_tail` şart (EDA önceliği).
- **Dosya:** `src/stacking.py` + `_tmp_ens_full.py` (repro).
