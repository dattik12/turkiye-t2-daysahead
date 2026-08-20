# Model Araştırması — LGBM vs XGB vs CatBoost (STLF, 24k satır, 1000 gün, H1/H2 ayrı)

**Veri:** ~24k saat (1000 gün), ~35-50 feature, horizon 1 (lag24 var) / 2 (lag24 yok) ayrı modeller. Hedef: 365 gün MAPE %1.

## 1) Karakteristik

| Model | Ağaç büyütme | Güçlü yanı (bizim durumda) | Zayıf / dikkat | Kaynak |
|---|---|---|---|---|
| **LightGBM** | leaf-wise, histogram | En hızlı, derin `num_leaves` ile ince etkileşimler (hour×temp, bayram×hour) iyi — STLF SOTA'ların çoğu LGBM/HGBR | Leaf-wise overfit'e meyilli → `min_child_samples` + `colsample` şart | Ke et al. NeurIPS 2017 |
| **XGBoost** | level-wise (hist), sparsity-aware | Düzenli derinlik + `reg_alpha/lambda` ile bayram gibi az örnekli rejimlerde daha stabil, `gamma` ile gürültü budama | `max_depth` 8+ hızla şişer, CatBoost kadar kategorik sevmez | Chen & Guestrin KDD 2016 |
| **CatBoost** | ordered boosting, oblivious trees | `hour/weekday/month` + bayram flag'lerini **cat_features** olarak native işler — one-hot kaybı yok, arife/bayram etkisini saatle birlikte iyi yakalar | `iterations` 1500'de LGBM'den yavaş, `border_count` düşükte kuantizasyon kaybı | Prokhorenkova et al. NeurIPS 2018 |

**Sonuç bizim veride:** LGBM = ana motor (hız + esneklik), XGB = regularize yedek (bayram kuyruğunda stabil), CatBoost = kategorik/calendrical uzman.

## 2) Hiperparametre — Optuna aralıkları (bizim ölçekte makul)

### LGBM (`LGB_PARAMS`)
```
n_estimators      800–2500 (early_stopping 50, ama biz fixed 1500 prior)
learning_rate     0.02–0.08  (log)
num_leaves        31–150     (prior 95)
max_depth         -1, 6–12   (-1 = leaves'e bırak)
colsample_bytree  0.6–1.0    (prior 0.8)
subsample         0.6–1.0    (prior 0.8, subsample_freq 1)
min_child_samples 10–100     (prior 50)
reg_alpha/lambda  0–1.0      (L1/L2)
```
Validasyon: TimeSeriesSplit(5), horizon ayrı. Trial 30–50 TPE, pruner median.

### XGB (`XGB_PARAMS`)
```
n_estimators      800–2500
learning_rate     0.02–0.10 log
max_depth         4–12       (prior 8)
colsample_bytree  0.6–1.0    (prior 0.8)
subsample         0.6–1.0    (prior 0.85)
min_child_weight  1–10       (prior 3)
reg_alpha         0–2.0      (prior 0.1)
reg_lambda        0–2.0      (prior 1.0)
gamma             0–0.3
tree_method       hist
```

### CatBoost (`CAT_PARAMS`)
```
iterations        800–2500
learning_rate     0.02–0.10 log
depth             4–10       (prior 8)
l2_leaf_reg       1–10       (prior 3)
random_strength   0–3        (prior 1.0)
bagging_temperature 0–1      (prior 0.5)
border_count      32–255     (prior 128)
```

**Not:** `n_estimators/iterations` 1500 prior'ı ~40dk eğitim demek; Optuna'da early stopping yoksa trial başı ~2dk (1000 gün × 5 fold değil, tek expanding window kullan).

## 3) Per-model feature set (aynı olmak zorunda değil — çeşitlilik = ensemble kazancı)

| Model | Önerilen set | Neden |
|---|---|---|
| **LGBM** | `BASE_H1/H2` + `CONTRA` + `HOLIDAY_COLS` (`is_holiday_tail`, `is_bridge`, `is_holiday_effect`) + tüm hava (`temp2`, `temp_hour`, `wdyn`) | En geniş seti kaldırır, leaf-wise ince interaksiyonları yakalar. `daylight_fraction`, `temp_spread_maxmin` gibi manyaklar LGBM'e kazandırır. |
| **XGBoost** | `BASE_H1/H2` (lean: `lag336` düşür) + `HOLIDAY_COLS` + sade hava (`temperature_2m`, `CDD/HDD`, `temp_hour`) | Regularisation + sade lag → bayram gibi az örnekte overfit az. EDA'da `bridge-day` zaten %1.88 → XGB'e `is_bridge` yeterli, ` CONTRA`'yı kıs. |
| **CatBoost** | `BASE_H1/H2` + `HOLIDAY_COLS` + `cat_features = [hour,weekday,month,is_holiday,is_arife,is_after,is_holiday_tail,is_bridge,is_ramadan]` | CatBoost'un ordered TS ile kategorikleri encoding'siz öğrenmesi; `hour × is_holiday_effect` interaksiyonunu otomatik bulur. Sayısal hava minimal tut, kategoriklere odaklan. |

Ortak hepsi: `daylight_fraction`, `CDD/HDD`, `temp2`/`temp_hour`. Ayrışma: LGBM'e geniş manyak, XGB'e lean+reg, CatBoost'a kategorik.

## 4) Ensemble çeşitliliği — korelasyonu düşürme

Hedef: 3 modelin **hata korelasyonu 0.65–0.80** civarı (şu an tek LGBM varyantları 0.95+). Düşük korelasyon → simple average bile ~0.15–0.30pp kazandırır (sentetik deney: simple avg 0.22pp).

- **Farklı feature:** yukarıdaki gibi lean vs rich vs cat
- **Farklı loss/sampling:** LGBM `min_child_samples` 50 vs XGB `min_child_weight` + `reg` vs CatBoost `bagging_temperature`
- **Farklı horizon:** H1 ve H2 zaten ayrı model — H2'de `lag24` yok, `pred_d1` feed var; bu doğal çeşitlilik
- **Farklı seed/bagging:** her model farklı `random_state` + subsample

Kaçınılacak: aynı feature + aynı depth → korelasyon 0.9+ → ensemble boşa.

## 5) EDA'dan çıkarım (bayram kuyruğu)

EDA `eda_segments.md`: normal (strict) %1.72, arife %7.00, bayram %6.49. `is_holiday_tail` (+2 gün) ve `is_holiday_effect` bu segmenti normalleştirmede kritik — **3 modele de ekle**. `is_bridge` zaten %1.88 ile sorun değil ama CatBoost'a cat olarak eklenmesi per-hour ağırlık yerine geçer.

## 6) Önerilen üretim

- Üretimde `ENSEMBLE_MODELS = ["lgbm","xgb","catboost"]`, per-horizon Ridge stacking (TimeSeriesSplit OOF, leak-free) — `STACKING_ALPHAS 0.01–100`. Basit ortalama fallback.
- Her horizon için ayrı `alpha/coef` (H1 vs H2 farklı hata dağılımı: H2 bayramda +1.98pp).
- Per-model Optuna sonuçları `data/results/{lgbm,xgb,catboost}_tuning.md`'ye, nihai stacking `stacking_weights.json`'a.
