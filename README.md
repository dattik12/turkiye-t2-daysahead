# 🇹🇷 tr-t2-daysahead — Türkiye Ulusal T+1/T+2 Gün Öncesi Elektrik Talep Tahmini

Her gece **03:00 (Türkiye)** [EPİAŞ Şeffaflık Platformu](https://seffaflik.epias.com.tr)'ndan
(eptr2) **bir önceki günün tamamlanmış tüketimini** çeker, veri setini günceller ve **T+2**
(ve T+1) gün öncesi saatlik tahmin üretir. Gerçekleşen veri geldikçe tahminlerle eşleştirilir,
**MAPE günlük loglanır**, TEİAŞ'ın kendi gün öncesi planıyla (load-plan) kıyaslanır.

> "Veri seti son günü = 10 Ağustos ise bize lazım olan = **12 Ağustos**."

## Nasıl çalışıyor

```
03:00 TR her gün
  1. eptr2 -> rt-cons + load-plan (önceki günün 24 saati tamamlandı)
  2. veri seti güncelle  → data/dataset/tr_consumption_hourly.parquet
  3. hava güncelle       → data/weather/* (OpenMeteo ECMWF IFS, 10 şehir nüfus ağırlıklı)
  4. karar günü D = son TAM gün
  5. T+1 ve T+2 saatlik tahmin (D+1 ve D+2)  → data/results/forecast_results.csv
  6. geçmiş tahminleri gerçekleşenle eşleştir → data/results/mape_history.csv
```

## Model (v4.3 + v5.2 ensemble)

- Her ufuk için **ayrı LightGBM** (D+1 ve D+2 modelleri).
- **D+2 modeli lag24 KULLANMAZ** (D+1 yaşanmadı) → recursive olmadan, dağılım kaymasız.
- **BASE model** (v4.3): elle ayarlı bayram paketi — `is_arife`, `is_holiday`, `prev_week_holiday`,
  `next_day_holiday`, `days_to_holiday`, `ramadan` + lag/rolling/günlük özetler.
- **CONT model** (v5.2): BASE + `same_hour_median_3d/7d`, `daylight_fraction`, spatial sıcaklık
  yayılımı, `wet_bulb`, `temp_anomaly_30d`, segment etkileşimleri.
- Ensemble: iki modelin ortalaması. Haftalık yeniden eğitim ("yakın veriyle adaptasyon" dersi).

### Ölçülen performans (üretim protokolü backtest)

| Senaryo | MAPE |
|---|---|
| T+1 (D+1) | ~%1.7 |
| T+2 (D+2) | ~%2.2–2.4 |
| D+2 bayram-dışı | ~%2.0 |
| D+2 Kurban penceresi | ~%4 |
| TEİAŞ load-plan (D+1, benchmark) | ~%2.8 |

Ayrıntılar: `ROADMAP.md` ve `data/results/backtest_sim_2025_2026.csv`.

## Repo yapısı

```
src/             config, data (eptr2+OpenMeteo), features, models, forecast, scoring
scripts/         bootstrap.py (ilk tam kurulum)
                 daily_run.py  (günlük 03:00 orkestrasyonu)
                 backtest_sim.py (2025→ bugün simülasyonu)
.github/workflows/daily.yml  (cron 03:00 TR, sonuçları otomatik commit eder)
data/
  dataset/       tr_consumption_hourly.parquet (rt-cons + load-plan, 2016→ bugün)
  weather/       OpenMeteo ECMWF IFS tarihi hava (eğitim) — parquet
  results/       forecast_results.csv   (her saatlik tahmin + actual + mape)
                 mape_history.csv       (target×horizon MAPE günlüğü)
                 backtest_sim_2025_2026.csv
```

## Kurulum (yerel)

```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt      # lightgbm 4.5.0 + numpy<2 (kritik combo)
# .env oluştur: EPTR_USERNAME / EPTR_PASSWORD (EPİAŞ Şeffaflık kayıtlı e-posta + şifre)
python -m scripts.bootstrap    # ilk: 2016→ bugün tam veri + hava geçmişi
python -m scripts.daily_run    # günlük tahmin
python -m scripts.backtest_sim # 2025→ bugün simülasyon CSV
```

## GitHub Actions

- `EPTR_USERNAME` ve `EPTR_PASSWORD` → **repo Settings → Secrets**
- Cron: her gün **00:00 UTC = 03:00 Türkiye** (Actions zamanlamada birkaç dakika gecikebilir)
- Sonuçlar her koşuda otomatik commit edilir → `data/results/` her gün güncellenir
- Ayrıca **Actions → daily-t2-forecast → Run workflow** ile elle tetiklenebilir

## Veri kaynakları

- EPİAŞ Şeffaflık (eptr2, Apache-2.0): `rt-cons` (gerçekleşen, hedef), `load-plan` (TEİAŞ, benchmark)
- OpenMeteo Historical-Forecast API `ecmwf_ifs`: eğitim havası (üretimle aynı model ailesi → bias minimum)
- OpenMeteo Forecast API `ecmwf_ifs`: hedef gün havası (gerçek kullanımda TAHMİN)
- Özel gün takvimi: `turkiye_ozel_gun_saatlik_talep_feature_takvimi` (doküman kaynağı)
