# ⚡ Türkiye Ulusal T+1/T+2 Elektrik Talep Tahmini

Her gece **03:00 (Türkiye)** 🇹🇷 EPİAŞ'tan bir önceki günün tüketimini çeker, veriyi günceller ve
**T+2 (ve T+1)** saatlik tahminini üretir. Gerçekleşen veri geldikçe tahminlerle eşleştirilir ve
**MAPE her gün loglanır** — TEİAŞ'ın kendi planıyla karşılaştırmalı.

> 🎯 *"Veri seti son günü 18 Ağustos ise bize lazım olan 20 Ağustos."*

---

## 📊 Performans

### Backtest · 2025-01-01 → 2026-08-18 (593 karar günü)

| Ufuk | MAPE | MAE |
|:---|---:|---:|
| **T+1** | **%1.63** | 649 MW |
| **T+2** | **%2.27** | 910 MW |
| T+2 bayram-dışı | %2.15 | — |
| T+2 Kurban/bayram | %4.50 | — |
| 🏛️ TEİAŞ load-plan (T+1, rakip) | **%3.01** | — |

> T+1'de resmi TEİAŞ planını **~1.4 puan** geçiyoruz. Saatlik detay: `data/results/backtest_sim_2025_2026.csv`

### 🔴 Canlı MAPE (bugünden itibaren)

Her sabah 03:00'te yapılan tahmin, o günün gerçekleşeni belli olunca `mape_history.csv`'e düşer.

| Karar | Hedef | Ufuk | MAPE |
|:--|:--|:--:|--:|
| 18 Ağu | 19 Ağu | T+1 | _gerçekleşen bekleniyor_ |
| 18 Ağu | 20 Ağu | T+2 | _gerçekleşen bekleniyor_ |

---

## 🔄 Her sabah ne oluyor?

```
03:00 TR
 1. EPİAŞ → dünün tamamlanmış tüketimi + TEİAŞ planı
 2. Hava → OpenMeteo ECMWF IFS (10 şehir, nüfus ağırlıklı)
 3. Karar günü D = son tam gün
 4. T+1 ve T+2 tahmini üret
 5. Geçmiş tahminleri gerçekleşenle eşleştir → MAPE logu
```

Tahminler `forecast_results.csv`'e yazılır, sonuçlar GitHub Actions ile **otomatik commit** edilir.
Manuel tetik: *Actions → daily-t2-forecast → Run workflow*.

---

## 🧠 Model (tek satır)

Her ufuk için **ayrı LightGBM** — D+2 modeli `lag24` **kullanmaz** (o saat daha yaşanmadı — leak yok),
elle ayarlı bayram paketi + OpenMeteo havasıyla. Haftalık yeniden eğitim. Deneysel "manyak"
feature'lar (`ROADMAP.md`) regülasyon ister, şimdilik üretimde kapalı.

---

## 📁 Yapı

```
src/        config · data (EPİAŞ+OpenMeteo) · features · model · forecast · scoring
scripts/    bootstrap (ilk kurulum) · daily_run (03:00) · backtest_sim
.github/    daily.yml — cron 03:00 TR + otomatik commit
data/       dataset/ (2016→ bugün) · weather/ · results/ (forecast + MAPE + backtest)
```

## 🚀 Kurulum

```bash
pip install -r requirements.txt        # lightgbm 4.5.0 + numpy<2 (kritik combo)
# .env: EPTR_USERNAME / EPTR_PASSWORD (EPİAŞ Şeffaflık)
python -m scripts.bootstrap            # ilk tam veri (2016→bugün)
python -m scripts.daily_run            # günlük tahmin
```

GitHub Actions için `EPTR_USERNAME` ve `EPTR_PASSWORD` → **Settings → Secrets**.

---

## 🗂️ Veri kaynakları

- **EPİAŞ Şeffaflık** (eptr2, Apache-2.0) — `rt-cons` (hedef), `load-plan` (TEİAŞ, rakip)
- **OpenMeteo** `ecmwf_ifs` — tarihi hava (eğitim) + hedef gün havası (tahmin), 10 il nüfus-ağırlıklı
