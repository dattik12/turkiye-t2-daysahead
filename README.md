# ⚡ Türkiye Ulusal T+1/T+2 Elektrik Talep Tahmini

[![daily-t2-forecast](https://github.com/dattik12/turkiye-t2-daysahead/actions/workflows/daily.yml/badge.svg)](https://github.com/dattik12/turkiye-t2-daysahead/actions/workflows/daily.yml)

Her gün **09:30 + 17:00 (Türkiye)** 🇹🇷 EPİAŞ'tan tüketim + RİTM, OpenMeteo ECMWF IFS'ten
10 il nüfus-ağırlıklı hava çekilir; **T+1/T+2** saatlik tahmin üretilir. Gerçekleşen veri
geldikçe tahminlerle eşleştirilir ve **MAPE her gün loglanır** — TEİAŞ'ın kendi planıyla karşılaştırmalı.
Sabah koşusu GÖP-öncesi baz, akşam koşusu LEP'li güncellemedir (06:00 UTC IFS + güncel RİTM).

> 🎯 *"Veri seti son günü 18 Ağustos ise bize lazım olan 20 Ağustos."*

![Performans](docs/performance.png)

---

## 📊 Sonuçlar · Backtest 2025-01-01 → 2026-08-18 (597 karar günü, saatlik 14.280 nokta)

| Ufuk | MAPE | MAE | Saatlerin payı |
|:---|---:|---:|:---|
| **T+1** | **%1.68** | 670 MW | %72,2'si ±%2 bandında |
| **T+2** | **%2.26** | 908 MW | %75,6'sı ±%3 bandında |
| T+2 bayram-dışı | %2.15 | — | — |
| T+2 Kurban/bayram | %4.50 | — | — |
| 🏛️ TEİAŞ load-plan (T+1, rakip) | **%3.01** | — | — |

- T+1'de resmî TEİAŞ planını **~1,4 puan** geçiyoruz — farkın en büyük olduğu dönemler
  havanın ani döndüğü geçiş ayları (soldaki grafikte gri barların öne çıktığı aylar).
- Rakamlar `data/exports/backtest_t1_t2_2025_2026.csv` üzerinden
  [`scripts/make_readme_chart.py`](scripts/make_readme_chart.py) ile yeniden üretilebilir.

### 📤 Başka ekiplerin kullanımı için temiz çıktılar (`data/exports/`)

| Dosya | İçerik | Satır |
|:--|:--|--:|
| `backtest_t1_t2_2025_2026.csv` | Saat başına tek satır: gerçekleşen + T+1 + T+2 + TEİAŞ planı | 14.280 |
| `backtest_daily_summary_2025_2026.csv` | Günlük özet: ortalamalar + günlük MAPE | 594 |

- `t1_forecast_mw` = bir gün önceki kararın tahmini; `t2_forecast_mw` = iki gün önceki kararın tahmini.
- `gerceklesen_mw` iki ufuk için aynı → kendi T+1 ve T+2 modelini tek gerçekle eğitebilirsiniz.
- Yeniden üretim: `python -m scripts.export_backtest`

### 🔴 Canlı MAPE

Her koşunun tahmini, gerçekleşen belli olunca `mape_history.csv`'e düşer.
Süreç GitHub Actions'ta koştuğu için tahminler ve skorlar otomatik commit edilir — bu repo aynı
zamanda modelin **canlı kanıt defteridir**.

---

## 🔄 Her gün ne oluyor? (dual-cron: 09:30 + 17:00 TSİ)

1. **Çekim** — EPİAŞ Şeffaflık (`eptr2`) ile gerçek tüketim + TEİAŞ planı + RİTM rüzgar; OpenMeteo ECMWF IFS ile 10 il nüfus-ağırlıklı hava.
2. **Tahmin** — tüketim v4.5 (ufuk-güvenli LightGBM + kapılı MoE), GES (PR-kalibreli fiziksel), RES (RİTM-tabanlı dual-blend).
3. **PTF store** — 13 kolonluk kontrat (`build_ptf_features`, non-blocking) + sert check (`check_ptf_latest`).
4. **Arşiv** — IFS tahmin snapshot'ı (`fcst_snapshots.parquet`) lead-time hizalı eğitim için birikir.
5. **Skorlama** — tahmin vs gerçekleşen → `mape_history.csv` + otomatik commit.

## 🧠 Model kararları (ve nedenleri)

- **Ufuk başına ayrı model.** T+2 modeli `lag24` **kullanmaz** — tahmin anında o saat daha
  yaşanmamıştır; kullanmak veri sızıntısı olur. Bu tek karar, T+2'nin "kağıt üstünde iyi,
  sahada çöken" modellerden farklılaşmasını sağlıyor.
- **Elle ayarlı bayram paketi.** Türkiye'de kayan dini bayramlar hazır takvimlerde eski yıla göre
  duruyor; karar günü bazlı manuel eşleme bayram MAPE'sini %4,5'e indirdi (hazır takvimde daha kötüydü).
- **Deneysel feature'lar üretimde kapalı.** A/B'de generalize etmeyen agresif sürekli feature'lar
  (`ROADMAP.md`) regülasyon şartına bağlı — ana dalda değil.
- **v4.5 kapılı MoE + guardrail.** Bayram/Ramazan'da rejim uzmanı, normalde saat uzmanı
  (çift düzeltme yok); ±%1,5 cap. 60-gün A/B: T+1 −0,009 / T+2 −0,020, hiçbir pencerede zarar yok.
- **Termal feature'lar kapalı.** Temmuz A/B'de T+1 +0,118pp zarar verdi (Ağustos'ta −0,012 fayda
  zararı ödemedi); kod bayrakla duruyor, ilk-sıcak rejiminde kalibre edilmeden açılmayacak.
- **Tek seed (42).** 3-seed ensemble A/B'de gürültü çıktı (T+1 +0,012), maliyet 3× — kapatıldı.

## 📁 Yapı

```
src/        config · data (EPİAŞ+OpenMeteo) · features (+ptf_store) · model · forecast · scoring
(v4.5: ufuk-güvenli lag seti, kapılı rejim-MoE + guardrail, bayram v2, rezidü düzeltme; termal kapalı;
`scripts/_leak_check.py` ile statik+dinamik leak denetimi)
src/consumption/  v4.3 tuketim hatti referans paketi (moduller tasinmadi)
src/solar/        GES: radyasyon (OpenMeteo SW) + sıcaklık derate + PR=0.921 (Tem'26 çapası) + zenit gece maskesi
src/wind/         RES: 7-hub NNLS fit-ağırlık + IFS/GFS çift NWP + dual-LGBM blend (RİTM taban)
scripts/    bootstrap · daily_run (dual-cron) · build_ptf_features · collect_fcast_snapshot ·
check_ptf_latest · backtest_sim · export_backtest · make_readme_chart · train_wind
.github/    daily.yml — cron 09:30 + 17:00 TR + PTF adimi (non-blocking) + sert check + otomatik commit
data/       dataset/ (2016→bugün) · weather/ (+fcst_snapshots) · results/ · exports/ · forecast/ptf_features/
docs/       performance.png (README grafiği, betikten üretilir)
```

## ⚡ PTF Input Feature Engine

Tüketim hattı (v4.5) aynen durur; depo ayrıca PTF modeline girdi üretir:

```bash
python -m scripts.build_ptf_features            # son karar gunu
python -m scripts.build_ptf_features 2026-09-02 # belirli karar gunu
```

```python
# Downstream handoff: tek satirlik dogrulamali okuma
from src.features.ptf_store import load_ptf_features
df = load_ptf_features()  # 48 satir x 13 kolon
```

Çıktı: `data/forecast/ptf_features/archive/ptf_features_YYYY-MM-DD.parquet`
(+csv) + `latest.*` kopyalari — 13 kolonluk data contract:
`datetime (Europe/Istanbul, PK) | horizon (T+1/T+2) | consumption/solar/wind/
residual/renewable_generation (float32 MW) | renewable_penetration |
solar_status (ok/zero_night/unconfigured) | wind_status (blend/ritm/
fallback) | is_peak_hour (08<=saat<20, uint8) | residual_ramp_1h | solar_ramp_1h
(blok-ici saatlik turevler, float32)`.
Export oncesi validate(): 48 satir, gap/dup yok, toplamsallik, residual>0,
gece GES=0, penetrasyon [0,1) — ihlalde yazim YOK.
GES: PR=0.921 (Tem'26 çapası + derate); saatlik şekil 66 lisanslı santrale karşı r=0.88
(şafak 1s gecikme → santral-ağırlıklı radyasyon TODO). RES: fit-ağırlıklı dual-blend **%9.22** vs
RİTM %9.57 (adaptif spread-blend; `wind_status=blend`). IFS tahmin snapshot'ları
(`data/weather/fcst_snapshots.parquet`) her koşuda birikir — lead-time hizalı eğitimin hammaddesi.

## 🚀 Kurulum

```bash
pip install -r requirements.txt        # lightgbm 4.5.0 + numpy<2 (kritik combo)
# .env: EPTR_USERNAME / EPTR_PASSWORD (EPİAŞ Şeffaflık)
python -m scripts.bootstrap            # ilk tam veri (2016→bugün)
python -m scripts.daily_run            # günlük tahmin
```

GitHub Actions için `EPTR_USERNAME` ve `EPTR_PASSWORD` → **Settings → Secrets**.

## 🗂️ Veri kaynakları

- **EPİAŞ Şeffaflık** (eptr2, Apache-2.0) — `rt-cons` (hedef), `load-plan` (TEİAŞ, rakip)
- **OpenMeteo** `ecmwf_ifs` — tarihi hava (eğitim) + hedef gün havası (tahmin), 10 il nüfus-ağırlıklı
