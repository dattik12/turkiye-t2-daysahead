# AUDIT — Phase 0

**Kapsam:** Canlı çalışan üç tahmin modülü (TALEP, GÜNEŞ, RÜZGAR) ve bunların PTF modeline bağlanışı.
**Tarih:** 2026-11-15 · **Mod:** salt-okuma, üretim kodu/config değiştirilmedi.
**Kanıt kaynakları:** `src/`, `scripts/`, `docs/PHASE_LOG.md`, üretim logları (`data/logs/morning_*.log`), ölçüm çıktıları (365 günlük walk-forward).

> **Ölçüm disiplini:** Bu belgede "ölçüldü" yazan her sayı bir koşudan gelir; "ölçülmedi" yazanlar tahmin edilmemiştir, AÇIK SORULAR'a taşınmıştır.

---

## 1. ENVANTER

### 1.1 TALEP (Load)

| | |
|---|---|
| **Entry point** | `scripts/morning_run.py` |
| **Schedule** | Windows Görev Zamanlayıcı `TR_Morning_Yarim_Tahmin`, günlük **09:00 TR**, SYSTEM |
| **Sarmalayıcı** | `run_morning.bat` (çıkış kodu korunur, `LAST_FAILURE.txt`'ye hata izi) |
| **Mimari** | Topluluk: LightGBM (w=0.10) + CatBoost (w=0.20) + MLP (w=0.70). MLP 3 tohum ortalaması `[7,17,27]` |
| **Feature sayısı** | **83** (lag48/72/96/120/144/168/336, lag24 maskeli; aynı-saat ortalamaları; anchor rolling özetleri; takvim; hava + HDD/CDD) |
| **Eğitim penceresi** | `ensemble_morning.TRAIN_DAYS = 2000` gün (~5,5 yıl) — pencere taramasıyla seçildi |
| **Retrain** | **Her koşuda** tam yeniden eğitim (artımlı değil) |
| **Çıktı** | `data/forecast/morning/yarin_YYYY-MM-DD.parquet` (24×7) + `data/results/morning_history.csv` (skor defteri) |
| **PTF'ye bağlanış** | Doğrudan değil: `morning_ptf.py` kendi modelini kurar ve kontrata yazar |

**Not:** `src/config.py:TRAIN_DAYS = 1000` tanımlı ama topluluk yolu `ensemble_morning.TRAIN_DAYS = 2000` kullanıyor. İki sabit çelişiyor → AÇIK SORULAR #1.

### 1.2 GÜNEŞ (Solar)

| | |
|---|---|
| **Entry point** | `scripts/solar_live.py` (canlı) · `scripts/solar_mnwp.py` (backtest/tanı) |
| **Schedule** | `run_morning.bat` içinde, `morning_ptf`'den **önce** |
| **Mimari** | LightGBM, CF-normalize (kapasite zarfına bölünmüş) |
| **Feature sayısı** | **28** (multi-NWP GHI/bulut/sıcaklık, clear-sky indeksi, `cf_lag48/72/168` ufuk-maskeli, `ghi_7d_same_hour`, `cf_same_hour_7d`, fizik terimleri) |
| **Eğitim penceresi** | `TRAIN_D = 540` gün, `STEP_D = 7` |
| **Retrain** | Her koşuda |
| **Çıktı** | `data/forecast/solar_live.parquet` (48×4) |
| **PTF'ye bağlanış** | `morning_ptf.py` `SOLAR_ML` olarak okur; 40/48 saatten az kapsama varsa **fizik formülüne** düşer |

**Kapasite zarfı:** `rolling(30*24).quantile(0.99).shift(1)` → son zarf **3.671 MW** (259 → 4.168 aralığı).
**Gece maskesi:** `solar_elevation <= 0` → zorunlu 0.

### 1.3 RÜZGAR (Wind)

| | |
|---|---|
| **Entry point** | `scripts/wind_live_v5.py` (canlı, `status="ml_v5_multinwp"`) |
| **Schedule** | `run_morning.bat` içinde, `morning_ptf`'den **önce** |
| **Mimari** | LightGBM, IEC power curve, CF-normalize |
| **Feature sayısı** | **30** (3 NWP × 100m rüzgar hızı, spread, `cf_lag24/48/168` ufuk-maskeli, `cf_7d`, ramp) |
| **Eğitim penceresi** | `TRAIN_D = 365` gün, `STEP_D = 7` |
| **Retrain** | Her koşuda |
| **Çıktı** | `data/forecast/wind_live.parquet` (48×4) |
| **PTF'ye bağlanış** | `morning_ptf.py`: **T+1 = 0,70×RİTM + 0,30×ML**; **T+2 = saf ML**. Kapsama yetersizse **iklim** fallback |

**Birim notu:** NWP girişi km/h, model içinde `/3.6` ile m/s'ye çevriliyor (geçmişte bu hata power curve'ü 25 km/h üzerinde sıfırlıyordu).

### 1.4 PTF Kontratı

| | |
|---|---|
| **Entry point** | `scripts/morning_ptf.py` |
| **Çıktı** | `data/forecast/ptf_features/latest.parquet` — **48 satır × 13 kolon** |
| **Arşiv** | `data/forecast/ptf_features/archive/ptf_YYYY-MM-DD.parquet` + sidecar |
| **Bütünlük** | `residual = consumption − (solar + wind)`, tolerans < 0,01 MW (fiilen 0,003906) |
| **Koruma** | Güneş tamamen 0 ise **kontrat yazılmaz** (`SystemExit`), önceki korunur |

---

## 2. DATA LINEAGE

| Input | Kaynak | Granülarite | Gecikme | Koşu anında var mı |
|---|---|---|---|---|
| Tüketim gerçekleşen (`rt_cons`) | EPİAŞ şeffaflık | saatlik, ulusal | ~1–2 saat | ✅ karar günü **07:00**'ye kadar |
| YG planı (`load_plan` / `lep`) | EPİAŞ (TEİAŞ planı) | saatlik | gün-öncesi | ✅ benchmark olarak |
| RİTM | EPİAŞ | saatlik | ~24 saat ufuk | ✅ T+1 için |
| NWP (yük) | Open-Meteo `historical-forecast-api` (arşiv) + `forecast` (canlı) | saatlik, 10 il nüfus-ağırlıklı | — | ✅ |
| NWP (rüzgar) | `C:\projeler\multinwp\wind` — IFS/GFS/ICON, **10 il**, 100 m | saatlik | — | ✅ `pull_daily` her sabah tazeler |
| NWP (güneş) | `lead_weather` 10 il + `multinwp/solar` | saatlik | — | ✅ |
| Takvim | `build_hourly_calendar` | saatlik, 2018–2028 | — | ✅ |

### 2.1 Hava verisi — leakage sorusuna NET cevap

| Modül | Backtest/eğitimde kullanılan | Vintage durumu |
|---|---|---|
| **TALEP** | `lead_weather/*.parquet` → `_previous_day1` / `_previous_day2` | ✅ **Gerçek geçmiş FORECAST vintage'ı.** `historical-forecast-api` "o gün için 1/2 gün önce yayınlanan tahmin" verir. Reanalysis (ERA5) **kullanılmıyor** — `archive-api` bilinçli olarak dışlandı. |
| **RÜZGAR** | `multinwp/wind/*.parquet` | ⚠️ **Vintage bilgisi YOK.** Arşivde her saat için tek değer var; hangi NWP run'ından geldiği yazmıyor. |
| **GÜNEŞ** | `multinwp/solar`, `lead_weather` | ⚠️ Aynı belirsizlik. |

**Sonuç:** Talep backtest'i vintage açısından **adil**. Rüzgar/güneş backtest'i **doğrulanamaz** — "%15,53" ve "%10,76" iyimser olabilir. Bu bir varsayım değil, ölçülemeyen bir belirsizlik: arşivde lead metadata'sı yok.

**Koordinatlar ve aggregate:** Rüzgar 10 il (Balıkesir, İzmir, Çanakkale, Manisa, Hatay, Aydın, İstanbul, Kırklareli, Afyon, Kayseri) — kurulu güç ağırlığı **değil**, düz ortalama + `arch_mean`. Güneş 8 il GES kurulu gücü ağırlıklı (Konya 26, Ankara 13, Kayseri 11, Antalya 10, İzmir 8, Adana 7, Gaziantep 7, Bursa 5, İstanbul 3, Trabzon 1). Yük 10 il **nüfus** ağırlıklı.

**Tutarsızlık:** Rüzgar il ağırlıkları kurulu güce göre değil → AÇIK SORULAR #4.

### 2.2 Bilinen kırılganlık

`WEATHER_NATIONAL_PARQUET` (`data/weather/histfc_national.parquet`) **gelecek satırı içermez**; hedef günün havası koşu anında canlı API'den belleğe eklenir ve diske yazılmaz. Bu, geçmiş bir koşunun havasını sonradan **yeniden üretmeyi imkânsız** kılıyordu. `weather_consumed/*.parquet` arşivi bu boşluğu 9 Kasım'dan itibaren kapatıyor.

---

## 3. TARGET KALİTESİ

| Konu | Talep | Güneş | Rüzgar |
|---|---|---|---|
| Hedef seri | `rt_cons` (EPİAŞ gerçekleşen tüketim) | `sun` (EPİAŞ ulusal güneş üretimi) | `wind` (EPİAŞ ulusal rüzgar üretimi) |
| Kısıntı/YAT etkisi | Yok | ⚠️ **Ölçülmedi** — curtailment hedefe gömülü olabilir | ⚠️ **Ölçülmedi** |
| Kurulu kapasite serisi | — | ❌ **Yok** — `rolling_30d_p99` zarfı vekil | ❌ **Yok** — aynı vekil |
| Lisanslı/lisanssız | — | ⚠️ **Belirsiz** — EPİAŞ `sun` serisinin kapsamı doğrulanmadı | ⚠️ Belirsiz |
| Outlier | 2026-11-03 üretimi arızalıydı (akşam koşusu ürünü ölçülmüş) | — | — |
| Eksik veri | `lag` maskeleri ile yönetiliyor | Gece 0 zorunlu | — |

**Kapasite zarfı neden riskli:** `rolling(30g).p99` gerçek kurulu gücü değil, **son 30 günün en yüksek üretimini** izler. Kurulu güç arttıkça zarf gecikmeli takip eder; bulutlu bir ay zarfı düşürür → CF yapay olarak yükselir. Gerçek kapasite serisi (TEİAŞ/ETKB aylık) varsa bu değiştirilmeli.

---

## 4. METRİKLER

### 4.1 Ölçülmüş ana skorlar (365 gün walk-forward)

| Modül | Ufuk | WAPE / MAPE | nMAE | bias | Not |
|---|---|---|---|---|---|
| **Talep** | T+1 | **%1,66** | 684–757 MW | −22 MW | canlı-eşdeğer (y 07:00 kesik) |
| **Talep** | T+2 | **%1,82** | 875 MW | −40 MW | per-horizon eğitim |
| Talep (son 90g) | T+1 / T+2 | %1,51 / %1,66 | — | — | iki yarı: 1,82 / 1,88 |
| **Güneş** | T+1 gündüz | **%10,76** | 109 MW | −8 MW | tüm gün %11,33 |
| **Güneş** | T+2 gündüz | **%10,86** | 110 MW | −10 MW | — |
| **Rüzgar** | T+1 | **%15,53** | 711 MW | −22 MW | — |
| **Rüzgar** | T+2 | **%15,63** | 716 MW | −23 MW | — |

**Benchmark:** TEİAŞ gün-öncesi planı, 8 günde **%1,62**. Aynı 8 günde bizim eski kod %3,18 idi; düzeltme sonrası beklenen ~%1,85.

### 4.2 Talep — saat bazlı (T+1, 365 gün)

En iyi: 02–04 (%1,00–1,02). En kötü: **08–09 (%2,18–2,23)**. Gece ~%1,0, gündüz ~%2,1. Sabit bias -650 MW'a kadar çıkıyordu (düzeltme öncesi).

### 4.3 Rüzgar — hata yapısı

| Boyut | Bulgu |
|---|---|
| Havza | **Marmara +246 MW**, **Ege +253 MW** (yüksek hızda); Akdeniz düz |
| NWP spread | düşük 623 → yüksek **807 MW** (1,30x) |
| Rampa (p90 1.282 MW) | 886 vs 681 MW (1,30x) |
| Cut-out (>25 m/s) | **sadece 3 saat** (%0,02) — reddedildi |
| Saat bias | gece −125…−255, öğle +133…+180 |

### 4.4 Güneş — hata yapısı

En iyi saatler **12–14** (%9,63–9,75). Gün batımında patlıyor (18: %17,29, 19: %33,11) ama bias küçük → saçılma, sistematik değil. **Curtailment şüphesi reddedildi** (öğle bias'ı −8…−43 MW).

### 4.5 ÖLÇÜLMEDİ — bu turda yapılmayanlar

Aşağıdakiler talep edildi ancak **bu denetimde ölçülmedi** (uydurulmadı):
- Ay × saat kırılımı (üç modül)
- Hafta içi / hafta sonu / tatil kırılımı (yük için kısmen biliniyor)
- Hava rejimi kırılımı (rüzgar için spread kırılımı var, diğerleri yok)
- **Residual load (talep − rüzgar − güneş) MAE/MAPE** — hiç ölçülmedi
- **Üç modülün hata korelasyonu** — hiç ölçülmedi
- Güneş/rüzgar için canlı skor (5 gün birikti, anlamlı değil)

Bu maddeler ayrı bir analiz script'i gerektirir (`scripts/audit_phase0.py`), mevcut kod değiştirilmeden koşulabilir.

---

## 5. HİPOTEZ LİSTESİ — beklenen kazanç / maliyet sırasıyla

> **Not:** GBM'in mevcut lag + takvim feature'larından zaten türetebileceği fikirler (ör. "saat ortalaması ekle", "hafta sonu bayrağı ekle") listeye **alınmadı** — model bunları hâlihazırda görüyor.

| # | Hipotez | Hedeflenen hata | Beklenen kazanç | Maliyet | Leakage riski |
|---|---|---|---|---|---|
| 1 | **Rüzgar gerçek vintage NWP** — arşivi lead-bazlı topla (D-1 00z run'ı), backtest'i yeniden kur | Rüzgar T+1/T+2 (711/716 MW) | Yüksek (skor gerçeğe iner, iyileştirme yönü netleşir) | Yüksek (ücretli veri / uzun çekim) | **Yok** — aksine mevcut iyimserliği kaldırır |
| 2 | **Residual load için doğrudan model** — talep/güneş/rüzgar ayrı ayrı değil, farkı doğrudan tahmin et | PTF'nin gerçek girdisi | Orta-yüksek (hata bileşenleri korele ise toplam hata düşer) | Orta (yeni model + backtest) | Düşük |
| 3 | **Gerçek kurulu kapasite serisi** (güneş/rüzgar, aylık TEİAŞ/ETKB) — p99 zarfı yerine | Güneş/rüzgar CF hatası | Orta (zarf gecikmesi sistematik bias yaratıyor) | Düşük-orta (veri temini) | Yok |
| 4 | **Rüzgar il ağırlıkları** — düz ortalama yerine kurulu güç ağırlıklı | Rüzgar (Marmara/Ege +500 MW hata) | Orta | Düşük (mevcut veriyle) | Yok |
| 5 | **Rüzgar rejim-bağımlı karışım** — NWP spread yüksekken RİTM ağırlığını değiştir | Rüzgar T+1 | Orta (spread 1,30x ayrışıyor) | Düşük (mevcut seriler) | Düşük (walk-forward şart) |
| 6 | **Talep T+2 horizon-farkında feature seti** — lag72+ ile ayrı eğitim (şu an T+1 semantiği taşınıyor) | Talep T+2 (875 MW) | Orta (%1,82 → ~%1,6) | Orta | Düşük |
| 7 | **Kısıntı (curtailment) düzeltmesi** — hedef seriden bilinen kısıntıyı arındır | Güneş/rüzgar hedef kalitesi | Orta ama **önce ölçülmeli** | Düşük-orta (veri) | Yok |
| 8 | **Hava sağlayıcı çeşitlendirme** — Open-Meteo tek nokta arızası (9 Kasım'da kontrat yazılmadı) | Sistem erişilebilirliği | Skor değil, **riski** düşürür | Orta | Yok |
| 9 | **Ramp odaklı örnek ağırlıklandırma** — büyük rampa saatlerinde hata 1,30x | Talep/rüzgar rampa saatleri | Düşük-orta | Düşük | Düşük |
| 10 | **Topluluk ağırlıklarının canlı veriyle yeniden kalibrasyonu** — şu an backtest ağırlıkları (0.10/0.20/0.70) | Talep | Düşük | Düşük | Düşük (trailing fit şart) |

### Elenen fikirler (kanıtla)
Tatil ağırlıklandırma · rejim ağırlıkları · MAPE-uyumlu amaç fonksiyonu · Chronos-2/TSFM · XGBoost · MoE · güneşte yeni mimari — **hepsi denendi, kazanmadı** (`docs/PHASE_LOG.md`).

---

## AÇIK SORULAR

1. **`TRAIN_DAYS` çelişkisi** — `src/config.py` 1000, `ensemble_morning.py` 2000. Canlı koşuda hangisi etkin? Kod okuması `ensemble_morning`'i işaret ediyor ama `morning_run` hangisini kullanıyor, doğrulanmadı.
2. **Rüzgar/güneş NWP vintage'ı** — arşivde lead metadata'sı yok. Hangi run'dan geldiği bilinmiyor. Bu doğrulanmadan rüzgar/güneş backtest skorları **sertifika edilemez**.
3. **`sun` serisinin kapsamı** — lisanslı mı, lisanssız üretim dahil mi? 27.445 MW `SOLAR_CAPACITY_MW` sabiti hangi kaynaktan, hangi tarihte?
4. **Rüzgar il ağırlıkları** — 10 il düz ortalamayla toplanıyor; kurulu güç ağırlıkları mevcut mu, neden kullanılmıyor?
5. **Kısıntı/YAT** — EPİAŞ `sun`/`wind` serilerinde kısıntı var mı? Varsa hedefe gömülü mü? Ölçülmedi.
6. **Residual load kalitesi** — hiç ölçülmedi. PTF'nin doğrudan girdisi olduğu halde.
7. **Üç modülün hata korelasyonu** — bilinmiyor. Yüksekse residual load hatası beklenenden büyük.
8. **TEİAŞ planının yayın saati** — bizimki 09:00. Onlarınki kaçta yayınlanıyor? Aynı bilgi setiyle yarıştığımız doğrulanmalı.
9. **Güneş `SHIFT_H = -1`** — hava serisinde 1 saat ileri kaydırma. Fiziksel gerekçesi (etiketleme konvansiyonu mu, gerçek offset mi) yazılı değil.
10. **`morning_ptf` T+1 RİTM ağırlığı 0,70** — hangi ölçümle seçildi? Güncel rüzgar skoruyla hâlâ optimal mi?
11. **Canlı skor birikimi** — 15 Kasım itibarıyla ~5 gün. 30 günlük anlamlı ölçüm için ne zaman hazır olacak?
12. **Kapasite zarfı `p99`** — bulutlu/rüzgarsız bir ayda zarf düşerse CF yapay yükselir. Bu etki ölçüldü mü?
