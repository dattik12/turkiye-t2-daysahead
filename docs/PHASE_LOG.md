# PHASE LOG — turkiye-t2-daysahead

Kararların ve ölçümlerin kalıcı kaydı. Context sıfırlanmalarına karşı.
Her giriş: tarih, karar, kanıt, geri alma yolu.

---

## FAZ 2 — KAPANDI (2026-11-05)

### Kazanç: rüzgar %32,59 → %14,67

| Model | WAPE | nMAE | Not |
|---|---|---|---|
| 10m / 10 il (lead_weather) | %32,59 | — | kullanılamaz |
| 100m / 39 il, tek NWP | %17,78 | 814 MW | power curve + 6 bölge |
| Multi-NWP, karışık dönem | %15,41 | 705 MW | 2023'te 2 model |
| **Multi-NWP, temiz dönem (2025+)** | **%14,67** | **718 MW** | 3 model tam, n=13.800 |
| RİTM (referans) | ~%13 | — | **ÖLÇÜLMEDİ** — eski notlardan; 24 Kasım'da aynı pencerede ölçülecek |

### KARAR: multi-NWP canlıda
- `scripts/wind_live_v5.py`, `status='ml_v5_multinwp'`
- Kontrat `wind_status='blend0.70_t1_ml_v5_multinwp_t2'`
- **Geri alma:** `wind_live.parquet`'i tek-NWP sürümle yeniden üret, `morning_ptf` çalıştır.

### NWP kapsam gerçeği (COVERAGE_MANIFEST.json)
- **2021: hiçbir model yok.** Arşiv çekilemez.
- **2022: sadece GFS.** IFS/ICON yok.
- **2023: GFS + ICON** (2 model).
- **2024Q2'dan itibaren: 3 model tam.**
- Sonuç: 2023 eğitim/eval verisi 2-model spread içeriyor. Temiz ölçüm 2025+ ile yapıldı.

### Faz 2'de düzeltilen hatalar
1. `wind_speed_100m` birimi **km/h**; power curve m/s varsayıyordu → 25 km/h üzeri `pc=0`. Düzeltme skoru değiştirmedi (model ham hızdan telafi ediyordu) ama fizik artık doğru.
2. Model kolon eşleşmesi: `split('_')[-1]` alt çizgili adlarda (`ecmwf_ifs025`) hiç eşleşmiyordu.
3. Güneş çekimi: çok değişkenli istek timeout yiyor → **değişken başına ayrı istek**.
4. tz karışımı → `object` dtype → `astype(float)`.
5. `solar_status` / `wind_status` etiketleri hardcode'du → artık üreten modülün dosyasından okunur.

---

## FAZ 3 — KARARLAR (2026-11-05 gece)

### 1. RİTM/ML karışım ağırlığı — ÖLÇÜLMEDİ

**Karar: `w=0.70` KALIYOR.**

Gerekçe: `wind_history.csv` 120 saat (~5 gün) arşiv içeriyor. Ön tanımlı kapı
**21 gün** idi. Pencere yetersiz → ölçüm yapılmadı, sadece durum raporlandı.

**Neden geriye dönük çekim yapılmadı:** RİTM'i bugün API'den geriye dönük çekmek
09:00'teki canlı tahmini değil **revize/tutarlılaştırılmış** seriyi verir → sızıntı.
Tek dürüst kaynak, koşu anında arşivlenen `ptf_store._log_wind()` kayıtları.

**Ne zaman ölçülebilir:** 2026-11-03 + 21 gün = **2026-11-24** sonrası.
O gün `w ∈ {0.40…0.80}` adım 0.05 taranır; sağlamlık kapısı: iki yarıda optimal w
±0.10 içinde VE toplam kazanç ≥ 0.3 puan → yeni w devreye girer, yoksa 0.70 kalır.

### 2. Rüzgar 2021-2022 arşivi — ÇEKİLEMEDİ

**Karar: iptal.** 2021'de hiçbir model yok; 2022'de yalnız GFS var. "Multi-NWP"
tanımına uymuyor. Eğitime koşulmadı (kurulu güç ~yarısı, dağılım kayar).

### 3. Güneş multi-NWP — KAPI GEÇMEDİ, TERFİ YOK

**Karar: `ml_v4_ges` KALIYOR.** Kanıt:

| Model | Tüm dönem | Tam kapsam (2025+) |
|---|---|---|
| ml_v4_ges (tek NWP, GES ağırlıklı) | %13,01 (tüm dönem) | **%11,39** — nMAE 116 MW (tüm gün) |
| multi-NWP (IFS+GFS+ICON) | %11,49 (tüm dönem) | **%10,69** — nMAE 109 MW (tüm gün) |
| **Fark (AYNI pencere)** | — | **−0,70 puan** |

Yarımlar (tam kapsam, gündüz):
- ml_v4_ges: 10,60 / 12,07
- multi-NWP: 10,19 / 11,13

Kapı **≥1 puan** istiyordu; gerçek kazanç **−0,70**. Terfi edilmedi.

**DERS (kritik):** İlk ölçümde "−1,52 puan" görünmüştü çünkü ml_v4_ges'in *tüm dönem*
skoru (%13,01) multi-NWP'nin *tam kapsam* skoru (%11,49) ile kıyaslanmıştı.
Kapının "aynı pencere" şartı tam bu tuzağı önledi. Kıyas her zaman aynı pencerede.

**Neden güneşte kazanç küçük, rüzgarda büyük:** Güneş zaten GES-ağırlıklı GHI +
bulut + clear-sky indeksi kullanıyor; tek NWP bu iş için yeterince iyi. Rüzgar ise
100m hızda model hatasına çok daha duyarlı — çoklu model spread'i orada gerçek bilgi
taşıyor (rüzgar temiz dönem: 17,78 → 14,67 = −3,11).

---

## OPERASYONEL KURALLAR (sertleştirme)

### Kill protokolü — log sessizliği ≠ ölü süreç
Python çıktısı tamponlanır. Bir süreci öldürmeden önce **üç kontrol zorunlu**:
1. Log dosyası mtime (yaş < 120 sn ise süreç canlıdır)
2. Çıktı dosyası mtime (sonuç dosyası güncelleniyor mu)
3. Süreç listesi (`Get-Process python`)

2026-11-05'te bu kural yokken `morning_ptf` gereksiz yere öldürüldü — iş aslında
16:04'te başarıyla bitmişti.

### Tüm uzun koşular `python -u`
Buffer'ı kapat; yoksa yukarıdaki yanlış teşhis tekrarlanır.

### Durum etiketleri tek kaynak
`solar_status` / `wind_status` üreten modülün kendi dosyasındaki `status`
kolonundan okunur. Hardcode yasak.

---

## KALICI KURALLAR (tekrar düşmemek için)

### Model karşılaştırma
1. **Her zaman AYNI PENCERE.** Yeni modelin "tam kapsam" skorunu eski modelin "tüm dönem"
   skoruyla kıyaslamak sahte kazanç üretir. Güneş multi-NWP'de −1,52 göründü, aynı
   pencerede −0,70 çıktı ve kapı geçmedi.
2. **Veri kapsaması dönem değiştiriyorsa yarım raporu zorunlu.** IFS 2024Q2'den önce yok.
3. **WAPE kapısı geçilmese de nMAE'ye bak** (güneş multi-NWP: WAPE −0,70, nMAE −6%).
4. **RİTM karışımını geriye dönük çekimle ölçme** — revize seri = sızıntı.

### Kill protokolü
Log sessizliği ≠ ölü süreç. Öldürmeden önce: log mtime (<120 sn), çıktı dosyası mtime,
süreç listesi. Uzun koşular `python -u`.

### Doğrulama
`scripts/verify_contract.py` (17 kapı). Denge 0,003906 MW = 2⁻⁸ = float32 tek ULP.

---

## ZAMANLANMIŞ GÖREV — kapsama kanıtı (2026-11-06 doğrulandı)

```
TaskName     : \TR_Morning_Yarim_Tahmin
Schedule Type: Daily          <-- HER GÜN, hafta sonu DAHİL
DaysInterval : 1
Next Run     : 6.11.2026 09:00
Last Run     : 5.11.2026 09:00:01   Last Result: 0 (başarılı)
```

**Hafta sonu açığı YOK.** Görev günlüktür, Cumartesi/Pazar da koşar.

**DÜZELTME:** Bir raporda "9 Kasım 09:00" yazılmıştı — HATALI. Sıradaki koşu her gün 09:00.

Sabah ürünü arşivi: `yarin_2026-11-03/04/05/06.parquet` (dördü de mevcut).

### Rapor rejimi standardı
- **Varsayılan: üretim rejimi (2025+)** — 3 NWP modeli tam.
- Tüm dönem skorları ek bilgi olarak, "tüm dönem" etiketiyle verilir.
- Her skora **pencere + gündüz/tüm-gün + n** yazılır. Etiketsiz sayı yasak.
- nMAE tek başına yazılmaz; kapsam (%) ile birlikte.

---

## FAZ 4 — RÜZGAR HATA YAPISI (2026-11-06)

Protokol: `wind_diag.py`, EVAL_START=2025-01-01, 3 model tam, **WAPE %14,67 birebir
yeniden üretildi** (protokol doğrulandı). Dump: `C:\projeler\wind_wf_preds.parquet`
(13.800 saat). Analiz: `wind_errstruct.py` — YENİ EĞİTİM YOK.

### Bulgular (MAE, n=13.800)

| Boyut | Sonuç |
|---|---|
| Genel | MAE **718 MW**, bias −31 MW |
| Gece / gündüz | 691 / 726 → **0,95x** (gündüz hafif kötü) |
| Saat bias | gece −125…−255 (eksik), öğle +133…+180 (fazla) |
| **Marmara** yüksek hız | 829 vs düşük 583 → **+246 MW** |
| **Ege** yüksek hız | 837 vs düşük 584 → **+253 MW** |
| Akdeniz | 716 vs 720 → **düz** |
| İç Anadolu | 788 vs 643 → +145 MW |
| **NWP spread** düşük/orta/yüksek | 623 / 725 / **807** → **1,30x** |
| Cut-out (>25 m/s) | **sadece 3 saat** (%0,02), MAE 3.591; 3 gün (Şubat 2026) |
| Büyük rampa (>p90 1.282 MW) | 886 vs 681 → **1,30x** |

### Hüküm
718 MW'ın kaynağı **yüksek rüzgarlı Marmara + Ege saatleri**. Akdeniz katkı vermiyor.
Cut-out ihmal edilebilir (3 saat) — "cut-out şüphesi" reddedildi.

### 24 KASIM ÖN KAYDI — w ölçümü
Aynı pencerede üç seri: RİTM-sadece, ML-sadece, blend w-eğrisi (0,40–0,80, adım 0,05).
**Ek hipotez (bu tanıdan):** NWP spread yüksekken (üst üçte bir) ML hatası 1,30x
büyüyor. RİTM'in spread rejimine duyarlılığı farklıysa **rejim-bağımlı w** kazanabilir.
Rejim kırılımı tablosu ölçüme EKLENİR; karar yine aynı sağlamlık kapısından geçer.

### Yük denetimi — GEÇERSİZ, YENİDEN GEREKLİ
`load_audit.py` 2025+ tarihler için `D.forecast_weather()` çağırıyordu; bu fonksiyon
**CANLI API** (`OM_FC_URL`) kullanıyor → geçmiş tarih için yanlış. Ayrıca `nat` tek
seri, T+1/T+2 lead ayrımı yok (iyimserlik).
**Doğru yol:** `lead_weather/<il>.parquet` içindeki `_previous_day1` / `_previous_day2`
serileri — lead ayrımı açık, arşiv tahmini. Yük denetimi bununla yeniden yazılmalı.

### Sistem track record (altın standart)
`yarin_*.parquet` arşivi + `rt_cons` gerçekleşeni — hava sızıntısı sorusu yok.
Şu an **2 gün** değerlendirilebilir:

| Dosya | n | WAPE | bias |
|---|---|---|---|
| yarin_2026-11-03 | 24 | %10,26 | **−4.059 MW** |
| yarin_2026-11-04 | 24 | %1,14 | −369 MW |

3 Kasım üretimi arızalı (ort 35.518 vs ertesi gün 39.755) — bu çalışma öncesi koşu.
**Track record henüz yok**; 30 gün birikince anlamlı.

---

## FAZ 4 — YÜK OUT-OF-TIME (SIZINTISIZ) ✅

`scripts/load_audit2.py` — `lead_weather` `_previous_day1` (T+1) ve `_previous_day2`
(T+2) serileri. Hiçbir gün gerçekleşen hava kullanılmaz. 364 gün, aylık yeniden eğitim.

### SONUÇ

| | WAPE | nMAE | bias | n |
|---|---|---|---|---|
| **T+1** | **%1,69** | 689 MW | **−357 MW** | 8.736 |
| **T+2** | **%1,69** | 690 MW | −357 MW | 8.736 |
| 1. yarı | %1,64 | | | 4.368 |
| 2. yarı | %1,73 | | | 4.368 |

**Hedef T+1 < %1,35 → TUTMADI** (1,69). Yarımlar dengeli (1,64 / 1,73), yani ölçüm sağlam.

### Kritik düzeltme: önceki skorlar iyimserdi
Canlı raporlarda geçen **%1,585 / %1,51** değerleri in-sample'dı. Sızıntısız
out-of-time gerçek **%1,69**. Fark +0,10-0,18 puan. Bu, projenin en büyük
"gerçek skor" düzeltmesidir.

### Hata yapısı
- **Saat:** gece %1,00-1,15 (iyi) → öğle **%2,08-2,23** (kötü). Ondalık fark 2x.
- **Bias saatleri:** 08-09'da **−634…−649 MW**, öğlen −500'ler, gece −100'ler.
  Yani model gündüz sistematik olarak **düşük** tahmin ediyor.
- Gece neredeyse kusursuz; sorun tamamen gündüz bloğunda.

### Neden önemli
Yük, hata bütçesinde rüzgardan sonra ikinci (~560-690 MW). Gündüz bias'ı
−650 MW'a ulaşıyor; bu, rezidüel yükün öğle çukurunu doğrudan bozuyor
(PTF öğle fiyatına duyarlı).

---

## FAZ 4 — GÜNEŞ TANILAMASI (mimari dokunuş YOK)

`scripts/solar_diag.py`, EVAL_START=2025-01-01, multi-NWP protokolü.

### T+1 / T+2 ayrı (gündüz)
| | Gündüz WAPE | nMAE | n |
|---|---|---|---|
| T+1 | %10,69 | 109 MW | 13.776 |
| T+2 | %10,75 | 109 MW | 13.800 |
| T+1 1./2. yarı | 10,19 / 11,13 | | |

### Saat bazlı (T+1, gündüz)
| Saat | WAPE | bias |
|---|---|---|
| 06-09 | 10,34-11,17 | +3…+10 / −10 |
| **12-14** | **9,63-9,75** | −8…−43 |
| 15-16 | 10,81-11,31 | −19 / +6 |
| 17 | 13,28 | +45 |
| **18** | **17,29** | +48 |
| **19** | **33,11** | +24 |

### Hükümler
- **Curtailment şüphesi REDDEDİLDİ.** Öğle bias'ı yalnızca **−8…−43 MW** (~2.000 MW
  tepeye göre %1-2). Sistematik negatif bias YOK.
- En iyi saatler **12-14** (%9,63-9,75) — tam öğle.
- Hata **gün batımında** patlıyor (18: %17,29, 19: %33,11) ama bias küçük (+48, +24):
  **saçılma**, sistematik değil. O saatlerde üretim düşük olduğu için WAPE doğal olarak şişiyor.
- Yani güneşte düzeltilecek sistematik bir kusur yok; kalan hata büyük ölçüde
  ölçümün doğası.

---

## SIZINTI #3 — `known_cutoff` HEDEFTEN TURETILIYORDU (2026-11-06)

### Mekanizma
`known_cutoff(target_idx)` cutoff'u HEDEF tarihten turetiriyordu:
`target - (h + 16)h`. T+1 icin D 08:00 cikar (dogru), **T+2 icin D+1 08:00**
cikar — yani T+2 tum feature'lar bir gun fazla bilgiyle hesaplaniyordu.
`ALWAYS_LAGS` (48+) kosulsuz uygulandigi icin T+2'de lag48 = D gunu = cutoff
sonrasi. Sonuc: **T+2 fiilen T+1 problemi olarak olculuyordu** — T+1=T+2
esitliginin sebebi buydu.

### Fix (3 tasarim sarti)
1. `known_cutoff(target_idx, decision_date)` — cutoff **karar gunu 08:00**
   (`LAST_KNOWN_HOUR = DECISION_HOUR - 1`), hedeften BAGIMSIZ. Magic 16/40 yok.
2. `_lag_known(L, h, horizons)` — esik = `horizons*24 + h - LAST_KNOWN_HOUR`
   (T+1 icin h+16'ya esit, geriye uyum). `ALWAYS_LAGS` de artik maskeli.
3. `build_features(..., decision_date=)` — cagiranlar acikca gecer.

### Regresyon testi (ZORUNLU) — `scripts/feat_check.py`
12 gun x 24 saat, N1/N2 arsiv havasi:

| | ESKI hash | YENI hash | NaN | Sonuc |
|---|---|---|---|---|
| **T+1** | `b582edaf5752694a` | `b582edaf5752694a` | 180→180 | **BIT-IDENTICAL** |
| T+2 | `0d801921c8ea236c` | `58e1b9f00e1a3eac` | 180→468 | leak kaldirildi |

NaN artisi tam 288 = lag24 (108) + lag48 (180) maskelendi. Beklenen deger.
**T+1 %1,69 AYNEN GECERLI — yeniden kosmaya gerek yok.**

### Cagri yerleri guncellendi
`scripts/morning_ptf.py:112` ve `scripts/load_audit2.py:92` → `decision_date=`.
`morning_run.py` T+1-only oldugu icin varsayilan (target-1 gun) dogru kalir.

### Etki yaricapi
- **Raporlanmis HER T+2 yuk sayisi gecersiz.** Yeni baseline asagida.
- Uretimde sizinti YOKTU (canlida `y` D+1'e uzanmaz → NaN), ama **NaN'li olu
  feature** vardi. Fix ile T+2 satirlari canli feature'la uretilecek → kalite
  muhtemelen ARTTI. Iki cumle ayni anda dogru.
- Egitim hala T+1 semantiginde (base-16). Bilinen uyumsuzluk, retrain YOK;
  horizon-farkinda egitim **Faz 5b adayi**.

### SINIF TARAMASI (ayni turda, zorunlu)
| Builder | Hedefden tureyen cutoff | Shiftsiz rolling | Ufuk-bagimsiz lag | Durum |
|---|---|---|---|---|
| load `src/morning.py` | vardi → **DUZELTILDI** | yok | vardi → **DUZELTILDI** | ✅ temiz |
| solar `solar_mnwp.py` | yok | tumu `.shift(1)` | uretim lag'i yok | ✅ **temiz** |
| wind `wind_mnwp.py` | yok | `.shift(1)` | `cf_lag24/48/168` + `cf_7d(k=1..7)` | ❌ **KIRLI** |

**Ruzgar da ayni sinifta:** `cf_lag24` T+2'de D+1 saatine denk gelir; `cf_7d`
icindeki k=1 terimi de oyle. Ruzgar T+2 sayilari da supheli — **Faz 5b'de
ayni maskeyle duzeltilmeli** (bu turda dokunulmadi, load fix'i muhurlensin once).

### SIZINTI #3 — YENI BASELINE (2026-11-06, `la4.log`)

| | WAPE | nMAE | bias | n | Not |
|---|---|---|---|---|---|
| **T+1** | **%1,67** | 684 MW | −349 MW | 8.736 | gecerli (feature bit-identical) |
| **T+2** | **%4,88** | 1.995 MW | −1.832 MW | 8.736 | **gercek T+2** (onceki 1,69 sahte) |
| T+1 1./2. yari | 1,63 / 1,71 | | | | dengeli |

T+1'de 1,69 → 1,67 farki **MLP tohum gurultusu** (ensemble'da mlp2 agirligi 0,70,
egitim stokastik). Feature hash'i bit-identical oldugu icin fix'ten gelmiyor.

### T+2 neden %4,88
Model hala T+1 semantigiyle egitiliyor (lag48/72/168 dolu). T+2 serve'te lag24
tamamen, lag48 kismen NaN → **egitim/serve kaymasi**. Bu kayma ÜRETIMDE DE VARDI
(canlida `y` D+1'e uzanmadigi icin ayni NaN'lar olusuyordu) — yani kontratin T+2
yuk satirlari aylardir bu rejimde uretiliyordu, sadece hic olculmemisti.

**Duzeltme yolu:** horizon-farkinda egitim (T+2 icin lag48+ / T+1 icin lag24+).
Faz 5b adayi. Bu turda retrain YOK (kullanici sarti).

### Ders
Iki turlu hata yapildi ve ikisi de testle yakalandi:
1. `decision_date=None` egitim yolunda tum 730 gune tek cutoff -> tum lag'ler NaN
   -> WAPE %489. Duzeltildi: egitim yolu eski per-satir anchor'u AYNEN koruyor.
2. Fix'in T+1'i etkilemedigi hash testiyle kanitlandi (b582edaf... degismedi).

### SINIF TARAMASI v2 — DUZELTME (2026-11-06)

Onceki "solar TEMIZ" hukmum **YANLISTI**. Ilk tarama `cf_lag` desenini solar'da
kacirdi (farkli yazim: `cf.get(t - timedelta)`). Dosya:satir tam tablo:

| Dosya:satir | Builder | Hedef seriden lag | Kirli ufuk |
|---|---|---|---|
| `src/morning.py:96` | load | lag24..lag336 | DUZELTILDI |
| `wind_mnwp.py:121-122` | wind | `cf_lag24/48/168` | **T+2** (24 de kirli) |
| `wind_mnwp.py:123` | wind | `cf_7d` k=1..7 (min 24h) | **T+2** |
| `wind100.py:91-92` | wind | `cf_lag24/48/168` | **T+2** |
| `wind_live.py:122-125` | wind | train lags | **T+2 serve** |
| `wind_live.py:152-155` | wind | predict lags | **T+2 serve** |
| `wind_live_v5.py:108-109` | wind | train lags | **T+2 serve** |
| `wind_live_v5.py:129-130` | wind | predict lags | **T+2 serve** |
| `solar_mnwp.py:231-232` | solar | `cf_lag48/72/168` | **T+2** (h>8 kismi) |
| `solar_mnwp.py:239-240` | solar | `cf_7d` k=1 (min 24h) | **T+2** |
| `solar_t6.py:163` | solar | `cf_lag*` | ayni |
| `solar_live.py:124-125` | solar | `cf_lag*`, `cf_7d` | **T+2 serve** |

**T+1 durumu:** `cf_lag48` T+1'de = D-1 -> guvenli. Ama `cf_7d`/`cf_lag24`
k=1 terimi T+1'de t-24 = D gunu → h>8 icin **kismi sizinti**.
Yani T+1 de tam temiz degil; etkisi olculmeli.

**Gunes notu:** `ghi_7d_same_hour` HAVA serisidir (hedef degil) — 09:00'te tum
ufuk icin bilinir, sizinti DEGIL. Yalnizca `cf_*` (uretim) lag'leri kirli.

**Ruzgar uretim T+2 skoru (`ml_v5_multinwp`) gercek degeri BILINMIYOR** — ayni
train/serve kaymasiyla calisiyor.

### KM 1 — RUZGAR MASKE SONRASI GERCEK SKOR (2026-11-06)

`wind_mnwp.py` ufuk-duyarli maske (`HZ` env): `cf_lag24/48/168` ve `cf_7d`
(k=1..7) artik `L >= HZ*24 + h - 8` ile maskeleniyor.

| | WAPE | nMAE | bias | n |
|---|---|---|---|---|
| Maskesiz (eski, tek sayi) | %15,41 | 705 MW | −18 | 21.792 |
| **T+1 (HZ=1)** | **%15,53** | 711 MW | −22 | 21.792 |
| **T+2 (HZ=2)** | **%15,63** | 716 MW | −23 | 21.792 |

**Sizinti bedeli yalnizca 0,12 / 0,22 puan.** Sebep: ruzgar hava-baskin bir seri,
`cf` lag'leri az tasiyor. Yukteki %1,67 → %4,88 cokusu ruzgarda YOK.

**Hala acik (bilinen, bu turda cozulmedi):** NWP arsivi o saat icin "en iyi
mevcut tahmin" — gercek gun-oncesi vintage degil. Yani her iki ufuk da
lead-time acisindan iyimser. Bu, lag sizintisindan AYRI bir konu.

### KM 1 — GERCEK SKOR TABLOSU v1.0 (2026-11-06)

Tum builder'larda ufuk-duyarli lag maskesi uygulandi. Uc seri, T+1/T+2 AYRI,
ayni protokoller.

| Seri | T+1 | T+2 | Metrik |
|---|---|---|---|
| **Yuk** | **%1,67** | **%4,88** | WAPE (tum saatler) |
|  | 684 MW | 1.995 MW | nMAE |
| **Ruzgar** | **%15,53** | **%15,63** | WAPE |
|  | 711 MW | 716 MW | nMAE |
| **Gunes (gunduz)** | **%10,76** | **%10,86** | WAPE |
|  | 109 MW | 110 MW | nMAE |
| Gunes (tum gun) | %11,33 | %11,43 | WAPE |

### Sizinti bedeli — seri bazinda
| Seri | T+1 | T+2 |
|---|---|---|
| Yuk | +0,00 | **+3,19 puan** |
| Ruzgar | +0,12 | +0,22 |
| Gunes | +0,07 | +0,11 |

**Desen net: sizinti yalniz YUKTE yikici.** Ruzgar ve gunes hava-baskin;
`cf` lag'leri az tasiyor, bu yuzden maskenin bedeli ~0,1 puan.

### Guneste maske ayrintisi
Iki turda uygulandi (ilk turda `cf_lag` blogu CRLF nedeniyle atlanmisti):
- yalniz `cf_same_hour_7d` maskeli: T+1 %10,76 / T+2 %10,82
- `cf_lag48/72/168` de maskeli: T+1 %10,76 / T+2 %10,86
`ghi_7d_same_hour` DOKUNULMADI (hava serisi, sizinti degil).

### Kalan bilinen iyimserlik (cozulmedi)
NWP arsivi gun-oncesi vintage degil ("en iyi mevcut tahmin") — ruzgar ve guneste
her iki ufuk da lead-time acisindan iyimser. Lag sizintisindan AYRI konu.

### KM 2 — T+2 YUK RETRAIN: KAPILAR GECTI ✅ (2026-11-06)

`scripts/load_t2_retrain.py` — per-horizon anchored egitim: her egitim satiri
kendi karar anindan (`hedef - 2 gun`, 08:00) insa edilir. Boylece lag72+
guvenli set kullanilir ve egitim/serve rejimi ayni olur.

| | WAPE | nMAE | bias | n |
|---|---|---|---|---|
| Eski T+2 (T+1 semantigi) | %4,88 | 1.995 MW | −1.832 MW | 8.736 |
| **Yeni T+2 (per-horizon)** | **%2,03** | **829 MW** | **−493 MW** | 8.736 |
| T+1 (dokunulmadi) | %1,67 | 684 MW | −349 MW | 8.736 |

**Kapi sonuclari:**
| Kapi | Esik | Sonuc |
|---|---|---|
| 4,88'den anlamli iyilesme | — | **−2,85 puan** ✅ |
| Iki yari dengeli | ayni yonde | 1,94 / 2,11 ✅ |
| bias | < −1.000 MW | −493 MW ✅ |

Beklenti 3-4 bandiydi; 2,03 geldi cunku per-horizon egitim hava sinyalini de
dogru kullaniyor (T+2'de tasiyici sinyal hava + gunluk ozetler).

Saatlik: en kotu 08-09 (%2,63-2,67, bias −815/−824), en iyi gece 02-04 (%1,35-1,36).
Gece-gunduz farki T+1'dekinden kucuk — lag kaybi geceyi daha cok vuruyor.

### Kontrat baglantisi
`morning_ptf.py`: `tr_dec = None if off == 1 else (tr_idx.normalize() - off gun)`.
T+1 yolu DEGISMEDI (decision_date=None → eski per-satir anchor). T+2 yolu artik
per-horizon egitimli model kullaniyor.
