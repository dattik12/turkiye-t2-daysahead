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
| RİTM (referans) | ~%13 | — | |

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
| ml_v4_ges (tek NWP, GES ağırlıklı) | %13,01 | **%11,39** |
| multi-NWP (IFS+GFS+ICON) | %11,49 | **%10,69** |
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
