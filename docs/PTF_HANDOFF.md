# PTF DEVİR — Aşağı Akım Sözleşmesi

Bu dosya, fiziksel denge katmanının çıktısını **fiyat modelleme** tarafına devretmek için yazıldı.
Yukarı akım (yük / güneş / rüzgar / hava) bu repoda; aşağı akım (GÖP fiyat) bu repoda **yoktur**.

Son güncelleme: **2026-11-04**

---

## 1. Ne alacaksın

Tek dosya, iki format:

```
data/forecast/ptf_features/latest.parquet
data/forecast/ptf_features/latest.csv
```

Arşiv: `data/forecast/ptf_features/archive/ptf_features_<tarih>_<sabah|aksam>.parquet`

**48 satır** = iki gün × 24 saat. `horizon` kolonu hangi gün olduğunu söyler.

---

## 2. Kolon sözleşmesi

| Kolon | Tip | Anlam |
|---|---|---|
| `datetime` | datetime (+03:00) | Hedef saat, TR saati |
| `horizon` | metin | `T+1` = karar günü, `T+2` = yarın |
| `consumption_pred_mw` | float32 | Yük tahmini (MW) |
| `solar_pred_mw` | float32 | GES üretimi (MW) — **bkz. §5 uyarı** |
| `wind_pred_mw` | float32 | RES üretimi (MW) |
| `renewable_generation_mw` | float32 | güneş + rüzgar |
| `residual_load_mw` | float32 | **yük − güneş − rüzgar** ← fiyatın ana girdisi |
| `renewable_penetration` | float32 | yenilenebilir / yük (0–1) |
| `residual_ramp_1h` | float32 | rezidüel yükün saatlik değişimi (MW/sa) |
| `solar_ramp_1h` | float32 | GES saatlik değişimi (MW/sa) |
| `is_peak_hour` | bool | 08:00 ≤ saat < 20:00 |
| `solar_status` | kategori | `ok` / `unconfigured` |
| `wind_status` | kategori | `blend` / `ritm` / `fallback` |

**Kimlik kontrolü:** `consumption − solar − wind − residual_load == 0`. Sağlamıyorsa dosya bozuk.

---

## 3. Güncelleme ritmi

| Koşu | Zaman (TR) | Ne yapar |
|---|---|---|
| Sabah | **09:00** | Yarının 24 saati (Windows görev `TR_Morning_Yarim_Tahmin`) |
| Akşam | **17:00** UTC 14:00 | PTF özellikleri yeniden üretilir |
| CI | 06:30 UTC | GitHub Actions yedek hat |

Her koşuda `latest.parquet` üzerine yazılır, eski hâli `archive/` altına kopyalanır.
**Hedef tarih** ile adlandırılır; `T+2` etiketi kullanılmıyor (kafa karıştırıyor).

---

## 4. Ölçülmüş kalite (720 gün, sızıntısız walk-forward)

| Eksen | Bizim | TEİAŞ / RİTM | Not |
|---|---|---|---|
| Yük (tüm günler) | **1,51%** | 2,86% | MAPE; bias düzeltmesi dahil |
| Yük (normal gün) | ~1,45% | ~2,75% | |
| Yük (tatil) | ~3,52% | ~4,67% | |
| Rüzgar | ~13% | ~13% | berabere; düşük rüzgarda RİTM önde |
| GES | **doğrulanamadı** | — | bkz. §5 |

Ölçüm protokolü ve betikler: `scripts/audit_experts.py`, `scripts/audit_bias.py`.

---

## 5. BİLİNEN KUSURLAR — bunları bilerek kullan

Bunlar gizlenmiyor; fiyat modelinde buna göre ağırlık ver.

1. **GES seviyesi doğrulanamadı.** Elimizdeki gerçekleşme yalnızca lisanslı santraller
   (kurulu gücün ~%15'i). Ulusal GES toplamı için bağımsız ölçüm yok. `SOLAR_PR=0.921`
   sabiti hiç fit edilmedi.
2. **GES şekli öğlen fazla, sabah az.** Lisanslı filoya göre 07:00'de %58 düşük,
   12:00'de %14 yüksek. Düzeltme tablosu hazır ama **uygulanmadı** — tablo doğudaki
   santrallerden fit edildiği için coğrafi kayma taşıyor (Erzurum 41°D vs İstanbul 29°D).
3. **Rüzgar düşük üretimde zayıf.** 0–1000 MW bandında RİTM bizden ~10 pp iyi;
   9000 MW üzerinde biz öndeyiz. Karışım ağırlığı bu gerçeğe göre ayarlanmadı.
4. **Tatil günleri yapısal olarak kötü.** Ağırlıklandırma çare olmadı; tatil MAPE
   normalin ~2,4 katı.
5. **Hava girdisi analiz kalitesinde değil.** Backtest analiz havası, canlı tahmin havası
   kullanır. Ölçülen ceza +0,04 pp.

---

## 6. Sınır — neye dokunma

- `src/`, `scripts/`, `models/` yukarı akım. Fiyat modeli bunları **değiştirmemeli**.
- Fiyat modelini bu repoya koyma; ayrı repo. Bu repo yalnız fiziksel denge üretir.
- `data/results/*.csv` skor defterleri. Elle düzenleme yok.
- `.env` repoda **yok** ve olmayacak. EPİAŞ kimliği RDP'de.

---

## 7. Doğrulama

```bash
git clone https://github.com/dattik12/turkiye-t2-daysahead.git
cd turkiye-t2-daysahead
pip install -r requirements.txt
python -m scripts.verify_all      # 19 kontrol, hepsi geçmeli
python -m scripts.check_ptf_latest --max-age-days 2
```

`verify_all` şunları kontrol eder: dosya tazeliği, kolon şeması, kimlik denklemi,
sızıntı testi, model bütünlüğü, skor defterleri.

---

## 8. İletişim

Soru / hata: **Çağatay Bayrak** — cagatay.bayrak@mrc-tr.com

Bir şey ters görünüyorsa önce `docs/SYSTEM_STATE.md` oku: kapanan hipotezler,
reddedilen yöntemler ve gerekçeleri orada.
