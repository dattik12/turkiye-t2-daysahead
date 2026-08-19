# ROADMAP — tr-t2-daysahead

Modeli adim adim gelistirme yolu (kanit-odakli; her madde backtest ile dogrulanir).

## Durum: pipeline CANLI (v1)
- [x] EPIAS gunluk cekim @03:00 TR (rt-cons + load-plan) + veri seti guncelleme
- [x] OpenMeteo ECMWF IFS hava (tarihi + hedef forecast), 10 sehir nufus agirlikli
- [x] T+1/T+2 ayri LightGBM + ensemble (BASE v4.3 + CONT v5.2)
- [x] D+2'de lag24'suz, leak-free, recursive-yok
- [x] Gunluk MAPE logu + TEIAS load-plan kiyasi
- [x] Simule backtest (2025→ bugün) CSV'i
- [x] GitHub Actions cron + otomatik commit

## Kisa vade (mevcut baseline'i saglamlastir)
- [ ] Bayram mini-model (Kurban/Ramazan günleri icin ayri rejim; bashlangicta %4-7 bandi)
- [ ] Gunluk yeniden egitim (Actions'ta su an haftaliktir; gunluk hale getir — ~-0.1pp)
- [ ] OpenMeteo Previous-Runs `previous_day2` lead-time hizali hava (T+2 hata dagilimini birebir ogretir)
- [ ] Bayram takvimini `turkiye_ozel_gun...` xlsx'ten oto-doldur (yillik guncelleme derdi biter)

## Orta vade (multimodel + regularization)
- [ ] **Multimodel ensemble sistemi**: LightGBM + XGBoost + HGBR + LSTM/GRU (PyTorch) + istatistiksel (SARIMA/HW)
      -> hafif meta-model (stacking / ağırlıklı ortalama) — ODTÜ 2026 meta-ogrenme referans
- [ ] **Hyperparameter tuning**: Optuna (TPE) veya CMA-ES (Karamollaoglu 2026 yaklasimi) per model
- [ ] **Regularization**: erken durdurma (vars), min_child_samples taramasi, bogusozluk
      (colsample/subsample), L1/L2, buyuk num_leaves cezasi; ozellik azaltma (SHAP tabanli)
- [ ] Egitim penceresi dinamik: rejim-degisim agirlikli eski veri (bantlanmis agirlik) deneyi
- [ ] Veri: uecm (uzlasilmis) ile rt-cons'un son T+10'da revizyonunu modellemek

## Uzun vade
- [ ] Olasiliksal tahmin (quantile regression / konformal) — belirsizlik bandi
- [ ] Bolgesel alt-modeller (OEDAS/ADM/GDZ -> butunlestirme) opsiyonu
- [ ] Canli dashboard / rapor (tahmin vs TEIAS vs gerceklesen grafigi)
- [ ] Entrypoint: TEIAS/EPIAS yuk planina karsi tarafsiz (ghost) skor tablosu

## Metrik tanimlari
- MAPE = mean(|pred-actual|/actual * 100) — saatlik, ufuk bazinda (T+1 / T+2 ayri)
- Bayram-disi MAPE: dini bayram pencereleri haric (gercek operasyonel kalite)
- Benchmark: TEIAS load-plan (hep D+1; D+2 icin resmi kaynak yok)
