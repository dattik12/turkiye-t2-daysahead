"""Turkiye toplam GES (solar) tahmin modulu — v1 iskelet.

v1 kapsami (backfill sonraya birakildi):
  - Radyasyon hatti: OpenMeteo shortwave_radiation (tarihce + tahmin), mevcut
    10-sehir ulusal agirlik altyapisi yeniden kullanilir (bkz. radiation.py).
  - Guc modeli: lineer radyasyon->MW (pr + kurulu guc). Kalibrasyon katsayisi
    backfill asamasinda fit edilecek (bkz. model.fit_scale).
  - Gerceklesme: EPiAS'ta ulusal saatlik GES endpoint'i yok (rt-gen santral
    bazli); pull_actual best-effort dener, bulamazsa None doner — pipeline'i
    bloklamaz (bkz. pull_actual.py).
"""
from . import radiation, model, pull_actual

__all__ = ["radiation", "model", "pull_actual"]
