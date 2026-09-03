"""EPiAS GES gerceklesme cekimi — v1 best-effort (pipeline'i ASLA bloklamaz).

Durum: eptr2'de ulusal toplam saatlik GES endpoint'i YOK.
  - `rt-gen` santral bazlidir (pp_id gerekir): pp-list'ten gunes santrallerini
    toplayip bulk cekmek gerekir (palmali/yavas; backfill'de sertlestirilecek).
  - `ren-rt-gen` YEKDEM santral bazlidir (ulusal toplami vermez).

Bu modul dener; basaramazsa (creds yok / endpoint yok / hata) (None, sebep)
doner. Cagiran taraf solar'i NaN + status ile isaretler, tuketim hatti aynen akar.
"""
from __future__ import annotations
import pandas as pd


def pull_solar_actual(start: str, end: str) -> tuple[pd.DataFrame | None, str]:
    """[start,end] ulusal saatlik GES gerceklesme denemesi.

    Donus: (df | None, sebep). df varsa kolonu `solar_actual_mw`, index naive TR saati.
    """
    try:
        from eptr2 import EPTR2
        eptr = EPTR2(use_dotenv=True, recycle_tgt=True, dotenv_path=".env", tgt_path=".")
        pp = eptr.call("pp-list")
    except Exception as ex:
        return None, f"eptr2/pp-list erisilemedi: {str(ex)[:100]}"
    try:
        names = " ".join(str(c) for c in pp.columns)
        cand = [c for c in pp.columns if any(k in str(c).lower() for k in ("fuel", "yakit", "type", "tip"))]
        if not cand:
            return None, f"pp-list yakit kolonu yok (kolonlar: {names[:120]})"
        fcol = cand[0]
        solar_pp = pp[pp[fcol].astype(str).str.contains("güneş|gunes|solar", case=False, na=False)]
        if solar_pp.empty:
            return None, "pp-list'te gunes santrali bulunamadi"
        idcol = next((c for c in pp.columns if "id" in str(c).lower()), pp.columns[0])
        ids = solar_pp[idcol].tolist()
        frames = []
        for pid in ids:
            try:
                df = eptr.call("rt-gen", start_date=start, end_date=end, pp_id=pid)
                frames.append(df)
            except Exception:
                continue
        if not frames:
            return None, f"{len(ids)} gunes santralinden rt-gen alinamadi"
        return None, f"v1 iskelet: {len(ids)} santral listelendi, toplama backfill'de (ham cekim sertlestirilmedi)"
    except Exception as ex:
        return None, f"beklenmedik hata: {str(ex)[:100]}"
