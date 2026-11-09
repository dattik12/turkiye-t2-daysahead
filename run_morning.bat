@echo off
rem Gunluk sabah sozlesmesi kosucusu (SYSTEM gorevi bunu cagirir).
rem Ciktiyi tarihli log dosyasina yazar; basarisizlikta exit code korunur.
setlocal enabledelayedexpansion
cd /d C:\projeler\turkiye-t2-daysahead
if not exist data\logs mkdir data\logs
set STAMP=
for /f %%i in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmm"') do set STAMP=%%i
if "%STAMP%"=="" set STAMP=unknown
set LOG=data\logs\morning_%STAMP%.log

set PY=C:\Users\cagatay.bayrak\AppData\Local\Programs\Python\Python311\python.exe

rem Gunluk NWP cekimi (kota butceli, il basina tek istek). Bloklamaz:
rem basarisiz olursa arsivdeki veriyle devam edilir.
"%PY%" -u -m scripts.pull_daily >> "%LOG%" 2>&1

"%PY%" -u scripts\morning_run.py >> "%LOG%" 2>&1
set RC=%ERRORLEVEL%

rem Skor tahtasi: tahmin basariliysa defteri gerceklesen + TEIAS ile guncelle.
rem Bloklamaz -> RC'yi degistirmez.
if "%RC%"=="0" (
  "%PY%" -u -m scripts.morning_scoreboard >> "%LOG%" 2>&1
  rem 48 satirlik PTF kontrati (T+1 + T+2) — sabah modelinden, bloklamaz.
  rem Ruzgar ve gunes ML ciktilarini uret (morning_ptf bunlari okur). Bloklamaz.
  "%PY%" -u scripts\wind_live_v5.py >> "%LOG%" 2>&1
  "%PY%" -u scripts\solar_live.py >> "%LOG%" 2>&1
  "%PY%" -u -m scripts.morning_ptf >> "%LOG%" 2>&1
  set PTFRC=!ERRORLEVEL!
  if not "!PTFRC!"=="0" (
    echo [ptf] BASARISIZ rc=!PTFRC! >> "data\logs\LAST_FAILURE.txt"
    echo tarih=%STAMP% >> "data\logs\LAST_FAILURE.txt"
  )
  rem Gunluk kontrat arsivi + feature surum damgasi (PTF ablation izi; bloklamaz).
  "%PY%" -u -m scripts.archive_contract >> "%LOG%" 2>&1
  rem Sabah urununu GitHub'a tasi (ayri klon uzerinden; bloklamaz).
  call push_morning.bat >> "%LOG%" 2>&1
)

echo [wrapper] exit=%RC% log=%LOG% >> "%LOG%"
if not "%RC%"=="0" (
  echo [wrapper] BASARISIZ exit=%RC% >> "data\logs\LAST_FAILURE.txt"
  echo tarih=%STAMP% >> "data\logs\LAST_FAILURE.txt"
)
exit /b %RC%
