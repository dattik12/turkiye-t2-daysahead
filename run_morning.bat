@echo off
rem Gunluk sabah sozlesmesi kosucusu (SYSTEM gorevi bunu cagirir).
rem Ciktiyi tarihli log dosyasina yazar; basarisizlikta exit code korunur.
setlocal
cd /d C:\projeler\turkiye-t2-daysahead
if not exist data\logs mkdir data\logs
set STAMP=
for /f %%i in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmm"') do set STAMP=%%i
if "%STAMP%"=="" set STAMP=unknown
set LOG=data\logs\morning_%STAMP%.log

"C:\Users\cagatay.bayrak\AppData\Local\Programs\Python\Python311\python.exe" -u scripts\morning_run.py >> "%LOG%" 2>&1
set RC=%ERRORLEVEL%

echo [wrapper] exit=%RC% log=%LOG% >> "%LOG%"
if not "%RC%"=="0" (
  echo [wrapper] BASARISIZ exit=%RC% >> "data\logs\LAST_FAILURE.txt"
  echo tarih=%STAMP% >> "data\logs\LAST_FAILURE.txt"
)
exit /b %RC%
