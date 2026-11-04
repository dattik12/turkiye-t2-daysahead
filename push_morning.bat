@echo off
rem Sabah urununu GitHub'a tasir. RDP'de uretim klasoru git degil; ayri klon
rem (C:\projeler\gh_push) uzerinden push edilir. BLOKLAMAZ: hata olsa da exit 0.
setlocal
set GIT="C:\Program Files\Git\cmd\git.exe"
set SRC=C:\projeler\turkiye-t2-daysahead\data\forecast\morning
set DST=C:\projeler\gh_push\data\forecast\morning
set GIT_SSH_COMMAND=ssh -o StrictHostKeyChecking=no -o IdentitiesOnly=yes -i C:/Users/cagatay.bayrak/.ssh/gh_deploy

if not exist "%DST%" mkdir "%DST%"
set STAMP=unknown
for /f %%i in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmm"') do set STAMP=%%i

cd /d C:\projeler\gh_push
if not exist .git (
  echo [push] HATA: gh_push klonu yok, atlandi
  exit /b 0
)

%GIT% pull --rebase --autostash origin main >nul 2>&1
copy /Y "%SRC%\*.parquet" "%DST%\" >nul 2>&1
%GIT% add -f data/forecast/morning >nul 2>&1
%GIT% diff --cached --quiet
if %ERRORLEVEL%==0 (
  echo [push] degisiklik yok
  exit /b 0
)
%GIT% commit -q -m "morning: %STAMP% tam denge (yuk+gunes+ruzgar)"
%GIT% push origin main >nul 2>&1
if %ERRORLEVEL%==0 (echo [push] OK %STAMP%) else (echo [push] BASARISIZ %STAMP%)
exit /b 0
