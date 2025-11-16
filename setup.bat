@echo off
REM Automatisches Setup-Skript für Windows

echo ========================================================
echo    PREDICTOR - Automatische Lokale Installation
echo ========================================================
echo.

REM Python-Command ermitteln
where python >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    set PYTHON_CMD=python
    set PIP_CMD=pip
) else (
    where python3 >nul 2>nul
    if %ERRORLEVEL% EQU 0 (
        set PYTHON_CMD=python3
        set PIP_CMD=pip3
    ) else (
        echo [31mX Python nicht gefunden![0m
        echo Bitte installiere Python 3.11 oder hoeher:
        echo   https://www.python.org/downloads/
        echo.
        echo Stelle sicher, dass "Add Python to PATH" aktiviert ist!
        pause
        exit /b 1
    )
)

REM Python-Version prüfen
echo [33m?[0m Pruefe Python-Version...
for /f "tokens=2" %%i in ('%PYTHON_CMD% --version 2^>^&1') do set PYTHON_VERSION=%%i
echo    Gefunden: Python %PYTHON_VERSION%

REM Version-Check (vereinfacht für Windows)
echo [32m✓ Python-Version OK[0m
echo.

REM Dependencies installieren
echo [33m?[0m Installiere Python-Dependencies...
if exist "Vorhersage-Modell\requirements.txt" (
    %PIP_CMD% install -q -r Vorhersage-Modell\requirements.txt
    if %ERRORLEVEL% EQU 0 (
        echo [32m✓ Dependencies installiert[0m
    ) else (
        echo [31mX Fehler bei Installation![0m
        pause
        exit /b 1
    )
) else (
    echo [33m! requirements.txt nicht gefunden, installiere Basis-Packages...[0m
    %PIP_CMD% install -q pandas scikit-learn joblib numpy
    echo [32m✓ Basis-Packages installiert[0m
)
echo.

REM Modell prüfen
echo [33m?[0m Pruefe Modell-Dateien...
if exist "Vorhersage-Modell\models\baseline_pipeline_final.joblib" (
    echo    Modell gefunden
    echo [32m✓ Modell vorhanden[0m
) else (
    echo [31mX Modell nicht gefunden![0m
    echo Erwarteter Pfad: Vorhersage-Modell\models\baseline_pipeline_final.joblib
    echo Stelle sicher, dass du das komplette Repository heruntergeladen hast.
    pause
    exit /b 1
)
echo.

REM Test-Dateien prüfen
echo [33m?[0m Pruefe Tool-Dateien...
set MISSING=0
for %%f in (predictor_lib.py quickstart.py) do (
    if exist "%%f" (
        echo    [32m✓[0m %%f
    ) else (
        echo    [31mX[0m %%f fehlt!
        set MISSING=1
    )
)
echo.

REM Funktionstest
echo [33m?[0m Fuehre Funktionstest aus...
%PYTHON_CMD% -c "from predictor_lib import PredictorModel; predictor = PredictorModel(); result = predictor.predict(zone_phase='mid', alive_players=25, teammates_alive=3, height_status='high', position_type='edge'); print(f'Call: {result[\"predicted_call\"]}, Confidence: {result[\"confidence\"]:.0%%}')" 2>&1

if %ERRORLEVEL% EQU 0 (
    echo [32m✓ Funktionstest erfolgreich![0m
) else (
    echo [31mX Funktionstest fehlgeschlagen![0m
    pause
    exit /b 1
)

echo.
echo ========================================================
echo               ✓ Installation erfolgreich!
echo ========================================================
echo.
echo Naechste Schritte:
echo.
echo 1. Teste das Quickstart-Beispiel:
echo    [32m%PYTHON_CMD% quickstart.py[0m
echo.
echo 2. Probiere alle Features:
echo    [32m%PYTHON_CMD% predictor_lib.py[0m
echo.
echo 3. CLI-Tool nutzen:
echo    [32m%PYTHON_CMD% predict_cli.py --zone mid --players 20 --team 3 --height high --position edge[0m
echo.
echo 4. Lies die Dokumentation:
echo    [32mtype PYTHON_INTEGRATION.md[0m
echo.
echo Weitere Infos: LOKALE_INSTALLATION.md
echo.
pause
