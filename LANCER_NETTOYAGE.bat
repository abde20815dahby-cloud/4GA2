@echo off
chcp 65001 > nul
echo ======================================================================
echo    DAHBY PRO SUITE - MOTEUR UNIFIE DE NETTOYAGE DXF (FIX PROBLEM)
echo    Normes ANCFCC 100%% - Zéro Parasite, Zéro Déchet, Zéro Collision
echo ======================================================================
echo.

if "%~1"=="" (
    echo [USAGE] Glissez-déposez un fichier DXF ou un dossier sur ce fichier .bat,
    echo         ou lancez en ligne de commande :
    echo.
    echo         LANCER_NETTOYAGE.bat "chemin\vers\fichier.dxf"
    echo.
    echo Options :
    echo   1. Nettoyer un seul fichier DXF
    echo   2. Nettoyer tout un dossier de fichiers DXF
    echo.
    set /p choice="Choisissez une option (1 ou 2) : "
    if "%choice%"=="1" (
        set /p dxf_in="Entrez le chemin du fichier DXF : "
        python "%~dp0CLEAN_DXF_MASTER.py" --input "%dxf_in%"
    )
    if "%choice%"=="2" (
        set /p dir_in="Entrez le chemin du dossier contenant les DXF : "
        python "%~dp0CLEAN_DXF_MASTER.py" --batch-dir "%dir_in%"
    )
) else (
    if exist "%~1\*" (
        echo Nettoyage du dossier : %~1
        python "%~dp0CLEAN_DXF_MASTER.py" --batch-dir "%~1"
    ) else (
        echo Nettoyage du fichier : %~1
        python "%~dp0CLEAN_DXF_MASTER.py" --input "%~1"
    )
)

echo.
echo ======================================================================
echo    Nettoyage termine avec succes !
echo ======================================================================
pause
