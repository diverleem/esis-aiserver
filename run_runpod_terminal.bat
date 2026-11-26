@echo off
chcp 65001 > nul
title E.S.I.S Plan A: RunPod Terminal Monitor (Lab Riset Lt 5 - Mainstream Cam)
cls

echo ===============================================================================
echo   E.S.I.S (Electronic Smoking Identification System)
echo   PLAN A: RunPod Cloud Inference - Mode Terminal
echo ===============================================================================
echo.
echo  Target Kamera : Lab Riset Lantai 5
echo  Stream Source : Pinggy CLI MediaMTX (Channel 101 Mainstream Cam)
echo  API Server    : http://localhost:8001/
echo  Stream MJPEG  : http://localhost:8001/stream
echo.
echo  Pilihan Eksekusi:
echo   - Default (Pinggy Base dari .env) : Langsung berjalan
echo   - Custom Stream Link Pinggy CLI  : run_runpod_terminal.bat --stream-url ^<LINK_PINGGY_CAM^>
echo   - Mode Mock Webcam Simulasi      : run_runpod_terminal.bat --mock
echo.
echo  Tekan Ctrl+C di terminal untuk berhenti kapan saja.
echo ===============================================================================
echo.

cd /d "%~dp0"
python main_runpod.py --channel 101 --api-port 8001 --no-pinggy %*

pause
