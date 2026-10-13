@echo off
rem ============================================================
rem  LiveTalking one-click launcher
rem  Default: Mode C (MiMo-V2.5-TTS, voice "Bingtang")
rem  Double-click this file to start and auto-open the web page.
rem
rem  Switch TTS mode:   start.bat --mode b   (b=edge-tts, c=MiMo is default)
rem  Switch avatar:     start.bat --avatar wav2lip_avatar_long_hair_girl
rem  Other options:     see tools\start_livetalking.py
rem ============================================================
cd /d %~dp0
chcp 65001 >nul
if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] .venv not found. Please finish the setup in
    echo         docs\full-local-deployment-guide.md first.
    pause
    exit /b 1
)
".venv\Scripts\python.exe" "tools\start_livetalking.py" %*
if errorlevel 1 pause
