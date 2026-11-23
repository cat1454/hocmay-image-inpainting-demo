@echo off
chcp 65001 >nul
echo ====================================================================
echo   KHOI DONG WEB DEMO IMAGE INPAINTING (DILATED U-NET vs DEEPFILLV1 vs LAMA FFC)
echo   Mo trinh duyet tai: http://127.0.0.1:8000
echo ====================================================================
start "" "http://127.0.0.1:8000"
python "%~dp0web_demo\app.py"
pause
