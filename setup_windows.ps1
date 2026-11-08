# setup_windows.ps1 — إعداد وتشغيل ZILAL AI على Windows (VS Code / PowerShell).
# الاستخدام من مجلد المشروع:   powershell -ExecutionPolicy Bypass -File .\setup_windows.ps1
# يستخدم Python 3.12 (المختبَر). إن مُنع استيراد pandas بسياسة Application Control يجرّب بناءً آخر (pandas 2.2.3).
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not (Get-Command py -ErrorAction SilentlyContinue)) { throw "py launcher غير موجود. ثبّت Python 3.12 من python.org" }
py -3.12 --version
if ($LASTEXITCODE -ne 0) { throw "Python 3.12 غير مثبت. نفّذ: winget install Python.Python.3.12" }

if (Test-Path .venv) { Remove-Item -Recurse -Force .venv }
py -3.12 -m venv .venv
$py = Join-Path $PWD ".venv\Scripts\python.exe"
& $py -m pip install --upgrade pip
& $py -m pip install -r requirements.txt

& $py check_env.py
if ($LASTEXITCODE -ne 0) {
    Write-Host "`nتجربة بناء بديل لـ pandas (2.2.3)..." -ForegroundColor Yellow
    & $py -m pip install --force-reinstall --no-deps "pandas==2.2.3"
    & $py check_env.py
    if ($LASTEXITCODE -ne 0) {
        Write-Host "`nما زال الاستيراد ممنوعاً: السياسة تمنع الملف المترجم. اطلب من IT السماح بـ .venv\Lib\site-packages\pandas\_libs\*.pyd أو استخدم Colab/WSL." -ForegroundColor Red
        exit 1
    }
}
Write-Host "`nتشغيل ZILAL AI على http://localhost:8501" -ForegroundColor Green
& $py -m streamlit run app.py
