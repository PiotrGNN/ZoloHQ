<#
Creates fresh .venv on Python 3.11, installs runtime + dev deps.
If .venv exists – it is removed first.
Run from repo root:  ./scripts/bootstrap.ps1   [-Python "C:\Path\python.exe"]
#>

param(
  [string]$Python = ""
)

# --- Locate Python 3.11 ----------------------------------------------------
if (-not $Python) {
    try       { $Python = (& py -3.11 -c "import sys,os;print(sys.executable)") }
    catch     { $Python = "" }
}
if (-not (Test-Path $Python)) {
    Write-Host "`n⚠  Python 3.11 not auto-detected." -ForegroundColor Yellow
    $Python = Read-Host "➡  Provide full path to python.exe 3.11"
}
if (-not (Test-Path $Python)) {
    Write-Error "❌  Given path '$Python' is invalid. Aborting."; exit 1
}

Write-Host "⏳  Resetting virtual environment..." -ForegroundColor Yellow
if (Test-Path ".venv") {
    Remove-Item ".venv" -Recurse -Force
    Write-Host "🗑️   Old .venv removed."
}

$venvResult = & $Python -m venv .venv
if ($LASTEXITCODE -ne 0) {
    Write-Error "❌  Failed to create venv"; exit 1
}

# Ensure pip is installed in the new venv
$ensurePip = & $Python -m ensurepip
if ($LASTEXITCODE -ne 0) {
    Write-Error "❌  Failed to ensure pip in venv"; exit 1
}

# Activate venv for this session
$activateScript = ".\.venv\Scripts\Activate.ps1"
if (Test-Path $activateScript) {
    & $activateScript
} else {
    Write-Error "❌  Could not find venv activation script."; exit 1
}

Write-Host "📦  Installing runtime deps..." -ForegroundColor Cyan
# Upgrade pip using python -m pip to avoid self-modification error
& $Python -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) {
    Write-Error "❌  pip upgrade failed"; exit 1
}

pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) {
    Write-Error "❌  Runtime install failed"; exit 1
}

Write-Host "🔧  Installing dev deps..." -ForegroundColor Cyan
pip install -r requirements-dev.txt
if ($LASTEXITCODE -ne 0) {
    Write-Error "❌  Dev install failed"; exit 1
}

Write-Host "`n✅  ZoL0 dev env ready on Python 3.11" -ForegroundColor Green
