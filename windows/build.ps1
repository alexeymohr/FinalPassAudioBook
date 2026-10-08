#Requires -Version 5.1
<#
Builds the Windows one-click installer: fpab-gui.exe + fpab.exe (PyInstaller) and the
model weights, wrapped into FinalPassAudioBook-<version>-setup.exe (Inno Setup).

Run from a Windows checkout with uv installed:
    powershell -ExecutionPolicy Bypass -File windows\build.ps1

Needs git (uv fetches the pinned FinalPass with it) and Inno Setup 6.3 or later
(https://jrsoftware.org/isinfo.php). Everything else is fetched by uv.
#>
$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$root = Split-Path -Parent $here
Set-Location $root

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    throw "uv is required: https://docs.astral.sh/uv/getting-started/installation/"
}

Write-Host "==> Syncing the project environment (with the locked PyInstaller)"
uv sync --locked --extra windows-build

Write-Host "==> Fetching and verifying the model weights"
uv run --locked fpab setup-model

$version = (uv run --locked python -c "import finalpass_audiobook as m; print(m.__version__)").Trim()
Write-Host "==> Version $version"

Write-Host "==> Staging the model weights beside the app (the two verified files, nothing else)"
$staging = Join-Path $here "staging"
Remove-Item -Recurse -Force $staging -ErrorAction SilentlyContinue
$env:FPAB_STAGE = Join-Path $staging "models"
uv run --locked python -c @"
import os, shutil
from pathlib import Path
from finalpass_audiobook import breath_model, model
dest = Path(os.environ['FPAB_STAGE'])
for m in (model, breath_model):
    m.load()                                        # verifies size and SHA-256 before anything is copied
    src = m.weights_path()
    rel = src.relative_to(m.model_dir().parent.parent)   # keep the <model>/<revision>/<file> layout
    (dest / rel).parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dest / rel)
    print('staged', rel)
"@
if ($LASTEXITCODE -ne 0) { throw "staging the model weights failed" }

Write-Host "==> Building the executables (PyInstaller)"
Remove-Item -Recurse -Force (Join-Path $here "dist"), (Join-Path $here "build") -ErrorAction SilentlyContinue
uv run --locked pyinstaller (Join-Path $here "fpab-windows.spec") --noconfirm --clean `
    --distpath (Join-Path $here "dist") --workpath (Join-Path $here "build")

Write-Host "==> Building the installer (Inno Setup)"
$iscc = (Get-Command iscc -ErrorAction SilentlyContinue).Source
if (-not $iscc) {
    foreach ($p in @("${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe",
                     "$env:ProgramFiles\Inno Setup 6\ISCC.exe",
                     "$env:LOCALAPPDATA\Programs\Inno Setup 6\ISCC.exe")) {   # a per-user install
        if (Test-Path $p) { $iscc = $p; break }
    }
}
if (-not $iscc) { throw "Inno Setup 6 not found (https://jrsoftware.org/isinfo.php)" }
& $iscc "/DAppVersion=$version" (Join-Path $here "installer.iss")
if ($LASTEXITCODE -ne 0) { throw "Inno Setup failed" }

Write-Host ""
Write-Host "Installer: $(Join-Path $here "dist\FinalPassAudioBook-$version-setup.exe")"
