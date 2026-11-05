# Создаёт новую сборку и архив обновления, не затрагивая установленные версии.
param([string]$BuildPython = "")
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
if (-not $BuildPython) { $BuildPython = Join-Path $ProjectRoot ".venv-build\Scripts\python.exe" }
if (-not (Test-Path $BuildPython -PathType Leaf)) {
    Write-Host "Создаю окружение сборки..."
    python -m venv (Join-Path $ProjectRoot ".venv-build")
    if ($LASTEXITCODE -ne 0) { throw "Не удалось создать окружение." }
}
& $BuildPython -m PyInstaller --version | Out-Null
if ($LASTEXITCODE -ne 0) {
    Write-Host "Устанавливаю зависимости сборки..."
    & $BuildPython -m pip install -r (Join-Path $ProjectRoot "requirements.txt") pyinstaller
    if ($LASTEXITCODE -ne 0) { throw "Не удалось установить зависимости." }
}
$ModelSource = Join-Path $ProjectRoot "models\anti_uav_fast_320_int8_pc-v2.v16m"
foreach ($required in @($ModelSource, (Join-Path $ProjectRoot "settings.json"), (Join-Path $ProjectRoot "logo.png"))) {
    if (-not (Test-Path $required -PathType Leaf)) { throw "Не найден необходимый файл: $required" }
}
$Version = Get-Date -Format "yyyyMMdd_HHmmss_ffff"
$Work = Join-Path $ProjectRoot "tmp\build-$Version"
$Dist = Join-Path $ProjectRoot "dist"
$Release = Join-Path $Dist "VideoT16-$Version"
$Engine = Join-Path $Release "versions\$Version"
$Zip = Join-Path $Dist "update-unsigned-$Version.zip"
if ((Test-Path $Release) -or (Test-Path $Zip)) { throw "Сборка уже существует; выберите другое время запуска." }
New-Item -ItemType Directory -Path $Work, $Release -ErrorAction Stop | Out-Null
Write-Host "Собираю версию $Version в отдельном каталоге..."
& $BuildPython -m PyInstaller --windowed --onedir --name Engine `
    --distpath (Join-Path $Work "engine") --workpath (Join-Path $Work "engine-work") `
    --specpath $Work --paths $ProjectRoot --paths (Join-Path $ProjectRoot "app") `
    --hidden-import cv2_enumerate_cameras (Join-Path $ProjectRoot "Run.py")
if ($LASTEXITCODE -ne 0) { throw "Не удалось собрать программу." }
New-Item -ItemType Directory -Path (Split-Path -Parent $Engine) -Force | Out-Null
Move-Item (Join-Path $Work "engine\Engine") $Engine
foreach ($item in @(@("launch.py", "Run"), @("update.py", "Update"))) {
    Write-Host "Собираю $($item[1]).exe..."
    & $BuildPython -m PyInstaller --windowed --onefile --name $item[1] `
        --distpath $Release --workpath (Join-Path $Work ($item[1] + "-work")) `
        --specpath $Work (Join-Path $ProjectRoot $item[0])
    if ($LASTEXITCODE -ne 0) { throw "Не удалось собрать $($item[1]).exe." }
}
[System.IO.File]::WriteAllText((Join-Path $Release "current.txt"), "$Version`n", [System.Text.Encoding]::ASCII)
Copy-Item (Join-Path $ProjectRoot "settings.json") $Release
Copy-Item (Join-Path $ProjectRoot "logo.png") $Release
New-Item -ItemType Directory -Path (Join-Path $Release "models") | Out-Null
Copy-Item $ModelSource (Join-Path $Release "models")
Write-Host "Создаю отдельный архив обновления..."
& $BuildPython (Join-Path $ProjectRoot "update.py") --pack $Engine $Zip $Version
if ($LASTEXITCODE -ne 0) { throw "Не удалось создать архив обновления." }
Write-Host "Готова первоначальная сборка: $Release"
Write-Host "Неподписанный архив обновления: $Zip; перед передачей подпишите его у владельца."
Write-Host "Создаю офлайн-установщик..."
& (Join-Path $ProjectRoot "build_installer.ps1") -ReleaseRoot $Release -BuildPython $BuildPython
if ($LASTEXITCODE -ne 0) { throw "Не удалось собрать установщик." }
