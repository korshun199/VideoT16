# Собирает только обновление программы в новый каталог, не затрагивая установки.
param([string]$BuildPython = "", [string]$Version = "")
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
if (-not (Test-Path (Join-Path $Root "runtime\_embedded_model_key.py") -PathType Leaf)) {
    throw "В исходниках обновления отсутствует секретный ключевой модуль сборки."
}
if (-not $BuildPython) {
    $BuildVenv = Join-Path (Split-Path -Parent $Root) ".venv-build"
    $BuildPython = Join-Path $BuildVenv "Scripts\python.exe"
    if (-not (Test-Path $BuildPython -PathType Leaf)) {
        $BasePython = Get-Command python.exe -ErrorAction SilentlyContinue
        if (-not $BasePython) { throw "Не найден Python 3 для подготовки среды сборки." }
        Write-Host "Создаю общее виртуальное окружение сборки..."
        & $BasePython.Source -m venv $BuildVenv
        if ($LASTEXITCODE -ne 0) { throw "Не удалось создать среду сборки." }
    }
}
if (-not (Test-Path $BuildPython -PathType Leaf)) { throw "Не найден Python окружения сборки: $BuildPython" }
Write-Host "Проверяю зависимости в изолированной среде сборки..."
& $BuildPython -m pip install -r (Join-Path $Root "requirements.txt") pyinstaller
if ($LASTEXITCODE -ne 0) { throw "Не удалось установить зависимости сборки." }
if (-not $Version) { $Version = Get-Date -Format "yyyyMMdd_HHmmss_ffff" }
if ($Version -notmatch '^[A-Za-z0-9_-]{1,64}$') { throw "Некорректная версия обновления." }
$Work = Join-Path $Root "tmp\update-$Version"
$Dist = Join-Path $Root "dist"
$Zip = Join-Path $Dist "update-unsigned-$Version.zip"
if (Test-Path $Zip) { throw "Такой архив уже существует; он не перезаписан." }
New-Item -ItemType Directory -Path $Work, $Dist -Force | Out-Null
Write-Host "Собираю обновление $Version..."
& $BuildPython -m PyInstaller --windowed --onedir --name Engine `
    --distpath (Join-Path $Work "engine") --workpath (Join-Path $Work "engine-work") `
    --specpath $Work --paths $Root --paths (Join-Path $Root "app") `
    --hidden-import cv2_enumerate_cameras (Join-Path $Root "Run.py")
if ($LASTEXITCODE -ne 0) { throw "Не удалось собрать программу." }
& $BuildPython (Join-Path $Root "update.py") --pack (Join-Path $Work "engine\Engine") $Zip $Version
if ($LASTEXITCODE -ne 0) { throw "Не удалось создать архив обновления." }
Write-Host "Готово: $Zip"
Write-Host "Для передачи испытателям подпишите архив на компьютере владельца."
