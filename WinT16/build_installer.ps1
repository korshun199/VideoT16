# Упаковывает проверенный переносимый комплект в один офлайн-установщик.
param([Parameter(Mandatory=$true)][string]$ReleaseRoot, [string]$BuildPython = "")
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
if (-not $BuildPython) { $BuildPython = Join-Path $Root ".venv-build\Scripts\python.exe" }
if (-not (Test-Path $BuildPython -PathType Leaf)) { throw "Не найден Python окружения сборки." }
$ReleaseRoot = (Resolve-Path $ReleaseRoot).Path
Push-Location $Root
try {
    & $BuildPython -c "import sys; from pathlib import Path; from install import validate_payload; validate_payload(Path(sys.argv[1])); print('Payload OK')" $ReleaseRoot
    if ($LASTEXITCODE -ne 0) { throw "Комплект содержит лишние или отсутствующие файлы." }
    $Version = Get-Date -Format "yyyyMMdd_HHmmss_ffff"
    $Work = Join-Path $Root "tmp\installer-$Version"
    $Dist = Join-Path $Root "dist"
    $Name = "Install-$Version"
    if (Test-Path (Join-Path $Dist ($Name + '.exe'))) { throw "Такой установщик уже существует." }
    New-Item -ItemType Directory -Path $Work, $Dist -Force | Out-Null
    Write-Host "Собираю офлайн-установщик $Name.exe..."
    & $BuildPython -m PyInstaller --windowed --onefile --name $Name `
        --distpath $Dist --workpath (Join-Path $Work 'work') --specpath $Work `
        --add-data ($ReleaseRoot + ';payload') (Join-Path $Root 'install.py')
    if ($LASTEXITCODE -ne 0) { throw "Не удалось собрать установщик." }
    Write-Host "Готово: $(Join-Path $Dist ($Name + '.exe'))"
} finally {
    Pop-Location
}
