param(
    [string]$PythonExe = ".\.venv\Scripts\python.exe",
    [string]$AppName = "DesktopGirls",
    [string]$Version = "0.1.0.0"
)

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
$BuildRoot = Join-Path $RepoRoot "build"
$NuitkaRoot = Join-Path $BuildRoot "nuitka"
$ReleaseRoot = Join-Path $BuildRoot "release"
$ReleaseDir = Join-Path $ReleaseRoot $AppName
$OutputExe = Join-Path $NuitkaRoot "$AppName.exe"

New-Item -ItemType Directory -Force -Path $BuildRoot | Out-Null
Remove-Item -Recurse -Force $NuitkaRoot -ErrorAction SilentlyContinue
Remove-Item -Recurse -Force $ReleaseDir -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force -Path $NuitkaRoot | Out-Null
New-Item -ItemType Directory -Force -Path $ReleaseDir | Out-Null

Write-Host "[build] compiling with Nuitka..."
& $PythonExe -m nuitka `
    --onefile `
    --enable-plugin=pyside6 `
    --windows-console-mode=disable `
    --assume-yes-for-downloads `
    --msvc=latest `
    --company-name=$AppName `
    --product-name=$AppName `
    --file-version=$Version `
    --product-version=$Version `
    --onefile-tempdir-spec="{CACHE_DIR}/$AppName/{VERSION}" `
    --output-dir=$NuitkaRoot `
    --output-filename="$AppName.exe" `
    (Join-Path $RepoRoot "main.py")

if (-not (Test-Path -LiteralPath $OutputExe)) {
    throw "Nuitka build finished without producing $OutputExe"
}

Copy-Item -LiteralPath $OutputExe -Destination (Join-Path $ReleaseDir "$AppName.exe")

$ExternalDirs = @("models", "tools", "dancer")
foreach ($DirName in $ExternalDirs) {
    $SourceDir = Join-Path $RepoRoot $DirName
    if (Test-Path -LiteralPath $SourceDir) {
        Write-Host "[build] copying $DirName/"
        Copy-Item -Recurse -Force -LiteralPath $SourceDir -Destination (Join-Path $ReleaseDir $DirName)
    }
}

Write-Host "[build] release directory ready: $ReleaseDir"
