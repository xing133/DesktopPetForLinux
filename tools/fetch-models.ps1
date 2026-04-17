param(
    [switch]$Force,
    [string[]]$Only
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

try {
    [Net.ServicePointManager]::SecurityProtocol = `
        [Net.ServicePointManager]::SecurityProtocol -bor `
        [Net.SecurityProtocolType]::Tls12
} catch {
    # PowerShell 7 on newer runtimes may not need or expose this in the same way.
}

$RepoRoot = Split-Path -Parent $PSScriptRoot

$ModelSpecs = @(
    @{
        Name = "rvm-fp16"
        FileName = "rvm_mobilenetv3_fp16.onnx"
        Url = "https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0.0/rvm_mobilenetv3_fp16.onnx"
        TargetPath = Join-Path $RepoRoot "models\rvm\rvm_mobilenetv3_fp16.onnx"
        Sha256 = "6A0D5CE6CC17702613BE548559879B4521ED424CFE14DDC48D1ACAA44D616F64"
    },
    @{
        Name = "rvm-fp32"
        FileName = "rvm_mobilenetv3_fp32.onnx"
        Url = "https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0.0/rvm_mobilenetv3_fp32.onnx"
        TargetPath = Join-Path $RepoRoot "models\rvm\rvm_mobilenetv3_fp32.onnx"
        Sha256 = "88D4531297118F595BF2FD60F6F566AEC2E559393802D1F436C380F0CBBD2828"
    },
    @{
        Name = "u2net-human-seg"
        FileName = "u2net_human_seg.onnx"
        Url = "https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net_human_seg.onnx"
        TargetPath = Join-Path $RepoRoot "models\u2net\u2net_human_seg.onnx"
        Sha256 = "01EB6A29A5C4D8EDB30B56ADAD9BB3A2A0535338E480724A213E0ACFD2D1C73C"
    }
)

function Get-NormalizedHash {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path
    )

    return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToUpperInvariant()
}

function Test-TargetFile {
    param(
        [Parameter(Mandatory = $true)]
        [hashtable]$Spec
    )

    if (-not (Test-Path -LiteralPath $Spec.TargetPath -PathType Leaf)) {
        return $false
    }

    $ActualHash = Get-NormalizedHash -Path $Spec.TargetPath
    if ($ActualHash -eq $Spec.Sha256) {
        Write-Host "[models] ok: $($Spec.Name) -> $($Spec.TargetPath)"
        return $true
    }

    Write-Warning "[models] checksum mismatch for existing file: $($Spec.TargetPath)"
    Write-Warning "[models] expected: $($Spec.Sha256)"
    Write-Warning "[models] actual:   $ActualHash"
    return $false
}

function Download-Model {
    param(
        [Parameter(Mandatory = $true)]
        [hashtable]$Spec
    )

    $TargetDir = Split-Path -Parent $Spec.TargetPath
    $TempPath = "$($Spec.TargetPath).download"

    New-Item -ItemType Directory -Force -Path $TargetDir | Out-Null
    if (Test-Path -LiteralPath $TempPath) {
        Remove-Item -LiteralPath $TempPath -Force
    }

    Write-Host "[models] downloading $($Spec.Name)"
    Write-Host "[models] source: $($Spec.Url)"
    Invoke-WebRequest -Uri $Spec.Url -OutFile $TempPath

    $ActualHash = Get-NormalizedHash -Path $TempPath
    if ($ActualHash -ne $Spec.Sha256) {
        Remove-Item -LiteralPath $TempPath -Force -ErrorAction SilentlyContinue
        throw (
            "[models] checksum mismatch for $($Spec.Name). " +
            "expected=$($Spec.Sha256), actual=$ActualHash"
        )
    }

    Move-Item -LiteralPath $TempPath -Destination $Spec.TargetPath -Force
    Write-Host "[models] saved: $($Spec.TargetPath)"
}

$SelectedSpecs = $ModelSpecs
if ($Only -and $Only.Count -gt 0) {
    $Wanted = [System.Collections.Generic.HashSet[string]]::new(
        [System.StringComparer]::OrdinalIgnoreCase
    )
    foreach ($Token in $Only) {
        if ([string]::IsNullOrWhiteSpace($Token)) {
            continue
        }
        [void]$Wanted.Add($Token.Trim())
    }

    $SelectedSpecs = @(
        foreach ($Spec in $ModelSpecs) {
            if ($Wanted.Contains($Spec.Name) -or $Wanted.Contains($Spec.FileName)) {
                $Spec
            }
        }
    )

    if ($SelectedSpecs.Count -eq 0) {
        $KnownNames = ($ModelSpecs | ForEach-Object { $_.Name }) -join ", "
        throw "No model matched -Only. Known names: $KnownNames"
    }
}

foreach ($Spec in $SelectedSpecs) {
    if (-not $Force -and (Test-TargetFile -Spec $Spec)) {
        continue
    }

    Download-Model -Spec $Spec
}

Write-Host "[models] done."
