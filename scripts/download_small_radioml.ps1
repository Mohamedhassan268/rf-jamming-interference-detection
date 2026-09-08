param(
    [int]$SamplesPerCondition = 64,
    [int]$Seed = 42,
    [switch]$InspectOnly
)

$ErrorActionPreference = "Stop"
$repositoryRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$taskTemp = Join-Path $repositoryRoot "data/tmp"
$statusPath = Join-Path $repositoryRoot "runs/radioml-subset.status"
$logPath = Join-Path $repositoryRoot "runs/radioml-subset.log"
New-Item -ItemType Directory -Path $taskTemp -Force | Out-Null
New-Item -ItemType Directory -Path (Split-Path $statusPath) -Force | Out-Null
Remove-Item -LiteralPath $statusPath -Force -ErrorAction SilentlyContinue

$env:TEMP = $taskTemp
$env:TMP = $taskTemp
$env:HF_HOME = Join-Path $taskTemp "huggingface-disabled"
Set-Location -LiteralPath $repositoryRoot
Start-Transcript -Path $logPath -Force | Out-Null

try {
    $arguments = @(
        "scripts/download_radioml_subset.py",
        "--max-transfer-mib", "150"
    )
    if ($InspectOnly) {
        $arguments += "--inspect-only"
    }
    else {
        $arguments += @(
            "--output", "data/raw/radioml2018.01a_small.hdf5",
            "--samples-per-condition", "$SamplesPerCondition",
            "--seed", "$Seed"
        )
    }
    python @arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Subset construction failed with exit code $LASTEXITCODE"
    }
    "PASS" | Set-Content -LiteralPath $statusPath
    Write-Host "RadioML subset construction passed." -ForegroundColor Green
}
catch {
    "FAIL: $($_.Exception.Message)" | Set-Content -LiteralPath $statusPath
    Write-Error $_
    exit 1
}
finally {
    Stop-Transcript | Out-Null
}
