$ErrorActionPreference = "Stop"
$repositoryRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$taskTemp = Join-Path $repositoryRoot "data/tmp"
$runRoot = Join-Path $repositoryRoot "runs"
$statusPath = Join-Path $runRoot "pipeline-smoke.status"
$logPath = Join-Path $runRoot "pipeline-smoke.log"

New-Item -ItemType Directory -Path $taskTemp -Force | Out-Null
New-Item -ItemType Directory -Path $runRoot -Force | Out-Null
Remove-Item -LiteralPath $statusPath -Force -ErrorAction SilentlyContinue
$env:TEMP = $taskTemp
$env:TMP = $taskTemp
Set-Location -LiteralPath $repositoryRoot
Start-Transcript -Path $logPath -Force | Out-Null

try {
    Write-Host "=== Two-epoch CPU pipeline smoke run ===" -ForegroundColor Cyan
    python scripts/train.py `
        --config configs/smoke.yaml `
        --experiment-id pipeline_smoke `
        --runs-dir runs
    if ($LASTEXITCODE -ne 0) {
        throw "Smoke training failed with exit code $LASTEXITCODE"
    }

    $runDirectory = Get-ChildItem -LiteralPath $runRoot -Directory -Filter "*_pipeline_smoke" |
        Sort-Object LastWriteTimeUtc -Descending |
        Select-Object -First 1
    if ($null -eq $runDirectory) {
        throw "Smoke training did not create a run directory."
    }

    Write-Host "=== Held-out smoke evaluation ===" -ForegroundColor Cyan
    python scripts/evaluate.py `
        --config configs/smoke.yaml `
        --checkpoint (Join-Path $runDirectory.FullName "best_model.pt") `
        --experiment-id pipeline_smoke_do_not_report `
        --results-csv (Join-Path $runDirectory.FullName "smoke_results.csv") `
        --device cpu
    if ($LASTEXITCODE -ne 0) {
        throw "Smoke evaluation failed with exit code $LASTEXITCODE"
    }

    "PASS: $($runDirectory.FullName)" | Set-Content -LiteralPath $statusPath
    Write-Host "Smoke pipeline passed: $($runDirectory.FullName)" -ForegroundColor Green
}
catch {
    "FAIL: $($_.Exception.Message)" | Set-Content -LiteralPath $statusPath
    Write-Error $_
    exit 1
}
finally {
    Stop-Transcript | Out-Null
}
