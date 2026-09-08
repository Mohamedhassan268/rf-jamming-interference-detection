$ErrorActionPreference = "Stop"
$repositoryRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$taskTemp = Join-Path $repositoryRoot "data/tmp"
$runRoot = Join-Path $repositoryRoot "runs"
$statusPath = Join-Path $runRoot "full-baseline.status"
$logPath = Join-Path $runRoot "full-baseline.log"

New-Item -ItemType Directory -Path $taskTemp -Force | Out-Null
New-Item -ItemType Directory -Path $runRoot -Force | Out-Null
Remove-Item -LiteralPath $statusPath -Force -ErrorAction SilentlyContinue
$env:TEMP = $taskTemp
$env:TMP = $taskTemp
Set-Location -LiteralPath $repositoryRoot
Start-Transcript -Path $logPath -Force | Out-Null

try {
    Write-Host "=== First full baseline: training and validation only ===" -ForegroundColor Cyan
    Write-Host "Started: $([DateTime]::UtcNow.ToString('o')) | temporary storage: $taskTemp"
    $before = @(Get-ChildItem -LiteralPath $runRoot -Directory -Filter "*_baseline" | ForEach-Object FullName)
    python scripts/train.py `
        --config configs/baseline.yaml `
        --experiment-id baseline `
        --runs-dir runs
    if ($LASTEXITCODE -ne 0) {
        throw "Baseline training failed with exit code $LASTEXITCODE"
    }

    $runDirectory = Get-ChildItem -LiteralPath $runRoot -Directory -Filter "*_baseline" |
        Where-Object { $_.FullName -notin $before } |
        Sort-Object LastWriteTimeUtc -Descending |
        Select-Object -First 1
    if ($null -eq $runDirectory) {
        throw "Baseline training did not create a new run directory."
    }

    Write-Host "=== Final balanced test (selected checkpoint) ===" -ForegroundColor Cyan
    python scripts/evaluate.py `
        --config configs/baseline.yaml `
        --checkpoint (Join-Path $runDirectory.FullName "best_model.pt") `
        --experiment-id first_full_baseline `
        --results-csv experiments/results.csv `
        --device cpu
    if ($LASTEXITCODE -ne 0) {
        throw "Balanced test evaluation failed with exit code $LASTEXITCODE"
    }

    Write-Host "=== Full 15-condition test-only stress evaluation ===" -ForegroundColor Cyan
    python scripts/evaluate_stress.py `
        --config configs/baseline.yaml `
        --checkpoint (Join-Path $runDirectory.FullName "best_model.pt") `
        --device cpu
    if ($LASTEXITCODE -ne 0) {
        throw "Stress evaluation failed with exit code $LASTEXITCODE"
    }

    "PASS: $($runDirectory.FullName)" | Set-Content -LiteralPath $statusPath
    Write-Host "Full baseline passed: $($runDirectory.FullName)" -ForegroundColor Green
    Write-Host "Finished: $([DateTime]::UtcNow.ToString('o'))"
}
catch {
    "FAIL: $($_.Exception.Message)" | Set-Content -LiteralPath $statusPath
    Write-Error $_
    exit 1
}
finally {
    Stop-Transcript | Out-Null
}
