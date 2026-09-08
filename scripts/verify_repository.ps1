$ErrorActionPreference = "Stop"
$repositoryRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$runDirectory = Join-Path $repositoryRoot "runs"
$logPath = Join-Path $runDirectory "lightweight-checks.log"
$statusPath = Join-Path $runDirectory "lightweight-checks.status"

New-Item -ItemType Directory -Path $runDirectory -Force | Out-Null
Remove-Item -LiteralPath $statusPath -Force -ErrorAction SilentlyContinue
Set-Location -LiteralPath $repositoryRoot
Start-Transcript -Path $logPath -Force | Out-Null

function Invoke-RepositoryCheck {
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][scriptblock]$Command
    )

    Write-Host ""
    Write-Host "=== $Name ===" -ForegroundColor Cyan
    & $Command
    if ($LASTEXITCODE -ne 0) {
        throw "$Name failed with exit code $LASTEXITCODE"
    }
}

try {
    Invoke-RepositoryCheck "Editable installation" { python -m pip install -e ".[dev]" }
    Invoke-RepositoryCheck "Unit tests" { python -m pytest }
    Invoke-RepositoryCheck "Package imports" { python -c "import src; import src.data; import src.models; import src.training; import src.evaluation; import src.utils; print(src.__version__)" }
    Invoke-RepositoryCheck "Configuration loading" { python -c "from src.utils.config import load_config; c = load_config('configs/baseline.yaml'); print(c['provenance']); print(c['dataset']['task_type'])" }
    Invoke-RepositoryCheck "Model architecture summary" { python scripts/summarize_model.py }
    Invoke-RepositoryCheck "RadioML preparation CLI" { python scripts/prepare_radioml.py --help }
    Invoke-RepositoryCheck "Capture preparation CLI" { python scripts/prepare_captures.py --help }
    Invoke-RepositoryCheck "Training CLI" { python scripts/train.py --help }
    Invoke-RepositoryCheck "Source evaluation CLI" { python scripts/evaluate.py --help }
    Invoke-RepositoryCheck "Shifted evaluation CLI" { python scripts/evaluate_shift.py --help }
    Invoke-RepositoryCheck "Domain analysis CLI" { python scripts/analyze_domain_shift.py --help }
    Write-Host ""
    Write-Host "=== Intentional unverified-input gate ===" -ForegroundColor Cyan
    $previousErrorPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    $gateOutput = python scripts/train.py --config configs/baseline.yaml 2>&1 | Out-String
    $gateExitCode = $LASTEXITCODE
    $ErrorActionPreference = $previousErrorPreference
    Write-Host $gateOutput
    if ($gateExitCode -eq 0) {
        throw "Training unexpectedly ran with unverified input configuration"
    }
    if ($gateOutput -notmatch "Input window length has not yet been verified") {
        throw "Training failed without the expected input-verification explanation"
    }
    "PASS" | Set-Content -LiteralPath $statusPath
    Write-Host ""
    Write-Host "All lightweight repository checks passed." -ForegroundColor Green
}
catch {
    "FAIL: $($_.Exception.Message)" | Set-Content -LiteralPath $statusPath
    Write-Error $_
    exit 1
}
finally {
    Stop-Transcript | Out-Null
}
