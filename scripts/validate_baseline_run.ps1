param(
    [Parameter(Mandatory = $true)]
    [string]$RunDirectory
)

$ErrorActionPreference = "Stop"
$run = (Resolve-Path -LiteralPath $RunDirectory).Path
$required = @(
    "config.yaml",
    "best_model.pt",
    "history.csv",
    "metrics.json",
    "balanced_test_predictions.csv",
    "jammer_jsr_results.csv",
    "source_snr_results.csv",
    "confusion_matrix.png",
    "training_curves.png",
    "detection_rate_vs_jsr.png",
    "detection_rate_vs_source_snr.png",
    "metadata.json",
    "source_indices.npz",
    "split_audit.json",
    "sanity_checks.json"
)

foreach ($name in $required) {
    $path = Join-Path $run $name
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        throw "Missing required artifact: $path"
    }
    if ((Get-Item -LiteralPath $path).Length -eq 0) {
        throw "Empty required artifact: $path"
    }
}

$metadata = Get-Content -Raw -LiteralPath (Join-Path $run "metadata.json") | ConvertFrom-Json
$metrics = Get-Content -Raw -LiteralPath (Join-Path $run "metrics.json") | ConvertFrom-Json
$split = Get-Content -Raw -LiteralPath (Join-Path $run "split_audit.json") | ConvertFrom-Json
$sanity = Get-Content -Raw -LiteralPath (Join-Path $run "sanity_checks.json") | ConvertFrom-Json
$predictions = @(Import-Csv -LiteralPath (Join-Path $run "balanced_test_predictions.csv"))
$detailed = @(Import-Csv -LiteralPath (Join-Path $run "jammer_jsr_results.csv"))
$sourceSnr = @(Import-Csv -LiteralPath (Join-Path $run "source_snr_results.csv"))

if ($metadata.parameter_count -ne 240962) { throw "Unexpected model parameter count." }
if (-not $split.leakage_check_passed) { throw "Leakage audit did not pass." }
if (-not $sanity.train.condition_balance_passed -or -not $sanity.validation.condition_balance_passed) {
    throw "Jammer-condition balance audit did not pass."
}
if (-not $sanity.determinism.sample_generated_twice_identically -or -not $sanity.jsr_check.passed) {
    throw "Determinism or JSR audit did not pass."
}
if ($predictions.Count -ne 2766) { throw "Balanced prediction row count is not 2,766." }
if (@($predictions | Where-Object true_label -eq "0").Count -ne 1383 -or
    @($predictions | Where-Object true_label -eq "1").Count -ne 1383) {
    throw "Balanced predictions are not exactly 50/50."
}
if ($detailed.Count -ne 2160) { throw "Detailed stress result does not contain 2,160 groups." }
if ($sourceSnr.Count -ne 6) { throw "Source-SNR result does not contain six levels." }
if ($metrics.stress_test_metrics.jammed_variants -ne 20745) {
    throw "Stress evaluation does not contain 20,745 jammed variants."
}

Write-Host "BASELINE ARTIFACT VALIDATION PASSED" -ForegroundColor Green
Write-Host "Run: $run"
Write-Host "Balanced predictions: $($predictions.Count) (1,383 clean + 1,383 jammed)"
Write-Host "Detailed stress groups: $($detailed.Count); stress variants: $($metrics.stress_test_metrics.jammed_variants)"
