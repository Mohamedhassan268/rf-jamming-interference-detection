# Pipeline Smoke Validation

On 2026-09-09, commit `8f688bc8a4a2c8505e4aef50934d22fcd7eaf1f4` completed the visible `scripts/run_smoke.ps1` workflow on CPU.

The run used `configs/smoke.yaml`: 32 deterministically selected source windows per modulation class, 768 source windows total, paired clean/jammed generation, two epochs, and a preserved held-out split. It produced both checkpoints, training history, resolved configuration, metadata, held-out metrics, a confusion matrix, and an isolated smoke-results CSV.

This establishes software-path functionality only. The sample cap, two-epoch duration, omitted SNR metadata, and lack of a predeclared scientific stopping rule make its numerical metrics unsuitable for publication, model comparison, or a CV claim. Local run artifacts remain ignored under `runs/`.
