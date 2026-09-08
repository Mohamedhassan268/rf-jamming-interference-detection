# RF Jamming & Interference Detection under Train-to-Field Domain Shift

This repository defines a reproducible binary RF experiment using RadioML 2018.01A source windows: distinguish an unchanged source window (`clean`) from a window containing a controlled synthetic jammer (`jammed`). The study is motivated by earlier compact-CNN work whose artifacts could not be recovered, with particular attention to degradation when a model moves from controlled synthetic training data toward captured RF conditions.

## Research Motivation

RF classifiers can perform well in-distribution while failing when receiver characteristics, channel conditions, gain, interference structure, noise, or acquisition settings change. This repository is designed to measure that source-to-target gap, inspect likely causes, and support controlled ablations. It does not claim to have solved domain adaptation.

## Current Status

Implemented:

- Strict RadioML 2018.01A HDF5 adapter with explicit schema and shape validation
- Internal raw-I/Q convention `[N, 2, T]`
- Explicit binary label generation after source-window splitting
- Deterministic tone, chirp, and barrage-noise injection with recorded JSR
- Configurable preprocessing and RF augmentations, disabled by default
- Provisional Conv1D reconstruction baseline
- Seeded train/validation/test splitting and validation-loss model selection
- Source-domain metrics, per-SNR and per-jammer/JSR detection rates, calibration, and confusion matrices
- Labeled captured-RF adapter and untouched-model target evaluation path
- Append-only experiment records and lightweight dataset-free tests

Pending:

- Recovery of the original project artifacts
- Recovery of the original architecture, preprocessing, and split protocol
- Reproducible source and captured-domain experiments
- Publication of actual metrics and figures

No RF training run has been completed in this repository.

For the current two-class default architecture, the trainable-parameter count is exactly `240,962` (`240,192 + 385 × C` with `C=2`). This is the count for the new provisional model, not evidence for the historical approximately-250K model.

## Pipeline

The implemented code expects raw I/Q windows and predicts the newly defined binary labels `clean=0` and `jammed=1`.

```text
Raw I/Q [N, 2, T]
        ↓
Configurable preprocessing
        ↓
Provisional compact CNN
        ↓
Binary clean / synthetically jammed prediction
        ↓
Held-out source-domain evaluation
        ↓
Untouched shifted-domain evaluation
        ↓
Descriptive failure analysis
```

## Provenance

Recovered facts from prior project material:

- RadioML 2018.01A was used.
- A compact CNN was used, targeting approximately 250K parameters.
- USRP-class SDR deployment was an intended target.
- Performance degraded outside the controlled training distribution/on captured RF.
- Train-to-field mismatch and architecture choices were identified as contributing issues.

The new binary task, synthetic jammer families, JSR grid, Conv1D layout, `[N, 2, T]` interface, seed `42`, 70/15/15 split, Adam defaults, and disabled-by-default preprocessing are new experimental choices—not recovered historical facts. See [docs/task_definition.md](docs/task_definition.md) and [docs/original_experiment.md](docs/original_experiment.md).

## Reproducibility

From the repository root, these lightweight commands are dataset-free and tested:

```bash
python -m pip install -e ".[dev]"
python -m pytest
python scripts/summarize_model.py
python -c "from src.utils.config import load_config; print(load_config('configs/baseline.yaml')['seed'])"
```

On Windows, the same repository checks plus every CLI help command can be run together in a visible terminal:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/verify_repository.ps1
```

Every CLI exposes its current interface through `--help`:

```bash
python scripts/prepare_radioml.py --help
python scripts/prepare_captures.py --help
python scripts/train.py --help
python scripts/evaluate.py --help
python scripts/evaluate_shift.py --help
python scripts/analyze_domain_shift.py --help
```

The default configuration uses a deterministic compact subset: 9,216 source windows, all 24 modulation classes, 64 windows per selected modulation/SNR condition, and SNR levels `[-20, -10, 0, 10, 20, 30]` dB. The ignored local HDF5 is 61.11 MiB. Recreate it without downloading the 21 GB source file:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/download_small_radioml.ps1
```

The downloader uses HTTP byte ranges, writes temporary and final data only under `data/` on D: when the repository is on D:, rejects full-file responses, and enforces a 150 MiB transfer ceiling. Validate the configured schema with:

```bash
python scripts/prepare_radioml.py --config configs/baseline.yaml --validate-configured-schema
```

The dataset itself is excluded from Git. Its local provenance JSON records source and output hashes, selection parameters, shapes, and license.

On CPU-only systems, validate the complete train/checkpoint/evaluate path with a deliberately non-reportable two-epoch smoke run:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_smoke.ps1
```

## Results

> Numerical results will be added after the new experiment is run reproducibly. No placeholder accuracy values are reported.

`experiments/results.csv` currently contains only its schema. There are no project accuracy, F1, calibration, training-history, or train-to-field gap measurements to report.

## Domain Shift

The implemented evaluation path can compare held-out synthetically labeled RadioML windows with a compatible labeled capture session using the same trained checkpoint, class order, input shape, and preprocessing. It records accuracy, macro/weighted F1, ECE, per-SNR metrics when available, and confusion matrices. Descriptive I/Q distribution plots are also supported.

This is planned experimental capability, not a completed domain-shift result. No capture data is currently present, and no adaptation method has been evaluated.

## Limitations

- The exact historical task and label construction remain unknown; the current binary task is new.
- Training jammers are simplified synthetic tone, chirp, and barrage models, not captured field interference.
- The new compact subset is verified, but it represents only six of the full dataset's SNR levels and 64 contiguous source windows per selected modulation/SNR condition.
- The current CNN is technically reasonable but provisional; it is not recovered original code.
- RadioML is not redistributed or automatically downloaded.
- Real captured RF and its acquisition metadata have not been recovered.
- Live USRP inference and real-time latency have not been demonstrated.
- Sample-level source splitting is implemented; parent-waveform grouping requires metadata not currently available.
- The GitHub Actions workflow is prepared locally but is not published because the current GitHub token lacks workflow-write scope.

## Repository Structure

```text
.
├── configs/                 experiment and domain-shift configuration
├── data/                    data policy, ignored raw/processed locations, capture schema
├── docs/                    provenance, dataset, domain-shift, and experiment documentation
├── experiments/             append-only results index
├── figures/                 ignored generated figures
├── scripts/                 preparation, training, evaluation, analysis, model summary
├── src/
│   ├── data/                RadioML/capture adapters, transforms, splits
│   ├── evaluation/          metrics, per-SNR analysis, calibration
│   ├── models/              provisional compact CNN
│   ├── training/            validation-selected training loop
│   └── utils/               config, reproducibility, experiment metadata
└── tests/                   dataset-free unit and smoke tests
```

## Roadmap

- [x] Create strict data-adapter and configuration infrastructure
- [x] Implement provisional compact-CNN and preprocessing modules
- [x] Add validation-selected training and preserved-split evaluation
- [x] Add metrics, calibration, capture loading, and experiment tracking
- [x] Add dataset-free automated tests
- [x] Define a new binary task and leakage-safe synthetic label-generation protocol
- [ ] Inspect and validate the actual RadioML 2018.01A file
- [ ] Recover or document differences from the original architecture
- [ ] Run and report a held-out source-domain baseline
- [ ] Recover and validate captured RF plus acquisition metadata
- [ ] Measure the untouched source-to-target performance gap
- [ ] Run controlled preprocessing, augmentation, and architecture ablations
- [ ] Publish CI after GitHub workflow-write access is available

Code is released under the MIT License. Dataset licensing and capture-publication rights must be established separately.
