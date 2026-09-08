# RF Jamming & Interference Detection under Train-to-Field Domain Shift

This repository reconstructs and extends a compact RF classifier originally developed using RadioML 2018.01A, with particular attention to the degradation that occurs when a model moves from a controlled training distribution toward captured RF conditions. The original task semantics and artifacts have not yet been recovered, so the current implementation is explicitly a provisional reconstruction rather than a claimed reproduction.

## Research Motivation

RF classifiers can perform well in-distribution while failing when receiver characteristics, channel conditions, gain, interference structure, noise, or acquisition settings change. This repository is designed to measure that source-to-target gap, inspect likely causes, and support controlled ablations. It does not claim to have solved domain adaptation.

## Current Status

Implemented:

- Strict RadioML 2018.01A HDF5 adapter with explicit schema and shape validation
- Internal raw-I/Q convention `[N, 2, T]`
- Configurable preprocessing and RF augmentations, disabled by default
- Provisional Conv1D reconstruction baseline
- Seeded train/validation/test splitting and validation-loss model selection
- Source-domain metrics, per-SNR metrics, calibration, and confusion matrices
- Labeled captured-RF adapter and untouched-model target evaluation path
- Append-only experiment records and lightweight dataset-free tests

Pending:

- Recovery of the exact prediction task and label-construction procedure
- Validation against the actual RadioML file and original project artifacts
- Recovery of the original architecture, preprocessing, and split protocol
- Reproducible source and captured-domain experiments
- Publication of actual metrics and figures

No RF training run has been completed in this repository.

For the current default architecture, the trainable-parameter count is `240,192 + 385 × C`, where `C` is the verified number of output classes. Because `C` is unresolved, reporting one exact project model size would be misleading.

## Pipeline

The implemented code expects raw I/Q windows. The semantic output classes remain unresolved.

```text
Raw I/Q [N, 2, T]
        ↓
Configurable preprocessing
        ↓
Provisional compact CNN
        ↓
Configured RF class / interference label (definition: TBD)
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

Reconstruction choices—not recovered historical facts—include the current Conv1D layout, `[N, 2, T]` interface, seed `42`, 70/15/15 split, Adam defaults, batch size, and disabled-by-default preprocessing. See [docs/original_experiment.md](docs/original_experiment.md).

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

Training is intentionally blocked until the required task, class, schema, and input-length fields in `configs/baseline.yaml` are replaced with verified facts. The program reports each unresolved field rather than selecting labels silently. RadioML schema inspection can run before those facts are known when an authorized HDF5 path is supplied:

```bash
python scripts/prepare_radioml.py --config configs/baseline.yaml --input PATH_TO_RADIOML_HDF5
```

This last command is a usage template because the dataset path is machine-specific and no RadioML file is present in the repository.

## Results

> Numerical results will be added after reconstruction of the original experiment and reproducible reruns. No placeholder accuracy values are reported.

`experiments/results.csv` currently contains only its schema. There are no project accuracy, F1, calibration, training-history, or train-to-field gap measurements to report.

## Domain Shift

The implemented evaluation path can compare a preserved RadioML test split with a labeled capture session using the same trained checkpoint, class order, input shape, and preprocessing. It records accuracy, macro/weighted F1, ECE, per-SNR metrics when available, and confusion matrices. Descriptive I/Q distribution plots are also supported.

This is planned experimental capability, not a completed domain-shift result. No capture data is currently present, and no adaptation method has been evaluated.

## Limitations

- The exact original task and label construction remain unknown.
- It is not known whether RadioML labels were used directly or transformed into a genuine interference task.
- The RadioML subset, SNR range, input length, and original split are unresolved.
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
- [ ] Recover the original task and label-construction procedure
- [ ] Inspect and validate the actual RadioML 2018.01A file
- [ ] Recover or document differences from the original architecture
- [ ] Run and report a held-out source-domain baseline
- [ ] Recover and validate captured RF plus acquisition metadata
- [ ] Measure the untouched source-to-target performance gap
- [ ] Run controlled preprocessing, augmentation, and architecture ablations
- [ ] Publish CI after GitHub workflow-write access is available

Code is released under the MIT License. Dataset licensing and capture-publication rights must be established separately.
