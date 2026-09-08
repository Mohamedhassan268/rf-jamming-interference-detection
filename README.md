# RF Jamming & Interference Detection under Train-to-Field Domain Shift

A reproducible PyTorch pipeline for reconstructing a compact RF classifier trained on RadioML 2018.01A and evaluating its behavior under shifted or captured RF conditions.

> The original classifier showed a substantial train-to-field performance gap when moved from its controlled training distribution to captured RF. This repository reconstructs that experiment and studies the sources of the mismatch rather than reporting only in-distribution performance.

## Status

The repository provides configuration, strict data adapters, preprocessing, a provisional model, validation-selected training, preserved-split source evaluation, untouched captured-domain evaluation, experiment recording, and lightweight CI. No project training has been run and no numerical results are claimed. The exact task, labels, dataset subset, original architecture, and capture settings remain `TBD` pending recovery of original artifacts.

## Known facts

- The original training source was RadioML 2018.01A.
- The original model was a compact CNN with roughly 250K parameters.
- It was designed with a USRP-class deployment target in mind.
- Performance degraded on captured RF due to train-to-field mismatch and architecture choices that did not generalize well.

The `CompactRFNet` included here is a new, configurable provisional baseline, not a claim about the recovered original architecture.

## Repository layout

```text
configs/       YAML experiment definitions
data/          local dataset locations and data policy
docs/          reconstruction notes and research protocol
experiments/   append-only experiment index
figures/       generated figures (no fabricated outputs)
scripts/       preparation, training, evaluation, and analysis CLIs
src/           reusable data, model, evaluation, and utility modules
tests/         lightweight tests that require no RF dataset
```

## Setup

Python 3.10 or newer is required.

```bash
python -m venv .venv
python -m pip install -e ".[dev]"
pytest
```

Obtain RadioML 2018.01A through an authorized source and place it under `data/raw/`; raw data is deliberately ignored by Git. See `data/README.md`.

## Expected workflow

```bash
python scripts/prepare_radioml.py --config configs/baseline.yaml
python scripts/train.py --config configs/baseline.yaml
python scripts/evaluate.py --config configs/baseline.yaml --checkpoint runs/.../best_model.pt
python scripts/evaluate_shift.py --config configs/domain_shift.yaml --checkpoint runs/.../best_model.pt
python scripts/analyze_domain_shift.py --source source.npy --target target.npy
```

Commands validate required settings and fail with readable `TBD` errors. Training writes its exact split and dataset-index fingerprint into each checkpoint; evaluation refuses to proceed if the selected dataset has changed.

## Dataset policy

RadioML and captured RF are not committed. Dataset licensing and permission to publish captures must be established independently of the MIT code license. Capture metadata must record only known facts; unknown values remain `null` or `TBD`.

## Unresolved details

- Exact task and label construction
- Modulation classes, SNR range, input length, and source split
- Original preprocessing and CNN architecture
- Training protocol and all source/target metrics
- Capture hardware, signal settings, labels, and publication rights
- Corrective changes and corrected architecture/results

See `docs/original_experiment.md` for the complete boundary between known facts and reconstruction choices.
Use `docs/recovery_checklist.md` to record artifact-backed answers before enabling training.

## Limitations

This repository currently supplies infrastructure, not reproduced evidence. It does not claim real-time inference, USRP deployment, real-world data availability, successful domain adaptation, or measured accuracy. The RadioML adapter supports documented HDF5-style arrays but must inspect actual files before accepting them; incompatible shapes are rejected rather than reshaped silently.

## Roadmap

1. Recover the original task, artifacts, and acquisition metadata.
2. Validate the loader against an authorized RadioML 2018.01A copy.
3. reproduce a held-out source-domain baseline.
4. Measure the untouched model on a separately held target domain.
5. Run controlled preprocessing, augmentation, and architecture ablations.
