# Original-artifact recovery worksheet

This worksheet is intentionally factual. Enter `unknown` when a fact cannot be recovered and cite the source of every recovered value (file path, notebook cell, log, screenshot, or written recollection).

## Artifact inventory

| Artifact | Location | Present? | Notes |
|---|---|---:|---|
| Original Python scripts | Not found in searched local/GitHub locations | No | Search completed 2026-09-08 |
| Original notebooks | Not found in searched local/GitHub locations | No | Unrelated FPGA/medical notebooks were excluded |
| Model checkpoints | Not found in searched local/GitHub locations | No | Unrelated vision checkpoints were excluded |
| Training logs/history | Not found in searched local/GitHub locations | No | |
| Metrics/confusion matrices | Not found in searched local/GitHub locations | No | |
| RadioML source file | Not found in searched local locations | No | Do not commit it if later recovered |
| Captured/shifted RF | Not found in searched local locations | No | |
| Capture labels | Not found in searched local locations | No | |
| Corrected-model artifacts | Not found in searched local/GitHub locations | No | |
| Project summary/index | External local project index | Yes | Repeats only the high-level RadioML/~250K/domain-mismatch claim |

### Excluded candidate

A separate USRP-2920 spectrum scanner/dashboard was inspected. It is rule-based, does not use the RadioML CNN, and is not treated as evidence for this experiment. Its hardware and acquisition settings must not be copied into this repository.

## Task and labels

- Exact prediction target: TBD
- Number and names of classes: TBD
- Clean-example definition: TBD
- Jammer/interference-example definition: TBD
- Label-construction code or procedure: TBD
- Were RadioML modulation labels used directly? TBD
- Were signals mixed or jammer waveforms injected? TBD

## RadioML selection

- Source filename/format: TBD
- HDF5 samples key: TBD
- HDF5 labels key: TBD
- HDF5 SNR key, if present: TBD
- Class-index-to-name mapping: TBD
- Modulation subset: TBD
- SNR subset: TBD
- Window length: TBD
- Maximum/examples per class: TBD

## Original experiment

- Input orientation/representation: TBD
- Preprocessing and order: TBD
- Augmentations: TBD
- Exact architecture: TBD
- Trainable parameter count: approximately 250K (exact value TBD)
- Split ratios and grouping unit: TBD
- Random seed(s): TBD
- Optimizer, learning rate, batch size, epochs: TBD
- Model-selection and early-stopping rule: TBD
- Held-out source metrics: TBD

## Captured/shifted evaluation

- Data format and session boundaries: TBD
- Label mapping to source task: TBD
- Receiver/SDR: TBD
- Sample rate, center frequency, bandwidth, gain: TBD
- Antenna, environment, and dates: TBD
- Untouched-model metrics: TBD
- Permission to publish: TBD

## Correction

- Exact preprocessing change: TBD
- Exact augmentation change: TBD
- Exact architecture change: TBD
- Data or split changes: TBD
- Selection protocol: TBD
- Corrected source and target metrics: TBD

## Evidence rule

A recollection may be documented as a recollection, but it must not be presented as file-verified. New reconstruction choices belong in configuration and experiment notes, not in the recovered-facts column.
