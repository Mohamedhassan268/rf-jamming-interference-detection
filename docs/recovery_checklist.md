# Original-artifact recovery worksheet

This worksheet is intentionally factual. Enter `unknown` when a fact cannot be recovered and cite the source of every recovered value (file path, notebook cell, log, screenshot, or written recollection).

## Artifact inventory

| Artifact | Location | Present? | Notes |
|---|---|---:|---|
| Original Python scripts | TBD | TBD | |
| Original notebooks | TBD | TBD | |
| Model checkpoints | TBD | TBD | |
| Training logs/history | TBD | TBD | |
| Metrics/confusion matrices | TBD | TBD | |
| RadioML source file | TBD | TBD | Do not commit it |
| Captured/shifted RF | TBD | TBD | |
| Capture labels | TBD | TBD | |
| Corrected-model artifacts | TBD | TBD | |

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

