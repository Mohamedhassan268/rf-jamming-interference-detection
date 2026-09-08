# Original Experiment Reconstruction

This document is the provenance boundary for the repository. On 2026-09-08, the inspected workspace contained no original Python code, notebooks, model weights, RF captures, training logs, metrics, confusion matrices, or figures. Statements below are therefore limited to the supplied prior-project material.

## Recovered Facts

The following facts are supported by the prior-project material:

- RadioML 2018.01A was used as the original training source.
- The task concerned jamming/interference detection or classification from RF signal data.
- A compact CNN was used.
- The original model targeted approximately 250K trainable parameters; its exact count is unknown.
- USRP-class SDR deployment was an intended target.
- Performance degraded outside the original controlled distribution/on captured RF.
- Train-to-field domain mismatch and architecture choices were identified as contributing issues.
- The project was iterated after those issues were diagnosed.

These facts do not establish the exact task, labels, architecture, hardware, acquisition settings, or numerical results.

## Reconstruction Choices

The following are new implementation choices made while building this repository. They must not be cited as properties of the original experiment:

- Raw-I/Q internal convention `[N, 2, T]`
- Conv1D with I and Q as input channels
- Convolution channels `[64, 128, 256]`
- Kernel sizes `[7, 5, 3]`
- Max-pooling factors `[2, 2, 2]`
- Batch normalization, ReLU, adaptive average pooling, dropout `0.3`, and a 384-unit classifier layer
- Default seed `42`
- Default 70/15/15 train/validation/test fractions
- Adam, learning rate `0.001`, zero weight decay, batch size `128`, 50-epoch cap, and patience `8`
- Validation-loss checkpoint selection
- Disabled-by-default preprocessing and augmentation
- Current capture-session directory format and exact-label matching
- HDF5 adapter interface and dataset-index fingerprinting

The current `ProvisionalCompactRFNet` is a technically reasonable reconstruction baseline. It is not recovered original code. With the present defaults its trainable-parameter count is `240,192 + 385 × C`, where `C` is the output-class count; `C` remains unresolved.

## Unresolved Original Task

- Exact prediction target: TBD
- Binary detection, jammer-type classification, modulation classification, or another task: TBD
- Number and semantic names of classes: TBD
- Definition of clean examples: TBD
- Definition of jammer/interference examples: TBD
- Label-construction procedure: TBD
- Whether RadioML modulation labels were used directly as a proxy: TBD
- Whether signals were mixed or synthetic jammer waveforms were injected: TBD

Until these are resolved, the repository must not claim that its executable model performs a verified jamming-detection task.

## Unresolved Dataset Details

- RadioML filename/storage schema: TBD
- Exact subset and sample counts: TBD
- Modulation classes: TBD
- SNR values/range: TBD
- Input window length and original orientation: TBD
- Class balance and subsampling: TBD
- Parent-waveform/group metadata needed for leakage-safe splitting: TBD

## Unresolved Preprocessing and Training

- Original normalization, DC handling, and augmentation: TBD
- Exact original CNN architecture and parameter count: TBD
- Original split ratios/grouping rules and random seeds: TBD
- Optimizer, learning rate, batch size, epoch count, and early stopping: TBD
- Model-selection protocol: TBD

## Unresolved Results

- Held-out in-distribution accuracy/F1 and per-SNR results: TBD
- Initial captured/shifted-domain metrics: TBD
- Exact contribution of each diagnosed mismatch: TBD
- Corrective preprocessing, augmentation, or architecture changes: TBD
- Corrected source/target metrics: TBD

No numerical project result is currently available.

## Unresolved Capture and Deployment Details

- Capture files, format, labels, and session boundaries: TBD
- Receiver/SDR model: TBD
- Sample rate, center frequency, bandwidth, gain, and antenna: TBD
- Capture environment and dates: TBD
- Permission to publish captured RF: TBD
- Whether a model ever ran in an SDR pipeline: TBD
- Latency, throughput, compute, and memory measurements: TBD

The only currently supportable deployment wording is **USRP-class deployment target**.
