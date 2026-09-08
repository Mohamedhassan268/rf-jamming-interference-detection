# Original Experiment Reconstruction

This document separates facts present in the implementation plan from new reconstruction choices. No original code, notebooks, weights, captures, logs, metrics, or figures were present in the inspected workspace on 2026-09-08.

## Task Definition

Known: jamming/interference detection or classification from raw RF signal data.

Exact prediction target: TBD. It is not known whether this was binary detection, jammer-type classification, modulation classification, or another task.

## Label Construction

TBD. It is unknown how clean, jammer, or interference labels were created, whether RadioML modulation labels were used as a proxy, or whether signals were mixed or injected.

## Dataset

- RadioML 2018.01A
- exact subset: TBD
- modulation classes: TBD
- SNR range: TBD
- sample counts and balance: TBD
- dataset storage schema used by the original code: TBD

## Input Representation

TBD. The reconstruction uses `[N, 2, T]` internally, with I and Q as Conv1D channels, solely as a provisional implementation choice. It is not a recovered historical fact.

## Preprocessing

TBD. The reconstructed library provides independently switchable no normalization, per-window RMS normalization, DC removal, amplitude scaling, phase rotation, AWGN, and normalized frequency offset. None is asserted to have been used originally.

## Original Model

- compact CNN
- approximately 250K parameters
- exact architecture: TBD unless recovered
- intended as a small model with a USRP-class deployment target

`CompactRFNet` is a new configurable provisional baseline, not the recovered original architecture. Its Conv1D layers, batch normalization, pooling, global average pooling, dropout, and dense head are implementation choices.

## Training Protocol

TBD. Seed 42 and the 70/15/15 split shown in configuration are safe reconstruction defaults/placeholders, not recovered facts. Optimizer, schedule, epochs, batch size, validation logic, and original data-leakage controls are unverified.

## In-Distribution Result

TBD. No accuracy, F1, confusion matrix, logs, or saved predictions were found.

## Shifted / Real-Capture Evaluation

Known: performance degraded outside the original training distribution when tested against captured RF.

Exact result: TBD. The capture data, labels, acquisition metadata, evaluation split, and raw predictions were not found.

## Diagnosed Causes

- train-to-field domain mismatch
- architecture choices that did not generalize well

The exact diagnostic evidence and contribution of each cause are TBD.

## Corrective Changes

TBD unless recovered from files. No corrected-model design or successful intervention is claimed.

## Corrected Result

TBD.

## Remaining Unknowns

- Exact output task, number of classes, and semantic class names
- Clean/interference/jammer label generation procedure
- RadioML modulation subset, SNR values, input length, sample counts, and source-file schema
- Whether interference was mixed, injected, or represented by another labeling rule
- Original input orientation, preprocessing, normalization, and augmentation
- Exact original and corrected architectures and parameter counts
- Train/validation/test ratios, grouping rules, seeds, optimizer, and stopping protocol
- Original held-out metrics and per-SNR behavior
- Captured-data files, format, labels, session boundaries, and publication rights
- Receiver/SDR model, sample rate, center frequency, bandwidth, gain, antenna, environment, and dates
- Exact corrective changes, selection protocol, and corrected metrics
- Whether any model was executed in an SDR pipeline or had latency measured

