# New Experimental Task Definition

## Scope and Provenance

This is a new experiment designed in this repository after the original task and label-generation artifacts could not be recovered. It is not a reconstruction of the historical label space.

## Prediction Task

Given one fixed-length raw I/Q window, predict one of two labels:

| ID | Label | Construction |
|---:|---|---|
| 0 | `clean` | The selected RadioML window without an injected jammer |
| 1 | `jammed` | The same kind of RadioML window with one configured synthetic jammer added at a recorded JSR |

Here, `clean` means **not synthetically jammed by this repository**. It does not mean noiseless, impairment-free, interference-free in every physical sense, or captured in a real receiver environment. RadioML channel/noise effects already present in a source window are retained.

## Leakage Control

Source RadioML windows are divided into train, validation, and test indices before labels are generated. Each selected source window then produces exactly two examples inside one split:

1. an unchanged clean example;
2. a deterministically generated jammed example.

This pairing creates a balanced binary task while ensuring clean and jammed versions of the same source window cannot cross split boundaries. If parent-waveform or session identifiers become available, splitting should be upgraded from sample-level to group-level before any reported experiment.

## Signal Model

For source window `s[n]` and a unit-power jammer `j[n]`, the jammed sample is

```text
x[n] = s[n] + α j[n]
```

where `α` is computed from the source and jammer window powers to achieve the configured jammer-to-signal ratio:

```text
JSR_dB = 10 log10(P_jammer / P_signal)
```

The implementation records both the requested and numerically achieved JSR.

## Initial Jammer Families

- `tone`: one constant-frequency complex sinusoid, with random phase and normalized frequency away from a configurable DC exclusion region;
- `chirp`: a complex linear-frequency sweep with random direction, start frequency, sweep magnitude, and phase;
- `barrage`: complex white Gaussian noise normalized to unit average complex power.

These are controlled synthetic interference models. They are not claimed to reproduce a specific field jammer, standard, or receiver impairment.

## Initial Design Grid

The provisional configuration uses:

```text
jammer families: tone, chirp, barrage
JSR values (dB): -10, -5, 0, 5, 10
label balance:   one clean + one jammed example per source window
```

This grid is a new baseline design choice, not a recovered setting and not claimed to be optimal. Results should be stratified by jammer family, JSR, source modulation ID, and RadioML SNR when available.

## Reproducibility

Jammer family, JSR, and waveform parameters are deterministic functions of the configured label-generation seed and source index. Metadata returned with each example includes:

- binary label and name;
- source sample ID and source modulation metadata when available;
- label-generation seed;
- jammer family;
- requested and achieved JSR;
- sampled tone/chirp parameters.

## Evaluation Boundary

The first result should be a held-out **synthetic in-distribution** result. It must not be called real-world or captured-RF performance. A later labeled capture session may be evaluated without fine-tuning to measure a source-to-target gap, but only if its `clean`/`jammed` labels are operationally compatible with this task.
