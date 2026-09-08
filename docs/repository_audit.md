# Repository Credibility Audit

Audit date: 2026-09-08

## Five Largest Weaknesses Before This Audit

1. The exact task and label construction were unknown, so genuine jamming detection could not be distinguished from a modulation-label proxy.
2. The README mixed tested infrastructure with future experiment commands that could not run using unresolved configuration.
3. Provisional numeric defaults were documented in prose but insufficiently labeled inside executable configuration.
4. No RF dataset, completed experiment, metrics, or figures existed.
5. A CI workflow existed only locally while the public README described lightweight CI as implemented.

## Audit Table

| Item | Current state | Supported fact / new assumption | Action needed |
| ---- | ------------- | ------------------------------- | ------------- |
| README | Rewritten to separate implemented code, recovered facts, pending work, and tested commands | Mixed: RadioML/domain-gap story is recovered; pipeline details are reconstruction | Add real results only after verified reruns |
| Task definition | Broad RF jamming/interference objective documented; exact target unknown | Recovered broad objective; exact labels unsupported | Highest priority: recover what the model predicted |
| Dataset loader | Strict HDF5 adapter; validates `[N,2,T]` or `[N,T,2]`, filters classes/SNR, preserves metadata | Entire adapter interface is a reconstruction choice | Validate against the actual authorized RadioML file |
| Label construction | Not implemented or inferred | Unknown | Recover label-generation code/procedure before training |
| Preprocessing | None/RMS normalization, DC removal, amplitude, phase, AWGN, and frequency-offset operations | New reconstruction infrastructure; none is claimed original | Run one-factor-at-a-time ablations after baseline recovery |
| Model architecture | `ProvisionalCompactRFNet`, configurable Conv1D; `240,192 + 385 × C` trainable parameters | Compact CNN/~250K target recovered; exact layers are new | Resolve class count, then compare with original code |
| Training script | Config-driven Adam training, validation-loss selection, early stopping, saved metadata/splits | Training protocol is new | Do not call it historical baseline until protocol is recovered |
| Evaluation script | Fingerprint-checked held-out source evaluation with standard metrics and calibration | New evaluation implementation | Run only after verified task/data configuration |
| Domain-shift script | Untouched checkpoint evaluation for exact-mapped capture labels | Original degradation is recovered; current implementation is new | Recover captures and verify source/target label equivalence |
| Configs | Unknown executable fields are `null`; defaults marked provisional; explicit validation | Numeric defaults are new assumptions | Replace nulls only with evidenced values |
| Tests | Dataset-free tests cover transforms, loaders, shapes, metrics, config, and training loop | Synthetic software verification, not RF evidence | Add integration tests against metadata-only fixtures as needed |
| Experiment tracking | Append-only CSV schema plus per-run config/history/checkpoint metadata | New infrastructure | Populate only from actual runs |
| Figures | Directory exists; no project figure is committed | No result exists | Generate figures from saved predictions after experiments |
| Real-capture support | Session loader accepts samples, labels, and open-ended metadata | Session format is a reconstruction choice | Recover data, acquisition facts, labels, and publication rights |
| Dependencies | Python 3.10+, PyTorch, NumPy, pandas, scikit-learn, matplotlib, PyYAML, h5py, pytest | New reproducible environment choice | Pin a lockfile only when experiment environment is established |
| Documentation | Dataset, domain shift, deployment, experiment protocol, provenance, recovery worksheet | Recovered facts explicitly separated from design choices | Fill citations/evidence paths as artifacts are recovered |
| GitHub Actions | Workflow prepared locally but ignored; not present on public GitHub | New CI configuration | Publish after GitHub authentication gains workflow-write scope |

## CV-Readiness Verdict

**NOT YET**

The codebase is inspectable and honest, but its central scientific object is unresolved. Exact blockers:

1. No verified task or label-construction definition.
2. No actual RadioML file has been inspected through the adapter.
3. No historically faithful architecture or documented difference analysis.
4. No reproducible held-out source-domain result.
5. No recovered, labeled capture data or measured train-to-field gap.
6. No actual result figures.

Publishing the link on a CV now would present competent infrastructure, but not yet a completed RF research project.
