# Dataset

RadioML 2018.01A is the known original training source. Its exact local representation, subset, class semantics, SNR selection, input length, and licensing/redistribution terms for this project remain `TBD`.

The internal convention is `[N, 2, T]`. This is a reconstruction interface, not evidence of the original representation. The HDF5 adapter inspects real keys and shapes, accepts `[N, 2, T]` or `[N, T, 2]`, transposes only the latter after validation, and never silently reshapes incompatible data.

Captured or shifted RF uses session directories described in `data/README.md`. Unknown acquisition fields remain unknown. Adjacent windows from a common waveform or session should be split as a group to prevent leakage.

