# Dataset

RadioML 2018.01A is the known original training source. The historical project's exact subset and protocol remain unknown. For the separate new experiment, the compact local representation, class order, SNR selection, input length, provenance, and license have now been verified as documented below.

The internal convention is `[N, 2, T]`. This is a reconstruction interface, not evidence of the original representation. The HDF5 adapter inspects real keys and shapes, accepts `[N, 2, T]` or `[N, T, 2]`, transposes only the latter after validation, and never silently reshapes incompatible data.

Captured or shifted RF uses session directories described in `data/README.md`. Unknown acquisition fields remain unknown. Adjacent windows from a common waveform or session should be split as a group to prevent leakage.

For the new binary task, source RadioML indices are split before label generation. Each source window produces one unchanged `clean` example and one deterministic synthetic `jammed` example within the same split. See `docs/task_definition.md`. RadioML modulation labels are retained as metadata/stratification information; they are not the model target.

## Compact Local Subset

The default configuration uses `data/raw/radioml2018.01a_small.hdf5`, created by `scripts/download_radioml_subset.py` through byte-range reads from an exact full-file mirror. It contains 9,216 source windows: all 24 modulation labels, six SNR levels (`-20`, `-10`, `0`, `10`, `20`, and `30` dB), and 64 windows per modulation/SNR condition. Its verified shape is `X=[9216,1024,2]`, `Y=[9216,24]`, and `Z=[9216,1]`.

The local file SHA-256 is `40f5245781f4f983ab5c46a382e6010453160bfc5523f16829998e8a7ff0b0eb`. The mirrored full-file SHA-256 advertised by its storage pointer is `e3dd0bef66a3426959ee66a1709a8c0a95d4f8395d18aaf6f1214bdbc763bd38`. The subset is licensed CC BY-NC-SA 4.0; its license and machine-readable provenance are stored beside the ignored HDF5.

This compact subset is suitable for development and a clearly labeled starter baseline. It is not equivalent to evaluating all 2,555,904 source windows or all 26 SNR levels. The selection uses one seeded contiguous block per modulation/SNR condition; possible adjacency correlation remains a limitation until parent-waveform metadata or a stronger grouping protocol is available.
