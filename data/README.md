# Data policy and layout

Raw datasets are not versioned in this repository.

```text
data/raw/        authorized RadioML 2018.01A files
data/processed/  locally generated indexes/caches
data/captures/   session-based captured or shifted RF
```

Obtain RadioML 2018.01A from an authorized source and confirm its license before use or redistribution. The loader inspects candidate HDF5 datasets and requires an explicit selection if names are ambiguous.

Captured sessions follow:

```text
session_001/
  metadata.json
  samples.npy
  labels.csv
```

`samples.npy` must use `[N, 2, T]` or `[N, T, 2]`; the latter is explicitly transposed after validation. `labels.csv` must contain one row per sample and a `label` column. Additional columns and metadata keys are preserved. Copy `metadata.example.json` when documenting a new session, replacing only facts that are known.

Never label synthetic samples as real-world captures. Publication permission for captured RF is `TBD`.

