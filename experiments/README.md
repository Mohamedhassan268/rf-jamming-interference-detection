# Experiment records

`results.csv` is an append-only index. Each completed run should also retain its resolved configuration, seed, UTC timestamp, Git commit (when available), model parameter count, training history, checkpoint, and metrics under a separate `runs/` directory.

Empty metric fields mean that an experiment has not produced that measurement; they must not be replaced by estimates. Tune against validation data and reserve final source and target test sets.

