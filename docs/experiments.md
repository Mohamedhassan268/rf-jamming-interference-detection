# Experiment protocol

Every run should save the resolved YAML configuration, seed, UTC timestamp, Git commit when available, model architecture and parameter count, training history, checkpoint, and machine-readable metrics. The shared CSV is append-only.

Start with E0, the historically faithful baseline once recoverable. Then change one factor at a time: RMS normalization, DC removal, phase rotation, frequency offset, amplitude scaling, AWGN, combined augmentation, recovered architecture correction, and architecture plus the best validated augmentation. Do not label the provisional network E0 until historical fidelity is established.

Never use training accuracy as the result, repeatedly tune on a final test set, or fill missing measurements with estimates. Report negative findings and class imbalance.

The training CLI selects checkpoints using validation loss and deliberately leaves test metrics empty. Source evaluation reuses the checkpoint's exact test indices and verifies a fingerprint of the filtered source indices. Target evaluation applies the source-trained preprocessing and refuses capture labels that do not exactly match the checkpoint class mapping.
