# Domain-shift protocol

Train only on the declared source training split. Select preprocessing, augmentation, and architecture using source validation data or a separately declared target-development set. Measure the untouched source-trained model on the held target test data before any target-domain fine-tuning.

Compare source and target accuracy, macro F1, weighted F1, confusion matrices, and confidence/calibration when label spaces genuinely correspond. Where metadata permits, stratify by SNR and capture session.

Descriptive diagnostics may compare I/Q amplitude, magnitude, phase, power, PSD, and embeddings. Such plots can suggest mismatch mechanisms but do not by themselves establish causality. Hardware impairment, channel, noise, sampling, and label-definition hypotheses require controlled tests.

No source-target metrics or successful adaptation are currently available.

