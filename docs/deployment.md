# Deployment

The current model is only a USRP-class deployment target. It has not been run in a USRP or other live SDR pipeline, and preprocessing, inference, end-to-end latency, throughput, compute use, and memory use have not been measured.

A future path is: SDR capture through UHD/GNU Radio, fixed I/Q buffering, the exact trained preprocessing pipeline, host-side PyTorch or ONNX inference, and output logging. Claims of live or real-time operation require an implemented pipeline and reproducible benchmarks.

