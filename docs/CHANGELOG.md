# Changelog

All notable changes to the Bolt project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),

and this project adheres to [Semantic Versioning](<https://semver.org/spec/v2.0.0.html>).



## [0.2.0] - 2020-03-06

### Added

- Support fp32 on ARM CPU
- Support fp16 on ARM MALI GPU
- Support memory reuse for feature maps and weight-sharing between operators
- Support dynamic input size
- Support CPU affinity setting
- Support convolution algorithm auto-tuning (runtime or full parameter space search)
- Support Java and C API


## [0.1.0] - 2019-12-01

### Added

- Support Caffe/ ONNX/ Tflite
- Support fp16/int8/binary
- Support Sequential/CNN/LSTM (common models of CV and NLP)
