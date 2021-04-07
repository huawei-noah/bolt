# Changelog

All notable changes to the Bolt project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),

and this project adheres to [Semantic Versioning](<https://semver.org/spec/v2.0.0.html>).


## [1.2.0] - 2021-3-15

### Added

- Support x86 compilation and cross-compialtion for ios/android on MacOs
- Support x86 compilation and cross-compilation for android on Windows
- Support MTK armv7 cross compilation toolchains on Linux by using linux-armv7_blank target
- Add Gitbook for user reference
- Support image nearest Resize and align_corners Resize
- Support more graph optimizations : Transpose+Concat+Transpose, Padding+Transpose, HardSwish-Fusion, Relu6-Fusion, Resize-Fusion, SwapTransposeEltwise, SwapPadTranspose, Convolution+Eltwise, Transpose+Matmul
- Support more operators: 3D-convolution, Where, SoftPlus, Exp, Split, Tdnn, Dropout, TopK, SpaceToBatchNd, BatchToSpaceNd, Abs, Equal, Sign, Resize(more mode)
- Support more networks on CPU: Reactnet, Tdnn, ShuffleNet, DenseNet, Hrnet, Efficientnet, Noah KWS2.0
- Support more networks on mali GPU : TinyBert, nmt
- Add more kit android/iOS demos : [Simple-Image-Classification](../kit/Android/SimpleImageClassification), [Image-SuperResolution](../kit/Android/CameraEnlarge), [Image-Classification](../kit/Android/SimpleImageClassification)
- Support float16, int8 model storage on any hardware
- Add Flow Java API

### Changed

- Change install, GPU library process shell script
- Optimize TfSlice with 75%+ speed-up on cpu
- Optimize Concat with 50%+ speed-up on cpu
- Optimize Deconvolution with 10%+ speed-up on cpu
- Optimize YoloDetection network with 15%+ speed-up on cpu
- Optimize resnet50 from 90ms+ to 70ms+ on x86, faster than openvino
- Optimize mobilenet v1/v2 with 10%+ speed-up on x86
- Optimize tts-melgan network from 200ms+ to 160ms on x86
- Optimize model read time
- Change Java API package name and use com.huawei.noah, split single API file to 6 files.

### Fixed

- Fix length of op/tensor name > 128 not-supporting bug
- Fix Caffe input dims extraction bug
- Fix Concat with single input in onnx converter
- Fix padding(nhwc) not-supporting bug
- Fix relu6 insertion in tflite converter 
- Fix GRU, LSTM LBR_GRU model converter and inference bug
- Fix X86 convolution, fully connected operators inference bug

### Removed

- Remove third party library FFTW and using FFTS for ASR example


## [1.0.0] - 2020-11-20

### Added

- Support fp32 on X86 AVX2 CPU
- Support partial fp32 operator(convolution, lstm) multi-threads parallel
- Support Tensorflow model
- Support more networks(Pointnet, ...)
- Support more networks int8 inference(TinyBert, NMT, ASR)
- Support time-series data acceleration
- Support Apple IOS phone


## [0.3.0] - 2020-06-01

### Added

- Optimized fp16 on ARM MALI GPU
- Support fp32 on ARMv7 CPU
- Support int8 PTQ calibration
- Support more networks(SSD, ASR, TTS)
- Support image classification task on ARM MALI GPU


## [0.2.0] - 2020-03-06

### Added

- Support fp32 on ARMv8 CPU
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
