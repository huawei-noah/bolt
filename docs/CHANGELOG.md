# Changelog
---

All notable changes to the Bolt project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),

and this project adheres to [Semantic Versioning](<https://semver.org/spec/v2.0.0.html>).

### [1.5.0] - 2023-2-23

#### Added

- Support Python API 
- Support AVX-VNNI and ARMv9 instruction set
- Support Intel Desktop GPU (float16 and float32)
- Support Windows on arm platform
- Support more operators : Random, Sin, Cos, Einsum, Elu, UnPooling, Flatten, ConvertColor, BilateralSliceApply, Lut
- Support more networks : ViTAE, CMT, EfficientFormer, ConvTT, Wenet, NFM, AFM, ONN, wide&deep, DeepFM, MMOE, etc
- Improve multi-threads parallel inference performance on CPU
- Add simple chinese deployment guide
- Support model file compatibility
- Support using outer memory(CPU array or OpenCL cl_mem) by using SetInputOutput API
- Support data type and format transform by using C API

#### Changed

- TensorDesc's dim array is changed to 20.
- Remove __FILE__ macro usage and warning log under release mode
- change enum data and operator parameter size

#### Fixed

- Fix GPU resize bug
- Fix GPU concurrent inference bug
- Fix ONNX converter bug
- Add missed chinese automatic speech recognition model


### [1.3.0] - 2022-2-28

#### Added

- Support on-device training for MLP, CNN(lenet, resnet50, mobilnetv1), transformer/bert(text to speech)
- Support change model input and output names in X2bolt
- Support more graph optimizations : Transpose+Convolution, Swish, Quantization, Power+Scale
- Support dynamic output related operators : Shape, ConstantOfShape, GenerateProposals, NonZero, NonMaxSuppression, Reshape, etc
- Support more operators : GridSample, CumSum, OneHot, Round, Floor, Ceil
- Support more networks on CPU : yolov2, yolov3, yolov4, yolov5, faster-rcnn, mask-rcnn, retinanet, dfsmn, frill, conformer, unet, etc
- Support Armv8 int8 to accelerate NLP network
- Improve inference performance on avx2 CPU
- Support netron to visualize bolt model
- Support not to bind CPU core
- Add C API MemoryCheck to check bolt memory leak

#### Changed

- X2bolt add -I and -O options to change model input and output names.
- X2bolt add -t option to convert model for on-device training.
- C API CreateModel and AllocAllResultHandle return value is set to NULL when unsuccessful.
- install.sh add --neon option to close arm neon acceleration on old platform.
- some operator parameter defination

#### Fixed

- Fix GPU depth2space and deconv bug
- Fix GPU preprocess tool on armv8 platform bug
- Fix x86 Sigmoid precision
- Fix C API CloneResultHandle bug
- Fix mobilnetv1 int8 inference
- Fix Java API build bug on Windows
- Fix ONNX converter deconv, pooling parameter bug

#### Removed

- Equal operator is replaced with Check.


### [1.2.1] - 2021-9-11

#### Added

- Support more graph optimizations : Convolution+Convolution, LayerNorm
- Support more operators : ROIAlign, GenerateProposals, Reciprocal, Not, Log, ReductionL2, InstanceNorm, Expand, Gather, Scatter
- Support more operators(PReLU) process NCHW input data.
- Support ONNX share weight between Linear, MatMul, Gemm and Gather
- Support more networks on CPU : vision transformer(ViT, TNT), recommendation networks
- Support more networks on GPU : ASR, Faster_RCNN
- Support Armv7 int8 to accelerate NLP network(50%+ speed-up)
- Support X86 AVX512 int8 to accelerate NLP network(3x+ speed-up)
- Support using image on Qualcomm GPU, add GPU image manage methods
- Improve inference performance on Qualcomm GPU
- Add more kit android/iOS demos : [Chinese ASR](../kit/Android/ChineseSpeechRecognition), [Face Detection](../kit/Android/FaceDetection), [Sentiment Analysis](../kit/Android/Semantics)
- Try to bind core when using GPU

#### Changed

- Replace *mali* option with *gpu* in install shell script, and remove default target option setting
- Change data format *NCWHC4* TO *NCHWC4* for GPU
- Simplified tensor padding method with *OclMemory* for GPU
- Tool [preprocess_ocl](../inference/engine/tools/preprocess_ocl) produces algofile and *xxxlib.so* before, for now algofile has been packaged into this *xxxlib.so*
- Add *BNN_FP16* option in X2bolt tool to convert ONNX 1-bit model
- Replace original *INT8* option with *INT8_FP16* in post_training_quantization tool to convert int8+float16 hybrid inference model, and add *INT8_FP32* option to convert int8+float32 hybrid inference model.
- Add shell environment variable *BOLT_INT8_STORAGE_ERROR_THRESHOLD* to control post_training_quantization convert int8 model, default value is 0.002. post_training_quantization will use int8 storage when when quantization error lower than BOLT_INT8_STORAGE_ERROR_THRESHOLD.

#### Fixed

- Fix PReLU 2d, 3d support
- Fix Resize bug on some mode
- Fix ONNX converter read Squeeze, UnSqueeze, Deconv parameter bug
- Fix Arm Sigmoid precision
- Fix ONNX RNN optimizer, and add support for NCHWC8 input data
- Fix Concat with weight tensor in onnx converter
- Simplify C API example

#### Removed

- 


### [1.2.0] - 2021-3-15

#### Added

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

#### Changed

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

#### Fixed

- Fix length of op/tensor name > 128 not-supporting bug
- Fix Caffe input dims extraction bug
- Fix Concat with single input in onnx converter
- Fix padding(nhwc) not-supporting bug
- Fix relu6 insertion in tflite converter 
- Fix GRU, LSTM LBR_GRU model converter and inference bug
- Fix X86 convolution, fully connected operators inference bug

#### Removed

- Remove third party library FFTW and using FFTS for ASR example


### [1.0.0] - 2020-11-20

#### Added

- Support fp32 on X86 AVX2 CPU
- Support partial fp32 operator(convolution, lstm) multi-threads parallel
- Support Tensorflow model
- Support more networks(Pointnet, ...)
- Support more networks int8 inference(TinyBert, NMT, ASR)
- Support time-series data acceleration
- Support Apple IOS phone


### [0.3.0] - 2020-06-01

#### Added

- Optimized fp16 on ARM MALI GPU
- Support fp32 on ARMv7 CPU
- Support int8 PTQ calibration
- Support more networks(SSD, ASR, TTS)
- Support image classification task on ARM MALI GPU


### [0.2.0] - 2020-03-06

#### Added

- Support fp32 on ARMv8 CPU
- Support fp16 on ARM MALI GPU
- Support memory reuse for feature maps and weight-sharing between operators
- Support dynamic input size
- Support CPU affinity setting
- Support convolution algorithm auto-tuning (runtime or full parameter space search)
- Support Java and C API


### [0.1.0] - 2019-12-01

#### Added

- Support Caffe/ ONNX/ Tflite
- Support fp16/int8/binary
- Support Sequential/CNN/LSTM (common models of CV and NLP)
