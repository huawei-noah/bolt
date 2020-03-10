# Introduction

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Bolt is a light-weight library for mobile devices. Bolt, as a universal deployment tool for all kinds of neural networks, aims to minimize the inference runtime as much as possible. Higher speed, better security and more efficient memory management are the advantages that Bolt strives to provide. Feel free to make good use of issue submission, or join our QQ chatroom (Chinese): 833345709.

# Features

- ### Overview

  Bolt is highly optimized for ARMv8.2 CPUs, supporting fast inference of FP16, INT8 and BNN networks. Recently, FP32 functionality has been integrated, which also works on ARMv8 devices.
  
  Bolt has its own format of model storage, which helps reduce the memory footprint by storing in FP16 and 1-bit representations when possible. We provide model converters for the following formats:
  
  - caffe
  - onnx
  - tflite
  
  For PyTorch and TensorFlow models, please try to convert them to the onnx format first. We also had some success in converting these models into customized caffe models.
  
- ### Verified Networks

  Bolt has shown its high performance in the inference of common CV and NLP neural networks. Some of the representative networks that we have verified are listed below. You can find detailed benchmark information in [docs/BENCHMARK.md](docs/BENCHMARK.md).
  
  - Squeezenet (full-network int8 quantization)
  - Mobilenet v1 - v3
  - Resnet50, [Ghostnet](https://github.com/huawei-noah/ghostnet) (plus FPN detection)
  - Birealnet18 (BNN)
  - Bert, TinyBert, Albert
  - Neural Machine Translation

- ### Inference Graph Optimizers

  Apart from the refined acceleration of convolutions and GeMM for the supported data precisions, Bolt has a sophisticated inference graph optimizer. As shown in [model-tools/include](model-tools/include), classic operator fusion is supported. Bolt is also equipped with a Memory Reuse Optmizer, which reassigns the space occupied by a feature map as soon as it is no longer needed as input or output. Most networks that we tested benefit from a two-third reduction in feature map storage.

- ### Thread Affinity Setting

  Users can specify the preferred policy (high-performance or low-power). Bolt will select the most suitable core and set the thread affinity.

- ### Algorithm Tuning

  Bolt can tailor-make the algorithm configuration for your specific target device.

# Documentation

- ### Installation

Bolt provides [install.sh](install.sh) for fast installation. The major third-party dependency is protobuf, and some other may come from the original model format that you want to use. You may also need libjpeg for building [tests/classification](tests).

After configuring [bolt.cmake](bolt.cmake), the compilation can be as simple as:

```shell
./install.sh -t 48 -c llvm
```

For more details, please refer to [docs/INSTALL.md](docs/INSTALL.md)

- ### User Guide

As a user, what you are normally concerned about include the following 4 parts:

- API (We guarantee that the C API will not be changed in the future)
- Model Preparation
- Model Conversion
- Model Inference

For the details, please refer to [docs/USER_HANDBOOK.md](docs/USER_HANDBOOK.md)

- ### Developer Guide

  We welcome all kinds of contribution. Before that, let's get familiar with the project structure.

- ##### Project Structure

  - [uni](uni) hosts the common headers that are used in the project.
  - [gcl](gcl) hosts the setup of MALI GPU environment.
  - [image](image) hosts common preprocessing routines for image inputs (e.g. bilinear interpolation).
  - [blas-enhance](blas-enhance) hosts the fast implementation of matrix-matrix multiplication and matrix-vector multiplication of FP32, FP16 and INT8. It is referenced by some of the operators in [tensor_computing](tensor_computing).
  - [tensor_computing](tensor_computing) hosts the implementation for individual operators.
  - [model-tools](model-tools) hosts everything related to model conversion and optimization.
  - [inference](inference) hosts the inference engine of neural networks.
  - Lastly, [tests](tests) include all the unit tests for the above functionalities.

  To support your own network, you can first try to convert it with the provided tools. If an operator is missing, you can first add the conversion to [model-tools](model-tools). You may then implement the missing computation routine in [tensor_computing](tensor_computing). Please also define a class for your new operator in [inference](inference).

- ##### Contribution

All contributions are welcomed. For the details, please refer to [docs/DEVELOPER.md](docs/DEVELOPER.md) 

- ### Benchmark

We provide a detailed benchmark report for your reference. For more testing information please refer to [docs/BENCHMARK.md](docs/BENCHMARK.md) .

# Road Map

#### v0.3.0

Future Release 2020-04-01

- GPU

# Who are using Bolt

- HUAWEI CBG
- HUAWEI PORTFOLIO

# FAQ

1. More details about dependency libraries for cross-compilation?

   The major dependency is Protobuf. Protoc should be the x86 version but protbuf should be the ARM version.

2. Requirements on tensor dimensions?

   For optimal performance, Bolt requires the number of output channels to be divisible by 8. For compatibility, Bolt will try to pad the output channels of convolution layers to the nearest multiple of 8. You can turn on DEBUG in [bolt.cmake](bolt.cmake) to check the actual dimensions.

3. Restrictions for BNN?

   For BNN convolution layers, the number of output channels must be divisible by 32.

4. Restrictions on quantization (int8)?

   For the time being, Bolt only supports post-training int8 quantization. If quantization is activated, the second convolution layer will quantize the tensors to 8-bit integers. For now, int8 operators include Convolution, Pooling and Concatenation (end-to-end support for Squeezenet). If your network includes other operators, you may need to add type casting in the front of those operators. The quantization method is symmetrical for both activation and weight.

5. Requirements for fp16 and int8?

   Only arm-v8.2 supports fp16 and int8 dotprod instructions. 

6. Restrictions for MALI?

   Only llvm compilation supports MALI computing.

# Acknowledgement

Bolt refers to the following projects: [caffe](https://github.com/BVLC/caffe), [onnx](https://github.com/onnx/onnx), [protobuf](https://github.com/protocolbuffers/protobuf), [flatbuffers](https://github.com/google/flatbuffers), [ncnn](https://github.com/Tencent/ncnn), [mnn](https://github.com/alibaba/MNN), [dabnn](https://github.com/JDAI-CV/dabnn).

# License

The MIT License(MIT)
