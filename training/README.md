# ![Raul](docs/raul_logo.png) Raul

![cmake](https://img.shields.io/badge/Cmake-3.11-blue?logo=CMake)
![c++](https://img.shields.io/badge/C++-17-blue?logo=c%2B%2B)
![c++](https://img.shields.io/badge/Android%20NDK-r22-blue)
![c++](https://img.shields.io/badge/Clang-11.0.0-blue)
![c++](https://img.shields.io/badge/GCC-9.2.1-blue)
![c++](https://img.shields.io/badge/Visual%20Studio-19-blue?logo=Visual%20Studio)
![MIT license](https://img.shields.io/badge/License-MIT-blue.svg)

**Cross-platform on-device training library for mobile and IoT devices**

Raul on-device training library is C++ based software designed to train complex neural network (NN) topologies using minimal external dependencies and minimize ROM/RAM footprint. The current implementation of Raul version is CPU-based with BLAS-compatible mathematical back-end used for intensive mathematical operations: matrix multiplication and element-wise vector operations.

**Features**

- CPU-based computations
- BLAS-compatible math back-end
- 90+ NN layers
- 9 NN optimization algorithms
    - lr schedulers
    - gradient clipping
    - regularization methods
- quantization-aware training
- 6 verified complex topologies (BERT, ResNet, Tacotron 2, NIN, MobileNet 2, MobileNet 3, Transformer)
- Memory efficiency strategies
- Gradient checkpoints 
- Workflows aka dynamic allocations 


## Usage

```cmake
cmake_minimum_required(VERSION 3.11)
project(sample)

add_subdirectory(raul)

add_executable(app main.cpp)
target_link_libraries(app PRIVATE Raul)

```

## Build

### Reqirements

- Build system
    - cmake 3.11+
- Compilers
    - Clang 11.0.0+
    - GCC 9.2.1+
    - Visual Studio 16 (2019)+
    - Android NDK r22+ (r21 without assets related tests)

### Host build

```sh
cmake -B build -S raul
cmake --build build --target Raul --parallel
```

This short example shows how to configure, generate a project for default build system and buld using `cmake`. Here, `raul` is a directory with repository root and `build` is an output build directory. All required dependencies will be downloaded; a connection must be established.

### Cross-platform Android build

Android NDK is required.

```sh
cmake -B build -S raul -DCMAKE_BUILD_TYPE=Release -DCMAKE_TOOLCHAIN_FILE=%path to android.toolchain.cmake% -DRAUL_CONFIG_BLAS_VENDOR=Huawei -DRAUL_CONFIG_ENABLE_OPENMP=ON -DANDROID_ABI=arm64-v8a -DANDROID_NATIVE_API_LEVEL=19 -DANDROID_STL=c++_static
cmake --build build --target Raul --parallel
```

[Read more information](docs/build.md)


