# Prerequisites

- CMake

  We use [cmake v3.15.1](https://cmake.org/files/v3.15/cmake-3.15.1-Linux-x86_64.tar.gz) to build Bolt. After installing the cmake, you need to set shell environment **PATH** to find it. You can use this simple test to confirm you have installed it successfully.
  
  ```shell
  cmake -version
  ```

- GNU make

  We use [GNU make v3.81](http://ftp.gnu.org/gnu/make/make-3.81.tar.gz) to build Bolt. After installing the make, you also need to set shell environment **PATH**. Simple test:
  
  ```shell
  make -version
  ```

- Cross compiler

    If you plan to directly compile Bolt on ARM platform and run on ARM, you can use gcc and skip this section.

    NDK compiler uses Android NDK toolchains to build Bolt for Java APIs required by Android applications. GNU compiler uses gcc to build Bolt for simple tests. Please choose according to your scenario.
    
    - Android NDK compiler
      
        We use Android NDK [android-ndk-r20](https://dl.google.com/android/repository/android-ndk-r20b-linux-x86_64.zip?hl=zh-cn) to build Bolt. After installing the Android NDK, you need to set shell environment **PATH** to find *aarch64-linux-android21-clang++*. Simple test:
        
        ```shell
        aarch64-linux-android21-clang++ --version
        ```
    
    - GNU compiler
      
        We use GNU compiler [gcc-arm-8.3-2019.03-x86_64-aarch64-linux-gnu](https://developer.arm.com/-/media/Files/downloads/gnu-a/8.3-2019.03/binrel/gcc-arm-8.3-2019.03-x86_64-aarch64-linux-gnu.tar.xz?revision=2e88a73f-d233-4f96-b1f4-d8b36e9bb0b9&la=en&hash=167687FADA00B73D20EED2A67D0939A197504ACD) to build Bolt. You need to set shell environment **PATH** to find *aarch64-linux-gnu-g++*. Simple test:
        
        ```shell
        aarch64-linux-gnu-g++ --version
        ```
    
- ADB

  We use [ADB](https://developer.android.com/studio/command-line/adb.html) tool to transfer the executables to android mobile phones and run the program. You also need to set shell environment **PATH**. Simple test:
  
  ```shell
  # this will list all available android devices
  adb devives
  ```

- Optional
    - Java SDK
    
        If you want to compile Java programs, you need to download and install [Java SE SDK](https://www.oracle.com/java/technologies/oracle-java-archive-downloads.html). After installing the SDK, you need to set shell environment **PATH** to find it. Simple test:
        ```shell
        java -version
        ```
        
    - Android dx
    
        If you want to directly run *jar* file on Android device, you can use [Android dx tool](https://developer.android.com/studio/releases/build-tools). We currently use Android *v28.0.3* build tools. After installing the *dx* tool, you also need to set shell environment **PATH**. Simple test:
        ```shell
        dx --version
        ```
    
- Third party library

  We provide a simple [install shell script](../third_party/install.sh) to install third party libraries(*protoc, protobuf, flatbuffers, tensorflow-lite, jpeg, ARM GPU OpenCL*) to the [third_party](third_party) directory and generate a shell script to set up compilation environment. You can choose between LLVM and GCC. Here is an example of installation for LLVM.

  ```shell
  ./third_party/install.sh -c llvm -t 33
  ```

# Download and Build Bolt

We provide a simple shell script [install.sh](../install.sh) to build and install the Bolt library, and you can modify it according to your scenario and environment. Please refer to the options section of [bolt.cmake](../bolt.cmake) and configure accordingly. Here we give an example of building Bolt with LLVM.

```shell
git clone https://github.com/huawei-noah/bolt.git
cd bolt
./install.sh -c llvm -t 33
```

We will install Bolt to *install_llvm* directory, you will find these subdirectories in it.

- bin
    - *tinybert* for intention identification
        
    - *nmt* for machine translation
    
    - *classification* for computer vision classification task
    
    
- include
    - C API
    - Java API

- lib: all static and shared library
- tools
    - *caffe2bolt* for converting caffe model to bolt model
    
    - *onnx2bolt* for converting onnx model to bolt model
    
    - *tflite2bolt* for converting tflite model to bolt model
    
    - *tensorflow2caffe* for converting tensorflow model to caffe model

    - *pytorch2caffe* for converting pytorch model to caffe model

    - *tensor_computing_library_search* for performance tuning of the operator library
      
If you want to build operator and API tests, please turn on the *BUILD_TEST* option and rebuild Bolt. These programs will be installed to *tests/bin* directory.

## Options

Here we list all options in [bolt.cmake](../bolt.cmake).

| options               | default | note                                          |
| --------------------- | ------- | --------------------------------------------- |
| USE_CROSS_COMPILE     | OFF     | use cross compile or not                      |
| USE_GNU_GCC           | OFF     | use GNU gcc compler or not                    |
| USE_LLVM_CLANG        | OFF     | use LLVM clang compiler or not                |
| USE_DEBUG             | OFF     | use debug information or not                  |
| USE_DYNAMIC_LIBRARY   | OFF     | use dynamic library or not                    |
| USE_CAFFE             | ON      | use caffe model as input or not               |
| USE_ONNX              | ON      | use onnx model as input or not                |
| USE_TFLITE            | ON      | use tflite model as input or not              |
| USE_NEON              | ON      | use ARM NEON instruction or not               |
| USE_FP32              | OFF     | use FP32 implementation or not                |
| USE_FP16              | ON      | use FP16 implementation or not                |
| USE_F16_MIX_PRECISION | ON      | use ARM NEON mixed-precision (F16/F32) or not |
| USE_INT8              | ON      | use INT8 implementation or not                |
| BUILD_TEST            | OFF     | build unit test or not                        |
| USE_OPENMP            | ON      | use OPENMP for parallel or not                |
| USE_MALI              | ON      | use MALI GPU for parallel or not              |


## Environment variables

We reserve some shell environment variable for Bolt.

- *Bolt_ROOT*: Bolt project home directory, set by user or Bolt.
- *Bolt_TensorComputing_LibraryAlgoritmMap*: a path on the target device set by user to save tensor_computing library performance tuning result.
