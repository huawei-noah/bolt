# Bolt build guide

## Prerequisites

### cmake

We use [cmake](https://cmake.org/) and [GNU make](https://www.gnu.org/software/make/) to build project.We have tested it on cmake v3.15.1 and GNU make v3.81. If your software is not the latest version, you may download the latest software or modify the bolt.cmake below as well as the releated CMakeLists.txt

### compiler

We use ARM 64bit compiler to compiler the project. You can compile the project on ARM development board by using direct compilation. You can also compile the project on X86 server by using crossing compilation and push the program to the android phone by using [ADB](https://developer.android.google.cn/studio/command-line/adb) tool. If you want to use cross compilation, you can download cross compiler from https://developer.arm.com/tools-and-software/open-source-software/developer-tools/gnu-toolchain/gnu-a/downloads, we use gcc-arm-8.3-2019.03-x86_64-aarch64-linux-gnu.

### image project prerequisites

#### CImg

Download the CImg header file from [Github](https://github.com/dtschump/CImg), and set the **CImg_ROOT** shell enviroment variable.

```
export CImg_ROOT=<directory>
```

Please make sure you have installed **png, jpeg, X11, Xau, xcb, zlib** libraries. If you use direct compilation, you can install dependency library by using apt-get or yum. If you use cross compilation, you can refer the [build_dependency.sh](https://github.com/huawei-noah/bolt/blob/master/image/dependency/build_dependency.sh) in image project. You need to set these shell environment variables after installation.

```
export ZLIB_ROOT=<directory>
export XAU_ROOT=<directory>
export XCB_PROTO_ROOT=<directory>
export XCB_ROOT=<directory>
export X11_ROOT=<directory>
export PNG_ROOT=<directory>
export JPEG_ROOT=<directory>

export LD_LIBRARY_PATH=${ZLIB_ROOT}/lib:${XAU_ROOT}/lib:${XCB_PROTO_ROOT}/lib:${XCB_ROOT}/lib:${X11_ROOT}/lib:${PNG_ROOT}/lib:${JPEG_ROOT}/lib:$LD_LIBRARY_PATH
```

### model-tools project prerequisites

Please make sure you have installed **protobuf** library and **protoc** program. If you use direct compilation, you can install dependency library by using apt-get or yum. If you use cross compilation, you can refer the [build_dependency.sh](https://github.com/huawei-noah/bolt/blob/master/model-tools/dependency/build_dependency.sh) in model-tools project. You need to set these shell environment variables after installation.

```
export PROTOC_ROOT=<directory>
export Protobuf_ROOT=<directory>

export PATH=${PROTOC_ROOT}/bin:$PATH
export PATH=${Protobuf_ROOT}/bin:$PATH
export LD_LIBRARY_PATH=${Protobuf_ROOT}/lib:$LD_LIBRARY_PATH
```

## Download and Build

### Download

```
git clone https://github.com/huawei-noah/bolt.git
cd bolt
export BOLT_ROOT=`PWD`
```

**please checkout to specific branch after download.**

## Configure

Please modify these configure files according to your environment.

### bolt.cmake

content

```
option(USE_CROSS_COMPILE "set use cross compile or not" ON)
option(USE_DEBUG "set use debug information or not" OFF)
option(USE_DYNAMIC_LIBRARY "set use dynamic library or not" OFF)

# model-tools variable
option(USE_CAFFE "set use caffe model as input or not" ON)
option(USE_ONNX "set use onnx model as input or not" ON)

# blas-enhance tensor_computing
option(USE_NEON "set use ARM NEON FP16 instruction or not" ON)
option(USE_INT8 "set use ARM NEON INT8 instruction or not" ON)


if (USE_CROSS_COMPILE)
    set(CMAKE_SYSTEM_NAME Linux)
    exec_program("which aarch64-linux-gnu-gcc" OUTPUT_VARIABLE aarch64-linux-gnu-gcc_absolute_path)
    exec_program("which aarch64-linux-gnu-g++" OUTPUT_VARIABLE aarch64-linux-gnu-g++_absolute_path)
    set(CMAKE_C_COMPILER ${aarch64-linux-gnu-gcc_absolute_path})
    set(CMAKE_CXX_COMPILER ${aarch64-linux-gnu-g++_absolute_path})
endif(USE_CROSS_COMPILE)

function (set_policy)
    cmake_policy(SET CMP0074 NEW)
endfunction(set_policy)
```

### Build

```
mkdir build
cd build
cmake .. -DCMAKE_INSTALL_PREFIX=${BOLT_ROOT}
make -j8
make install
```

