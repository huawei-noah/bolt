# bolt build guide

## Prerequisites

### image project prerequisites

#### CImg

Download the CImg header file from [Github](https://github.com/dtschump/CImg), and set the **CImg_ROOT** shell enviroment variable.

```bash
export CImg_ROOT=<directory>
```

Please make sure you have installed **png, jpeg, X11, Xau, xcb, zlib** libraries.

##### direct compile

to be implemented

##### cross compile

You can refer the [dependency/build_dependency.sh]() in image project. You need to set these shell environment variables after install.

```bash
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

Please make sure you have install **protobuf** library and **protoc** exe.

#### direct compile

to be implemented

#### cross compile

You can refer the [dependency/build_dependency.sh]() in model-tools project. You need to set these shell environment variables after installation.

```bash
export PROTOC_ROOT=<directory>
export Protobuf_ROOT=<directory>

export PATH=${PROTOC_ROOT}/bin:$PATH
export PATH=${Protobuf_ROOT}/bin:$PATH
export LD_LIBRARY_PATH=${Protobuf_ROOT}/lib:$LD_LIBRARY_PATH

```

## Download and Build

### Download

```bash
mkdir bolt
cd bolt
export BOLT_ROOT=`PWD`

git clone https://gitlab.huawei.com/ee/universal/uni
git clone https://gitlab.huawei.com/ee/deployment/model-tools
git clone https://gitlab.huawei.com/ee/computing/image
git clone https://gitlab.huawei.com/ee/computing/blas-enhance
git clone https://gitlab.huawei.com/ee/computing/tensor_computing
git clone https://gitlab.huawei.com/ee/deployment/engine
```

**please checkout to specific branch after download.**

## Configure

Please add these files.

### bolt.cmake

content

```bash
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
### CMakeLists.txt

content

```bash
cmake_minimum_required(VERSION 3.2)

file(GLOB BOLT_CONFIGURE_FILE $ENV{BOLT_ROOT}/bolt.cmake ${BOLT_ROOT}/bolt.cmake)
if (BOLT_CONFIGURE_FILE)
    include(${BOLT_CONFIGURE_FILE})
endif (BOLT_CONFIGURE_FILE)


project(bolt C CXX)


add_subdirectory(blas-enhance)
add_subdirectory(model-tools)
add_subdirectory(tensor_computing)
add_subdirectory(image)
add_subdirectory(engine)

add_dependencies(tensor_computing blas-enhance)
add_dependencies(tensor_computing_static blas-enhance_static)
add_dependencies(model-tools model-tools_caffe)
add_dependencies(model-tools_static model-tools_caffe_static)
add_dependencies(engine tensor_computing model-tools image)
add_dependencies(engine_static tensor_computing_static model-tools_static image_static)

install(TARGETS blas-enhance blas-enhance_static tensor_computing tensor_computing_static
                model-tools model-tools_static model-tools_caffe model-tools_caffe_static
                image image_static
                engine engine_static
                test_completeUT
                classification
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib)
```

### Build

```bash
mkdir build
cd build
cmake .. -DCMAKE_INSTALL_PREFIX=${BOLT_ROOT}
make -j8
make install
```

