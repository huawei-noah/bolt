option(USE_CROSS_COMPILE "set use cross compile or not" OFF)
option(USE_DEBUG "set use debug information or not" OFF)
option(USE_DYNAMIC_LIBRARY "set use dynamic library or not" OFF)

# model-tools variable
option(USE_CAFFE "set use caffe model as input or not" ON)
option(USE_ONNX "set use onnx model as input or not" ON)

# blas-enhance tensor_computing
option(USE_NEON "set use ARM NEON FP16 instruction or not" ON)
option(USE_INT8 "set use ARM NEON INT8 instruction or not" ON)
option(BUILD_TEST "set to build unit test or not" ON)


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