option(USE_CROSS_COMPILE "set use cross compile or not" ON)
option(USE_GNU_GCC "set use GNU gcc compiler or not" OFF)
option(USE_LLVM_CLANG "set use LLVM clang compiler or not" OFF)
option(USE_DEBUG "set use debug information or not" OFF)
option(USE_DYNAMIC_LIBRARY "set use dynamic library or not" OFF)

# model-tools variable
option(USE_CAFFE "set use caffe model as input or not" ON)
option(USE_ONNX "set use onnx model as input or not" ON)
option(USE_TFLITE "set use tflite model as input or not" ON)

# blas-enhance tensor_computing
option(USE_NEON "set use ARM NEON instruction or not" ON)
option(USE_FP32 "set use ARM NEON FP32 instruction or not" ON)
option(USE_FP16 "set use ARM NEON FP16 instruction or not" ON)
option(USE_F16_MIX_PRECISION "set use ARM NEON mix precision f16/f32 instruction or not" ON)
option(USE_INT8 "set use ARM NEON INT8 instruction or not" ON)
option(BUILD_TEST "set to build unit test or not" OFF)
option(USE_OPENMP "set use OpenMP for parallel or not" ON)
option(USE_MALI "set use mali for parallel or not" ON)

set(BOLT_ROOT $ENV{BOLT_ROOT})

if (USE_CROSS_COMPILE)
    set(CMAKE_SYSTEM_NAME Linux)

    unset(cc_absolute_path)
    unset(cxx_absolute_path)
    unset(ar_absolute_path)

    if (USE_GNU_GCC)
        if (DEFINED cc_absolute_path)
            message(FATAL_ERROR "
FATAL: compiler is duplicated,
       please don't set both <USE_GNU_GCC> and <USE_LLVM_CLANG>.
            ")
        endif (DEFINED cc_absolute_path)
        exec_program("which aarch64-linux-gnu-gcc" OUTPUT_VARIABLE cc_absolute_path)
        exec_program("which aarch64-linux-gnu-g++" OUTPUT_VARIABLE cxx_absolute_path)
        exec_program("which aarch64-linux-gnu-ar" OUTPUT_VARIABLE ar_absolute_path)
    endif(USE_GNU_GCC)
    if (USE_LLVM_CLANG)
        if (DEFINED cc_absolute_path)
            message(FATAL_ERROR "
FATAL: compiler is duplicated,
       please don't set both <USE_GNU_GCC> and <USE_LLVM_CLANG>.
            ")
        endif (DEFINED cc_absolute_path)
        exec_program("which aarch64-linux-android21-clang" OUTPUT_VARIABLE cc_absolute_path)
        exec_program("which aarch64-linux-android21-clang++" OUTPUT_VARIABLE cxx_absolute_path)
        exec_program("which aarch64-linux-android-ar" OUTPUT_VARIABLE ar_absolute_path)
    endif(USE_LLVM_CLANG)

    if (NOT DEFINED cc_absolute_path)
        message(FATAL_ERROR "
FATAL: compiler is missing,
       please set <USE_GNU_GCC> or <USE_LLVM_CLANG>.
        ")
    endif(NOT DEFINED cc_absolute_path)
    set(CMAKE_C_COMPILER ${cc_absolute_path})
    set(CMAKE_CXX_COMPILER ${cxx_absolute_path})
endif(USE_CROSS_COMPILE)

function (set_policy)
    cmake_policy(SET CMP0074 NEW)
endfunction(set_policy)

macro (set_c_cxx_flags)
    set(COMMON_FLAGS "-Wall -Wextra -O3")
    
    if (BUILD_TEST)
        set(COMMON_FLAGS "${COMMON_FLAGS} -D_BUILD_TEST")
    endif(BUILD_TEST)

    if (USE_DEBUG)
        set(COMMON_FLAGS "${COMMON_FLAGS} -D_DEBUG")
    endif(USE_DEBUG)
    
    if (USE_MALI)
        set(COMMON_FLAGS "${COMMON_FLAGS} -D_USE_MALI")
    endif(USE_MALI)
    
    if (USE_NEON)
        set(COMMON_FLAGS "${COMMON_FLAGS} -D_USE_NEON")
    
        if (USE_FP32)
            set(COMMON_FLAGS "${COMMON_FLAGS} -D_USE_FP32")
        endif (USE_FP32)
    
        if (USE_FP16)
            set(COMMON_FLAGS "${COMMON_FLAGS} -D_USE_FP16")
            if (USE_F16_MIX_PRECISION)
                set(COMMON_FLAGS "${COMMON_FLAGS} -D_USE_F16_MIX_PRECISION")
            endif (USE_F16_MIX_PRECISION)
            if (USE_INT8)
                set(COMMON_FLAGS "${COMMON_FLAGS} -D_USE_INT8 -march=armv8.2-a+fp16+dotprod")
            else (USE_INT8)
                set(COMMON_FLAGS "${COMMON_FLAGS} -march=armv8.2-a+fp16")
            endif (USE_INT8)
        endif (USE_FP16)
    endif(USE_NEON)

    if (USE_CAFFE)
        set(COMMON_FLAGS "${COMMON_FLAGS} -D_USE_CAFFE_MODEL")
    endif()
    if (USE_ONNX)
        set(COMMON_FLAGS "${COMMON_FLAGS} -D_USE_ONNX_MODEL")
    endif()
    if (USE_TFLITE)
        set(COMMON_FLAGS "${COMMON_FLAGS} -D_USE_TFLITE_MODEL")
    endif()

    set(CMAKE_C_FLAGS   "${CMAKE_C_FLAGS} ${COMMON_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${COMMON_FLAGS} -std=c++17")

    if (USE_DEBUG)
        set(CMAKE_BUILD_TYPE "Debug")
    else (USE_DEBUG)
        set(CMAKE_BUILD_TYPE "MinSizeRel")
    endif (USE_DEBUG)
endmacro(set_c_cxx_flags)

macro (set_test_c_cxx_flags)
    if (${USE_DYNAMIC_LIBRARY} STREQUAL "OFF")
        if (USE_CROSS_COMPILE)
            if (USE_GNU_GCC)
                set(COMMON_FLAGS "${COMMON_FLAGS} -static")
            endif(USE_GNU_GCC)
        endif(USE_CROSS_COMPILE)
    endif(${USE_DYNAMIC_LIBRARY} STREQUAL "OFF")
    
    if (USE_LLVM_CLANG)
        if (${USE_DYNAMIC_LIBRARY} STREQUAL "OFF")
            set(COMMON_FLAGS "${COMMON_FLAGS} -Wl,-allow-shlib-undefined, -static-libstdc++")
        else (${USE_DYNAMIC_LIBRARY} STREQUAL "OFF")
            set(COMMON_FLAGS "${COMMON_FLAGS} -Wl,-allow-shlib-undefined")
        endif(${USE_DYNAMIC_LIBRARY} STREQUAL "OFF")
    endif(USE_LLVM_CLANG)

    set(CMAKE_C_FLAGS   "${CMAKE_C_FLAGS} ${COMMON_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${COMMON_FLAGS}")
endmacro (set_test_c_cxx_flags)

macro (set_project_install_directory)
    SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
    SET(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)
endmacro (set_project_install_directory)
