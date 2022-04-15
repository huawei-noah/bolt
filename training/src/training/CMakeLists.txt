# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

cmake_minimum_required(VERSION 3.10...3.20 FATAL_ERROR)

############################
##  OpenCL interface
############################
# Deprecated (move to separate opencl file)
if (RAUL_CONFIG_ENABLE_OPENCL)
    set(RAUL_CONFIG_OPENCL_VERSION "210" CACHE STRING "Set OpenCL target version")
    set_property(CACHE RAUL_CONFIG_OPENCL_VERSION PROPERTY STRINGS "100;110;120;200;210;220;300")
endif ()

############################
##  BLAS dev interface
############################
set(RAUL_CONFIG_BLAS_VENDOR "None" CACHE STRING "Set BLAS vendor")
set_property(CACHE RAUL_CONFIG_BLAS_VENDOR PROPERTY STRINGS "None;Custom;OpenBLAS;Huawei;OpenBLAS (Internal);Auto")

if (RAUL_CONFIG_BLAS_VENDOR STREQUAL "OpenBLAS")
    set(BLA_VENDOR OpenBLAS)
endif ()

if (RAUL_CONFIG_BLAS_VENDOR STREQUAL "Huawei")
    set(BLA_VENDOR Huawei)
    set(CMAKE_LIBRARY_PATH ${CMAKE_SOURCE_DIR}/src/thirdParty/blas-enhance/lib/android)
    set(CMAKE_INCLUDE_PATH ${CMAKE_SOURCE_DIR}/src/thirdParty/blas-enhance/include/android)
endif ()
if (RAUL_CONFIG_BLAS_VENDOR STREQUAL "OpenBLAS (Internal)")
    if (NOT RAUL_CONFIG_ENABLE_OPENMP AND NOT MSVC)
        message(STATUS "OpenBLAS (Internal): set RAUL_CONFIG_ENABLE_OPENMP=ON")
        set(RAUL_CONFIG_ENABLE_OPENMP ON CACHE BOOL "" FORCE)
    endif ()
    if (MSVC)
        if ("${CMAKE_VS_PLATFORM_NAME}" STREQUAL "Win32")
            set(BLAS_LIBRARIES ${CMAKE_SOURCE_DIR}/src/thirdParty/openblas/lib/msvc/x32/openblas.lib)
        else ()
            set(BLAS_LIBRARIES ${CMAKE_SOURCE_DIR}/src/thirdParty/openblas/lib/msvc/x64/openblas.lib)
        endif ()
        set(BLAS_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/src/thirdParty/openblas/include/host)
    elseif (MINGW)
        set(BLAS_LIBRARIES ${CMAKE_SOURCE_DIR}/src/thirdParty/openblas/lib/mingw/libopenblas.a)
        set(BLAS_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/src/thirdParty/openblas/include/host)
    elseif (ANDROID)
        set(BLAS_LIBRARIES ${CMAKE_SOURCE_DIR}/src/thirdParty/openblas/lib/android/libopenblas.so)
        set(BLAS_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/src/thirdParty/openblas/include/android)
    elseif (UNIX)
        set(BLAS_LIBRARIES ${CMAKE_SOURCE_DIR}/src/thirdParty/openblas/lib/linux/libopenblas.a)
        set(BLAS_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/src/thirdParty/openblas/include/host)
    else ()
        message(WARNING "OpenBLAS (Internal): unknown platform")
    endif ()
endif ()

if (RAUL_CONFIG_BLAS_VENDOR STREQUAL "Custom")
    set(BLAS_LIBRARIES "" CACHE FILEPATH "Path to BLAS library file")
    set(BLAS_INCLUDE_DIR "" CACHE PATH "Path to BLAS library include directory")
endif ()

############################
##  Modules
############################
add_subdirectory(external/yato)

if (RAUL_CONFIG_ENABLE_OPENMP)
    include(external/openmp/CMakeLists.txt)
endif ()

if (NOT RAUL_CONFIG_BLAS_VENDOR STREQUAL "None")
    include(external/blas/CMakeLists.txt)
endif ()

############################
##  Library sources
############################
add_library(Raul SHARED)
include(sources.cmake)

############################
##  Version
############################
include(cmake/git.cmake)
configure_file(
        training/version/Version.cpp.in
        ${CMAKE_CURRENT_BINARY_DIR}/training/version/Version.cpp
)
target_sources(Raul PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/training/version/Version.cpp)

############################
##  OpenCL Kernel Splitting (deprecated)
############################
if (RAUL_CONFIG_ENABLE_OPENCL)
    include(cmake/kernels.cmake)
    split_kernels(training/opencl/kernels/kernel_def.h 15000)
    configure_file(
            training/opencl/kernels/kernel_chunks.h.in
            ${CMAKE_CURRENT_BINARY_DIR}/training/opencl/kernels/kernel_chunks.h
    )
    target_sources(Raul PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/training/opencl/kernels/kernel_chunks.h ${Raul_sources})
endif ()

############################
##  Library build config
############################
target_include_directories(Raul
        PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/training/opencl> # deprecated
        $<INSTALL_INTERFACE:raul>
        )

target_link_libraries(Raul
        PUBLIC
        libyato
        $<$<BOOL:${BLAS_FOUND}>:BLAS::BLAS>
        $<$<BOOL:${OpenMP_CXX_FOUND}>:OpenMP::OpenMP_CXX>
        $<$<BOOL:${RAUL_CONFIG_ENABLE_OPENCL}>:${CMAKE_DL_LIBS}>
        )

target_compile_definitions(Raul
        PUBLIC
        $<$<BOOL:${RAUL_BUILD_C_API}>:API_EXPORTS>
        $<$<BOOL:${RAUL_CONFIG_ENABLE_OPENCL}>:CL_TARGET_OPENCL_VERSION=${RAUL_CONFIG_OPENCL_VERSION}>
        )

target_compile_features(Raul PUBLIC cxx_std_17)
set_target_properties(Raul PROPERTIES OUTPUT_NAME "raul")

if (RAUL_CONFIG_ENABLE_FP16 AND ANDROID)
    target_compile_definitions(Raul PUBLIC __ARM_FEATURE_FP16_VECTOR_ARITHMETIC=1)
    target_compile_options(Raul PUBLIC -march=armv8.2-a+fp16+dotprod)
endif ()

if (MSVC)
    set_target_properties(Raul PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)
endif ()

# Parallelize mode
if (RAUL_CPU_CORES)
    target_compile_options(Raul PUBLIC $<$<CXX_COMPILER_ID:MSVC>:/MP${RAUL_CPU_CORES}>)
endif ()

# Pedantic mode (more warnigns and warnings as errors)
if (RAUL_CONFIG_ENABLE_PEDANTIC)
    target_compile_options(Raul PUBLIC
            $<$<OR:$<CXX_COMPILER_ID:GNU>,$<CXX_COMPILER_ID:Clang>>:-Wall;-Wextra;-pedantic;-Werror>
            $<$<CXX_COMPILER_ID:MSVC>:/W4;/WX>
            )

    target_compile_definitions(Raul PUBLIC
            $<$<CXX_COMPILER_ID:MSVC>:_CRT_SECURE_NO_WARNINGS>
            )
endif ()

if (CMAKE_BUILD_TYPE STREQUAL Release)
    add_custom_command(TARGET Raul POST_BUILD
            COMMAND ${CMAKE_STRIP} $<TARGET_FILE:Raul>)
endif ()

############################
##  Building layout
############################
set_target_properties(Raul PROPERTIES
        ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
        )

############################
##  Install
############################
if (RAUL_INSTALL_ENABLE_SUBDIRS)
    install(TARGETS Raul EXPORT raul-export)
else ()
    install(TARGETS Raul EXPORT raul-export DESTINATION ".")
endif ()

############################
##  Appearance in IDEs
############################
if (MSVC)
    target_sources(Raul INTERFACE raul.natvis)
endif ()

# Static sources
get_target_property(raul-src-static Raul SOURCES)
list(FILTER raul-src-static EXCLUDE REGEX "${CMAKE_CURRENT_BINARY_DIR}/.*")
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR}/training FILES ${raul-src-static})
# Generated sources
get_target_property(raul-src-generated Raul SOURCES)
list(FILTER raul-src-generated INCLUDE REGEX "${CMAKE_CURRENT_BINARY_DIR}/.*")
source_group(TREE ${CMAKE_CURRENT_BINARY_DIR}/training FILES ${raul-src-generated})

set_target_properties(Raul PROPERTIES FOLDER Raul)
