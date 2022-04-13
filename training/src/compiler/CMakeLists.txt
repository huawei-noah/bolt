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
##  BLAS dev interface
############################
set(RAUL_CONFIG_BLAS_VENDOR "None" CACHE STRING "Set BLAS vendor")
set_property(CACHE RAUL_CONFIG_BLAS_VENDOR PROPERTY STRINGS "None;Custom;OpenBLAS;Auto")

if (RAUL_CONFIG_BLAS_VENDOR STREQUAL "OpenBLAS")
    set(BLA_VENDOR OpenBLAS)
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

message(STATUS "BLAS: ${RAUL_CONFIG_BLAS_VENDOR}")

if (NOT RAUL_CONFIG_BLAS_VENDOR STREQUAL "None")
    include(external/blas/CMakeLists.txt)
endif ()

############################
##  Library sources
############################
add_library(Raul-Compiler OBJECT)
include(sources.cmake)

############################
##  Library build config
############################
target_include_directories(Raul-Compiler
        PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
        $<INSTALL_INTERFACE:raul/compiler>
        )

target_compile_features(Raul-Compiler PUBLIC cxx_std_17)


target_link_libraries(Raul-Compiler
        PUBLIC
        Raul-Frontend
        Raul-System
        libyato
        $<$<BOOL:${BLAS_FOUND}>:BLAS::BLAS>
        $<$<BOOL:${OpenMP_CXX_FOUND}>:OpenMP::OpenMP_CXX>
        )

set_target_properties(Raul-Compiler PROPERTIES POSITION_INDEPENDENT_CODE ON)

# Parallelize mode
if (RAUL_CPU_CORES)
    target_compile_options(Raul-Compiler PUBLIC $<$<CXX_COMPILER_ID:MSVC>:/MP${RAUL_CPU_CORES}>)
endif ()

# Pedantic mode (more warnigns and warnings as errors)
if (RAUL_CONFIG_ENABLE_PEDANTIC)
    target_compile_options(Raul-Compiler PUBLIC
            $<$<OR:$<CXX_COMPILER_ID:GNU>,$<CXX_COMPILER_ID:Clang>>:-Wall;-Wextra;-pedantic;-Werror>
            $<$<CXX_COMPILER_ID:MSVC>:/W4;/WX>
            )

    target_compile_definitions(Raul-Compiler PUBLIC
            $<$<CXX_COMPILER_ID:MSVC>:_CRT_SECURE_NO_WARNINGS>
            )
endif ()

############################
##  Appearance in IDEs
############################
# Static sources
get_target_property(raul-compiler-src-static Raul-Compiler SOURCES)
list(FILTER raul-compiler-src-static EXCLUDE REGEX "${CMAKE_CURRENT_BINARY_DIR}/.*")
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR}/training FILES ${raul-compiler-src-static})
# Generated sources
get_target_property(raul-compiler-src-generated Raul-Compiler SOURCES)
list(FILTER raul-compiler-src-generated INCLUDE REGEX "${CMAKE_CURRENT_BINARY_DIR}/.*")
source_group(TREE ${CMAKE_CURRENT_BINARY_DIR}/training FILES ${raul-compiler-src-generated})

set_target_properties(Raul-Compiler PROPERTIES FOLDER Raul)