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

if (CMAKE_SOURCE_DIR STREQUAL CMAKE_BINARY_DIR)
    message(FATAL_ERROR "Do not build in-source")
endif ()

############################
##  Project
############################
project(
        Raul
        VERSION 1
        LANGUAGES CXX
)

############################
##  Includes
############################
include(CMakeDependentOption)

############################
##  Options: Modules
############################
option(RAUL_BUILD_TESTS "Build tests" OFF)
option(RAUL_BUILD_EXPERIMENTS "Build experiments" OFF)

############################
##  Options: Raul
############################
option(RAUL_BUILD_C_API "Build with C API" OFF)

option(RAUL_CONFIG_ENABLE_OPENMP "Build with OpenMP" OFF)
option(RAUL_CONFIG_ENABLE_OPENCL "Build with OpenCL" ON)
option(RAUL_CONFIG_ENABLE_FP16 "Build with FP16" ON)

option(RAUL_CONFIG_ENABLE_PEDANTIC "Build in pedantic mode" ON)
option(RAUL_CONFIG_ENABLE_PARALLEL_BUILD "Build in parallelize mode" ON)

############################
##  Options: Dev
############################
option(RAUL_CONFIG_DEV_ENABLE_CPPCHECK "Enable cppcheck if available" OFF)
option(RAUL_CONFIG_DEV_ENABLE_CLANG_TIDY "Enable clang-tidy if available" OFF)

############################
##  Options: Tests
############################
cmake_dependent_option(RAUL_TESTS_CONFIG_ENABLE_VERBOSE "Enable verbose test" OFF RAUL_BUILD_TESTS OFF)
cmake_dependent_option(RAUL_TESTS_CONFIG_ENABLE_LIBJPG "Enable LibJpeg" OFF RAUL_BUILD_TESTS OFF)
cmake_dependent_option(RAUL_TESTS_CONFIG_ENABLE_SCENARIOUS "Enable test scenarious" OFF RAUL_BUILD_TESTS OFF)

cmake_dependent_option(RAUL_TESTS_BUILD_CORE "Build core library unit tests" ON RAUL_BUILD_TESTS OFF)
cmake_dependent_option(RAUL_TESTS_BUILD_ACTIVATIONS "Build activation functions unit tests" ON RAUL_BUILD_TESTS OFF)
cmake_dependent_option(RAUL_TESTS_BUILD_INITIALIZERS "Build initializers unit tests" ON RAUL_BUILD_TESTS OFF)
cmake_dependent_option(RAUL_TESTS_BUILD_LAYERS "Build layers unit tests" ON RAUL_BUILD_TESTS OFF)
cmake_dependent_option(RAUL_TESTS_BUILD_LOSS "Build loss functions unit tests" ON RAUL_BUILD_TESTS OFF)
cmake_dependent_option(RAUL_TESTS_BUILD_META "Build meta layers unit tests" ON RAUL_BUILD_TESTS OFF)
cmake_dependent_option(RAUL_TESTS_BUILD_OPTIMIZERS "Build optimizers unit tests" ON RAUL_BUILD_TESTS OFF)
cmake_dependent_option(RAUL_TESTS_BUILD_TOPOLOGIES "Build topologies unit tests" ON RAUL_BUILD_TESTS OFF)
cmake_dependent_option(RAUL_TESTS_BUILD_POSTPROCESSING "Build postprocessing unit tests" ON RAUL_BUILD_TESTS OFF)


############################
##  Options: Installation
############################
option(RAUL_INSTALL_ENABLE_SUBDIRS "Install executables and libraries to bin and lib subdirectories" OFF)
cmake_dependent_option(RAUL_INSTALL_TESTS "Install tests" OFF RAUL_BUILD_TESTS OFF)

############################
##  Configuration
############################
if (RAUL_INSTALL_ENABLE_SUBDIRS)
    include(GNUInstallDirs)
endif ()

if (RAUL_CONFIG_ENABLE_PARALLEL_BUILD)
    include(cmake/core-counter.cmake)
endif ()

############################
##  Add: Linters
############################
if (RAUL_CONFIG_DEV_ENABLE_CPPCHECK)
    include(cmake/cppcheck.cmake)
endif ()

if (RAUL_CONFIG_DEV_ENABLE_CLANG_TIDY)
    include(cmake/clang-tidy.cmake)
endif ()

############################
##  Add: Modules
#############################
add_subdirectory(src/training)

if (RAUL_BUILD_TESTS)
    enable_testing()
    add_subdirectory(src/applications/tests)
endif ()

if (RAUL_BUILD_EXPERIMENTS)
    add_subdirectory(src/applications/experiments/lstm_values_distr)
endif ()

if (RAUL_BUILD_SSD_DEMO)
	add_subdirectory(src/applications/ssd_demo)
endif ()

############################
##  Add:  Test Scenarious
############################
if (RAUL_TESTS_CONFIG_ENABLE_SCENARIOUS)
    include(cmake/testing.cmake)
    add_test_target(test-gpu-unit "Run all gpu unit tests" INCLUDE GpuUnit)
    add_test_target(test-cpu-unit "Run all cpu unit tests" INCLUDE Unit EXCLUDE GpuUnit)
    add_test_target(test-unit-optimizers "Run all optimizers unit tests" INCLUDE TestOptimizer.*Unit)
endif ()

############################
##  Add: Build Scenarious
############################
include(cmake/ndk.cmake)
if (RAUL_CONFIG_NDK_PATH AND NOT MSVC)
    add_custom_target(build-android-tests
            COMMAND ${CMAKE_COMMAND} -E remove ${CMAKE_CURRENT_BINARY_DIR}/CMakeCache.txt
            COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles
            COMMAND ${CMAKE_COMMAND}
            -DCMAKE_LIBRARY_PATH=${CMAKE_SOURCE_DIR}/src/thirdParty/blas-enhance/lib/android
            -DCMAKE_INCLUDE_PATH=${CMAKE_SOURCE_DIR}/src/thirdParty/blas-enhance/include/android
            -DCMAKE_BUILD_TYPE=Release
            -DCMAKE_TOOLCHAIN_FILE="${RAUL_CONFIG_NDK_PATH}/build/cmake/android.toolchain.cmake"
            -DANDROID_ABI=arm64-v8a
            -DANDROID_NATIVE_API_LEVEL=19
            -DANDROID_STL=c++_static
            -DRAUL_CONFIG_ENABLE_OPENMP=ON
            -DRAUL_CONFIG_BLAS_VENDOR=Huawei
            -B ${CMAKE_CURRENT_BINARY_DIR}
            -S ${CMAKE_CURRENT_SOURCE_DIR}
            COMMAND ${CMAKE_COMMAND} --build ${CMAKE_CURRENT_BINARY_DIR} --target RaulTests --parallel
            COMMENT "Build Android release (lib+tests)"
            )
endif ()

