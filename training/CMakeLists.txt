# Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.
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
        Training
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
option(TRAINING_BUILD_C_API "Build training C API" OFF)
cmake_dependent_option(TRAINING_BUILD_DEMO "Build training C demo" OFF TRAINING_BUILD_C_API OFF)

############################
##  Options: Raul
############################
option(RAUL_CONFIG_ENABLE_OPENMP "Build with OpenMP" OFF)
option(RAUL_CONFIG_ENABLE_FP16 "Build with FP16" ON)
option(RAUL_BUILD_TESTS "Build tests" OFF)

option(RAUL_CONFIG_ENABLE_PEDANTIC "Build in pedantic mode" ON)
option(RAUL_CONFIG_ENABLE_PARALLEL_BUILD "Build in parallelize mode" ON)

option(RAUL_CONFIG_ENABLE_IO_JSON "Build with support loading topology from json" OFF)

############################
##  Options: Dev
############################
option(RAUL_CONFIG_DEV_ENABLE_CPPCHECK "Enable cppcheck if available" OFF)
option(RAUL_CONFIG_DEV_ENABLE_CLANG_TIDY "Enable clang-tidy if available" OFF)

############################
##  Options: Tests
############################
cmake_dependent_option(RAUL_TESTS_CONFIG_ENABLE_VERBOSE "Enable verbose test" OFF RAUL_BUILD_TESTS OFF)
cmake_dependent_option(RAUL_TESTS_CONFIG_ENABLE_SCENARIOS "Enable test scenarios" OFF RAUL_BUILD_TESTS OFF)

cmake_dependent_option(RAUL_TESTS_BUILD_CORE "Build core library unit tests" ON RAUL_BUILD_TESTS OFF)
cmake_dependent_option(RAUL_TESTS_BUILD_ACTIVATIONS "Build activation functions unit tests" ON RAUL_BUILD_TESTS OFF)
cmake_dependent_option(RAUL_TESTS_BUILD_INITIALIZERS "Build initializers unit tests" ON RAUL_BUILD_TESTS OFF)
cmake_dependent_option(RAUL_TESTS_BUILD_LAYERS "Build layers unit tests" ON RAUL_BUILD_TESTS OFF)
cmake_dependent_option(RAUL_TESTS_BUILD_LOSS "Build loss functions unit tests" ON RAUL_BUILD_TESTS OFF)
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
add_subdirectory(src)
if (TRAINING_BUILD_C_API)
    add_subdirectory(api)
endif ()
if (TRAINING_BUILD_DEMO)
    add_subdirectory(demos)
endif ()


############################
##  Add:  Test Scenarios
############################
if (RAUL_TESTS_CONFIG_ENABLE_SCENARIOS)
    include(cmake/testing.cmake)
    add_test_target(test-cpu-unit "Run all cpu unit tests" INCLUDE Unit)
    add_test_target(test-unit-optimizers "Run all optimizers unit tests" INCLUDE TestOptimizer.*Unit)
endif ()

############################
##  Add: Tools
#############################
include(docs/docs.cmake)
include(cmake/clang-format.cmake)
include(cmake/cmake-gui.cmake)
