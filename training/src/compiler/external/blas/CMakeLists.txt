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

cmake_minimum_required(VERSION 3.11...3.20 FATAL_ERROR)

message(STATUS "Preparing BLAS")

set(BLAS_SHARED ON)

if (BLAS_LIBRARIES AND BLAS_INCLUDE_DIR)
    get_filename_component(BLAS_CUSTOM_LIB_EXT ${BLAS_LIBRARIES} EXT)
    if (BLAS_CUSTOM_LIB_EXT STREQUAL ".a" OR BLAS_CUSTOM_LIB_EXT STREQUAL ".lib")
        set(BLAS_SHARED OFF)
    endif ()
    set(BLAS_FOUND "TRUE")
    message(STATUS "Custom BLAS: ${BLAS_LIBRARIES}")
elseif (RAUL_CONFIG_BLAS_VENDOR MATCHES "^Huawei$")
    if (BOLT_ROOT)
        set(BLAS_FOUND "TRUE")
        set(BLAS_LIB "${CMAKE_BINARY_DIR}/compute/blas_enhance/src/${CMAKE_SHARED_LIBRARY_PREFIX}blas_enhance${CMAKE_SHARED_LIBRARY_SUFFIX}")
        set(BLAS_INCLUDE_DIR "${BOLT_ROOT}/compute/blas_enhance/include;${BOLT_ROOT}/common/uni/include;${BOLT_ROOT}/common/memory/include")
    else()
        find_library(BLAS_LIB NAMES blas_enhance NO_CMAKE_FIND_ROOT_PATH)
        find_path(BLAS_INCLUDE_DIR NAMES blas_enhance.h NO_CMAKE_FIND_ROOT_PATH)
    endif()

    if (BLAS_LIB AND BLAS_INCLUDE_DIR)
        set(BLAS_FOUND "TRUE")
    endif ()
    if (BLAS_FOUND)
        message(STATUS "Found Huawei BLAS Enhance: ${BLAS_LIB}")
        set(BLAS_LINKER_FLAGS "-D_BLAS_ENHANCE")
        set(BLAS_LIBRARIES "${BLAS_LIB}")
    endif ()
else ()
    find_package(BLAS REQUIRED)
    find_path(BLAS_INCLUDE_DIR NAMES "cblas.h" PATHS ${BLAS_INCLUDE_SEARCH_PATHS})
    if (NOT BLAS_INCLUDE_DIR)
        set(BLAS_FOUND OFF)
    endif ()
endif ()

if (BLAS_FOUND)
    if (NOT TARGET BLAS::BLAS)
        if (BLAS_SHARED)
            add_library(BLAS::BLAS SHARED IMPORTED)
        else ()
            add_library(BLAS::BLAS STATIC IMPORTED)
        endif ()
        set_target_properties(BLAS::BLAS PROPERTIES IMPORTED_LOCATION ${BLAS_LIBRARIES})
        target_compile_options(BLAS::BLAS INTERFACE ${BLAS_LINKER_FLAGS})
        target_include_directories(BLAS::BLAS INTERFACE ${BLAS_INCLUDE_DIR})
    endif ()
    target_compile_definitions(BLAS::BLAS INTERFACE _BLAS ${BLAS_COMPILER_DEFINITIONS})
endif ()

if (NOT BLAS_FOUND)
    message(WARNING "BLAS not found")
endif ()
