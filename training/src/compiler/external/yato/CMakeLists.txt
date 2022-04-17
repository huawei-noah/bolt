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

if (POLICY CMP0077)
    set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)
endif ()

include(FetchContent)

set(FETCHCONTENT_UPDATES_DISCONNECTED ON)
if (EXISTS ${BOLT_ROOT}/third_party/sources/Yato-9b5a49f6ec4169b67b9e5ffd11fdae9c238b0a3d.zip)
    FetchContent_Declare(yato URL ${BOLT_ROOT}/third_party/sources/Yato-9b5a49f6ec4169b67b9e5ffd11fdae9c238b0a3d.zip)
else ()
    FetchContent_Declare(yato
            GIT_REPOSITORY https://github.com/agruzdev/Yato.git
            GIT_TAG 9b5a49f6ec4169b67b9e5ffd11fdae9c238b0a3d
            GIT_SHALLOW FALSE
            GIT_PROGRESS FALSE
            )
endif ()

# Workaround for Yato until it became a library target
FetchContent_GetProperties(yato)
if (NOT yato_Target_POPULATED)
    message(STATUS "Preparing yato")
    set(YATO_BUILD_TESTS OFF)
    FetchContent_Populate(yato)
    add_library(libyato INTERFACE)
    target_include_directories(libyato INTERFACE ${yato_SOURCE_DIR}/include)
endif ()
