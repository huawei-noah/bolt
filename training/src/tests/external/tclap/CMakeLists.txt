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
FetchContent_Declare(tclap
        GIT_REPOSITORY https://github.com/mirror/tclap.git
        GIT_TAG v1.2.4
        GIT_SHALLOW FALSE
        GIT_PROGRESS FALSE
        )

FetchContent_GetProperties(tclap)
if (NOT tclap_Target_POPULATED)
    message(STATUS "Preparing tclap")
    FetchContent_Populate(tclap)
    add_library(libtclap INTERFACE)
    target_include_directories(libtclap INTERFACE ${tclap_SOURCE_DIR}/include)
endif ()