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

cmake_minimum_required(VERSION 3.11...3.20 FATAL_ERROR)

if (POLICY CMP0077)
    set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)
endif ()

if (POLICY CMP0054)
    set(CMAKE_POLICY_DEFAULT_CMP0054 NEW)
endif ()

include(FetchContent)

set(FETCHCONTENT_UPDATES_DISCONNECTED ON)
FetchContent_Declare(libjpeg-turbo
        GIT_REPOSITORY https://github.com/libjpeg-turbo/libjpeg-turbo.git
        GIT_TAG 2.0.4
        GIT_SHALLOW TRUE
        GIT_PROGRESS FALSE
        )

message(STATUS "Preparing libjpeg-turbo")
set(ENABLE_SHARED FALSE)
set(WITH_SIMD FALSE)
set(CMAKE_INSTALL_DOCDIR ${CMAKE_CURRENT_BINARY_DIR})
FetchContent_MakeAvailable(libjpeg-turbo)
get_target_property(turbojpeg-static-directories turbojpeg-static INCLUDE_DIRECTORIES)
target_include_directories(turbojpeg-static PUBLIC ${turbojpeg-static-directories})
set_target_properties(turbojpeg-static simd PROPERTIES FOLDER External/libjpeg-turbo)
