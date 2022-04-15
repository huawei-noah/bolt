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

include(FetchContent)

set(FETCHCONTENT_UPDATES_DISCONNECTED ON)
FetchContent_Declare(opencl-headers
        GIT_REPOSITORY https://github.com/KhronosGroup/OpenCL-Headers.git
        GIT_TAG v2021.04.29
        GIT_SHALLOW TRUE
        GIT_PROGRESS FALSE
        )

FetchContent_GetProperties(opencl-headers)
if (NOT opencl-headers_Target_POPULATED)
    message(STATUS "Preparing opencl headers")
    FetchContent_Populate(opencl-headers)
    add_library(libopencl-headers INTERFACE)
    target_include_directories(libopencl-headers INTERFACE ${opencl-headers_SOURCE_DIR})
    add_library(OpenCL::Headers ALIAS libopencl-headers)
endif ()

FetchContent_Declare(opencl-hpp
        GIT_REPOSITORY https://github.com/KhronosGroup/OpenCL-CLHPP.git
        GIT_TAG v2.0.14
        GIT_SHALLOW TRUE
        GIT_PROGRESS FALSE
        )

FetchContent_GetProperties(opencl-hpp)
if (NOT opencl-hpp_Target_POPULATED)
    message(STATUS "Preparing opencl API C++ bindings")
    FetchContent_Populate(opencl-hpp)
    add_library(libopencl-hpp INTERFACE)
    add_dependencies(libopencl-hpp OpenCL::Headers)
    target_include_directories(libopencl-hpp INTERFACE ${opencl-hpp_SOURCE_DIR}/include)
    target_link_libraries(libopencl-hpp INTERFACE OpenCL::Headers)
    target_compile_definitions(libopencl-hpp INTERFACE CL_HPP_TARGET_OPENCL_VERSION=210)
    add_library(OpenCL::CppBindings ALIAS libopencl-hpp)
endif ()