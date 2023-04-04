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
execute_process(COMMAND mkdir -p ${PROJECT_SOURCE_DIR}/cache)
if (EXISTS ${BOLT_ROOT}/third_party/sources/half-2.2.0.zip)
    file(COPY ${BOLT_ROOT}/third_party/sources/half-2.2.0.zip DESTINATION ${PROJECT_SOURCE_DIR}/cache)
else ()
    execute_process(COMMAND wget https://telkomuniversity.dl.sourceforge.net/project/half/half/2.2.0/half-2.2.0.zip WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/cache)
    if (EXISTS ${BOLT_ROOT}/third_party/sources)
        file(COPY ${PROJECT_SOURCE_DIR}/cache/half-2.2.0.zip DESTINATION ${BOLT_ROOT}/third_party/sources/)
    endif ()
endif ()
FetchContent_Declare(half URL ${PROJECT_SOURCE_DIR}/cache/half-2.2.0.zip)

FetchContent_GetProperties(half)
if (NOT half_Target_POPULATED)
    message(STATUS "Preparing half")
    FetchContent_Populate(half)
    add_library(libhalf INTERFACE)
    target_include_directories(libhalf INTERFACE ${half_SOURCE_DIR}/include)
endif ()

execute_process(COMMAND rm -rf ${PROJECT_SOURCE_DIR}/cache)
