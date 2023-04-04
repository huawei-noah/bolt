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

############################
##  Library sources
############################
add_library(Training-API OBJECT)
include(sources.cmake)

############################
##  Library build config
############################
target_include_directories(Training-API
        PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
        ${BOLT_ROOT}/common/model_spec/include 
        ${BOLT_ROOT}/common/uni/include
        ${BOLT_ROOT}/common/memory/include
        )

target_compile_features(Training-API PUBLIC cxx_std_17)

target_link_libraries(Training-API
        PUBLIC
        Raul
        ${model_spec_library}
        ${uni_library}
        )
if (ANDROID)
    target_link_libraries(Training-API PUBLIC log)
endif()

target_compile_definitions(Training-API
        PUBLIC
        API_EXPORTS
        )

set_target_properties(Training-API PROPERTIES POSITION_INDEPENDENT_CODE ON)

# Parallelize mode
if (RAUL_CPU_CORES)
    target_compile_options(Training-API PUBLIC $<$<CXX_COMPILER_ID:MSVC>:/MP${RAUL_CPU_CORES}>)
endif ()

# Pedantic mode (more warnings and warnings as errors)
if (RAUL_CONFIG_ENABLE_PEDANTIC)
    target_compile_options(Training-API PUBLIC
            $<$<OR:$<CXX_COMPILER_ID:GNU>,$<CXX_COMPILER_ID:Clang>>:-Wall;-Wextra;-pedantic;-Werror>
            $<$<CXX_COMPILER_ID:MSVC>:/W4;/WX>
            )

    target_compile_definitions(Training-API PUBLIC
            $<$<CXX_COMPILER_ID:MSVC>:_CRT_SECURE_NO_WARNINGS>
            )
endif ()
