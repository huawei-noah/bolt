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
##  Modules
############################
if (RAUL_CONFIG_ENABLE_IO_JSON)
    include(external/json/CMakeLists.txt)
endif ()

############################
##  Library sources
############################
add_library(Raul-Frontend OBJECT)
include(sources.cmake)

############################
##  Library build config
############################
target_include_directories(Raul-Frontend
        PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
        $<INSTALL_INTERFACE:raul/frontend>
        )


target_link_libraries(Raul-Frontend
        PUBLIC
        Raul-System
        $<$<BOOL:${RAUL_CONFIG_ENABLE_IO_JSON}>:libjson>)

target_compile_definitions(Raul-Frontend
        PUBLIC
        $<$<BOOL:${RAUL_CONFIG_ENABLE_IO_JSON}>:ENABLE_IO_JSON>
        )

target_compile_features(Raul-Frontend PUBLIC cxx_std_17)
set_target_properties(Raul-Frontend PROPERTIES POSITION_INDEPENDENT_CODE ON)


# Parallelize mode
if (RAUL_CPU_CORES)
    target_compile_options(Raul-Frontend PUBLIC $<$<CXX_COMPILER_ID:MSVC>:/MP${RAUL_CPU_CORES}>)
endif ()

# Pedantic mode (more warnigns and warnings as errors)
if (RAUL_CONFIG_ENABLE_PEDANTIC)
    target_compile_options(Raul-Frontend PUBLIC
            $<$<OR:$<CXX_COMPILER_ID:GNU>,$<CXX_COMPILER_ID:Clang>>:-Wall;-Wextra;-pedantic;-Werror>
            $<$<CXX_COMPILER_ID:MSVC>:/W4;/WX>
            )

    target_compile_definitions(Raul-Frontend PUBLIC
            $<$<CXX_COMPILER_ID:MSVC>:_CRT_SECURE_NO_WARNINGS>
            )
endif ()

############################
##  Appearance in IDEs
############################
# Static sources
get_target_property(raul-frontend-src-static Raul-Frontend SOURCES)
list(FILTER raul-frontend-src-static EXCLUDE REGEX "${CMAKE_CURRENT_BINARY_DIR}/.*")
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR}/training FILES ${raul-frontend-src-static})
# Generated sources
get_target_property(raul-frontend-src-generated Raul-Frontend SOURCES)
list(FILTER raul-frontend-src-generated INCLUDE REGEX "${CMAKE_CURRENT_BINARY_DIR}/.*")
source_group(TREE ${CMAKE_CURRENT_BINARY_DIR}/training FILES ${raul-frontend-src-generated})

set_target_properties(Raul-Frontend PROPERTIES FOLDER Raul)