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
add_subdirectory(system)
add_subdirectory(frontend)
add_subdirectory(compiler)
if (RAUL_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif ()

############################
##  Library sources
############################
add_library(Raul SHARED)

############################
##  Version
############################
include(cmake/git.cmake)
configure_file(
        Version.cpp.in
        ${CMAKE_CURRENT_BINARY_DIR}/Version.cpp
)
target_sources(Raul PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/Version.h ${CMAKE_CURRENT_BINARY_DIR}/Version.cpp)

############################
##  Library build config
############################
target_include_directories(Raul
        PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
        $<INSTALL_INTERFACE:raul>
        )

target_link_libraries(Raul
        PUBLIC
        Raul-System
        Raul-Frontend
        Raul-Compiler
        )

target_compile_features(Raul PUBLIC cxx_std_17)
set_target_properties(Raul PROPERTIES OUTPUT_NAME "raul")

if (RAUL_CONFIG_ENABLE_FP16 AND ANDROID)
    target_compile_definitions(Raul PUBLIC __ARM_FEATURE_FP16_VECTOR_ARITHMETIC=1)
    target_compile_options(Raul PUBLIC -march=armv8.2-a+fp16+dotprod)
endif ()

if (MSVC)
    set_target_properties(Raul PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)
endif ()

# Parallelize mode
if (RAUL_CPU_CORES)
    target_compile_options(Raul PUBLIC $<$<CXX_COMPILER_ID:MSVC>:/MP${RAUL_CPU_CORES}>)
endif ()

# Pedantic mode (more warnigns and warnings as errors)
if (RAUL_CONFIG_ENABLE_PEDANTIC)
    target_compile_options(Raul PUBLIC
            $<$<OR:$<CXX_COMPILER_ID:GNU>,$<CXX_COMPILER_ID:Clang>>:-Wall;-Wextra;-pedantic;-Werror>
            $<$<CXX_COMPILER_ID:MSVC>:/W4;/WX>
            )

    target_compile_definitions(Raul PUBLIC
            $<$<CXX_COMPILER_ID:MSVC>:_CRT_SECURE_NO_WARNINGS>
            )
endif ()

if (CMAKE_BUILD_TYPE STREQUAL Release)
    add_custom_command(TARGET Raul POST_BUILD
            COMMAND ${CMAKE_STRIP} $<TARGET_FILE:Raul>)
endif ()

############################
##  Building layout
############################
set_target_properties(Raul PROPERTIES
        ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
        )

############################
##  Install
############################
if (RAUL_INSTALL_ENABLE_SUBDIRS)
    install(TARGETS Raul EXPORT raul-export)
else ()
    install(TARGETS Raul EXPORT raul-export DESTINATION "lib")
endif ()

############################
##  Appearance in IDEs
############################
if (MSVC)
    target_sources(Raul INTERFACE raul.natvis)
endif ()

# Static sources
get_target_property(raul-src-static Raul SOURCES)
list(FILTER raul-src-static EXCLUDE REGEX "${CMAKE_CURRENT_BINARY_DIR}/.*")
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${raul-src-static})
# Generated sources
get_target_property(raul-src-generated Raul SOURCES)
list(FILTER raul-src-generated INCLUDE REGEX "${CMAKE_CURRENT_BINARY_DIR}/.*")
source_group(TREE ${CMAKE_CURRENT_BINARY_DIR} FILES ${raul-src-generated})

set_target_properties(Raul PROPERTIES FOLDER Raul)
