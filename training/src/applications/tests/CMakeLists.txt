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

cmake_minimum_required(VERSION 3.10...3.20 FATAL_ERROR)

############################
##  Options
############################
option(RAUL_TESTS_CONFIG_ENABLE_LIBJPG "Use LibJpeg" OFF)

option(RAUL_TESTS_BUILD_CORE "Build core library unit tests" ON)
option(RAUL_TESTS_BUILD_ACTIVATIONS "Build activation functions unit tests" ON)
option(RAUL_TESTS_BUILD_INITIALIZERS "Build initializers unit tests" ON)
option(RAUL_TESTS_BUILD_LAYERS "Build layers unit tests" ON)
option(RAUL_TESTS_BUILD_LOSS "Build loss functions unit tests" ON)
option(RAUL_TESTS_BUILD_META "Build meta layers unit tests" ON)
option(RAUL_TESTS_BUILD_OPTIMIZERS "Build optimizers unit tests" ON)
option(RAUL_TESTS_BUILD_TOPOLOGIES "Build topologies unit tests" ON)
option(RAUL_TESTS_BUILD_POSTPROCESSING "Build postprocessing unit tests" ON)

############################
##  Modules
############################
add_subdirectory(external/googletest EXCLUDE_FROM_ALL)
add_subdirectory(external/tclap EXCLUDE_FROM_ALL)

if (RAUL_TESTS_CONFIG_ENABLE_LIBJPG)
    add_subdirectory(external/libjpeg-turbo EXCLUDE_FROM_ALL)
endif ()

include(GoogleTest)

############################
##  Tests sources
############################
add_executable(RaulTests tests/main.cpp tests/GTestExtensions.h)
include(sources.cmake)

############################
##  Tests dependencies
############################
target_link_libraries(RaulTests
        PRIVATE
        gtest
        Raul
        libtclap
        $<$<BOOL:${RAUL_TESTS_CONFIG_ENABLE_LIBJPG}>:turbojpeg-static>
        )

target_include_directories(RaulTests
        PRIVATE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
        )

if (CMAKE_BUILD_TYPE STREQUAL Release)
    add_custom_command(TARGET RaulTests POST_BUILD
            COMMAND ${CMAKE_STRIP} $<TARGET_FILE:RaulTests>)
endif ()

############################
##  Building layout
############################
set_target_properties(RaulTests PROPERTIES
        ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
        )

############################
##  Install
############################
if (RAUL_INSTALL_TESTS)
    if (RAUL_INSTALL_ENABLE_SUBDIRS)
        install(TARGETS RaulTests EXPORT raul-export)
    else ()
        install(TARGETS RaulTests EXPORT raul-export DESTINATION ".")
    endif ()
endif ()


############################
##  Appearance in IDEs
############################
get_target_property(raul-tests-src RaulTests SOURCES)
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR}/tests FILES ${raul-tests-src})
set_target_properties(RaulTests PROPERTIES FOLDER Raul)

if (NOT ANDROID)
    gtest_discover_tests(RaulTests)
endif ()

message(STATUS "ASSETS: provide path in ENV 'RAUL_ASSETS'")
