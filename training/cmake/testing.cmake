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

set(CMAKE_CTEST_ENV_COMMAND ${CMAKE_COMMAND} -E env RAUL_ASSETS=${CMAKE_CURRENT_SOURCE_DIR}/assets ${CMAKE_CTEST_COMMAND})

function(add_test_target test_target_name test_target_comment)
    set(RUN_COMMAND ${CMAKE_CTEST_ENV_COMMAND} ${CMAKE_CURRENT_BINARY_DIR})
    set(oneValueArgs INCLUDE EXCLUDE)
    cmake_parse_arguments(ADD_TEST_TARGET "" "${oneValueArgs}" "" ${ARGN})
    if (ADD_TEST_TARGET_INCLUDE)
        set(RUN_COMMAND ${RUN_COMMAND} -R ${ADD_TEST_TARGET_INCLUDE})
    endif ()
    if (ADD_TEST_TARGET_EXCLUDE)
        set(RUN_COMMAND ${RUN_COMMAND} -E ${ADD_TEST_TARGET_EXCLUDE})
    endif ()
    if (RAUL_TESTS_CONFIG_ENABLE_VERBOSE)
        set(RUN_COMMAND ${RUN_COMMAND} -VV)
    endif ()
    add_custom_target(${test_target_name}
            COMMAND ${RUN_COMMAND} --parallel
            COMMENT ${test_target_comment}
            DEPENDS RaulTests
            )
    set_target_properties(${test_target_name} PROPERTIES FOLDER Scenarios)
endfunction()
