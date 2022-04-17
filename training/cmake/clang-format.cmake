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

find_program(RAUL_DEV_CLANG_FORMAT NAMES "clang-format"
        "clang-format-13"
        "clang-format-12"
        "clang-format-11")

mark_as_advanced(RAUL_DEV_CLANG_FORMAT)

if (RAUL_DEV_CLANG_FORMAT AND TARGET Raul)
    get_target_property(raul-src Raul SOURCES)
    list(FILTER raul-src EXCLUDE REGEX "${CMAKE_CURRENT_BINARY_DIR}/.*")
    get_target_property(raul-dir Raul SOURCE_DIR)
    set(raul-abs-src "")
    foreach (file ${raul-src})
        list(APPEND raul-abs-src ${raul-dir}/${file})
    endforeach ()

    add_custom_target(format
            COMMAND ${CMAKE_COMMAND} -E echo "Tool: ${RAUL_DEV_CLANG_FORMAT}"
            COMMAND ${RAUL_DEV_CLANG_FORMAT} -style=file -i ${raul-abs-src}
            COMMENT "Format raul sources"
            )
endif ()
