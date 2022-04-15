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

# cmake 3.10 doesn't have file(size ...) function
function(get_file_size var filename)
    file(READ "${filename}" content HEX)
    string(LENGTH "${content}" content_length)
    math(EXPR content_length "${content_length} / 2")
    set(${var} ${content_length} PARENT_SCOPE)
endfunction()
get_file_size(file_kernel_def_size ${CMAKE_CURRENT_SOURCE_DIR}/training/base/opencl/kernels/kernel_def.h)

function(split_kernels filename file_kernel_def_step)
    message(STATUS "Splitting ${filename} (deprecated)")
    get_file_size(file_kernel_def_size ${filename})
    set(kernel_def_chunks "")
    set(kernel_sources ${Raul_sources})
    foreach (offset RANGE 0 ${file_kernel_def_size} ${file_kernel_def_step})
        file(READ ${CMAKE_CURRENT_SOURCE_DIR}/training/base/opencl/kernels/kernel_def.h file_kernel_def_content OFFSET ${offset} LIMIT ${file_kernel_def_step})
        # exclude additional end of line
        string(SUBSTRING "${file_kernel_def_content}" 0 ${file_kernel_def_step} file_kernel_def_content)
        file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/training/base/opencl/kernels/kernel_def_${offset}.h "R\"(${file_kernel_def_content})\"\n")
        list(APPEND kernel_sources ${CMAKE_CURRENT_BINARY_DIR}/training/base/opencl/kernels/kernel_def_${offset}.h)
        list(APPEND kernel_def_chunks "#include \"kernel_def_${offset}.h\"")
    endforeach ()

    string(REPLACE ";" "\n,\n" kernel_def_chunks "${kernel_def_chunks}")
    list(APPEND Raul_sources ${CMAKE_CURRENT_BINARY_DIR}/training/base/opencl/kernels/kernel_chunks.h)
    set(Raul_sources ${kernel_sources} PARENT_SCOPE)
    set(kernel_def_chunks ${kernel_def_chunks} PARENT_SCOPE)
endfunction()

