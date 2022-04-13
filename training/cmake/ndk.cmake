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

set(ANDROID_NDK_SEARCH_PATHS
        /opt/android-ndk-r22/
        $ENV{NDK_PATH}
        )

if (WIN32)
    set(ANDROID_NDK_SEARCH_PATHS
            ${ANDROID_NDK_SEARCH_PATHS}
            $ENV{SystemDrive}/android-ndk-r22/
            $ENV{ProgramW6432}/android-ndk-r22/
            $ENV{ProgramFiles}/android-ndk-r22/
            $ENV{ProgramFiles\(x86\)}/android-ndk-r22/
            )
endif ()

set(RAUL_CONFIG_NDK_PATH "" CACHE PATH "Path to android NKD")
find_path(RAUL_CONFIG_NDK_PATH NAMES build/cmake/android.toolchain.cmake PATHS ${ANDROID_NDK_SEARCH_PATHS} NO_CMAKE_FIND_ROOT_PATH)
mark_as_advanced(RAUL_CONFIG_NDK_PATH)

if (RAUL_CONFIG_NDK_PATH)
    message(STATUS "Found Android NDK: ${RAUL_CONFIG_NDK_PATH}")
endif ()