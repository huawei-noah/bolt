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
target_sources(Raul-Frontend PRIVATE
        training/frontend/Types.h
        training/frontend/Path.h
        training/frontend/Frontend.h
        training/frontend/Declaration.h
        training/frontend/Generator.h
        training/frontend/Port.h
        training/frontend/processors/Processor.h
        training/frontend/processors/Processor.cpp
        training/frontend/processors/TextPrinter.h
        training/frontend/processors/DotLangPrinter.h
        training/frontend/Layers.h
        training/frontend/Graph.h
        )


if (RAUL_CONFIG_ENABLE_IO_JSON)
    target_sources(Raul-Frontend PRIVATE
            training/frontend/io/JSON.cpp
            training/frontend/io/JSON.h
            )
endif ()