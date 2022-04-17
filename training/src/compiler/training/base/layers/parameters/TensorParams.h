// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef TENSOR_LAYER_PARAMS_H
#define TENSOR_LAYER_PARAMS_H

#include <string>
#include <vector>

#include "BasicParameters.h"

#include <training/compiler/Workflow.h>

namespace raul
{

struct TensorParams : public BasicParams
{
    TensorParams() = delete;
    /**
     * @param outputs vector of names of output tensors
     * @param b batch size in elements
     * @param d depth size in elements
     * @param h height size in elements
     * @param w width size in elements
     */
    TensorParams(const Names& outputs,
                 size_t b,
                 size_t d,
                 size_t h,
                 size_t w,
                 Workflow::Usage u = raul::Workflow::Usage::Forward,
                 Workflow::Mode m = raul::Workflow::Mode::Read,
                 bool optimizeGraph = false,
                 bool optimizeMem = false,
                 bool trainable = false,
                 bool zero = false,
                 bool compress = false)
        : BasicParams({}, outputs)
        , shape(b, d, h, w)
        , init(false)
        , initValue(0_dt)
        , usage(u)
        , mode(m)
        , isOptimizeGraph(optimizeGraph)
        , isOptimizeMem(optimizeMem)
        , isTrainable(trainable)
        , isZero(zero)
        , isCompress(compress)
    {
    }

    /**
     * @param outputs vector of names of output tensors
     * @param shape shape of output tensors
     * @param initValue fill value
     */
    TensorParams(const Names& outputs,
                 raul::WShape shape,
                 raul::dtype initValue,
                 Workflow::Usage u = raul::Workflow::Usage::Forward,
                 Workflow::Mode m = raul::Workflow::Mode::Read,
                 bool optimizeGraph = false,
                 bool optimizeMem = false,
                 bool trainable = false,
                 bool zero = false,
                 bool compress = false)
        : BasicParams({}, outputs)
        , shape(shape)
        , init(true)
        , initValue(initValue)
        , usage(u)
        , mode(m)
        , isOptimizeGraph(optimizeGraph)
        , isOptimizeMem(optimizeMem)
        , isTrainable(trainable)
        , isZero(zero)
        , isCompress(compress)
    {
    }

    /**
     * @brief declares Tensor that will be filled externally
     * @param outputs vector of names of output tensors
     * @param shape shape of output tensors
     */
    TensorParams(const Names& outputs,
                 raul::WShape shape,
                 Workflow::Usage u = raul::Workflow::Usage::Forward,
                 Workflow::Mode m = raul::Workflow::Mode::Read,
                 bool optimizeGraph = false,
                 bool optimizeMem = false,
                 bool trainable = false,
                 bool zero = false,
                 bool compress = false)
        : BasicParams({}, outputs)
        , shape(shape)
        , init(zero)
        , initValue(0_dt)
        , usage(u)
        , mode(m)
        , isOptimizeGraph(optimizeGraph)
        , isOptimizeMem(optimizeMem)
        , isTrainable(trainable)
        , isZero(zero)
        , isCompress(compress)
    {
    }

    raul::WShape shape;
    bool init;
    raul::dtype initValue;

    Workflow::Usage usage;
    Workflow::Mode mode;

    bool isOptimizeGraph;
    bool isOptimizeMem;
    bool isTrainable;
    bool isZero;
    bool isCompress;

    void print(std::ostream& stream) const override;
};
} // raul namespace

#endif // TENSOR_LAYER_PARAMS_H
