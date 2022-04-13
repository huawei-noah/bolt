// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef TRAINABLEPOOL_2D_PARAMS_H
#define TRAINABLEPOOL_2D_PARAMS_H

#include <string>
#include <vector>

#include "TrainableParams.h"

namespace raul
{
struct TrainablePool2DParams : public TrainableParams
{
    TrainablePool2DParams() = delete;
    TrainablePool2DParams(const Names& inputs,
                          const Names& outputs,
                          size_t kernelW,
                          size_t kernelH,
                          size_t strideWidth,
                          size_t strideHeight,
                          size_t paddingWidth,
                          size_t paddingHeight,
                          bool frozen = false);
    TrainablePool2DParams(const Names& inputs, const Names& outputs, size_t kernelSize, size_t stride, size_t padding = 0, bool frozen = false);

    size_t kernelWidth = 0, kernelHeight = 0;
    size_t strideW = 0, strideH = 0;
    size_t paddingW = 0, paddingH = 0;

    void print(std::ostream& stream) const override;
};
} // raul namespace

#endif