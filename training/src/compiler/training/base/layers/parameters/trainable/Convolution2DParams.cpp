// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "Convolution2DParams.h"

namespace raul
{

Convolution2DParams::Convolution2DParams(const Names& inputs,
                                         const Names& outputs,
                                         size_t kernelW,
                                         size_t kernelH,
                                         size_t paramOutChannel,
                                         size_t strideW,
                                         size_t strideH,
                                         size_t paddingW,
                                         size_t paddingH,
                                         bool paramBias,
                                         bool quantizeWeights,
                                         size_t dilationW,
                                         size_t dilationH,
                                         size_t groups,
                                         bool frozen)
    : TrainablePool2DParams(inputs, outputs, kernelW, kernelH, strideW, strideH, paddingW, paddingH, frozen)
    , kernelsCount(paramOutChannel)
    , bias(paramBias)
    , quantizeWeights(quantizeWeights)
    , mDilationW(dilationW)
    , mDilationH(dilationH)
    , mGroups(groups)
{
}

Convolution2DParams::Convolution2DParams(const Names& inputs,
                                         const Names& outputs,
                                         size_t kernelSize,
                                         size_t paramOutChannel,
                                         size_t stride,
                                         size_t padding,
                                         bool paramBias,
                                         bool quantizeWeights,
                                         size_t dilation,
                                         size_t groups,
                                         bool frozen)
    : TrainablePool2DParams(inputs, outputs, kernelSize, stride, padding, frozen)
    , kernelsCount(paramOutChannel)
    , bias(paramBias)
    , quantizeWeights(quantizeWeights)
    , mDilationW(dilation)
    , mDilationH(dilation)
    , mGroups(groups)
{
}

void Convolution2DParams::print(std::ostream& stream) const
{
    TrainablePool2DParams::print(stream);
    stream << ", kernels: " << kernelsCount << ", bias: " << (bias ? "true" : "false") << ", quantized weights: " << (quantizeWeights ? "true" : "false");
}

} // namespace raul
