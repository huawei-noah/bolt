// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef TRANSPOSED_CONVOLUTION_2D_PARAMS_H
#define TRANSPOSED_CONVOLUTION_2D_PARAMS_H

#include <string>
#include <vector>

#include "TrainablePool2DParams.h"

namespace raul
{

/**
 * @param inputs vector of names of input tensors
 * @param outputs vector of names of output tensors
 * @param kernelW/H the size of the convolution kernel
 * @param paramOutChannel number of channels produced by the convolution
 * @param strideW/H the stride of the convolution
 * @param paddingW/H implicit zero padding to be added on both sides
 * @param outputPaddingW/H additional size added to one side of output shape
 * @param paramBias enable or disable bias usage
 * @param dilation spacing between the kernel points
 * @param quantizeWeights enable or disable weights quantization
 */
struct TransposedConvolution2DParams : public TrainablePool2DParams
{
    TransposedConvolution2DParams() = delete;

    TransposedConvolution2DParams(const Names& inputs,
                                  const Names& outputs,
                                  size_t kernelW,
                                  size_t kernelH,
                                  size_t paramOutChannel,
                                  size_t strideW,
                                  size_t strideH,
                                  size_t paddingW,
                                  size_t paddingH,
                                  size_t outputPaddingW,
                                  size_t outputPaddingH,
                                  bool paramBias,
                                  size_t dilationW,
                                  size_t dilationH,
                                  bool quantizeWeights = false,
                                  bool frozen = false);

    TransposedConvolution2DParams(const Names& inputs,
                                  const Names& outputs,
                                  size_t kernelSize,
                                  size_t paramOutChannel,
                                  size_t stride = 1,
                                  size_t padding = 0,
                                  size_t outputPadding = 0,
                                  bool paramBias = true,
                                  size_t dilation = 1,
                                  bool quantizeWeights = false,
                                  bool frozen = false);

    size_t kernelsCount = 0;
    size_t mOutputPaddingW;
    size_t mOutputPaddingH;
    bool bias;
    size_t mDilationW;
    size_t mDilationH;
    bool quantizeWeights;

    void print(std::ostream& stream) const override;
};

} // raul namespace

#endif