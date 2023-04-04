// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef CONVOLUTION_1D_PARAMS_H
#define CONVOLUTION_1D_PARAMS_H

#include <string>
#include <vector>

#include "TrainableParams.h"

namespace raul
{

/**
 * @param input name of input tensor
 * @param output name of output tensor
 * @param paramKernelSize the size of the convolution kernel
 * @param paramOutChannel number of channels produced by the convolution
 * @param paramStride the stride of the convolution
 * @param paramPadding implicit zero padding to be added on both sides
 * @param paramDilation the spacing between the kernel points
 * @param paramGroups controls the connections between inputs and outputs
 * @param paramBias enable or disable bias usage
 * @param paramQuantizeWeights enable or disable weights quantization
 *
 */
struct Convolution1DParams : public TrainableParams
{
    Convolution1DParams() = delete;

    Convolution1DParams(const BasicParams& params,
                        size_t paramKernelSize,
                        size_t paramOutChannel,
                        size_t paramStride = 1,
                        size_t paramPadding = 0,
                        size_t paramDilation = 1,
                        size_t paramGroups = 1,
                        bool paramBias = true,
                        bool paramQuantizeWeights = false,
                        bool paramTFStyle = false,
                        bool paramFrozen = false);

    Convolution1DParams(const Name& input,
                        const Name& output,
                        const Name& sharedLayer,
                        size_t kernelSize,
                        size_t outChannel,
                        size_t stride = 1,
                        size_t padding = 0,
                        size_t dilation = 1,
                        size_t groups = 1,
                        bool bias = true,
                        bool quantizeWeights = false,
                        bool tfStyle = false,
                        bool paramFrozen = false);

    Convolution1DParams(const Name& input,
                        const Name& output,
                        size_t kernelSize,
                        size_t outChannel,
                        size_t stride = 1,
                        size_t padding = 0,
                        size_t dilation = 1,
                        size_t groups = 1,
                        bool bias = true,
                        bool quantizeWeights = false,
                        bool tfStyle = false,
                        bool paramFrozen = false);

    size_t kernelSize;
    size_t kernelsCount;
    size_t stride;
    size_t padding;
    size_t dilation;
    size_t groups;
    bool useBias;
    bool quantizeWeights;
    bool tfStyle;

    void print(std::ostream& stream) const override;
};

} // raul namespace

#endif