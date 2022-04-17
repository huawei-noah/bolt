// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "Convolution1DParams.h"

namespace raul
{

Convolution1DParams::Convolution1DParams(const BasicParams& params,
                                         size_t paramKernelSize,
                                         size_t paramOutChannel,
                                         size_t paramStride,
                                         size_t paramPadding,
                                         size_t paramDilation,
                                         size_t paramGroups,
                                         bool paramBias,
                                         bool paramQuantizeWeights,
                                         bool paramTFStyle,
                                         bool paramFrozen)
    : TrainableParams(params, paramFrozen)
    , kernelSize(paramKernelSize)
    , kernelsCount(paramOutChannel)
    , stride(paramStride)
    , padding(paramPadding)
    , dilation(paramDilation)
    , groups(paramGroups)
    , useBias(paramBias)
    , quantizeWeights(paramQuantizeWeights)
    , tfStyle(paramTFStyle)
{
}

Convolution1DParams::Convolution1DParams(const Name& input,
                                         const Name& output,
                                         const Name& sharedLayer,
                                         size_t paramKernelSize,
                                         size_t paramOutChannel,
                                         size_t paramStride,
                                         size_t paramPadding,
                                         size_t paramDilation,
                                         size_t paramGroups,
                                         bool paramBias,
                                         bool paramQuantizeWeights,
                                         bool paramTFStyle,
                                         bool paramFrozen)
    : TrainableParams({ input }, { output }, sharedLayer, paramFrozen)
    , kernelSize(paramKernelSize)
    , kernelsCount(paramOutChannel)
    , stride(paramStride)
    , padding(paramPadding)
    , dilation(paramDilation)
    , groups(paramGroups)
    , useBias(paramBias)
    , quantizeWeights(paramQuantizeWeights)
    , tfStyle(paramTFStyle)
{
}

Convolution1DParams::Convolution1DParams(const Name& input,
                                         const Name& output,
                                         size_t kernelSize,
                                         size_t outChannel,
                                         size_t stride,
                                         size_t padding,
                                         size_t dilation,
                                         size_t groups,
                                         bool bias,
                                         bool quantizeWeights,
                                         bool tfStyle,
                                         bool paramFrozen)
    : Convolution1DParams({ { input }, { output } }, kernelSize, outChannel, stride, padding, dilation, groups, bias, quantizeWeights, tfStyle, paramFrozen)
{
}

void Convolution1DParams::print(std::ostream& stream) const
{
    BasicParams::print(stream);
    stream << " kernel_size: " << kernelSize << ", out_channels: " << kernelsCount << ", stride: " << stride << ", padding: " << padding << ", dilation: " << dilation;
    stream << ", groups: " << groups << ", bias: " << (useBias ? "true" : "false") << ", quantized weights: " << (quantizeWeights ? "true" : "false") << ", TF style: " << (tfStyle ? "true" : "false");
}

} // namespace raul
