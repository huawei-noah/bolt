// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "TransposedConvolution1DParams.h"

namespace raul
{

TransposedConvolution1DParams::TransposedConvolution1DParams(const raul::Name& input,
                                                             const raul::Name& output,
                                                             size_t paramKernelSize,
                                                             size_t paramOutChannel,
                                                             size_t paramStride,
                                                             size_t paramPadding,
                                                             size_t paramOutputPadding,
                                                             size_t paramDilation,
                                                             size_t paramGroups,
                                                             bool paramBias,
                                                             bool paramQuantizeWeights,
                                                             bool frozen)
    : Convolution1DParams({ { input }, { output } }, paramKernelSize, paramOutChannel, paramStride, paramPadding, paramDilation, paramGroups, paramBias, paramQuantizeWeights, false, frozen)
    , outputPadding(paramOutputPadding)
{
}

void TransposedConvolution1DParams::print(std::ostream& stream) const
{
    Convolution1DParams::print(stream);
    stream << "output padding = " << outputPadding;
}

} // namespace raul