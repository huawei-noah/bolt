// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef LINEAR_LAYER_PARAMS_H
#define LINEAR_LAYER_PARAMS_H

#include "TrainableParams.h"
#include <string>
#include <vector>

namespace raul
{
/**
 * @param inputs vector of names of input tensors
 * @param outputs vector of names of output tensors
 * @param paramOutputsCount output size in elements
 * @param paramBias enable or disable bias usage
 */
struct LinearParams : public TrainableParams
{
    LinearParams() = delete;

    LinearParams(const Name& input, const Name& output, size_t paramOutputsCount, bool paramBias = true, bool paramFrozen = false)
        : TrainableParams({ input }, { output }, paramFrozen)
        , outputsCount(paramOutputsCount)
        , bias(paramBias)
    {
    }

    LinearParams(const BasicParams& params, size_t paramOutputsCount, bool paramBias = true, bool paramFrozen = false)
        : TrainableParams(params, paramFrozen)
        , outputsCount(paramOutputsCount)
        , bias(paramBias)
    {
    }

    LinearParams(const Name& input, const Name& output, const Name& sharedLayer, size_t paramOutputsCount, bool paramBias = true, bool paramFrozen = false)
        : TrainableParams({ input }, { output }, sharedLayer, paramFrozen)
        , outputsCount(paramOutputsCount)
        , bias(paramBias)
    {
    }

    size_t outputsCount;
    bool bias;

    void print(std::ostream& stream) const override;
};

} // raul namespace

#endif // LINEAR_LAYER_PARAMS_H