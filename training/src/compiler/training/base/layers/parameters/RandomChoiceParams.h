// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef RANDOM_CHOICE_PARAMS_H
#define RANDOM_CHOICE_PARAMS_H

#include "BasicParameters.h"
#include <string>
#include <vector>

namespace raul
{
/**
 * @param output name of output tensor
 * @param ratiosOrProbabilities specify the ratio of choice
 * For N inputs could be:
 *   - N ratios
 *   - N probabilities
 *   - N-1 probabilities (last one will be calculated)
 */
struct RandomChoiceParams : public BasicParams
{
    RandomChoiceParams() = delete;
    RandomChoiceParams(const Names& inputs, const raul::Name& output, std::vector<float> ratiosOrProbabilities, size_t seed = std::random_device{}())
        : BasicParams(inputs, Names(1, output))
        , mRatios(ratiosOrProbabilities)
        , mSeed(seed)
    {
    }

    std::vector<float> mRatios;
    size_t mSeed;

    void print(std::ostream& stream) const override;
};

}

#endif // raul namespace
