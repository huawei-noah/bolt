// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef MULTIHEAD_ATTENTION_PARAMS_H
#define MULTIHEAD_ATTENTION_PARAMS_H

#include "TrainableParams.h"
#include <string>
#include <vector>

namespace raul
{
struct TrainableDropoutParams : public TrainableParams
{
    TrainableDropoutParams() = delete;
    TrainableDropoutParams(const Names& inputs, const Names& outputs, dtype paramProbability, bool frozen = false)
        : TrainableParams(inputs, outputs, frozen)
        , probability(paramProbability)
    {
    }

    dtype probability;

    void print(std::ostream& stream) const override;
};

/**
 * @param inputs vector of names of input tensors
 * @param outputs vector of names of output tensors
 * @param heads number of parallel heads
 * @param paramProbability probability of element to be zero-ed
 */
struct MultiHeadAttentionParams : public TrainableDropoutParams
{
    MultiHeadAttentionParams() = delete;
    MultiHeadAttentionParams(const Names& inputs, const raul::Name& output, unsigned int paramHeads = 1, float paramProbability = 0.1f, bool paramFinalTransform = true, bool frozen = false)
        : TrainableDropoutParams(inputs, Names(1, output), paramProbability, frozen)
        , heads(paramHeads)
        , finalTransform(paramFinalTransform)
    {
    }

    unsigned int heads;
    bool finalTransform;

    void print(std::ostream& stream) const override;
};

} // raul namespace

#endif // MULTIHEAD_ATTENTION_PARAMS_H