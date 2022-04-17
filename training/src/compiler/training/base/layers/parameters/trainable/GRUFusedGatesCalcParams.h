// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef GRU_FUSED_GATES_CALC_PARAMS_H
#define GRU_FUSED_GATES_CALC_PARAMS_H

#include "TrainableParams.h"
#include <string>
#include <vector>

namespace raul
{

/**
 * @param inputs vector of names of input tensors
 * @param output name output tensor
 * @param paramOutputsCount output size in elements of hidden linear layer
 * @param useBiasForInput enable or disable bias usage for input
 * @param useBiasForHidden enable or disable bias usage for hidden
 */
struct GRUFusedGatesCalcParams : public TrainableParams
{
    GRUFusedGatesCalcParams() = delete;

    GRUFusedGatesCalcParams(const Names& inputs, const Name& output, size_t paramOutputsCount, bool useBiasForInput = true, bool useBiasForHidden = true, bool paramFrozen = false)
        : TrainableParams(inputs, { output }, paramFrozen)
        , outputsCount(paramOutputsCount)
        , mUseBiasForInput(useBiasForInput)
        , mUseBiasForHidden(useBiasForHidden)
    {
    }

    GRUFusedGatesCalcParams(const BasicParams& params, size_t paramOutputsCount, bool useBiasForInput = true, bool useBiasForHidden = true, bool paramFrozen = false)
        : TrainableParams(params, paramFrozen)
        , outputsCount(paramOutputsCount)
        , mUseBiasForInput(useBiasForInput)
        , mUseBiasForHidden(useBiasForHidden)
    {
    }

    GRUFusedGatesCalcParams(const Names& inputs,
                            const Name& output,
                            const Name& sharedLayer,
                            size_t paramOutputsCount,
                            bool useBiasForInput = true,
                            bool useBiasForHidden = true,
                            bool paramFrozen = false)
        : TrainableParams(inputs, { output }, sharedLayer, paramFrozen)
        , outputsCount(paramOutputsCount)
        , mUseBiasForInput(useBiasForInput)
        , mUseBiasForHidden(useBiasForHidden)
    {
    }

    size_t outputsCount;
    bool mUseBiasForInput;
    bool mUseBiasForHidden;

    void print(std::ostream& stream) const override;
};

} // raul namespace

#endif // GRU_FUSED_GATES_CALC_PARAMS_H