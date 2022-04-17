// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef LSTM_CELL_PARAMS_H
#define LSTM_CELL_PARAMS_H

#include <string>
#include <vector>

#include "TrainableParams.h"

namespace raul
{

struct LSTMCellParams : public TrainableParams
{
    LSTMCellParams() = delete;
    /** Parameters for LSTM Cell layer
     * @param inputs vector of names of input tensors
     * @param outputs vector of names of output tensors
     * @param paramBias enable or disable bias usage
     * @param zoneout value (probability, optional)
     * @param useSingleParamTensor flag (default=false)
     * @param forgetBias value (default=0)
     */
    LSTMCellParams(const BasicParams& params,
                   bool paramBias = true,
                   dtype zoneout = 0.0_dt,
                   bool useSingleParamTensor = false,
                   dtype forgetBias = 0.0_dt,
                   bool frozen = false,
                   bool useFusion = false,
                   std::optional<size_t> hidden_size = std::nullopt)
        : TrainableParams(params, frozen)
        , mBias(paramBias)
        , mZoneout(zoneout)
        , mUseSingleParamTensor(useSingleParamTensor)
        , mForgetBias(forgetBias)
        , mUseFusion(useFusion)
        , mHiddenFeatures(hidden_size)
    {
    }

    /** Parameters for LSTM Cell layer
     * @param input name of input tensor
     * @param inputHidden name of input hidden state tensor
     * @param inputCell name of input cell state tensor
     * @param outputHidden name of output hidden state tensor
     * @param outputCell name of output cell state tensor
     * @param paramBias enable or disable bias usage
     * @param zoneout value (probability)
     * @param useSingleWeightTensor flag (default=false)
     * @param forgetBias value (default=0)
     */
    LSTMCellParams(const raul::Name& input,
                   const raul::Name& inputHidden,
                   const raul::Name& inputCell,
                   const raul::Name& outputHidden,
                   const raul::Name& outputCell,
                   const Names& weights,
                   bool paramBias = true,
                   dtype zoneout = 0.0_dt,
                   bool useSingleMatrix = false,
                   dtype forgetBias = 0.0_dt,
                   bool frozen = false,
                   bool useFusion = false)
        : TrainableParams({ input, inputHidden, inputCell }, { outputHidden, outputCell }, weights, frozen)
        , mBias(paramBias)
        , mZoneout(zoneout)
        , mUseSingleParamTensor(useSingleMatrix)
        , mForgetBias(forgetBias)
        , mUseFusion(useFusion)
    {
    }

    void print(std::ostream& stream) const override;

    bool mBias;
    dtype mZoneout;
    bool mUseSingleParamTensor;
    dtype mForgetBias;
    bool mUseFusion;
    std::optional<size_t> mHiddenFeatures;
};

} // raul namespace

#endif