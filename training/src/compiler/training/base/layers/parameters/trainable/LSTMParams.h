// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef LSTM_LAYER_PARAMS_H
#define LSTM_LAYER_PARAMS_H

#include <string>
#include <vector>

#include "TrainableParams.h"

namespace raul
{

struct LSTMParams : public TrainableParams
{
    LSTMParams() = delete;

    /** Parameters for LSTM layer
     * @param inputs vector of names of input tensors
     * @param outputs vector of names of output tensors
     * @param hiddenFeatures the number of features in the hidden state
     * @param paramBias enable or disable bias usage
     * @param reversed enable reversed LSTM
     */
    LSTMParams(const Names& inputs,
               const Names& outputs,
               const size_t hiddenFeatures,
               bool useGlobalFusion,
               bool paramBias = true,
               bool frozen = false,
               bool reversed = false,
               dtype zoneout = 0.0_dt,
               bool useSingleMatrix = false,
               dtype forgetBias = 0.0_dt,
               bool useFusion = false)
        : TrainableParams(inputs, outputs, frozen)
        , mHiddenFeatures(hiddenFeatures)
        , mBias(paramBias)
        , mReversed(reversed)
        , mZoneout(zoneout)
        , mUseSingleParamTensor(useSingleMatrix)
        , mForgetBias(forgetBias)
        , mUseGlobalFusion(useGlobalFusion)
        , mUseFusion(useFusion)
    {
        if (hiddenFeatures == 0U)
        {
            THROW_NONAME("LSTMLayerParams", "number of hidden features cannot be zero");
        }
    }

    /** Parameters for LSTM layer
     * @param input name of input tensor
     * @param output name of output tensor
     * @param inputHidden names of output tensors
     * @param inputCell names of output tensors
     * @param outputHidden names of output tensors
     * @param outputCell name of output tensors
     * @param paramBias enable or disable bias usage
     * @param reversed enable reversed LSTM
     */
    LSTMParams(const raul::Name& input,
               const raul::Name& inputHidden,
               const raul::Name& inputCell,
               const raul::Name& output,
               const raul::Name& outputHidden,
               const raul::Name& outputCell,
               bool useGlobalFusion,
               bool paramBias = true,
               bool frozen = false,
               bool reversed = false,
               dtype zoneout = 0.0_dt,
               bool useSingleMatrix = false,
               dtype forgetBias = 0.0_dt,
               bool useFusion = false)
        : TrainableParams({ input, inputHidden, inputCell }, { output, outputHidden, outputCell }, frozen)
        , mBias(paramBias)
        , mReversed(reversed)
        , mZoneout(zoneout)
        , mUseSingleParamTensor(useSingleMatrix)
        , mForgetBias(forgetBias)
        , mUseGlobalFusion(useGlobalFusion)
        , mUseFusion(useFusion)
    {
        for (const auto& name : getInputs())
        {
            if (name.empty())
            {
                THROW_NONAME("LSTMParams", "empty input name");
            }
        }

        for (const auto& name : getOutputs())
        {
            if (name.empty())
            {
                THROW_NONAME("LSTMParams", "empty output name");
            }
        }
    }

    /** Parameters for LSTM layer
     * @param inputs vector of names of input tensors
     * @param outputs vector of names of output tensors
     * @param paramBias enable or disable bias usage
     * @param reversed enable reversed LSTM
     *
     */
    LSTMParams(const Names& inputs,
               const Names& outputs,
               bool useGlobalFusion,
               bool paramBias = true,
               bool frozen = false,
               bool reversed = false,
               dtype zoneout = 0.0_dt,
               bool useSingleMatrix = false,
               dtype forgetBias = 0.0_dt,
               bool useFusion = false)
        : TrainableParams(inputs, outputs, frozen)
        , mHiddenFeatures(std::nullopt)
        , mBias(paramBias)
        , mReversed(reversed)
        , mZoneout(zoneout)
        , mUseSingleParamTensor(useSingleMatrix)
        , mForgetBias(forgetBias)
        , mUseGlobalFusion(useGlobalFusion)
        , mUseFusion(useFusion)
    {
    }

    void print(std::ostream& stream) const override;

    std::optional<size_t> mHiddenFeatures;
    bool mBias;
    bool mReversed;
    dtype mZoneout;
    bool mUseSingleParamTensor;
    dtype mForgetBias;
    bool mUseGlobalFusion;
    bool mUseFusion;
};

} // raul namespace

#endif