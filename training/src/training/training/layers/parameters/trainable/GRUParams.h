// Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef GRU_LAYER_PARAMS_H
#define GRU_LAYER_PARAMS_H

#include <string>
#include <vector>

#include "TrainableParams.h"

namespace raul
{

struct GRUParams : public TrainableParams
{
    GRUParams() = delete;

    /** Parameters for GRU layer
     * @param inputs vector of names of input tensors
     * @param outputs vector of names of output tensors
     * @param hiddenFeatures the number of features in the hidden state
     * @param paramBias enable or disable bias usage
     */
    GRUParams(const Names& inputs, const Names& outputs, const size_t hiddenFeatures, bool paramBias = true, bool frozen = false, bool useFusion = false)
        : TrainableParams(inputs, outputs, frozen)
        , mHiddenFeatures(hiddenFeatures)
        , mBias(paramBias)
        , mUseFusion(useFusion)
    {
        if (hiddenFeatures == 0U)
        {
            throw std::runtime_error("GRULayerParams[ctor]: number of hidden features cannot be zero");
        }
    }

    GRUParams(const Names& inputs, const Names& outputs, const size_t hiddenFeatures, LayerExecutionTarget layerExecTarget, bool paramBias = true, bool frozen = false, bool useFusion = false)
        : TrainableParams(inputs, outputs, layerExecTarget, frozen)
        , mHiddenFeatures(hiddenFeatures)
        , mBias(paramBias)
        , mUseFusion(useFusion)
    {
        if (hiddenFeatures == 0U)
        {
            throw std::runtime_error("GRULayerParams[ctor]: number of hidden features cannot be zero");
        }
    }

    /** Parameters for GRU layer
     * @param input name of input tensor
     * @param output name of output tensor
     * @param inputHidden names of output tensors
     * @param outputHidden names of output tensors
     * @param paramBias enable or disable bias usage
     */
    GRUParams(const raul::Name& input, const raul::Name& inputHidden, const raul::Name& output, const raul::Name& outputHidden, bool paramBias = true, bool frozen = false, bool useFusion = false)
        : TrainableParams({ input, inputHidden }, { output, outputHidden }, frozen)
        , mBias(paramBias)
        , mUseFusion(useFusion)
    {
        for (const auto& name : getInputs())
        {
            if (name.empty())
            {
                throw std::runtime_error("GRUParams[ctor]: empty input name");
            }
        }

        for (const auto& name : getOutputs())
        {
            if (name.empty())
            {
                throw std::runtime_error("GRUParams[ctor]: empty output name");
            }
        }
    }

    /** Parameters for GRU layer
     * @param inputs vector of names of input tensors
     * @param outputs vector of names of output tensors
     * @param paramBias enable or disable bias usage
     */
    GRUParams(const Names& inputs, const Names& outputs, bool paramBias = true, bool frozen = false, bool useFusion = false)
        : TrainableParams(inputs, outputs, frozen)
        , mHiddenFeatures(std::nullopt)
        , mBias(paramBias)
        , mUseFusion(useFusion)
    {
    }

    void print(std::ostream& stream) const override;

    std::optional<size_t> mHiddenFeatures;
    bool mBias;
    bool mUseFusion;
};

} // raul namespace

#endif