// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "GRULayer.h"

#include <algorithm>
#include <map>

#include <training/base/layers/basic/ConcatenationLayer.h>
#include <training/base/layers/basic/SlicerLayer.h>
#include <training/base/layers/basic/TensorLayer.h>
#include <training/base/layers/composite/rnn/GRUFusedLayer.h>

namespace raul
{

GRULayer::GRULayer(const Name& name, const GRUParams& params, NetworkParameters& networkParameters)
    : mIsExternalState(!params.mHiddenFeatures)
    , mLengthSequence(0)
    , mSequenceDimension("depth")
{
    this->verifyInOut(name, params);
    this->initLocalState(name, params, networkParameters);
    this->buildGRULayer(name, params, networkParameters);
}

void GRULayer::verifyInOut(const Name& lName, const GRUParams& params) const
{
    const size_t inOutNum = mIsExternalState ? 2U : 1U;
    if (params.getInputs().size() != inOutNum)
    {
        THROW("GRULayer", lName, "wrong number of input names");
    }
    if (params.getOutputs().size() != inOutNum)
    {
        THROW("GRULayer", lName, "wrong number of output names");
    }
}

void GRULayer::initLocalState(const Name& name, const GRUParams& params, NetworkParameters& networkParameters)
{
    const auto nameInput = params.getInputs()[0];

    if (!params.mUseGlobalFusion)
    {
        if (networkParameters.mWorkflow.getDepth(nameInput) != 1u && networkParameters.mWorkflow.getHeight(nameInput) != 1u)
        {
            THROW("GRULayer", name, "length of sequence should be placed in one dimension: depth or height");
        }

        if (networkParameters.mWorkflow.getDepth(nameInput) == 1u)
        {
            mLengthSequence = networkParameters.mWorkflow.getHeight(nameInput);
            mSequenceDimension = "height";
        }
        else
        {
            mLengthSequence = networkParameters.mWorkflow.getDepth(nameInput);
        }

        if (mLengthSequence == 0)
        {
            THROW("GRULayer", name, "length of sequence cannot be zero");
        }
    }

    if (params.mHiddenFeatures)
    {
        mHidden = name / "hidden_state";
        networkParameters.mWorkflow.add<TensorLayer>(mHidden, raul::TensorParams{ { mHidden }, raul::WShape{ BS(), 1, 1, *params.mHiddenFeatures }, 0.0_dt, DEC_FORW_READ });
    }
    else
    {
        mHidden = params.getInputs()[1];
    }
}

void GRULayer::buildGRULayer(const Name& name, const GRUParams& params, NetworkParameters& networkParameters) const
{
    size_t cnt = 0;
    const auto nameInput = params.getInputs()[0];
    const auto nameOutput = params.getOutputs()[0];

    if (!params.mUseGlobalFusion)
    {
        Names nameInputSlices(mLengthSequence);
        Names nameOutputSlices(mIsExternalState ? mLengthSequence - 1 : mLengthSequence);
        std::generate_n(nameInputSlices.begin(), mLengthSequence, [&, cnt]() mutable { return name / nameInput + "[" + Conversions::toString(cnt++) + "]"; });

        if (mIsExternalState)
        {
            std::generate_n(nameOutputSlices.begin(), mLengthSequence - 1, [&, cnt]() mutable { return mHidden + "[" + Conversions::toString(cnt++) + "]"; });
            nameOutputSlices.push_back(params.getOutputs()[1]);
        }
        else
        {
            std::generate_n(nameOutputSlices.begin(), mLengthSequence, [&, cnt]() mutable { return mHidden + "[" + Conversions::toString(cnt++) + "]"; });
        }

        networkParameters.mWorkflow.add<SlicerLayer>(name / "slice", SlicingParams(nameInput, nameInputSlices, mSequenceDimension));

        std::string nameHiddenStateIn = mHidden;

        auto nameUnrolledCellZero = name / "cell";
        Names nameUnrolledCellWeights = {
            nameUnrolledCellZero / "linear_ih::Weights", nameUnrolledCellZero / "linear_ih::Biases", nameUnrolledCellZero / "linear_hh::Weights", nameUnrolledCellZero / "linear_hh::Biases"
        };

        for (size_t i = 0; i < mLengthSequence; ++i)
        {
            const auto idx = Conversions::toString(i);
            auto nameUnrolledCell = i == 0 ? nameUnrolledCellZero : (name / "unrolled_cell[" + idx + "]");
            std::string nameHiddenStateOut = mHidden + "[" + idx + "]";

            auto inputIdx = i;
            if (i == mLengthSequence - 1)
            {
                if (mIsExternalState)
                {
                    GRUCellLayer(nameUnrolledCell,
                                 GRUCellParams(nameInputSlices[inputIdx],
                                               nameHiddenStateIn,
                                               params.getOutputs()[1],
                                               i == 0 ? Names{} : nameUnrolledCellWeights,
                                               params.mUseBiasForInput,
                                               params.mUseBiasForHidden,
                                               params.frozen,
                                               params.mUseFusion),
                                 networkParameters);
                }
                else
                {
                    GRUCellLayer(nameUnrolledCell,
                                 GRUCellParams(nameInputSlices[inputIdx],
                                               nameHiddenStateIn,
                                               nameHiddenStateOut,
                                               i == 0 ? Names{} : nameUnrolledCellWeights,
                                               params.mUseBiasForInput,
                                               params.mUseBiasForHidden,
                                               params.frozen,
                                               params.mUseFusion),
                                 networkParameters);
                }
            }
            else
            {
                GRUCellLayer(nameUnrolledCell,
                             GRUCellParams(nameInputSlices[inputIdx],
                                           nameHiddenStateIn,
                                           nameHiddenStateOut,
                                           i == 0 ? Names{} : nameUnrolledCellWeights,
                                           params.mUseBiasForInput,
                                           params.mUseBiasForHidden,
                                           params.frozen,
                                           params.mUseFusion),
                             networkParameters);

                nameHiddenStateIn = nameHiddenStateOut;
            }
        }

        networkParameters.mWorkflow.add<ConcatenationLayer>(name / "concat", BasicParamsWithDim(nameOutputSlices, { nameOutput }, mSequenceDimension));
    }
    else
    {
        networkParameters.mWorkflow.add<GRUFusedLayer>(name / "globalFusion", GRUParams(params), name, mHidden);
    }
}

} // namespace raul