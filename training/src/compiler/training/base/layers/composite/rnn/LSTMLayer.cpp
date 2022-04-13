// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "LSTMLayer.h"
#include "ZeroOutputLayer.h"

#include <algorithm>
#include <map>

#include <training/base/layers/basic/ConcatenationLayer.h>
#include <training/base/layers/basic/ReverseLayer.h>
#include <training/base/layers/basic/SlicerLayer.h>
#include <training/base/layers/basic/TensorLayer.h>
#include <training/base/layers/composite/rnn/LSTMFusedLayer.h>

namespace raul
{

LSTMLayer::LSTMLayer(const Name& name, const LSTMParams& params, NetworkParameters& networkParameters)
    : mIsExternalState(!params.mHiddenFeatures)
    , mLengthSequence(0)
    , mSequenceDimension("depth")
{
    try
    {

        this->verifyInOut(name, params);
        this->initLocalState(name, params, networkParameters);
        this->buildLSTMLayer(name, params, networkParameters);
    }
    catch (...)
    {
        THROW("LSTMLayer", name, "Cannot create LSTM layer");
    }
}

void LSTMLayer::verifyInOut(const Name& lName, const LSTMParams& params) const
{
    const size_t inOutNum = mIsExternalState ? 3U : 1U;
    if (params.getInputs().size() != inOutNum && params.getInputs().size() != inOutNum + 1u)
    {
        THROW("LSTM", lName, " wrong number of input names");
    }
    if (params.getOutputs().size() != inOutNum)
    {
        THROW("LSTM", lName, " wrong number of output names");
    }
}

void LSTMLayer::initLocalState(const Name& name, const LSTMParams& params, NetworkParameters& networkParameters)
{

    if (params.mUseGlobalFusion && params.mUseFusion)
    {
        THROW("LSTM", name, " use only one type of fusion");
    }

    if (!params.mUseGlobalFusion)
    {
        const auto nameInput = params.getInputs()[0];

        if (networkParameters.mWorkflow.getDepth(nameInput) != 1u && networkParameters.mWorkflow.getHeight(nameInput) != 1u)
        {
            THROW("LSTM", name, " length of sequence should be placed in one dimension: depth or height");
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
            THROW("LSTM", name, " length of sequence cannot be zero");
        }
    }

    if (params.mHiddenFeatures)
    {
        mState.hidden = name / "hidden_state";
        mState.cell = name / "cell_state";
        networkParameters.mWorkflow.add<TensorLayer>(mState.hidden, raul::TensorParams{ { mState.hidden }, raul::WShape{ BS(), 1, 1, *params.mHiddenFeatures }, 0.0_dt, DEC_FORW_READ });
        networkParameters.mWorkflow.add<TensorLayer>(mState.cell, raul::TensorParams{ { mState.cell }, raul::WShape{ BS(), 1, 1, *params.mHiddenFeatures }, 0.0_dt, DEC_FORW_READ });
    }
    else
    {
        mState.hidden = params.getInputs()[1];
        mState.cell = params.getInputs()[2];
    }
}

void LSTMLayer::buildLSTMLayer(const Name& name, const LSTMParams& params, NetworkParameters& networkParameters) const
{
    size_t cnt = 0;
    const auto nameInput = params.getInputs()[0];
    const auto nameOutput = params.getOutputs()[0];

    const auto useSeqLength = params.getInputs().size() == 2u || params.getInputs().size() == 4u;
    if (params.mReversed)
    {
        if (useSeqLength)
        {
            networkParameters.mWorkflow.add<ReverseLayer>(name / "reverse_input", BasicParams({ nameInput, params.getInputs().back() }, { name / "input_reversed" }));
        }
        else
        {
            networkParameters.mWorkflow.add<ReverseLayer>(name / "reverse_input", BasicParams({ nameInput }, { name / "input_reversed" }));
        }
    }

    if (!params.mUseGlobalFusion)
    {
        Names nameInputSlices(mLengthSequence);
        Names nameOutputSlices(mIsExternalState ? mLengthSequence - 1 : mLengthSequence);
        std::generate_n(nameInputSlices.begin(), mLengthSequence, [&, cnt]() mutable { return name / nameInput + "[" + Conversions::toString(cnt++) + "]"; });

        if (mIsExternalState)
        {
            std::generate_n(nameOutputSlices.begin(), mLengthSequence - 1, [&, cnt]() mutable { return mState.hidden + "[" + Conversions::toString(cnt++) + "]"; });
            nameOutputSlices.push_back(params.getOutputs()[1]);
        }
        else
        {
            std::generate_n(nameOutputSlices.begin(), mLengthSequence, [&, cnt]() mutable { return mState.hidden + "[" + Conversions::toString(cnt++) + "]"; });
        }

        networkParameters.mWorkflow.add<SlicerLayer>(name / "slice", SlicingParams(params.mReversed ? name / "input_reversed" : nameInput, nameInputSlices, mSequenceDimension));

        std::string nameHiddenStateIn = mState.hidden;
        std::string nameCellStateIn = mState.cell;

        auto nameUnrolledCellZero = name / "cell";

        for (size_t i = 0; i < mLengthSequence; ++i)
        {
            const auto idx = Conversions::toString(i);
            auto nameUnrolledCell = i == 0 ? nameUnrolledCellZero : (name / "unrolled_cell[" + idx + "]");
            std::string nameHiddenStateOut = mState.hidden + "[" + idx + "]";
            std::string nameCellStateOut = mState.cell + "[" + idx + "]";

            Names nameUnrolledCellWeights = {
                nameUnrolledCellZero / "linear_ih::Weights", nameUnrolledCellZero / "linear_ih::Biases", nameUnrolledCellZero / "linear_hh::Weights", nameUnrolledCellZero / "linear_hh::Biases"
            };
            if (params.mUseSingleParamTensor)
            {
                nameUnrolledCellWeights = { nameUnrolledCellZero / "linear::Weights", nameUnrolledCellZero / "linear::Biases" };
            }

            if (i == 0) nameUnrolledCellWeights.clear(); // to create tensors for first layer

            auto inputIdx = i;
            if (i == mLengthSequence - 1)
            {
                if (mIsExternalState)
                {
                    LSTMCellLayer(nameUnrolledCell,
                                  LSTMCellParams(nameInputSlices[inputIdx],
                                                 nameHiddenStateIn,
                                                 nameCellStateIn,
                                                 params.getOutputs()[1],
                                                 params.getOutputs()[2],
                                                 nameUnrolledCellWeights,
                                                 params.mBias,
                                                 params.mZoneout,
                                                 params.mUseSingleParamTensor,
                                                 params.mForgetBias,
                                                 params.frozen,
                                                 params.mUseFusion),
                                  networkParameters);
                }
                else
                {
                    LSTMCellLayer(nameUnrolledCell,
                                  LSTMCellParams(nameInputSlices[inputIdx],
                                                 nameHiddenStateIn,
                                                 nameCellStateIn,
                                                 nameHiddenStateOut,
                                                 nameCellStateOut,
                                                 nameUnrolledCellWeights,
                                                 params.mBias,
                                                 params.mZoneout,
                                                 params.mUseSingleParamTensor,
                                                 params.mForgetBias,
                                                 params.frozen,
                                                 params.mUseFusion),
                                  networkParameters);
                }
            }
            else
            {
                LSTMCellLayer(nameUnrolledCell,
                              LSTMCellParams(nameInputSlices[inputIdx],
                                             nameHiddenStateIn,
                                             nameCellStateIn,
                                             nameHiddenStateOut,
                                             nameCellStateOut,
                                             nameUnrolledCellWeights,
                                             params.mBias,
                                             params.mZoneout,
                                             params.mUseSingleParamTensor,
                                             params.mForgetBias,
                                             params.frozen,
                                             params.mUseFusion),
                              networkParameters);

                nameHiddenStateIn = nameHiddenStateOut;
                nameCellStateIn = nameCellStateOut;
            }
        }

        networkParameters.mWorkflow.add<ConcatenationLayer>(name / "concat",
                                                            BasicParamsWithDim(nameOutputSlices, { (useSeqLength || params.mReversed) ? name / "proto_output" : nameOutput }, mSequenceDimension));
    }
    else
    {
        LSTMParams lstmParams(params);
        if (useSeqLength || params.mReversed)
        {
            lstmParams.getOutputs()[0] = name / "proto_output";
        }
        networkParameters.mWorkflow.add<LSTMFusedLayer>(name / "globalFusion", lstmParams, name, mState.hidden, mState.cell);
    }

    if (useSeqLength)
    {
        networkParameters.mWorkflow.add<ZeroOutputLayer>(name / "zero", BasicParams({ name / "proto_output", params.getInputs().back() }, { params.mReversed ? name / "zeroed_output" : nameOutput }));
    }

    if (params.mReversed)
    {
        if (useSeqLength)
        {
            networkParameters.mWorkflow.add<ReverseLayer>(name / "reverse", BasicParams({ name / "zeroed_output", params.getInputs().back() }, { nameOutput }));
        }
        else
        {
            networkParameters.mWorkflow.add<ReverseLayer>(name / "reverse", BasicParams({ name / "proto_output" }, { nameOutput }));
        }
    }
}

} // namespace raul
