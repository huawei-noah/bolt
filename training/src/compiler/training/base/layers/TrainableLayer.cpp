// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "TrainableLayer.h"

namespace raul
{

TrainableLayer::TrainableLayer(const raul::Name& name, const std::string& typeName, const TrainableParams& params, NetworkParameters& networkParams, std::pair<bool, bool> doChecks)
    : BasicLayer(name, typeName, params, networkParams, doChecks)
    , mFrozen(params.frozen)
{
    MEASURE_BLOCK("TrainableLayer[" + mName + "::ctor]")

    if (!mSharedLayer.empty())
    {
        mWeightsName = mSharedLayer / "Weights";
        mBiasesName = mSharedLayer / "Biases";
    }
    else
    {
        if (mSharedWeights.empty() || (!mSharedWeights.empty() && mSharedWeights[0].empty()))
        {
            mWeightsName = mName / "Weights";
        }
        else
        {
            mWeightsName = mSharedWeights[0];
        }

        if (mSharedWeights.empty() || (mSharedWeights.size() > 1 && mSharedWeights[1].empty()))
        {
            mBiasesName = mName / "Biases";
        }
        else
        {
            if (mSharedWeights.size() < 2)
            {
                THROW("TrainableLayer", mName, "wrong number of weight names");
            }
            mBiasesName = mSharedWeights[1];
        }
    }
}
} // raul namespace
