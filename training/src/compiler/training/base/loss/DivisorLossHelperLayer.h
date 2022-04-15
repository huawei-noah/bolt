// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef DIVISOR_LOSS_HELPER_LAYER_H
#define DIVISOR_LOSS_HELPER_LAYER_H

#include <training/base/layers/BasicLayer.h>

#include "impl/DivisorLossHelperLayerCPU.h"

namespace raul
{

/**
 * @brief Helper layer for loss wrapper
 */
class DivisorLossHelperLayer : public raul::BasicLayer
{
  public:
    DivisorLossHelperLayer(const raul::Name& name, const raul::BasicParams& params, bool isCustomMean, const Name& inputName, raul::NetworkParameters& networkParameters)
        : BasicLayer(name, "DivisorLossHelperLayer", params, networkParameters, { false, false })
        , mIsCustomMean(isCustomMean)
        , mInputName(inputName)
    {
        auto prefix = "DivisorLossHelperLayer[" + mName + "::ctor]: ";

        if (mOutputs.size() != 1)
        {
            THROW("DivisorLossHelperLayer", name, "wrong number of output names");
        }
        if (mOutputs[0].empty())
        {
            THROW("DivisorLossHelperLayer", name, "empty output name");
        }

        DECLARE_IMPL(DivisorLossHelperLayer, DivisorLossHelperLayerCPU<MemoryManager>, DivisorLossHelperLayerCPU<MemoryManagerFP16>)

        mNetworkParams.mWorkflow.tensorNeeded(mName, mOutputs[0], raul::WShape{ 1u, 1u, 1u, 1u }, DEC_FORW_WRIT_NOMEMOPT);
    }

  private:
    bool mIsCustomMean;
    Name mInputName;

    template<typename MM>
    friend class DivisorLossHelperLayerCPU;
};

} // raul namespace

#endif // DIVISOR_LOSS_HELPER_LAYER_H