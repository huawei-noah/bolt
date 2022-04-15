// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef GRU_FUSED_LAYER_H
#define GRU_FUSED_LAYER_H

#include <training/base/common/Common.h>
#include <training/base/layers/TrainableLayer.h>
#include <training/base/layers/parameters/trainable/GRUParams.h>

namespace raul
{

class GRUFusedLayer : public TrainableLayer
{
  public:
    GRUFusedLayer(const Name& name, const GRUParams& params, const Name& basicName, const Name& nameHiddenStateIn, NetworkParameters& networkParameters);

    GRUFusedLayer(GRUFusedLayer&&) = default;
    GRUFusedLayer(const GRUFusedLayer&) = delete;
    GRUFusedLayer& operator=(const GRUFusedLayer&) = delete;

  private:
    bool mIsExternalState;
    size_t mLengthSequence;
    std::string mSequenceDimension;

    std::vector<Names> mInputsLocal;
    Names mOutputsLocal;

    size_t mOutputsCount;
    bool mUseBiasForInput;
    bool mUseBiasForHidden;
    Names mLinearIHTmp;
    Names mLinearHHTmp;

    Name mWeightsNameIH;
    Name mBiasesNameIH;
    Name mWeightsNameHH;
    Name mBiasesNameHH;

    Dimension mDirection;
    size_t mDimIndex = 0;
    size_t mCurrentInputMaxSize;

    template<typename MM>
    friend class GRUFusedLayerCPU;
};
} // raul namespace

#endif