// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef LSTM_FUSED_LAYER_H
#define LSTM_FUSED_LAYER_H

#include "training/base/layers/TrainableLayer.h"
#include "training/base/layers/parameters/trainable/LSTMParams.h"
#include <training/base/common/Common.h>

namespace raul
{
/**
 * @brief Long Short-Term Memory Layer - fused version
 *
 * This layer is a fused layer.
 *
 * @see
 * - S. Hochreiter and J. Schmidhuber, �Long Short-Term Memory� Neural Comput., vol. 9, no. 8, pp. 1735�1780, Nov. 1997, doi: 10.1162/neco.1997.9.8.1735.
 */
class LSTMFusedLayer : public TrainableLayer
{
  public:
    LSTMFusedLayer(const Name& name, const LSTMParams& params, const Name& basicName, const Name& nameHiddenStateIn, const Name& nameCellStateIn, NetworkParameters& networkParameters);

    LSTMFusedLayer(LSTMFusedLayer&&) = default;
    LSTMFusedLayer(const LSTMFusedLayer&) = delete;
    LSTMFusedLayer& operator=(const LSTMFusedLayer&) = delete;

  private:
    bool mIsExternalState;
    size_t mLengthSequence;
    std::string mSequenceDimension;

    std::vector<Names> mInputsLocal;
    std::vector<Names> mOutputsLocal;

    size_t mOutputsCount;
    bool mUseBias;
    dtype mZoneout;
    bool mUseSingleParamTensor;
    dtype mForgetBias;
    Names mTmpCalculationsName;
    Names mGatesName;

    // Needed if zoneout used
    Names mRandomNameHidden;
    Names mRandomNameCell;
    Names mNoZoneoutNewCellName;
    Names mNoZoneoutNewHiddenGradName;
    Names mNoZoneoutNewCellGradName;
    bool mUseZoneout;

    // Trainable names
    Names mTrainableParamsNames;

    Name mHidden;
    Name mCell;

    Dimension mDirection;
    size_t mDimIndex = 0;
    size_t mCurrentInputMaxSize;

    template<typename MM>
    friend class LSTMFusedLayerCPU;
    friend class LSTMFusedLayerCPUFP16;
};
} // raul namespace

#endif