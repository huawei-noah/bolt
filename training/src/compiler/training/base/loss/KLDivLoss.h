// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef KL_DIV_LOSS_H
#define KL_DIV_LOSS_H

#include "LossWrapper.h"
#include <training/base/common/Common.h>
#include <training/base/layers/BasicLayer.h>

namespace raul
{

/**
 * @brief KLDivLoss (Kullback-Leibler divergence Loss)
 *
 *  KL divergence is a useful distance measure for continuous distributions
 *  and is often useful when performing direct regression
 *  over the space of (discretely sampled) continuous output distributions.
 *  As with NLLLoss, the input given is expected to contain log-probabilities.
 *  Target values (labels) should be given as probabilities.
 *
 *  MEAN reduction works as 'batchmean' in torch
 *  @see
 */
class KLDivLoss : public BasicLayer
{
  public:
    KLDivLoss(const Name& name, const LossParams& params, NetworkParameters& networkParameters);

    KLDivLoss(KLDivLoss&&) = default;
    KLDivLoss(const KLDivLoss&) = delete;
    KLDivLoss& operator=(const KLDivLoss&) = delete;

  private:
    Name mInputName;
    std::string mLabelName;
    Name mOutputName;
    std::shared_ptr<LossWrapper<KLDivLoss>> wrapper;
    bool mIsFinal;

    size_t mDepth;
    size_t mHeight;
    size_t mWidth;

    template<typename MM>
    friend class KLDivLossCPU;
};

} // raul namespace

#endif