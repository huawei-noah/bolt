// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef SIGMOID_CROSS_ENTROPY_LOSS_H
#define SIGMOID_CROSS_ENTROPY_LOSS_H

#include "LossWrapper.h"
#include <training/base/common/Common.h>
#include <training/base/layers/BasicLayer.h>

namespace raul
{

/**
 * @brief SigmoidCrossEntropyLoss
 * Measures the probability error in discrete classification tasks in which each class is independent and not mutually exclusive.
 * For instance, one could perform multilabel classification where a picture can contain both an elephant and a dog at the same time.
 * @see
 */
class SigmoidCrossEntropyLoss : public BasicLayer
{
  public:
    SigmoidCrossEntropyLoss(const Name& name, const LossParams& params, NetworkParameters& networkParameters);

    SigmoidCrossEntropyLoss(SigmoidCrossEntropyLoss&&) = default;
    SigmoidCrossEntropyLoss(const SigmoidCrossEntropyLoss&) = delete;
    SigmoidCrossEntropyLoss& operator=(const SigmoidCrossEntropyLoss&) = delete;

  private:
    Name mInputName;
    std::string mLabelName;
    Name mOutputName;
    std::shared_ptr<LossWrapper<SigmoidCrossEntropyLoss>> wrapper;
    bool mIsFinal;

    size_t mDepth;
    size_t mHeight;
    size_t mWidth;

    template<typename MM>
    friend class SigmoidCrossEntropyLossCPU;
};

} // raul namespace

#endif