// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef L1_LOSS_H
#define L1_LOSS_H

#include "LossWrapper.h"
#include <training/base/common/Common.h>
#include <training/base/layers/BasicLayer.h>

namespace raul
{

/**
 * @brief L1Loss
 * Creates a criterion that measures the mean absolute error(MAE)
 * between each element in the input x and target y
 * @see
 */
class L1Loss : public BasicLayer
{
  public:
    L1Loss(const Name& name, const LossParams& params, NetworkParameters& networkParameters);

    L1Loss(L1Loss&&) = default;
    L1Loss(const L1Loss&) = delete;
    L1Loss& operator=(const L1Loss&) = delete;

  private:
    Name mInputName;
    std::string mLabelName;
    Name mOutputName;
    std::shared_ptr<LossWrapper<L1Loss>> wrapper;
    bool mIsFinal;

    size_t mDepth;
    size_t mHeight;
    size_t mWidth;

    template<typename MM>
    friend class L1LossLayerCPU;
};

} // raul namespace

#endif