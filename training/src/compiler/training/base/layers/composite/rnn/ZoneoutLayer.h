// Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef ZONEOUT_LAYER_H
#define ZONEOUT_LAYER_H

#include "training/base/layers/BasicLayer.h"
#include "training/base/layers/parameters/ZoneoutParams.h"
#include <training/base/common/Common.h>
#include <training/base/layers/basic/RandomSelectLayer.h>

namespace raul
{

/**
 * @brief Zoneout mechanism for RNN cells
 *
 * The main parameter is the probability of choosing a previous value. The probability must be in [0,1).
 *
 * @see
 * - D. Krueger et al., “Zoneout: Regularizing RNNs by Randomly Preserving Hidden Activations,” arXiv:1606.01305 [cs], Sep. 2017, Accessed: Jul. 15, 2020. [Online]. Available:
 * http://arxiv.org/abs/1606.01305.
 */
class ZoneoutLayer : public RandomSelectLayer
{
  public:
    ZoneoutLayer(const Name& name, const ZoneoutParams& params, NetworkParameters& networkParameters);

    ZoneoutLayer(ZoneoutLayer&&) = default;
    ZoneoutLayer(const ZoneoutLayer&) = delete;
    ZoneoutLayer& operator=(const ZoneoutLayer&) = delete;

    void forwardComputeImpl(NetworkMode mode) override;
};
} // raul namespace

#endif