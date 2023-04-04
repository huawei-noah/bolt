// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef RANDOM_SELECT_LAYER_H
#define RANDOM_SELECT_LAYER_H

#include <training/base/common/Common.h>
#include <training/base/layers/BroadcastingLayer.h>
#include <training/base/layers/parameters/RandomSelectParams.h>

namespace raul
{

/**
 * @brief Random Select Layer
 *
 * This layer returns a tensor of elements selected from either x or y, depending on dropout rate.
 * First input values selected at indices where condition is True, second input values - where condition is False.
 * All input tensors must be broadcastable.
 *
 */
class RandomSelectLayer : public BroadcastingLayer
{
  public:
    RandomSelectLayer(const Name& name, const RandomSelectParams& params, NetworkParameters& networkParameters);

    RandomSelectLayer(RandomSelectLayer&&) = default;
    RandomSelectLayer(const RandomSelectLayer&) = delete;
    RandomSelectLayer& operator=(const RandomSelectLayer&) = delete;

  private:
    raul::dtype mProbability;
    bool mBroadcast;
    Name mRandomName;

    template<typename MM>
    friend class RandomSelectLayerCPU;
};

} // raul namespace

#endif