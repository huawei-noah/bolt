// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef L2_SQUARED_NORM_LAYER_H
#define L2_SQUARED_NORM_LAYER_H

#include <training/base/common/Common.h>
#include <training/base/layers/BasicLayer.h>

namespace raul
{

/**
 * @brief L2 Squared Normalizing Layer
 *
 *  Computes half the L2 norm of a tensor without sqrt.
 *
 *  @see
 *  https://www.tensorflow.org/api_docs/python/tf/nn/l2_loss
 */
class L2SquaredNormLayer : public BasicLayer
{
  public:
    L2SquaredNormLayer(const Name& name, const BasicParams& params, NetworkParameters& networkParameters);

    L2SquaredNormLayer(L2SquaredNormLayer&&) = default;
    L2SquaredNormLayer(const L2SquaredNormLayer&) = delete;
    L2SquaredNormLayer& operator=(const L2SquaredNormLayer&) = delete;

    template<typename MM>
    friend class L2SquaredNormLayerCPU;
};

} // raul namespace

#endif