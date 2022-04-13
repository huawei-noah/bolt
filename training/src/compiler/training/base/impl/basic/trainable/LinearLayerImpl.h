// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef LINEAR_LAYER_IMPL_H
#define LINEAR_LAYER_IMPL_H

#include <training/base/layers/BasicImpl.h>

namespace raul
{

class LinearLayer;

/**
 * @brief Linear layer HW independent implementation
 */
template<typename MM>
class LinearLayerImpl : public BasicImpl
{
  public:
    LinearLayerImpl(LinearLayer& layer)
        : mLayer(layer)
    {
    }

    LinearLayerImpl(LinearLayerImpl&&) = default;
    LinearLayerImpl(const LinearLayerImpl&) = delete;
    LinearLayerImpl& operator=(const LinearLayerImpl&) = delete;

    void onBatchSizeChanged(size_t) override;
    void initNotBSTensors() override;
    void forwardComputeImpl(NetworkMode mode) override;
    void backwardComputeImpl() override;

  private:
    LinearLayer& mLayer;
};

} // raul namespace

#endif