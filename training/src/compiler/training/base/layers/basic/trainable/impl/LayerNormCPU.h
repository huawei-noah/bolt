// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef LAYERNORM_LAYER_CPU_H
#define LAYERNORM_LAYER_CPU_H

#include <training/base/layers/BasicImpl.h>

namespace raul
{
class LayerNormLayer;

/**
 * @brief Layer Normalization CPU implementation
 */
template<typename MM>
class LayerNormLayerCPU : public BasicImpl
{
  public:
    LayerNormLayerCPU(LayerNormLayer& layer)
        : mLayer(layer)
    {
    }

    LayerNormLayerCPU(LayerNormLayerCPU&&) = default;
    LayerNormLayerCPU(const LayerNormLayerCPU&) = delete;
    LayerNormLayerCPU& operator=(const LayerNormLayerCPU&) = delete;

    void initNotBSTensors() override;
    void forwardComputeImpl(NetworkMode mode) override;
    void backwardComputeImpl() override;

  private:
    LayerNormLayer& mLayer;
};

} // raul namespace

#endif