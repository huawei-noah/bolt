// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef CONVOLUTION_1D_LAYER_CPU_H
#define CONVOLUTION_1D_LAYER_CPU_H

#include <training/base/layers/BasicImpl.h>

namespace raul
{
class Convolution1DLayer;

/**
 * @brief Convolution1D layer CPU implementation
 */
template<typename MM>
class Convolution1DLayerCPU : public BasicImpl
{
  public:
    Convolution1DLayerCPU(Convolution1DLayer& layer)
        : mLayer(layer)
    {
    }

    Convolution1DLayerCPU(Convolution1DLayerCPU&&) = default;
    Convolution1DLayerCPU(const Convolution1DLayerCPU&) = delete;
    Convolution1DLayerCPU& operator=(const Convolution1DLayerCPU&) = delete;

    void forwardComputeImpl(NetworkMode mode) override;
    void backwardComputeImpl() override;

  private:
    Convolution1DLayer& mLayer;
};
} // raul namespace

#endif
