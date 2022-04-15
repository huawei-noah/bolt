// Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef LINEAR_LAYER_CPU_FP32_MASTER_H
#define LINEAR_LAYER_CPU_FP32_MASTER_H

#include <training/layers/BasicImpl.h>

namespace raul
{
class LinearLayer;

/**
 * @brief Linear layer CPU implementation transform master weights
 */
class LinearLayerCPUFP32Master : public BasicImpl
{
  public:
    LinearLayerCPUFP32Master(LinearLayer& layer);

    LinearLayerCPUFP32Master(LinearLayerCPUFP32Master&&) = default;
    LinearLayerCPUFP32Master(const LinearLayerCPUFP32Master&) = delete;
    LinearLayerCPUFP32Master& operator=(const LinearLayerCPUFP32Master&) = delete;

    void forwardComputeImpl(NetworkMode mode) override;
    void backwardComputeImpl() override;

  private:
    LinearLayer& mLayer;

    bool mInitMasterWeights;
};
} // raul namespace

#endif
