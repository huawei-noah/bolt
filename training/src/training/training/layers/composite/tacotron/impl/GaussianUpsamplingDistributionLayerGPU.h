// Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef GAUSSIAN_UPSAMPLING_DISTRIBUTION_LAYER_GPU_H
#define GAUSSIAN_UPSAMPLING_DISTRIBUTION_LAYER_GPU_H

#include <training/layers/BasicImpl.h>

namespace raul
{
class GaussianUpsamplingDistributionLayer;

class GaussianUpsamplingDistributionLayerGPU : public BasicImpl
{
  public:
    GaussianUpsamplingDistributionLayerGPU(GaussianUpsamplingDistributionLayer& layer)
        : mLayer(layer)
    {
    }

    GaussianUpsamplingDistributionLayerGPU(GaussianUpsamplingDistributionLayerGPU&&) = default;
    GaussianUpsamplingDistributionLayerGPU(const GaussianUpsamplingDistributionLayerGPU&) = delete;
    GaussianUpsamplingDistributionLayerGPU& operator=(const GaussianUpsamplingDistributionLayerGPU&) = delete;

    void forwardComputeImpl(NetworkMode mode) override;
    void backwardComputeImpl() override;

  private:
    GaussianUpsamplingDistributionLayer& mLayer;
};
} // raul namespace

#endif