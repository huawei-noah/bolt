// Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef DYNAMIC_DEPTHWISE_CONVOLUTION_2D_LAYER_GPU_H
#define DYNAMIC_DEPTHWISE_CONVOLUTION_2D_LAYER_GPU_H

#include <training/layers/BasicImpl.h>

namespace raul
{

class DynamicDepthwiseConvolution2DLayer;

class DynamicDepthwiseConvolution2DLayerGPU : public BasicImpl
{
  public:
    DynamicDepthwiseConvolution2DLayerGPU(DynamicDepthwiseConvolution2DLayer& layer);

    DynamicDepthwiseConvolution2DLayerGPU(DynamicDepthwiseConvolution2DLayerGPU&&) = default;
    DynamicDepthwiseConvolution2DLayerGPU(const DynamicDepthwiseConvolution2DLayerGPU&) = delete;
    DynamicDepthwiseConvolution2DLayerGPU& operator=(const DynamicDepthwiseConvolution2DLayerGPU&) = delete;

    void forwardComputeImpl(NetworkMode) override;
    void backwardComputeImpl() override;

  private:
    DynamicDepthwiseConvolution2DLayer& mLayer;
};

} // raul namespace
#endif