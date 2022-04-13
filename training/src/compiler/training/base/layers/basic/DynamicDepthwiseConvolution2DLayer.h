// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef DYNAMIC_DEPTHWISE_CONVOLUTION_2D_LAYER_H
#define DYNAMIC_DEPTHWISE_CONVOLUTION_2D_LAYER_H

#include <training/base/common/Common.h>
#include <training/base/layers/BasicLayer.h>

namespace raul
{

/**
 * @brief Channel-wise Dynamic Convolution 2D Layer
 *
 * The layer applies 2D convolution over input tensor, each channel processed separately
 * and can be multiplied N times
 */
class DynamicDepthwiseConvolution2DLayer : public BasicLayer
{
  public:
    DynamicDepthwiseConvolution2DLayer(const Name& name, const BasicParams& params, NetworkParameters& networkParameters);

    DynamicDepthwiseConvolution2DLayer(DynamicDepthwiseConvolution2DLayer&&) = default;
    DynamicDepthwiseConvolution2DLayer(const DynamicDepthwiseConvolution2DLayer&) = delete;
    DynamicDepthwiseConvolution2DLayer& operator=(const DynamicDepthwiseConvolution2DLayer&) = delete;

  private:
    Name mInputName;
    Name mFiltersName;
    Name mOutputName;

    size_t mInputDepth{ 0u };
    size_t mInputHeight{ 0u };
    size_t mInputWidth{ 0u };

    size_t mOutputHeight{ 0u };
    size_t mOutputWidth{ 0u };

    size_t mFilterHeight{ 0u };
    size_t mFilterWidth{ 0u };
    size_t mInChannels{ 0u };
    size_t mChannelMultiplier{ 0u };

    template<typename MM>
    friend class DynamicDepthwiseConvolution2DLayerCPU;
};

} // raul namespace

#endif