// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef CONVOLUTION_DEPTHWISE_LAYER_H
#define CONVOLUTION_DEPTHWISE_LAYER_H

#include <training/base/layers/TrainableLayer.h>
#include <training/base/layers/parameters/trainable/Convolution2DParams.h>

#include <training/base/common/Common.h>

namespace raul
{

/**
 * @brief Channel-wise Convolution 2D Layer
 *
 * The layer applies 2D convolution over input tensor, each channel processed separately.
 */
class ConvolutionDepthwiseLayer : public TrainableLayer
{
  public:
    ConvolutionDepthwiseLayer(const Name& name, const Convolution2DParams& params, NetworkParameters& networkParameters);

    ConvolutionDepthwiseLayer(ConvolutionDepthwiseLayer&&) = default;
    ConvolutionDepthwiseLayer(const ConvolutionDepthwiseLayer&) = delete;
    ConvolutionDepthwiseLayer& operator=(const ConvolutionDepthwiseLayer&) = delete;

    void forwardComputeImpl(NetworkMode mode) override;
    void backwardComputeImpl() override;

  private:
    ConvolutionDepthwiseLayer(const Name& name, const std::string& typeName, const Convolution2DParams& params, NetworkParameters& networkParameters);

    size_t mInputWidth;
    size_t mInputHeight;
    size_t mInputDepth;

    size_t mOutputWidth;
    size_t mOutputHeight;

    size_t mKernelWidth;
    size_t mKernelHeight;
    size_t mKernelsCount;

    size_t mStrideW;
    size_t mStrideH;

    size_t mPaddingW;
    size_t mPaddingH;

    bool mUseBias;

    Name mInputName;
    Name mOutputName;

    Names mIm2ColForward;

    Name mWeightsBackup;

    template<typename MM>
    friend class ConvolutionDepthwiseLayerCPU;
};

} // raul namespace
#endif