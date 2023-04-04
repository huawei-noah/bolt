// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef TRANSPOSED_CONVOLUTION_2D_LAYER_H
#define TRANSPOSED_CONVOLUTION_2D_LAYER_H

#include <training/base/layers/TrainableLayer.h>

#include <training/base/common/Common.h>
#include <training/base/layers/parameters/trainable/TransposedConvolution2DParams.h>

namespace raul
{

/**
 * @brief Transposed Convolution 2D Layer
 * Applies a 2D transposed convolution operator over an input image
 * composed of several input planes. This module can be seen as the
 * gradient of Convolution 2D Layer with respect to its input.
 * It is also known as a fractionally-strided convolution or a deconvolution
 * (although it is not an actual deconvolution operation).
 *
 */
class TransposedConvolution2DLayer : public TrainableLayer
{
  public:
    TransposedConvolution2DLayer(const Name& name, const TransposedConvolution2DParams& params, NetworkParameters& networkParameters);

    TransposedConvolution2DLayer(TransposedConvolution2DLayer&&) = default;
    TransposedConvolution2DLayer(const TransposedConvolution2DLayer&) = delete;
    TransposedConvolution2DLayer& operator=(const TransposedConvolution2DLayer&) = delete;

    void forwardComputeImpl(NetworkMode mode) override;
    void backwardComputeImpl() override;

  protected:
    TransposedConvolution2DLayer(const Name& name, const std::string& typeName, const TransposedConvolution2DParams& params, NetworkParameters& networkParameters);

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

    size_t mOutputPaddingW;
    size_t mOutputPaddingH;

    bool mUseBias;

    size_t mDilationW;
    size_t mDilationH;

    bool mQuantizeWeights;
    Name mWeightsBackup;

    size_t mEffectiveReceptiveFieldW;
    size_t mEffectiveReceptiveFieldH;

    bool mDilationEnabled;
    Name mDilationTensor;

    Name mInputName;
    Name mOutputName;
    Name mPrevLayerDeltaName;

    Names mIm2ColForward;
    Names mIm2ColBackward;
};

} // raul namespace
#endif