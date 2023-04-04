// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef CONVOLUTION_1D_LAYER_H
#define CONVOLUTION_1D_LAYER_H

#include <training/base/layers/TrainableLayer.h>

#include <training/base/common/Common.h>
#include <training/base/layers/basic/TransposeLayer.h>
#include <training/base/layers/parameters/trainable/TransposedConvolution1DParams.h>

namespace raul
{

/**
 * @brief Convolution1D Layer
 * Applies a 1D transposed convolution operator over an input image composed of several input planes.
 * This module can be seen as the gradient of Conv1d with respect to its input. It is also known as
 * a fractionally-strided convolution or a deconvolution (although it is not an actual deconvolution operation).
 *
 */
class TransposedConvolution1DLayer : public TrainableLayer
{
  public:
    TransposedConvolution1DLayer(const Name& name, const TransposedConvolution1DParams& params, NetworkParameters& networkParameters);

    TransposedConvolution1DLayer(TransposedConvolution1DLayer&&) = default;
    TransposedConvolution1DLayer(const TransposedConvolution1DLayer&) = delete;
    TransposedConvolution1DLayer& operator=(const TransposedConvolution1DLayer&) = delete;

    void forwardComputeImpl(NetworkMode mode) override;
    void backwardComputeImpl() override;

  protected:
    size_t mInputSize;
    size_t mInputChannels;

    size_t mOutputSize;
    size_t mOutputChannels;

    size_t mKernelSize;
    size_t mStride;
    size_t mPadding;
    size_t mOutputPadding;
    size_t mDilation;
    size_t mGroups;

    bool mUseBias;

    bool mQuantizeWeights;
    Name mWeightsBackup;

    size_t mEffectiveReceptiveField;

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