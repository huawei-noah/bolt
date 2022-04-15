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
#include <training/base/layers/parameters/trainable/Convolution1DParams.h>

//#define RAUL_NAIVE_CONV_FORWARD
//#define RAUL_NAIVE_CONV_BACKWARD

namespace raul
{

/**
 * @brief Convolution 1D Layer
 *
 * Applies a 1D convolution over an input signal composed of several input planes.
 * Supports 2 modes:
 *  1. PyTorch style: Input[N, C, 1, L1] (or [N, 1, C, L1]) -> Output[N, FILTERS, 1, L2] (or [N, 1, FILTERS, L2])
 *  2. TensorFlow style: Input[N, L1, 1, C] (or [N, 1, L1, C]) -> Output[N, L2, 1, FILTERS] (or [N, 1, L2, FILTERS])
 *
 */
class Convolution1DLayer : public TrainableLayer
{
  public:
    Convolution1DLayer(const Name& name, const Convolution1DParams& params, NetworkParameters& networkParameters);

    Convolution1DLayer(Convolution1DLayer&&) = default;
    Convolution1DLayer(const Convolution1DLayer&) = delete;
    Convolution1DLayer& operator=(const Convolution1DLayer&) = delete;

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
    size_t mDilation;
    size_t mGroups;

    bool mUseBias;

    bool mQuantizeWeights;
    Name mWeightsBackup;

    Name mInputName;
    Name mOutputName;

    size_t mEffectiveReceptiveField;
    bool mDilationEnabled;
    Name mDilationTensor;

    Names mIm2ColForward;
    Names mIm2ColBackward;
    Names mTempWeightsGrag;

    Name mTmpIm2ColName;
#if defined(RAUL_NAIVE_CONV_BACKWARD)
    Name mTmpForBackwardName;
#endif
    bool mTFStyle;
    Name mTempWeights; // for TF style

    template<typename MM>
    friend class Convolution1DLayerCPU;
};

} // raul namespace
#endif
