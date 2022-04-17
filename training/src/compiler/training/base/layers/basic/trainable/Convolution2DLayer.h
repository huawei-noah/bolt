// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef CONVOLUTION_2D_LAYER_H
#define CONVOLUTION_2D_LAYER_H

#include <training/base/layers/TrainableLayer.h>

#include <training/base/common/Common.h>
#include <training/base/layers/parameters/trainable/Convolution2DParams.h>

//#define RAUL_NAIVE_CONV_FORWARD
//#define RAUL_NAIVE_CONV_BACKWARD

namespace raul
{

/**
 * @brief Convolution 2D Layer
 *
 * The layer applies 2D convolution over input tensor, all channels convolved.
 */
class Convolution2DLayer : public TrainableLayer
{
  public:
    Convolution2DLayer(const Name& name, const Convolution2DParams& params, NetworkParameters& networkParameters);

    Convolution2DLayer(Convolution2DLayer&&) = default;
    Convolution2DLayer(const Convolution2DLayer&) = delete;
    Convolution2DLayer& operator=(const Convolution2DLayer&) = delete;

    size_t getInputWidth() const { return mInputWidth; }
    size_t getInputHeight() const { return mInputHeight; }
    size_t getInputDepth() const { return mInputDepth; }

    size_t getOutputWidth() const { return mOutputWidth; }
    size_t getOutputHeight() const { return mOutputHeight; }

    size_t getKernelWidth() const { return mKernelWidth; }
    size_t getKernelHeight() const { return mKernelHeight; }
    size_t getKernelsCount() const { return mKernelsCount; }

    size_t getStrideW() const { return mStrideW; }
    size_t getStrideH() const { return mStrideH; }

    size_t getPaddingW() const { return mPaddingW; }
    size_t getPaddingH() const { return mPaddingH; }

    size_t getDilationW() const { return mDilationW; }
    size_t getDilationH() const { return mDilationH; }

    size_t getGroups() const { return mGroups; }

    bool isUseBias() const { return mUseBias; }

    bool isQuantizeWeights() const { return mQuantizeWeights; }

    size_t getEffectiveReceptiveFieldW() const { return mEffectiveReceptiveFieldW; }
    size_t getEffectiveReceptiveFieldH() const { return mEffectiveReceptiveFieldH; }

    const Name& getInputName() const { return mInputName; }
    const Name& getOutputName() const { return mOutputName; }

    bool isDilationEnabled() const { return mDilationEnabled; }

    const Name& getDilationTensor() const { return mDilationTensor; }
    const Names& getIm2ColForward() const { return mIm2ColForward; }
    const Names& getIm2ColBackward() const { return mIm2ColBackward; }

#if defined(RAUL_NAIVE_CONV_BACKWARD)
    const Name& getTmpTensorName() const { return mTmpTensorName; }
#endif

    const Name& getWeightsBackup() const { return mWeightsBackup; }

  private:
    Convolution2DLayer(const Name& name, const std::string& typeName, const Convolution2DParams& params, NetworkParameters& networkParameters);

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

    size_t mDilationW;
    size_t mDilationH;

    size_t mGroups;

    bool mUseBias;

    bool mQuantizeWeights;

    // Taking into account dilation
    size_t mEffectiveReceptiveFieldW;
    size_t mEffectiveReceptiveFieldH;

    Name mInputName;
    Name mOutputName;

    bool mDilationEnabled;
    Name mDilationTensor;

    Names mIm2ColForward;
    Names mIm2ColBackward;
#if defined(RAUL_NAIVE_CONV_BACKWARD)
    Name mTmpTensorName;
#endif

    Name mWeightsBackup;
};

} // raul namespace
#endif