// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef LINEAR_LAYER_H
#define LINEAR_LAYER_H

#include <training/base/common/Common.h>
#include <training/base/layers/TrainableLayer.h>
#include <training/base/layers/parameters/trainable/LinearParams.h>

namespace raul
{
/**
 * @brief Linear transform
 *
 *  Linear transformation of rows of tensor.
 *  Batch, Depth and Height dimensions are kept unchanged.
 *  [N, C, H, W] -> [N, C, H, L]
 *
 *  Weights are stored transposed ([L, W]) as in torch.nn.Linear
 *
 */
class LinearLayer : public TrainableLayer
{
  public:
    LinearLayer(const Name& name, const LinearParams& params, NetworkParameters& networkParameters);

    LinearLayer(LinearLayer&&) = default;
    LinearLayer(const LinearLayer&) = delete;
    LinearLayer& operator=(const LinearLayer&) = delete;

    size_t getInputsCount() const { return mInputsCount; }
    size_t getOutputsCount() const { return mOutputsCount; }
    size_t getDepth() const { return mDepth; }
    size_t getHeight() const { return mHeight; }

    bool isUseBias() const { return mUseBias; }

    const Name& getInputName() const { return mInputName; }
    const Name& getOutputName() const { return mOutputName; }

    size_t getForwardAlignedWeightsSize() const { return mForwardAlignedWeightsSize; }
    size_t getBackwardAlignedWeightsSize() const { return mBackwardAlignedWeightsSize; }

    const Name& getTransposedWeightsName() const { return mTransposedWeightsName; }
    const Name& getTransposedPaddedWeightsName() const { return mTransposedPaddedWeightsName; }
    const Name& getPaddedWeightsName() const { return mPaddedWeightsName; }

    enum class WeightsUsage
    {
        Unchanged,
        Transposed,
        Padded
    };

    WeightsUsage getForwardWeightsUsage() const { return mForwardWeightsUsage; }
    WeightsUsage getBackwardWeightsUsage() const { return mBackwardWeightsUsage; }

  private:
    size_t mInputsCount;
    size_t mOutputsCount;
    size_t mDepth;
    size_t mHeight;

    bool mUseBias;

    Name mInputName;
    Name mOutputName;

    size_t mForwardAlignedWeightsSize = 0;
    size_t mBackwardAlignedWeightsSize = 0;

    Name mTransposedWeightsName;
    Name mTransposedPaddedWeightsName;
    Name mPaddedWeightsName;

    WeightsUsage mForwardWeightsUsage;
    WeightsUsage mBackwardWeightsUsage;
};
} // raul namespace

#endif