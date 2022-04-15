// Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

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

#include <training/common/Common.h>
#include <training/layers/TrainableLayer.h>
#include <training/layers/parameters/trainable/LinearParams.h>

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

  private:
    size_t mInputsCount;
    size_t mOutputsCount;
    size_t mDepth;
    size_t mHeight;

    bool mUseBias;

    Name mInputName;
    Name mOutputName;

    template<typename MM>
    friend class LinearLayerImpl;
    friend class LinearLayerCPUFP16;
    friend class LinearLayerCPUFP32;
    friend class LinearLayerCPUFP32Master;

    size_t mForwardAlignedWeightsSize = 0;
    size_t mBackwardAlignedWeightsSize = 0;

    Name mTransposedWeightsName;
    Name mTransposedPaddedWeightsName;
    Name mPaddedWeightsName;

    Name mForwardWeightsNameGpu;
    Name mBackwardWeightsNameGpu;

    enum class WeightsUsage
    {
        Unchanged,
        Transposed,
        Padded
    };

    WeightsUsage mForwardWeightsUsage;
    WeightsUsage mBackwardWeightsUsage;
};
} // raul namespace

#endif
