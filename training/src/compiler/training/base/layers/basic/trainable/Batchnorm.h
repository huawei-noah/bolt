// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef BATCHNORM_LAYER_H
#define BATCHNORM_LAYER_H

#include <training/base/layers/TrainableLayer.h>
#include <training/base/layers/parameters/trainable/BatchnormParams.h>

#include <training/base/common/Common.h>

namespace raul
{

/**
 * @brief Batch normalization layer
 *
 * The layer applies batch normalization over 4D input tensor ([batch, channel, 2D inputs]).
 */
class BatchNormLayer : public TrainableLayer
{
  public:
    BatchNormLayer(const Name& name, const BatchnormParams& params, NetworkParameters& networkParameters);

    BatchNormLayer(BatchNormLayer&&) = default;
    BatchNormLayer(const BatchNormLayer&) = delete;
    BatchNormLayer& operator=(const BatchNormLayer&) = delete;

  private:
    size_t mInputWidth;
    size_t mInputHeight;
    size_t mInputDepth;

    size_t mChosenDimSize;
    std::array<size_t, 2> mOtherDims;

    Name mInputName;
    Name mOutputName;

    dtype mMomentum;
    const dtype mEps;
    raul::Dimension mDimension;

    template<typename MM>
    friend class BatchNormLayerCPU;
};

}
#endif