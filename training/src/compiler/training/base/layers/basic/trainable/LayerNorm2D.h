// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef LAYERNORM2D_LAYER_H
#define LAYERNORM2D_LAYER_H

#include <training/base/layers/TrainableLayer.h>
#include <training/base/layers/parameters/trainable/LayerNormParams.h>

#include <training/base/common/Common.h>

namespace raul
{

/**
 * @brief Layer Normalization
 *
 * An alternative technique to BatchNorm to normalize the distribution of layer outputs that performs exactly the same during training and inference.
 * This is a cummulative implementation - mean and variance are calculated along width and height
 * Weights and biases still have size = width
 */
class LayerNorm2DLayer : public TrainableLayer
{
  public:
    LayerNorm2DLayer(const Name& name, const LayerNormParams& params, NetworkParameters& networkParameters);

    LayerNorm2DLayer(LayerNorm2DLayer&&) = default;
    LayerNorm2DLayer(const LayerNorm2DLayer&) = delete;
    LayerNorm2DLayer& operator=(const LayerNorm2DLayer&) = delete;

  private:
    size_t mInputWidth;
    size_t mInputHeight;

    size_t mOutputWidth;
    size_t mOutputHeight;
    size_t mOutputSize;
    size_t mInputDepth;

    Name mInputName;
    Name mOutputName;

    Name mVarianceName;
    Name mXHatName;
    Name mXHatNablaName;

    const dtype mEps;
    bool mTFStyle;
    bool mUseBesselCorrection;

    template<typename MM>
    friend class LayerNorm2DLayerCPU;
};

}
#endif