// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef RANDOM_TENSOR_LAYER_H
#define RANDOM_TENSOR_LAYER_H

#include <training/base/common/Common.h>
#include <training/base/layers/BasicLayer.h>
#include <training/base/layers/parameters/RandomTensorLayerParams.h>

namespace raul
{

/**
 * @brief RandomTensorLayer
 * Creates tensor with specified shape and
 * fills it by values from random normal distribution.
 */
class RandomTensorLayer : public BasicLayer
{
  public:
    RandomTensorLayer(const Name& name, const RandomTensorLayerParams& params, NetworkParameters& networkParameters);

    RandomTensorLayer(RandomTensorLayer&&) = default;
    RandomTensorLayer(const RandomTensorLayer&) = delete;
    RandomTensorLayer& operator=(const RandomTensorLayer&) = delete;

  private:
    size_t mDepth;
    size_t mHeight;
    size_t mWidth;

    raul::dtype mMean;
    raul::dtype mStdDev;
    std::default_random_engine mGenerator;

    template<typename MM>
    friend class RandomTensorLayerCPU;
};

} // raul namespace

#endif