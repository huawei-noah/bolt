// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef DROPOUT_LAYER_H
#define DROPOUT_LAYER_H

#include <training/base/layers/BasicLayer.h>
#include <training/base/layers/parameters/DropoutParams.h>

#include <training/base/common/Common.h>
#include <training/base/common/Random.h>

namespace raul
{

/**
 * @brief Dropout layer
 *
 * The main parameter is the probability of zeroing. It must be in [0, 1).
 */
class DropoutLayer : public BasicLayer
{
  public:
    DropoutLayer(const Name& name, const DropoutParams& params, NetworkParameters& networkParameters);

  private:
    // typedef std::vector<float> Random;
    // Random mRandom;

    dtype mScale;
    dtype mProbability;

    Name mInputName;
    Name mOutputName;

    std::mt19937_64 mState;

    Name mTmpBufferName;

    size_t mDepth;
    size_t mHeight;
    size_t mWidth;

    template<typename MM>
    friend class DropoutLayerCPU;
};

}
#endif