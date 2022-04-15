// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef SOFTPLUS_ACTIVATION_H
#define SOFTPLUS_ACTIVATION_H

#include <training/base/common/Common.h>
#include <training/base/layers/BasicLayer.h>
#include <training/base/layers/parameters/SoftPlusActivationParams.h>

namespace raul
{

/**
 * @brief SoftPlus activation function
 *
 * SoftPlus is a smooth approximation to the ReLU function and
 * can be used to constrain the output of a machine to always be positive.
 *
 * Function:
 *  \f[
 *      \mathrm{SoftPlus}(x) = \frac{1}{beta} * \log(1 + \exp(beta * x));
 *  \f]
 *
 * For TensorFlow behaviour use beta=1, threshold=MAX_FLOAT
 */
class SoftPlusActivation : public BasicLayer
{
  public:
    SoftPlusActivation(const Name& name, const SoftPlusActivationParams& params, NetworkParameters& networkParameters);

    SoftPlusActivation(SoftPlusActivation&&) = default;
    SoftPlusActivation(const SoftPlusActivation&) = delete;
    SoftPlusActivation& operator=(const SoftPlusActivation&) = delete;

  private:
    dtype mBeta;
    dtype mThreshold;

    Name mInputName;
    Name mOutputName;

    size_t mDepth;
    size_t mHeight;
    size_t mWidth;

    template<typename MM>
    friend class SoftPlusActivationCPU;
};

} // raul namespace

#endif