// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef RELU_ACTIVATION_IMPL_H
#define RELU_ACTIVATION_IMPL_H

#include <training/base/layers/BasicImpl.h>

namespace raul
{
class ReLUActivation;
class ReLU6Activation;

/**
 * @brief Rectified linear unit activation function layer HW independent implementation
 */
template<typename MM>
class ReLUActivationImpl : public BasicImpl
{
  public:
    ReLUActivationImpl(ReLUActivation& layer)
        : mLayer(layer)
    {
    }

    ReLUActivationImpl(ReLUActivationImpl&&) = default;
    ReLUActivationImpl(const ReLUActivationImpl&) = delete;
    ReLUActivationImpl& operator=(const ReLUActivationImpl&) = delete;

    void forwardComputeImpl(NetworkMode mode) override;
    void backwardComputeImpl() override;

  private:
    ReLUActivation& mLayer;
};

/**
 * @brief Rectified linear unit 6 activation function layer HW independent implementation
 */
template<typename MM>
class ReLU6ActivationImpl : public BasicImpl
{
  public:
    ReLU6ActivationImpl(ReLU6Activation& layer)
        : mLayer(layer)
    {
    }

    ReLU6ActivationImpl(ReLU6ActivationImpl&&) = default;
    ReLU6ActivationImpl(const ReLU6ActivationImpl&) = delete;
    ReLU6ActivationImpl& operator=(const ReLU6ActivationImpl&) = delete;

    void forwardComputeImpl(NetworkMode mode) override;
    void backwardComputeImpl() override;

  private:
    ReLU6Activation& mLayer;
};
} // raul namespace

#endif
