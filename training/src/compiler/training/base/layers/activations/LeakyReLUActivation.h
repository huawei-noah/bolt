// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef LEAKY_RELU_ACTIVATION_H
#define LEAKY_RELU_ACTIVATION_H

#include "training/base/layers/BasicLayer.h"
#include "training/base/layers/parameters/LeakyReLUParams.h"

#include <training/base/common/Common.h>

namespace raul
{

/**
 * @brief Leaky rectified linear unit activation function
 *
 * The layer applies the element-wise leaky rectified linear unit.
 */
class LeakyReLUActivation : public BasicLayer
{
  public:
    LeakyReLUActivation(const Name& name, const LeakyReLUParams& params, NetworkParameters& networkParameters);

    LeakyReLUActivation(LeakyReLUActivation&&) = default;
    LeakyReLUActivation(const LeakyReLUActivation&) = delete;
    LeakyReLUActivation& operator=(const LeakyReLUActivation&) = delete;

  protected:
    LeakyReLUActivation(const Name& name, const std::string& typeName, const LeakyReLUParams& params, NetworkParameters& networkParameters);

    dtype mNegativeSlope;

    Name mInputName;
    Name mOutputName;

    template<typename MM>
    friend class LeakyReLUActivationCPU;
};

} // raul namespace

#endif