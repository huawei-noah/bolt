// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef SWISH_ACTIVATION_CPU_H
#define SWISH_ACTIVATION_CPU_H

#include <training/base/layers/BasicImpl.h>

namespace raul
{

class SwishActivation;

/**
 * @brief SwishActivation layer CPU implementation
 */
template<typename MM>
class SwishActivationCPU : public BasicImpl
{
  public:
    SwishActivationCPU(SwishActivation& layer)
        : mLayer(layer)
    {
    }

    SwishActivationCPU(SwishActivationCPU&&) = default;
    SwishActivationCPU(const SwishActivationCPU&) = delete;
    SwishActivationCPU& operator=(const SwishActivationCPU&) = delete;

    void forwardComputeImpl(NetworkMode mode) override;
    void backwardComputeImpl() override;

  private:
    SwishActivation& mLayer;
};

} // raul namespace

#endif