// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef GELU_ACTIVATION_H
#define GELU_ACTIVATION_H

#include <training/base/layers/BasicLayer.h>

#include <training/base/common/Common.h>

namespace raul
{

/**
 * @brief Gaussian error linear activation function
 *
 * The layer applies the element-wise gaussian error linear function.
 */
class GeLUErf : public BasicLayer
{
  public:
    GeLUErf(const Name& name, const BasicParams& params, NetworkParameters& networkParameters);

    GeLUErf(GeLUErf&&) = default;
    GeLUErf(const GeLUErf&) = delete;
    GeLUErf& operator=(const GeLUErf&) = delete;

  protected:
    GeLUErf(const Name& name, const Name& typeName, const BasicParams& params, NetworkParameters& networkParameters);

    Name mInputName;
    Name mOutputName;

    size_t mDepth;
    size_t mHeight;
    size_t mWidth;

    template<typename MM>
    friend class GeLUErfCPU;
};

class GeLUTanh : public GeLUErf
{
  public:
    GeLUTanh(const Name& name, const BasicParams& params, NetworkParameters& networkParameters);

    GeLUTanh(GeLUTanh&&) = default;
    GeLUTanh(const GeLUTanh&) = delete;
    GeLUTanh& operator=(const GeLUTanh&) = delete;

    template<typename MM>
    friend class GeLUTanhCPU;
};

} // raul namespace
#endif