// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef REVERSE_LAYER_H
#define REVERSE_LAYER_H

#include <training/base/common/Common.h>
#include <training/base/layers/BasicLayer.h>

namespace raul
{

/**
 * @brief Reverse Layer
 *
 * Given a real length of a sequence, reverse tensor elements from range [0, real_length].
 * Works only when depth or height of the input tensor is equal to 1.
 *
 */

class ReverseLayer : public BasicLayer
{
  public:
    ReverseLayer(const Name& name, const BasicParams& params, NetworkParameters& networkParameters);

    ReverseLayer(ReverseLayer&&) = default;
    ReverseLayer(const ReverseLayer&) = delete;
    ReverseLayer& operator=(const ReverseLayer&) = delete;

  private:
    bool mReverseOnly;

    Name mInputName;
    Name mOutputName;
    template<typename MM>
    friend class ReverseLayerCPU;
};

} // raul namespace

#endif