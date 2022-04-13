// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef MAXPOOL_LAYER_H
#define MAXPOOL_LAYER_H

#include <training/base/layers/BasicLayer.h>

#include <training/base/common/Common.h>

namespace raul
{

/**
 * @brief 2D Max Pooling Layer
 *
 * The layer applies a 2D max-pooling over input signal with several channels.
 */
class MaxPoolLayer2D : public BasicLayer
{
  public:
    MaxPoolLayer2D(const Name& name, const Pool2DParams& params, NetworkParameters& networkParameters);

    MaxPoolLayer2D(MaxPoolLayer2D&&) = default;
    MaxPoolLayer2D(const MaxPoolLayer2D&) = delete;
    MaxPoolLayer2D& operator=(const MaxPoolLayer2D&) = delete;

    typedef std::vector<size_t> Indexes;

    void forwardComputeImpl(NetworkMode mode) override;
    void backwardComputeImpl() override;

  private:
    size_t mKernelWidth;
    size_t mKernelHeight;
    size_t mStrideW;
    size_t mStrideH;
    size_t mPaddingW;
    size_t mPaddingH;

    size_t mInputWidth;
    size_t mInputHeight;
    size_t mInputDepth;

    size_t mOutputWidth;
    size_t mOutputHeight;

    Indexes mIndexes;

    Name mInputName;
    Name mOutputName;
};

}
#endif