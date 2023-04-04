// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef LABEL_SMOOTHING_H
#define LABEL_SMOOTHING_H

#include <training/base/layers/BasicLayer.h>

#include <training/base/common/Common.h>

namespace raul
{
/*
 * @brief Label Smoothing
 *
 * The generalization and learning speed of a multi-class neural network can often be significantly improved
 * by using soft targets that are a weighted average of the hard targets and the uniform distribution over labels.
 * Smoothing the labels in this way prevents the network from becoming over-confident.
 *
 * Zero input vectors are a special case and are kept zero.
 *
 * Does nothing in Test Mode (as if smoothing is 0)
 *
 */
class LabelSmoothing : public BasicLayer
{
  public:
    LabelSmoothing(const Name& name, const LabelSmoothingParams& params, NetworkParameters& networkParameters);

    LabelSmoothing(LabelSmoothing&&) = default;
    LabelSmoothing(const LabelSmoothing&) = delete;
    LabelSmoothing& operator=(const LabelSmoothing&) = delete;

  private:
    Name mInputName;
    Name mOutputName;

    dtype mSmoothing;
    size_t mPaddingIdx;

    template<typename MM>
    friend class LabelSmoothingCPU;
};
} // raul namespace
#endif