// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef ATTENTION_LAYER_H
#define ATTENTION_LAYER_H

#include <training/base/common/Common.h>
#include <training/base/layers/parameters/DropoutParams.h>
#include <training/base/common/NetworkParameters.h>

namespace raul
{
/**
 * @brief Dot-Product Attention Layer With Scaling
 *
 * This layer is a dot-product attention layer with additional scaling.
 *
 *
 * Inputs: Query, Value, Key[, Mask] or Query[, Mask]. In latter case Value=Key=Query
 * Outputs: Attention[, Probabilities]
 *
 * @see
 * Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin, �Attention Is All You Need�
 */
class AttentionLayer
{
  public:
    AttentionLayer(const Name& name, const DropoutParams& params, NetworkParameters& networkParameters);

    AttentionLayer(AttentionLayer&&) = default;
    AttentionLayer(const AttentionLayer&) = delete;
    AttentionLayer& operator=(const AttentionLayer&) = delete;

  private:
};
} // raul namespace
#endif