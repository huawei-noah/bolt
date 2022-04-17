// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef TRANSPOSE_LAYER_H
#define TRANSPOSE_LAYER_H

#include <training/base/layers/BasicLayer.h>

#include <training/base/common/Common.h>

namespace raul
{
/**
 * @brief Transpose a tensor
 *
 * Setting both dimensions to Dimension::Default will swap W and H dimensions.
 */
class TransposeLayer : public BasicLayer
{
  public:
    TransposeLayer(const Name& name, const TransposingParams& params, NetworkParameters& networkParameters);

    TransposeLayer(TransposeLayer&&) = default;
    TransposeLayer(const TransposeLayer&) = delete;
    TransposeLayer& operator=(const TransposeLayer&) = delete;

  private:
    Name mInputName;
    Name mOutputName;

    size_t mDim1 = 0;
    size_t mDim2 = 0;

    template<typename MM>
    friend class TransposeLayerCPU;
};
} // raul namespace
#endif