// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef CUM_SUM_LAYER_H
#define CUM_SUM_LAYER_H

#include <training/base/common/Common.h>
#include <training/base/layers/BasicLayer.h>

namespace raul
{

/**
 * @brief CumSumLayer
 *
 * Returns the cumulative sum of elements of input in the dimension dim.
 *
 */
class CumSumLayer : public BasicLayer
{
  public:
    CumSumLayer(const Name& name, const BasicParamsWithDim& params, NetworkParameters& networkParameters);

    CumSumLayer(CumSumLayer&&) = default;
    CumSumLayer(const CumSumLayer&) = delete;
    CumSumLayer& operator=(const CumSumLayer&) = delete;

  private:
    raul::Dimension mDimension;

    template<typename MM>
    friend class CumSumLayerCPU;
};

} // raul namespace

#endif