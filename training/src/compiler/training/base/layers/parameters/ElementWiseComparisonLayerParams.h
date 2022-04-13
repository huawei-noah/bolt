// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef ELEMENT_WISE_COMPARISON_LAYER_PARAMS_H
#define ELEMENT_WISE_COMPARISON_LAYER_PARAMS_H

#include "BasicParameters.h"

namespace raul
{

/** Parameters for element-wise comparison layers
 * @param inputs vector of names of input tensors
 * @param outputs vector of names of output tensors
 * @param broadcast if true the layer does not throw exception when sizes do not match and tries broadcast tensors
 * @param comparator rule to compare input tensors elementwise with 12 predefined values (approximate "equal", "ne" (not equal), "less", "greater", "le" (less or equal), "ge" (greater or equal)
 * and their "exact" variants)
 * @param tolerance determine possible inaccuracy of comparison
 *
 * @note Broadcast can fail too
 * @see Tensor broadcast implementation
 */
struct ElementWiseComparisonLayerParams : public ElementWiseLayerParams
{
    ElementWiseComparisonLayerParams() = delete;

    ElementWiseComparisonLayerParams(const Names& inputs, const Names& outputs, const bool broadcast = true, const std::string comparator = std::string("equal"), const dtype tolerance = 0.0_dt)
        : ElementWiseLayerParams({ inputs, outputs }, broadcast)
        , mComparator(comparator)
        , mTolerance(tolerance)
    {
    }

    void print(std::ostream& stream) const override;

    std::string mComparator;
    raul::dtype mTolerance;
};

} // raul namespace

#endif // ELEMENT_WISE_COMPARISON_LAYER_PARAMS_H