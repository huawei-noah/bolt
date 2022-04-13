// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include <algorithm>

#include "training/base/common/quantization/SymmetricQuantizer.h"

namespace raul::quantization
{

void SymmetricQuantizer::quantize(TensorItr begin, TensorItr end)
{
    mScale = calcScale(begin, end);
    AffineQuantizer::quantize(begin, end);
}

void SymmetricQuantizer::backpropagate(TensorConstItr begin, TensorConstItr end, TensorConstItr delta_begin, TensorItr grad_begin)
{
    mScale = calcScale(begin, end);
    AffineQuantizer::backpropagate(begin, end, delta_begin, grad_begin);
}

dtype SymmetricQuantizer::calcScale(TensorConstItr begin, TensorConstItr end) const
{
    const auto& max_abs_value_ref = std::max_element(begin, end, [](dtype a, dtype b) { return std::abs(a) < std::abs(b); });
    const dtype max_abs_value = std::abs(*max_abs_value_ref);
    const dtype scale = (mQuantizedRange->max - mQuantizedRange->min) / max_abs_value / 2.0_dt;
    return scale;
}

RangeT SymmetricQuantizer::calcQuantizedRange() const
{
    if (mDigits == 0U)
    {
        THROW_NONAME("SymmetricQuantizer", "digits must be nonzero");
    }
    const auto max_signed_positive_value = std::pow(2, mDigits - 1U);
    if (mMode == Mode::restricted_range)
    {
        const auto max_value = max_signed_positive_value - 1;
        const auto min_value = -(max_signed_positive_value - 1);
        return { static_cast<dtype>(min_value), static_cast<dtype>(max_value) };
    }
    else
    {
        const auto max_value = max_signed_positive_value - 1;
        const auto min_value = -max_signed_positive_value;
        return { static_cast<dtype>(min_value), static_cast<dtype>(max_value) };
    }
}

} // raul::quantization