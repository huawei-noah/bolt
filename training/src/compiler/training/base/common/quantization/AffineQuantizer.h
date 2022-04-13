// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#pragma once
#include "training/base/common/Tensor.h"
#include "training/base/common/quantization/IQuantizer.h"
#include <functional>
#include <optional>
#include <utility>
#include <variant>

namespace raul::quantization
{

/**
 * @brief Affine quantization algorithm
 */
struct AffineQuantizer : IQuantizer
{
    AffineQuantizer(RoundFuncT roundFunc, const dtype scale, const dtype offset, std::optional<RangeT> quantized_range = std::nullopt)
        : mRoundFunc(std::move(roundFunc))
        , mScale(scale)
        , mOffset(offset)
        , mQuantizedRange(quantized_range)
    {
    }

    void quantize(TensorItr begin, TensorItr end) override;

    void dequantize(TensorItr begin, TensorItr end) override;

    void backpropagate(TensorConstItr begin, TensorConstItr end, TensorConstItr delta_begin, TensorItr grad_begin) override;

  protected:
    RoundFuncT mRoundFunc;
    dtype mScale;
    dtype mOffset;
    std::optional<RangeT> mQuantizedRange;
};

} // raul::quantization