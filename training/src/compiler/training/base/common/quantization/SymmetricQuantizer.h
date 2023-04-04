// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef SYMMETRIC_QUANTIZER_H
#define SYMMETRIC_QUANTIZER_H

#include <functional>
#include <optional>
#include <variant>

#include <training/base/common/Tensor.h>
#include <training/base/common/quantization/AffineQuantizer.h>
#include <training/base/common/quantization/IQuantizer.h>
#include <utility>

namespace raul::quantization
{

/** @brief Symmetric affine (linear) quantization
 *
 *  The quantizer performs two steps: symmetric mapping and rounding
 *
 *  1. Symmetric mapping:
 *  \f[
 *      x_{q} = \textrm{round}(\alpha x_{f}),
 *  \f]
 *  where \f$x_{f}\f$ is floating value, \f$x_{q}\f$ is quantized value,  \f$\alpha\f$ is a scale factor.
 *
 *  The scale factor depends on the mode of Symmetric quantizers.
 *
 *  **Full-range mode**
 *  \f[
 *      \alpha = \frac{1}{2}\frac{2^n-1}{\max{|{x_{f}|}}}.
 *  \f]
 *
 *  **Restricted-range mode**
 *  \f[
 *      \alpha = \frac{2^n-1}{\max{|{x_{f}|}}}.
 *  \f]
 *
 *  @see
 *  - @ref page_rounding "Rounding algorithms"
 *
 */
struct SymmetricQuantizer final : AffineQuantizer
{
    enum class Mode
    {
        full_range,
        restricted_range
    };

    SymmetricQuantizer(RoundFuncT roundFunc, const uint8_t digits = DEFAUL_QUANTIZATION_BITSIZE, const Mode mode = Mode::restricted_range)
        : AffineQuantizer(std::move(roundFunc), 0, 0, std::nullopt)
        , mDigits(digits)
        , mMode(mode)
    {
        mQuantizedRange = calcQuantizedRange();
    }

    void quantize(TensorItr begin, TensorItr end) override;
    void backpropagate(TensorConstItr begin, TensorConstItr end, TensorConstItr delta_begin, TensorItr grad_begin) override;

    dtype calcScale(TensorConstItr begin, TensorConstItr end) const;

  private:
    RangeT calcQuantizedRange() const;

    uint8_t mDigits;
    Mode mMode;
};

} // raul::quantization

#endif // SYMMETRIC_QUANTIZER_H