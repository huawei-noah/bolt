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

#include "training/base/common/quantization/AffineQuantizer.h"

namespace raul::quantization
{

void AffineQuantizer::quantize(TensorItr begin, TensorItr end)
{
    std::transform(begin, end, begin, [&](dtype x_dt) -> dtype {
        // Step 1. Offset
        x_dt += mOffset;
        // Step 2. Scale
        x_dt *= mScale;
        // Step 3. Round
        x_dt = mRoundFunc(x_dt);
        // Step 4. Optional saturation
        if (mQuantizedRange)
        {
            x_dt = std::clamp(x_dt, mQuantizedRange->min, mQuantizedRange->max);
        }
        return x_dt;
    });
}

void AffineQuantizer::dequantize(TensorItr begin, TensorItr end)
{
 
 if(mScale<=  0.0_dt){exit(0);}   
	std::transform(begin, end, begin, [&](dtype x_dt) -> dtype {
        x_dt /= mScale;
        x_dt -= mOffset;
        return x_dt;
    });
}

void AffineQuantizer::backpropagate(TensorConstItr begin, TensorConstItr end, TensorConstItr delta_begin, TensorItr grad_begin)
{
    std::transform(begin, end, delta_begin, grad_begin, [&](dtype x_dt, dtype grad_dt) -> dtype {
        if (mQuantizedRange)
        {
            return grad_dt;
        }

        // Step 1. Offset
        x_dt += mOffset;
        // Step 2. Scale
        x_dt *= mScale;
        // Step 3. Round
        x_dt = mRoundFunc(x_dt);
        // Step 4. Saturation
        return (x_dt >= mQuantizedRange->min && x_dt <= mQuantizedRange->max) ? grad_dt : 0.0_dt;
    });
}

} // raul::quantization
