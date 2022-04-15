// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "CosineAnnealing.h"
#include <iostream>

#define _USE_MATH_DEFINES
#include <math.h>

namespace raul::optimizers::Scheduler::Strategies
{

std::ostream& CosineAnnealing::as_ostream(std::ostream& out) const
{
    out << "Strategies::CosineAnnealing()";
    return out;
}

void CosineAnnealing::calculateLR(dtype baseLR, long long step, dtype& currentLR)
{
    Base::calculateLR(baseLR, step, currentLR); // value calculated here is not used due to the nature of this strategy

    float value = 1.f;

    size_t i = step - 1;

    if (static_cast<size_t>(i) < mWarmupLoops)
    {
        i = mWarmupLoops - 1 - i;
        value = static_cast<float>(cos(M_PI * i / mWarmupLoops));
        value = 0.5f * (value + 1.f);
        value = pow(value, mWarmupPow);
    }
    else
    {
        i = i - mWarmupLoops;
        value = static_cast<float>(cos(M_PI * i / mAnnealingLoops));
        value = 0.5f * (value + 1.f);
        value = pow(value, mAnnealingPow);
    }
    value = (mMaxA - mMinA) * value + mMinA;
    currentLR = baseLR * value;
}

} // raul::optimizers::Scheduler::Strategies