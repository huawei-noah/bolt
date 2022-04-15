// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include <training/base/common/Common.h>
#include <training/base/tools/ElementSequence.h>

#include <random>

using namespace raul;

MonotonicSequence::MonotonicSequence(uint32_t min, uint32_t max)
    : ElementSequence(min, max)
    , mMin(min)
    , mMax(max)
    , mCurrentElement(mMin)
{
}

uint32_t MonotonicSequence::getElement()
{
    if (mCurrentElement > mMax)
    {
        mCurrentElement = mMin;
    }

    return mCurrentElement++;
}

RandomSequence::RandomSequence(uint32_t min, uint32_t max)
    : ElementSequence(min, max)
{
    mElementSource.resize(max > min ? max - min + 1 : min - max + 1);
    std::generate(mElementSource.begin(), mElementSource.end(), [n = min]() mutable { return n++; });
    shuffle();
}

uint32_t RandomSequence::getElement()
{
    if (mCurrentIdx >= mElementSource.size())
    {
        mCurrentIdx = 0;
        shuffle();
    }
    return mElementSource[mCurrentIdx++];
}

void RandomSequence::shuffle()
{
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(mElementSource.begin(), mElementSource.end(), g);
}
