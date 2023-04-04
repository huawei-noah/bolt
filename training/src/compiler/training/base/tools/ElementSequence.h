// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef ELEMENTSEQUENCE_H
#define ELEMENTSEQUENCE_H

#include <cstdint>
#include <vector>

namespace raul
{

class ElementSequence
{
  public:
    ElementSequence(uint32_t, uint32_t) {}

    virtual uint32_t getElement() = 0;

    virtual ~ElementSequence() = default;
};

class MonotonicSequence : public ElementSequence
{
  public:
    MonotonicSequence(uint32_t min, uint32_t max);

    uint32_t getElement() final;

~MonotonicSequence(){}
  private:
    uint32_t mMin;
    uint32_t mMax;
    uint32_t mCurrentElement;
};

class RandomSequence : public ElementSequence
{
  public:
    RandomSequence(uint32_t min, uint32_t max);

    uint32_t getElement() final;
~ RandomSequence(){}
  private:
    void shuffle();

  private:
    std::vector<uint32_t> mElementSource;
    std::vector<uint32_t>::size_type mCurrentIdx = 0;
};

} // !namespace raul

#endif // ELEMENTSEQUENCE_H
