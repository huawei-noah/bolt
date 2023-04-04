// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef TENSOR_MEM_H
#define TENSOR_MEM_H

#include <vector>

#include <training/base/common/Common.h>

namespace raul
{

template<typename dt>
class TensorMem
{
  public:
    explicit TensorMem(AllocationMode allocMode)
        : mAllocationMode(allocMode)
        , mData(nullptr)
        , mSize(0)
    {
    }

    TensorMem(size_t size, dt filler)
        : mAllocationMode(AllocationMode::STANDARD)
        , mData(nullptr)
        , mMem(size, filler)
    {
        if (size != 0)
        {
            mData = &mMem[0];
        }

        mSize = size;
    }

    explicit TensorMem(std::initializer_list<dt> list)
        : mAllocationMode(AllocationMode::STANDARD)
        , mData(nullptr)
        , mMem(list)
    {
        if (!mMem.empty())
        {
            mData = &mMem[0];
        }

        mSize = mMem.size();
    }

    TensorMem(const dt* first, const dt* sesond)
        : mAllocationMode(AllocationMode::STANDARD)
        , mData(nullptr)
        , mMem(first, sesond)
    {
        if (!mMem.empty())
        {
            mData = &mMem[0];
        }

        mSize = mMem.size();
    }

    ~TensorMem(){}
    INLINE void resize(size_t size, dt* data)
    {
        if (mAllocationMode == AllocationMode::STANDARD)
        {
            mMem.resize(size);
            if (size != 0)
            {
                mData = &mMem[0];
            }
            else
            {
                mData = nullptr;
            }

            mSize = size;
        }
        else
        {
            mData = data;

            if (mData != nullptr)
            {
                mSize = size;

                std::fill(mData, mData + mSize, static_cast<dt>(0));
            }
            else
            {
                mSize = 0;
            }
        }
    }

    INLINE size_t size() const { return mSize; }

    INLINE bool empty() const { return mSize == 0; }

    INLINE dt* begin() { return &mData[0]; }

    INLINE const dt* begin() const { return &mData[0]; }

    INLINE dt* end() { return &mData[mSize]; }

    INLINE const dt* end() const { return &mData[mSize]; }

    INLINE void clear()
    {
        if (mAllocationMode == AllocationMode::STANDARD)
        {
            mMem.clear();
        }

        mData = nullptr;
        mSize = 0;
    }

    INLINE void shrink_to_fit()
    {
        if (mAllocationMode == AllocationMode::STANDARD)
        {
            mMem.shrink_to_fit();
        }
    }

    INLINE dt& operator[](size_t index) noexcept { return mData[index]; }

    INLINE const dt& operator[](size_t index) const noexcept { return mData[index]; }

    template<size_t NewDimsNum>
    INLINE auto reshape(const yato::dimensionality<NewDimsNum, size_t>& extents) const
    {
        return yato::array_view<const dt>(mData, yato::dims(mSize)).reshape(extents);
    }

    template<size_t NewDimsNum>
    INLINE auto reshape(const yato::dimensionality<NewDimsNum, size_t>& extents)
    {
        return yato::array_view<dt>(mData, yato::dims(mSize)).reshape(extents);
    }

    AllocationMode getAllocationMode() const { return mAllocationMode; }

  private:
    AllocationMode mAllocationMode;

    dt* mData;
    size_t mSize;

    // used in NORMAL AllocationMode
    std::vector<dt> mMem;
};
} // raul namespace

#endif
