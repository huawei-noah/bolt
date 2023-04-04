// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef BASIC_IMPL_H
#define BASIC_IMPL_H

#include <training/base/common/Common.h>

namespace raul
{

/**
 * @brief Base class for all implementations
 */
class BasicImpl
{
  public:
    virtual ~BasicImpl() = default;

    virtual void initNotBSTensors() {}
    virtual void onBatchSizeChanged(size_t /*newBatchSize*/) {}
    virtual void forwardComputeImpl(NetworkMode) = 0;
    virtual void backwardComputeImpl() = 0;
};

/**
 * @brief Stub class
 */
class NotImplemented : public BasicImpl
{
  public:
    template<typename T>
    NotImplemented(T&)
    {
    }
    void forwardComputeImpl(NetworkMode) {}
    void backwardComputeImpl() {}
 ~NotImplemented(){}
};

class DummyImpl : public BasicImpl
{
  public:
    template<typename T>
    DummyImpl(T&)
    {
    }
    ~DummyImpl(){}
    void forwardComputeImpl(NetworkMode) {}
    void backwardComputeImpl() {}

};

} // raul namespace

#endif // BASIC_IMPL_H
