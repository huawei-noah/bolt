// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef SYSTEM_ZIP_H
#define SYSTEM_ZIP_H

namespace raul::system
{
template<typename T>
using iter = std::conditional_t<std::is_const_v<std::remove_reference_t<T>>, typename std::decay_t<T>::const_iterator, typename std::decay_t<T>::iterator>;

template<typename A, typename B>
struct Zipper
{
    using IterA = iter<A>;
    using IterB = iter<B>;

    struct Iterator
    {
        Iterator(IterA initIterA, IterB initIterB)
            : iterA{ initIterA }
            , iterB{ initIterB }
        {
        }

        auto operator++()
        {
            ++iterA;
            ++iterB;
            return *this;
        }

        auto operator==(const Iterator& other) { return iterA == other.iterA || iterB == other.iterB; }
        auto operator!=(const Iterator& other) { return !(*this == other); }

        auto operator*() { return std::make_pair(*iterA, *iterB); }

        IterA iterA;
        IterB iterB;
    };

    Zipper(A initA, B initB)
        : a(initA)
        , b(initB)
    {
    }

    auto begin() { return Iterator{ std::begin(a), std::begin(b) }; }
    auto end() { return Iterator{ std::end(a), std::end(b) }; }

  private:
    A a;
    B b;
};

template<typename A, typename B>
auto zip(A a, B b)
{
    return Zipper<A, B>{ std::forward<A>(a), std::forward<B>(b) };
}

}

#endif // SYSTEM_ZIP_H
