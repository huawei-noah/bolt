// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef TENSORSTREAM_H
#define TENSORSTREAM_H

#include <iomanip>
#include <iostream>

namespace raul::io::tensor
{

enum class TensorView : long
{
    size = 0x1,
    content = 0x2,
    reduced = 0x4,
    scale = 0x8
};

inline TensorView operator|(TensorView a, TensorView b)
{
    return static_cast<TensorView>(static_cast<long>(a) | static_cast<long>(b));
}

int getStreamIndex();

struct setview
{
    explicit setview(TensorView flags)
        : flags(static_cast<long>(flags))
    {
    }

    explicit setview(long flags)
        : flags(flags)
    {
    }

    friend std::ostream& operator<<(std::ostream& os, const setview obj)
    {
        const auto index = getStreamIndex();
        os.iword(index) = obj.flags;
        return os;
    }

    long flags;
};

std::ostream& full(std::ostream& os);
std::ostream& compact(std::ostream& os);
std::ostream& brief(std::ostream& os);

bool isSetFlag(std::ostream& os, TensorView option);

} // namespace raul::io::tensor

#endif // TENSORSTREAM_H
