// Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "Miscellaneous.h"

#include <algorithm>
namespace raul::tacotron
{

template<typename T>
void sequence_mask(const T& lengths, size_t r, T& mask)
{
    size_t maxLen = static_cast<size_t>(*std::max_element(lengths.begin(), lengths.end()));

    // _round_up_tf
    size_t remainder = maxLen % r;
    if (remainder != 0)
    {
        maxLen = maxLen - remainder + r;
    }

    if (maxLen > mask.getHeight())
    {
        THROW_NONAME("Tacotron", "sequence longer than allocated maximum size");
    }

    auto mask2d = mask.reshape(yato::dims(lengths.size(), mask.size() / lengths.size()));
    std::fill(mask.begin(), mask.end(), static_cast<typename T::type>(0));
    for (size_t q = 0; q < lengths.size(); ++q)
    {
        size_t len = static_cast<size_t>(lengths[q]) * mask.getWidth();
        std::fill(mask2d[q].begin(), mask2d[q].begin() + len, static_cast<typename T::type>(1));
    }
}

template void sequence_mask<Tensor>(const Tensor& lengths, size_t r, Tensor& mask);
template void sequence_mask<TensorFP16>(const TensorFP16& lengths, size_t r, TensorFP16& mask);

} // namespace raul::tacotron
