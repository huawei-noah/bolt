// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include <tests/tools/TestTools.h>

#include <training/base/common/quantization/SymmetricQuantizer.h>

namespace UT
{

using namespace raul::quantization;

TEST(TestQuantization, SymmetricQuantizeSimpleUnit)
{
    PROFILE_TEST
    const auto digits = 8U;
    const auto round = static_cast<raul::dtype (*)(raul::dtype)>(&std::trunc);
    auto quantizer = raul::quantization::SymmetricQuantizer(round, digits);

    const raul::Tensor x{ 1.2_dt, 1.5_dt, 1.7_dt, -1.2_dt, -1.5_dt, -1.7_dt };
    raul::Tensor y = x;

    quantizer.quantize(y.begin(), y.end());

    // Check
    constexpr auto max_value = 1.7_dt;
    const auto max_mapped_value = static_cast<raul::dtype>(std::pow(2, digits - 1) - 1);
    const raul::dtype scale = max_mapped_value / max_value;

    for (size_t i = 0; i < y.size(); ++i)
    {
        raul::dtype golden_val = round(x[i] * scale);
        EXPECT_FLOAT_EQ(golden_val, y[i]);
    }
}

TEST(TestQuantization, SymmetricQudeqSimpleUnit)
{
    PROFILE_TEST
    const auto digits = 8U;
    const auto round = static_cast<raul::dtype (*)(raul::dtype)>(&std::trunc);
    auto quantizer = raul::quantization::SymmetricQuantizer(round, digits);

    const raul::Tensor x{ 1.2_dt, 1.5_dt, 1.7_dt, -1.2_dt, -1.5_dt, -1.7_dt };
    raul::Tensor y = x;

    quantizer.quantize(y.begin(), y.end());
    quantizer.dequantize(y.begin(), y.end());

    // Check
    constexpr auto max_value = 1.7_dt;
    const auto max_mapped_value = static_cast<raul::dtype>(std::pow(2, digits - 1) - 1);
    const raul::dtype scale = max_mapped_value / max_value;

    for (size_t i = 0; i < y.size(); ++i)
    {
        raul::dtype golden_val = round(x[i] * scale) / scale;
        EXPECT_FLOAT_EQ(golden_val, y[i]);
    }
}

} // UT namespace