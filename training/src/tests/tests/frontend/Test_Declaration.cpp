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

#include <training/frontend/Declaration.h>
#include <training/frontend/Layers.h>

namespace UT
{

TEST(TestDeclaration, PrimitiveLinearDeclarationUnit)
{
    using namespace raul::frontend;

    auto generateLinear = [](size_t features, bool bias)
    {
        auto x = Linear{ features };
        if (bias)
        {
            x = x.enableBias();
        }
        return x;
    };

    struct Checker : Processor
    {
        size_t features = 1;
        bool bias = false;

        auto setFeatures(size_t value)
        {
            features = value;
            return *this;
        }
        auto setBias(bool value)
        {
            bias = value;
            return *this;
        }

        void process(const LinearDeclaration& layer, std::optional<frontend::Path>) override
        {
            ASSERT_EQ(layer.type, Type::Linear);
            ASSERT_EQ(layer.features, features);
            ASSERT_EQ(layer.bias, bias);
            ASSERT_EQ(layer.inputs.size(), 1);
            ASSERT_EQ(layer.outputs.size(), 1);
        }
    };

    {
        auto features = 10;
        auto bias = false;
        auto layer = generateLinear(features, bias);
        auto checker = Checker().setBias(bias).setFeatures(features);
        layer.apply(checker);
    }

    {
        auto features = 20;
        auto bias = true;
        auto layer = generateLinear(features, bias);
        auto checker = Checker().setBias(bias).setFeatures(features);
        layer.apply(checker);
    }
}

} // UT namespace