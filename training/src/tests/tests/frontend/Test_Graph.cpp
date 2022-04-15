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

#include <training/frontend/Graph.h>
#include <training/frontend/Layers.h>
#include <training/frontend/processors/DotLangPrinter.h>
#include <training/frontend/processors/TextPrinter.h>

using namespace raul::frontend;

namespace UT
{

TEST(TestGraph, SequencialGraphUnit)
{
    EXPECT_NO_THROW(Graph({ Linear{ 1 }, Linear{ 2 }, Linear{ 3 }, Linear{ 4 }, Linear{ 5 } }));
}

TEST(TestGraph, InnerGraphUnit)
{

    EXPECT_NO_THROW(Graph({ Graph{ Linear{ 1 }, Linear{ 2 } }, Graph{ Linear{ 3 }, Linear{ 4 } }, Linear{ 5 } }));
}

TEST(TestGraph, GraphMapLikeInterfaceUnit)
{

    auto g = Graph{};
    g["linear_1"] = Linear{ 1 };
    g["linear_2"] = Linear{ 2 };
    g["linear_3"] = Linear{ 3 };
    g["linear_4"] = Linear{ 4 };
    g["linear_5"] = Linear{ 5 };

    struct Checker : Processor
    {
        size_t featuresValue = 1;
        void process(const LinearDeclaration& layer, std::optional<frontend::Path>) override
        {
            ASSERT_EQ(layer.features, featuresValue);
            ++featuresValue;
        }
    } checker;

    g.apply(checker);

    ASSERT_EQ(checker.featuresValue, 6);
}

TEST(TestGraph, GraphVectorLikeInterfaceUnit)
{
    auto l = Linear{ 0 };
    auto g = Graph{ l, l, l, l, l };

    g[0] = Linear{ 1 };
    g[1] = Linear{ 2 };
    g[2] = Linear{ 3 };
    g[3] = Linear{ 4 };
    g[4] = Linear{ 5 };

    struct Checker : Processor
    {
        size_t featuresValue = 1;
        void process(const LinearDeclaration& layer, std::optional<frontend::Path>) override
        {
            ASSERT_EQ(layer.features, featuresValue);
            ++featuresValue;
        }
    } checker;

    for (size_t i = 0; i < 5; ++i)
    {
        EXPECT_NO_THROW(g[i]);
        g[i].apply(checker);
    }

    EXPECT_THROW(g.at(5), std::out_of_range);
}

TEST(TestGraph, GraphNamedLayersUnit)
{

    auto l = Linear{ 1 };
    EXPECT_NO_THROW(Graph({ { "name 1", l }, { "name 2", l }, l }));
}

TEST(TestGraph, GraphNameCollisionUnit)
{
    auto l = Linear{ 1 };
    EXPECT_NO_THROW(Graph({ { "a", l }, { "b", l } }));
    EXPECT_THROW(Graph({ { "a", l }, { "a", l } }), raul::Exception);
}

TEST(TestGraph, GraphTextPrinterUnit)
{
    auto g1 = Graph{ Linear{ 1 }, Linear{ 2 } };
    auto g2 = Graph{ { "subgraph_1", g1 }, { "element", Linear{ 3 } }, Linear{ 4 } };

    std::stringstream ss;
    TextPrinter printer(ss);
    g2.apply(printer);

    ASSERT_STREQ(ss.str().c_str(),
                 "[subgraph_1:[Linear(name=0, features=1, bias=false),Linear(name=1, features=2, "
                 "bias=false)|Port(in)->Port(0::in),Port(0::out)->Port(1::in),Port(1::out)->Port(out)],Linear(name=element, features=3, bias=false),Linear(name=2, features=4, "
                 "bias=false)|Port(in)->Port(subgraph_1::in),Port(subgraph_1::out)->Port(element::in),Port(element::out)->Port(2::in),Port(2::out)->Port(out)]");
}

TEST(TestGraph, GraphPortsUnit)
{

    auto l = Linear{ 1 };
    auto g = Graph({ { "a", l }, { "b", l } }, { Port("in").to(Port("a", "in")), Port("b", "out").to(Port("out")) });

    std::stringstream ss;
    TextPrinter printer(ss);
    g.apply(printer);

    ASSERT_STREQ(ss.str().c_str(), "[Linear(name=a, features=1, bias=false),Linear(name=b, features=1, bias=false)|Port(in)->Port(a::in),Port(b::out)->Port(out)]");
}

} // UT namespace