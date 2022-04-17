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

#include <training/frontend/Frontend.h>
#include <training/compiler/FrontendCompiler.h>
#include <training/compiler/Workflow.h>

#include <training/frontend/processors/TextPrinter.h>

using namespace raul::frontend;

namespace UT
{

TEST(TestCompilation, EmptyGraphUnit)
{
    FrontendCompiler compiler;

    auto g = Graph{};

    EXPECT_NO_THROW(compiler.setTopology(g));
    EXPECT_NO_THROW(compiler.compile());

    const auto& work = compiler.getWorkflow();
    EXPECT_EQ(work.getSetOfLayers().size(), 0);
    EXPECT_EQ(work.getBatchSize(), 1);
}

TEST(TestCompilation, LinearUnit)
{
    FrontendCompiler compiler;

    auto g = Linear{ 3 };

    EXPECT_NO_THROW(compiler.setTopology(g));
    EXPECT_NO_THROW(compiler.compile());

    const auto& work = compiler.getWorkflow();
    EXPECT_EQ(work.getSetOfLayers().size(), 2); // Data (implicitly declares) + Linear
    EXPECT_EQ(work.getBatchSize(), 1);
}

TEST(TestCompilation, LinearSequenceUnit)
{
    FrontendCompiler compiler;

    auto g = Graph{ Linear{ 3 }, Linear{ 4 }, Linear{ 5 } };

    EXPECT_NO_THROW(compiler.setTopology(g));
    EXPECT_NO_THROW(compiler.compile());

    const auto& work = compiler.getWorkflow();
    EXPECT_EQ(work.getSetOfLayers().size(), 2); // Data (implicitly declares) + Linear
    EXPECT_EQ(work.getBatchSize(), 1);

    std::stringstream ss;
    work.printInfo(ss);

    ASSERT_STREQ(ss.str().c_str(),
                 "Data [in]: \n"
                 "\toutputs: Tensor 'in' (1,1,1,1)\n"
                 "Linear [0]: \n"
                 "\tinputs: Tensor 'in' (1,1,1,1)\n"
                 "\toutputs: Tensor '0::out' (1,1,1,3)\n"
                 "Linear [1]: \n"
                 "\tinputs: Tensor '0::out' (1,1,1,3)\n"
                 "\toutputs: Tensor '1::out' (1,1,1,4)\n"
                 "Linear [2]: \n"
                 "\tinputs: Tensor '1::out' (1,1,1,4)\n"
                 "\toutputs: Tensor '2::out' (1,1,1,5)\n");
}

TEST(TestCompilation, LinearSubgraphsUnit)
{
    FrontendCompiler compiler;

    auto g = Graph{ Graph{ Linear{ 3 } }, Graph{ Linear{ 4 } } };

    EXPECT_NO_THROW(compiler.setTopology(g));
    EXPECT_NO_THROW(compiler.compile());

    const auto& work = compiler.getWorkflow();
    EXPECT_EQ(work.getSetOfLayers().size(), 2); // Data (implicitly declares) + Linear
    EXPECT_EQ(work.getBatchSize(), 1);

    std::stringstream ss;
    work.printInfo(ss);

    ASSERT_STREQ(ss.str().c_str(),
                 "Data [in]: \n"
                 "\toutputs: Tensor 'in' (1,1,1,1)\n"
                 "Linear [0::0]: \n"
                 "\tinputs: Tensor 'in' (1,1,1,1)\n"
                 "\toutputs: Tensor '0::0::out' (1,1,1,3)\n"
                 "Linear [1::0]: \n"
                 "\tinputs: Tensor '0::0::out' (1,1,1,3)\n"
                 "\toutputs: Tensor '1::0::out' (1,1,1,4)\n");
}

TEST(TestCompilation, LinearSubgraphsMixedUnit)
{
    FrontendCompiler compiler;

    auto g = Graph{ Graph{ Linear{ 3 } }, Linear{ 4 }, Graph{ Linear{ 5 } }, Graph{ Linear{ 6 }, Linear{ 7 } } };

    EXPECT_NO_THROW(compiler.setTopology(g));
    EXPECT_NO_THROW(compiler.compile());

    const auto& work = compiler.getWorkflow();
    EXPECT_EQ(work.getSetOfLayers().size(), 2); // Data (implicitly declares) + Linear
    EXPECT_EQ(work.getBatchSize(), 1);

    std::stringstream ss;
    work.printInfo(ss);

    ASSERT_STREQ(ss.str().c_str(),
                 "Data [in]: \n"
                 "\toutputs: Tensor 'in' (1,1,1,1)\n"
                 "Linear [0::0]: \n"
                 "\tinputs: Tensor 'in' (1,1,1,1)\n"
                 "\toutputs: Tensor '0::0::out' (1,1,1,3)\n"
                 "Linear [1]: \n"
                 "\tinputs: Tensor '0::0::out' (1,1,1,3)\n"
                 "\toutputs: Tensor '1::out' (1,1,1,4)\n"
                 "Linear [2::0]: \n"
                 "\tinputs: Tensor '1::out' (1,1,1,4)\n"
                 "\toutputs: Tensor '2::0::out' (1,1,1,5)\n"
                 "Linear [3::0]: \n"
                 "\tinputs: Tensor '2::0::out' (1,1,1,5)\n"
                 "\toutputs: Tensor '3::0::out' (1,1,1,6)\n"
                 "Linear [3::1]: \n"
                 "\tinputs: Tensor '3::0::out' (1,1,1,6)\n"
                 "\toutputs: Tensor '3::1::out' (1,1,1,7)\n");
}

TEST(TestCompilation, LinearNamedSubgraphsParametrizedUnit)
{
    FrontendCompiler compiler;

    auto genLinearSequence = [](size_t n)
    {
        Graph g{};
        for (size_t i = 0; i < n; ++i)
        {
            g.insert(Linear{ i + 1 }.enableBias());
        }

        return g;
    };

    auto g = Graph{ { "block_a", genLinearSequence(5) }, { "block_b", genLinearSequence(3) } };

    EXPECT_NO_THROW(compiler.setTopology(g));
    EXPECT_NO_THROW(compiler.compile());

    const auto& work = compiler.getWorkflow();
    EXPECT_EQ(work.getSetOfLayers().size(), 2); // Data (implicitly declares) + Linear
    EXPECT_EQ(work.getBatchSize(), 1);

    std::stringstream ss;
    work.printInfo(ss);

    ASSERT_STREQ(ss.str().c_str(),
                 "Data [in]: \n"
                 "\toutputs: Tensor 'in' (1,1,1,1)\n"
                 "Linear [block_a::0]: \n"
                 "\tinputs: Tensor 'in' (1,1,1,1)\n"
                 "\toutputs: Tensor 'block_a::0::out' (1,1,1,1)\n"
                 "Linear [block_a::1]: \n"
                 "\tinputs: Tensor 'block_a::0::out' (1,1,1,1)\n"
                 "\toutputs: Tensor 'block_a::1::out' (1,1,1,2)\n"
                 "Linear [block_a::2]: \n"
                 "\tinputs: Tensor 'block_a::1::out' (1,1,1,2)\n"
                 "\toutputs: Tensor 'block_a::2::out' (1,1,1,3)\n"
                 "Linear [block_a::3]: \n"
                 "\tinputs: Tensor 'block_a::2::out' (1,1,1,3)\n"
                 "\toutputs: Tensor 'block_a::3::out' (1,1,1,4)\n"
                 "Linear [block_a::4]: \n"
                 "\tinputs: Tensor 'block_a::3::out' (1,1,1,4)\n"
                 "\toutputs: Tensor 'block_a::4::out' (1,1,1,5)\n"
                 "Linear [block_b::0]: \n"
                 "\tinputs: Tensor 'block_a::4::out' (1,1,1,5)\n"
                 "\toutputs: Tensor 'block_b::0::out' (1,1,1,1)\n"
                 "Linear [block_b::1]: \n"
                 "\tinputs: Tensor 'block_b::0::out' (1,1,1,1)\n"
                 "\toutputs: Tensor 'block_b::1::out' (1,1,1,2)\n"
                 "Linear [block_b::2]: \n"
                 "\tinputs: Tensor 'block_b::1::out' (1,1,1,2)\n"
                 "\toutputs: Tensor 'block_b::2::out' (1,1,1,3)\n");
}

TEST(TestCompilation, LinearSubgraphsPortsUnit)
{
    FrontendCompiler compiler;

    auto g = Graph({ { "a", Linear{ 1 } }, { "b", Linear{ 2 } } }, { Port{ "in" }.to(Port{ "a", "in" }), Port{ "in" }.to(Port{ "b", "in" }) });

    EXPECT_NO_THROW(compiler.setTopology(g));
    EXPECT_NO_THROW(compiler.compile({ 1, 2, 3, 4 }));

    const auto& work = compiler.getWorkflow();
    EXPECT_EQ(work.getSetOfLayers().size(), 2); // Data (implicitly declares) + Linear
    EXPECT_EQ(work.getBatchSize(), 1);

    std::stringstream ss;
    work.printInfo(ss);

    ASSERT_STREQ(ss.str().c_str(),
                 "Data [in]: \n"
                 "\toutputs: Tensor 'in' (1,2,3,4)\n"
                 "Linear [a]: \n"
                 "\tinputs: Tensor 'in' (1,2,3,4)\n"
                 "\toutputs: Tensor 'a::out' (1,2,3,1)\n"
                 "Linear [b]: \n"
                 "\tinputs: Tensor 'in' (1,2,3,4)\n"
                 "\toutputs: Tensor 'b::out' (1,2,3,2)\n");
}

} // UT namespace