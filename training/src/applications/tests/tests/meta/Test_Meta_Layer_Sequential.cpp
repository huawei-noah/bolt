// Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include <tests/tools/TestTools.h>

#include <tests/GTestExtensions.h>
#include <training/initializers/ConstantInitializer.h>
#include <training/initializers/RandomUniformInitializer.h>
#include <training/layers/composite/meta/SequentialMetaLayer.h>
#include <training/network/Layers.h>
#include <training/network/Workflow.h>

// d.polubotko(TODO): implement
#if 0

namespace UT
{

TEST(TestSequentialMetaLayer, SimpleSequenceUnit)
{
    PROFILE_TEST
    // Port of TestSequential.VectorUnit
    using namespace raul;

    const Tensor inData = { 111._dt, 112._dt, 121._dt, 122._dt, 131._dt, 132._dt, 141._dt, 142._dt, 211._dt, 212._dt, 221._dt, 222._dt, 231._dt, 232._dt, 241._dt, 242._dt };

    std::vector<OperatorDef> v;
    v.emplace_back(OperatorDef{ "t1", TRANSPOSE_LAYER, createParam(TransposingParams{ "in", "out1", Dimension::Width, Dimension::Height }) });
    v.emplace_back(OperatorDef{ "t2", TRANSPOSE_LAYER, createParam(TransposingParams{ "out1", "out2", Dimension::Height, Dimension::Depth }) });
    v.emplace_back(OperatorDef{ "t3", TRANSPOSE_LAYER, createParam(TransposingParams{ "out2", "out3", Dimension::Width, Dimension::Height }) });
    v.emplace_back(OperatorDef{ "t4", TRANSPOSE_LAYER, createParam(TransposingParams{ "out3", "out", Dimension::Width, Dimension::Depth }) });
    
    //d.polubotko(TODO): implement
    /*
    Workflow netdef;
    netdef.add<raul::DataLayer>("data", raul::DataParams{ { "in" }, 1, 4, 2 });
    netdef.addOp("seq", SEQUENTIAL_META_LAYER, createParam(SequentialParams(std::move(v), {})));

    netdef.printInfo(std::cout);

    Graph g(std::move(netdef), 2U, raul::CompressionMode::NONE, raul::CalculationMode::DETERMINISTIC, nullptr);

    g.getMemoryManager()["in"] = TORANGE(inData);

    g.printInfo(std::cout);

    EXPECT_EQ(g[1].getInputs(), Names{ "in" });
    EXPECT_EQ(g[4].getOutputs(), Names{ "out" });

    g.forward(NetworkMode::Train);

    auto& in = g.getMemoryManager()["in"];
    auto& out = g.getMemoryManager()["out"];

    for (size_t i = 0; i < 4; ++i)
        EXPECT_EQ(in.getShape()[i], out.getShape()[i]);

    for (size_t i = 0; i < out.size(); ++i)
        EXPECT_EQ(in[i], out[i]);

    printf(" - SequentialLayer forward is Ok.\n");

    g.getMemoryManager().createTensor(raul::Name("out").grad(), out.getShape(), out);

    g.backward();

    auto& in_nabla = g.getMemoryManager()[raul::Name("in").grad()];

    for (size_t i = 0; i < 4; ++i)
        EXPECT_EQ(in.getShape()[i], in_nabla.getShape()[i]);

    for (size_t i = 0; i < in.size(); ++i)
        EXPECT_EQ(in[i], in_nabla[i]);

    printf(" - SequentialLayer backward is Ok.\n");*/
}

TEST(TestSequentialMetaLayer, NestedStackUnit)
{
    PROFILE_TEST
    // Port of TestSequential.VectorUnit
    using namespace raul;

    const Tensor inData = { 111._dt, 112._dt, 121._dt, 122._dt, 131._dt, 132._dt, 141._dt, 142._dt, 211._dt, 212._dt, 221._dt, 222._dt, 231._dt, 232._dt, 241._dt, 242._dt };

    std::vector<OperatorDef> v1;
    v1.emplace_back(OperatorDef{ "t1", TRANSPOSE_LAYER, createParam(TransposingParams{ "in", "out1", Dimension::Width, Dimension::Height }) });
    v1.emplace_back(OperatorDef{ "t2", TRANSPOSE_LAYER, createParam(TransposingParams{ "out1", "out2", Dimension::Height, Dimension::Depth }) });

    std::vector<OperatorDef> v2;
    v2.emplace_back(OperatorDef{ "t3", TRANSPOSE_LAYER, createParam(TransposingParams{ "out2", "out3", Dimension::Width, Dimension::Height }) });
    v2.emplace_back(OperatorDef{ "t4", TRANSPOSE_LAYER, createParam(TransposingParams{ "out3", "out", Dimension::Width, Dimension::Depth }) });

    std::vector<OperatorDef> v3;
    v3.emplace_back(OperatorDef{ "s1", SEQUENTIAL_META_LAYER, createParam(SequentialParams{ std::move(v1), {} }) });
    v3.emplace_back(OperatorDef{ "s2", SEQUENTIAL_META_LAYER, createParam(SequentialParams{ std::move(v2), {} }) });

    //d.polubotko(TODO): implement
    /*
    NetDef netdef;
    netdef.add<raul::DataLayer>("data", raul::DataParams{ { "in" }, 1, 4, 2 });
    netdef.addOp("seq", SEQUENTIAL_META_LAYER, createParam(SequentialParams(std::move(v3), {})));

    netdef.printInfo(std::cout);

    Graph g(std::move(netdef), 2U, raul::CompressionMode::NONE, raul::CalculationMode::DETERMINISTIC, nullptr);

    g.getMemoryManager()["in"] = TORANGE(inData);

    g.printInfo(std::cout);

    EXPECT_EQ(g[1].getInputs(), Names{ "in" });
    EXPECT_EQ(g[4].getOutputs(), Names{ "out" });

    g.forward(NetworkMode::Train);

    auto& in = g.getMemoryManager()["in"];
    auto& out = g.getMemoryManager()["out"];

    for (size_t i = 0; i < 4; ++i)
        EXPECT_EQ(in.getShape()[i], out.getShape()[i]);

    for (size_t i = 0; i < out.size(); ++i)
        EXPECT_EQ(in[i], out[i]);

    printf(" - SequentialLayer forward is Ok.\n");

    g.getMemoryManager().createTensor(raul::Name("out").grad(), out.getShape(), out);

    g.backward();

    auto& in_nabla = g.getMemoryManager()[raul::Name("in").grad()];

    for (size_t i = 0; i < 4; ++i)
        EXPECT_EQ(in.getShape()[i], in_nabla.getShape()[i]);

    for (size_t i = 0; i < in.size(); ++i)
        EXPECT_EQ(in[i], in_nabla[i]);

    printf(" - SequentialLayer backward is Ok.\n");*/
}

} // UT namespace

#endif
