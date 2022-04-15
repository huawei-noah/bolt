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
#include <training/layers/composite/meta/StackedMetaLayer.h>
#include <training/network/Layers.h>
#include <training/network/Workflow.h>

namespace UT
{

TEST(TestStackedMetaLayer, SimpleStackUnit){
    PROFILE_TEST
    // d.polubotko(TODO): implement
    /*
    const auto stackSize = 5U;

    raul::Workflow netdef;
    raul::StackedParams params{ LINEAR_LAYER, createParam(raul::LinearParams{ { "in" }, { "out" }, 1 }), stackSize };

    netdef.add<raul::DataLayer>("data", raul::DataParams{ { "in" }, 1, 1, 1 });
    netdef.addOp("stack", STACKED_META_LAYER, createParam(params));


    EXPECT_EQ(netdef.size(), 2U);

    raul::Graph graph(std::move(netdef), 1);

    graph.printInfo(std::cout);

    EXPECT_EQ(graph.getNodes().size(), stackSize + 1U);
    */
}

TEST(TestStackedMetaLayer, NestedStackUnit){
    PROFILE_TEST
    // d.polubotko(TODO): implement
    /*
    const auto stackSizeInner = 3U;
    const auto stackSizeOuter = 2U;

    raul::Workflow netdef;
    raul::StackedParams paramsInner{ LINEAR_LAYER, createParam(raul::LinearParams{ { "in" }, { "out" }, 1 }), stackSizeInner };
    raul::StackedParams paramsOuter{ STACKED_META_LAYER, createParam(paramsInner), stackSizeOuter };

    netdef.add<raul::DataLayer>("data", raul::DataParams{ { "in" }, 1, 1, 1 });
    netdef.addOp("stack", STACKED_META_LAYER, createParam(paramsOuter));


    EXPECT_EQ(netdef.size(), 2U);

    raul::Graph graph(std::move(netdef), 1);

    graph.printInfo(std::cout);

    EXPECT_EQ(graph.getNodes().size(), stackSizeInner * stackSizeOuter + 1U);
    */
}

TEST(TestStackedMetaLayer, MultiLayerLSTMBuildUnit){
    PROFILE_TEST

    // d.polubotko(TODO): implement
    /*
    const auto layers = 5U;
    const auto hiddenSize = 10U;
    const auto batchSize = 1U;

    raul::Workflow netdef;
    raul::StackedParams params{ LSTM_LAYER, createParam(raul::LSTMParams{ { "in" }, { "out" }, hiddenSize, true }), layers };

    netdef.add<raul::DataLayer>("data", raul::DataParams{ { "in" }, 1, 1, 1 });
    netdef.addOp("stack", STACKED_META_LAYER, createParam(params));


    EXPECT_EQ(netdef.size(), 2U);

    raul::Graph graph(std::move(netdef), batchSize);

    graph.printInfo(std::cout);

    EXPECT_EQ(graph.getNodes().size(), layers + 1U);

    raul::initializers::ConstantInitializer initializer{ 1.0_dt };
    initializer(graph.getMemoryManager()["in"]);

    graph.forward(raul::NetworkMode::Train);
    */
}

TEST(TestStackedMetaLayer, MultiLayerLSTMRandUnit)
{
    PROFILE_TEST

    // d.polubotko(TODO): implement
    /*
    // Test parameters
    const auto eps = 1e-6_dt;
    const size_t input_size = 11U;
    const size_t hidden_size = 7U;
    const size_t sequence_length = 5U;
    const size_t batch_size = 2U;

    const auto minVal = -1.0_dt;
    const auto maxVal = 1.0_dt;

    raul::initializers::RandomUniformInitializer initializer{ minVal, maxVal };



    // Reference Network
    raul::NetDef netdef;
    netdef.add<raul::DataLayer>("data", raul::DataParams{ { "in" }, sequence_length, 1, input_size });
    netdef.add<raul::DataLayer>("data_state", raul::DataParams{ { "in_hidden", "in_cell" }, 1, 1, hidden_size });
    netdef.addOp("lstm::0", LSTM_LAYER, createParam(raul::LSTMParams{ { "in", "in_hidden", "in_cell" }, { "inter", "inter_hidden", "inter_cell" } }));
    netdef.addOp("lstm::1", LSTM_LAYER, createParam(raul::LSTMParams{ { "inter", "inter_hidden", "inter_cell" }, { "out", "out_hidden", "out_cell" } }));
    raul::Graph graph(std::move(netdef), batch_size, raul::CompressionMode::NONE, raul::CalculationMode::DETERMINISTIC, nullptr);

    auto& memory_manager = graph.getMemoryManager();
    initializer(memory_manager["in"]);

    memory_manager.createTensor(raul::Name("out").grad(), batch_size, sequence_length, 1, hidden_size);
    initializer(memory_manager[raul::Name("out").grad()]);
    initializer(memory_manager[raul::Name("in_hidden")]);
    initializer(memory_manager[raul::Name("in_cell")]);

    for (auto& [param, grad] : graph.getTrainableParameters())
    {
        std::fill(param.begin(), param.end(), 1.0_dt);
    }

    // Stacked network
    raul::NetDef netdef_stacked;
    raul::StackedParams params{ LSTM_LAYER, createParam(raul::LSTMParams{ { "in", "in_hidden", "in_cell" }, { "out", "out_hidden", "out_cell" } }), 2U };

    netdef_stacked.add<raul::DataLayer>("data", raul::DataParams{ { "in" }, sequence_length, 1, input_size });
    netdef_stacked.add<raul::DataLayer>("data_state", raul::DataParams{ { "in_hidden", "in_cell" }, 1, 1, hidden_size });
    netdef_stacked.addOp("lstm", STACKED_META_LAYER, createParam(params));

    raul::Graph graph_stacked(std::move(netdef_stacked), batch_size, raul::CompressionMode::NONE, raul::CalculationMode::DETERMINISTIC, nullptr);

    auto& memory_manager_stacked = graph_stacked.getMemoryManager();
    memory_manager_stacked["in"] = TORANGE(memory_manager["in"]);
    memory_manager_stacked.createTensor(raul::Name("out").grad(), batch_size, sequence_length, 1, hidden_size);
    memory_manager_stacked[raul::Name("out").grad()] = TORANGE(memory_manager[raul::Name("out").grad()]);
    memory_manager_stacked[raul::Name("in_hidden")] = TORANGE(memory_manager[raul::Name("in_hidden")]);
    memory_manager_stacked[raul::Name("in_cell")] = TORANGE(memory_manager[raul::Name("in_cell")]);

    for (auto& [param, grad] : graph_stacked.getTrainableParameters())
    {
        std::fill(param.begin(), param.end(), 1.0_dt);
    }

    // Apply
    graph.forward(raul::NetworkMode::Train);
    graph_stacked.forward(raul::NetworkMode::Train);

    ASSERT_FLOAT_TENSORS_EQ(graph.getMemoryManager()["out"], graph_stacked.getMemoryManager()["out"], eps);

    graph.backward();
    graph_stacked.backward();

    // Checks
    const auto& inputs_grad = memory_manager[raul::Name("in").grad()];
    const auto& inputs_grad_stacked = memory_manager_stacked[raul::Name("in").grad()];

    ASSERT_FLOAT_TENSORS_EQ(inputs_grad, inputs_grad_stacked, eps);*/
}

} // UT namespace
