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
#include <training/layers/composite/meta/RecurrentMetaLayer.h>
#include <training/layers/composite/meta/SequentialMetaLayer.h>
#include <training/layers/composite/rnn/LSTMCellLayer.h>
#include <training/network/Layers.h>
#include <training/network/Workflow.h>

namespace UT
{

TEST(TestRecurrentMetaLayer, RnnLinearBuildUnit){
    PROFILE_TEST

    // d.polubotko(TODO): implement

    /*
    const auto sequenceSize = 5U;


    raul::NetDef netdef;
    raul::RecurrentParams params{ LINEAR_LAYER, createParam(raul::LinearParams{ { { "in" }, { "out" }, { "Weights", "Biases" } }, 1 }), "in", "out", sequenceSize, raul::Dimension::Depth };

    netdef.add<raul::DataLayer>("data", raul::DataParams{ { "in" }, sequenceSize, 1, 1 });
    netdef.addOp("rnn", RECURRENT_META_LAYER, createParam(params));

    EXPECT_EQ(netdef.size(), 2U);

    raul::Graph graph(std::move(netdef), 1);

    graph.printInfo(std::cout);

    EXPECT_EQ(graph.getNodes().size(), sequenceSize + 2U + 1U);

    EXPECT_FALSE(graph[0].getLayer()->isShared());
    EXPECT_FALSE(graph[1].getLayer()->isShared());
    EXPECT_FALSE(graph[2].getLayer()->isShared());
    EXPECT_TRUE(graph[3].getLayer()->isShared());
    EXPECT_TRUE(graph[4].getLayer()->isShared());
    EXPECT_TRUE(graph[5].getLayer()->isShared());
    EXPECT_TRUE(graph[6].getLayer()->isShared());
    EXPECT_FALSE(graph[7].getLayer()->isShared());
    */
}

TEST(TestRecurrentMetaLayer, RnnSequencedLinearBuildUnit){
    PROFILE_TEST

    // d.polubotko(TODO): implement
    /*
    const auto sequenceSize = 5U;
    const auto groupSize = 2U;


    raul::NetDef netdef;

    raul::LinearParams paramLinear1{ { "in" }, { "inter" }, 1 };
    raul::LinearParams paramLinear2{ { "inter" }, { "out" }, 1 };
    raul::SequentialParams paramsSeq{ { raul::OperatorDef{ "linear_1", LINEAR_LAYER, createParam(paramLinear1) }, raul::OperatorDef{ "linear_2", LINEAR_LAYER, createParam(paramLinear2) } },
                                      { "Weights", "Biases" } };
    raul::RecurrentParams paramsRNN{ SEQUENTIAL_META_LAYER, createParam(paramsSeq), "in", "out", sequenceSize, raul::Dimension::Depth };

    netdef.add<raul::DataLayer>("data", raul::DataParams{ { "in" }, sequenceSize, 1, 1 });
    netdef.addOp("rnn", RECURRENT_META_LAYER, createParam(paramsRNN));

    EXPECT_EQ(netdef.size(), 2U);

    raul::Graph graph(std::move(netdef), 1);

    graph.printInfo(std::cout);

    EXPECT_EQ(graph.getNodes().size(), sequenceSize * groupSize + 2U + 1U);

    EXPECT_FALSE(graph[0].getLayer()->isShared());
    EXPECT_FALSE(graph[1].getLayer()->isShared());
    EXPECT_FALSE(graph[2].getLayer()->isShared());
    EXPECT_FALSE(graph[3].getLayer()->isShared());
    EXPECT_TRUE(graph[4].getLayer()->isShared());
    EXPECT_TRUE(graph[5].getLayer()->isShared());
    EXPECT_TRUE(graph[6].getLayer()->isShared());
    EXPECT_TRUE(graph[7].getLayer()->isShared());
    EXPECT_TRUE(graph[8].getLayer()->isShared());
    EXPECT_TRUE(graph[9].getLayer()->isShared());
    EXPECT_TRUE(graph[10].getLayer()->isShared());
    EXPECT_TRUE(graph[11].getLayer()->isShared());
    EXPECT_FALSE(graph[12].getLayer()->isShared());

    EXPECT_EQ(graph[4].getLayer()->getSharedWeights()[0], graph[2].getName() / "Weights");
    EXPECT_EQ(graph[4].getLayer()->getSharedWeights()[1], graph[2].getName() / "Biases");
    EXPECT_EQ(graph[5].getLayer()->getSharedWeights()[0], graph[3].getName() / "Weights");
    EXPECT_EQ(graph[5].getLayer()->getSharedWeights()[1], graph[3].getName() / "Biases");

    EXPECT_EQ(graph[6].getLayer()->getSharedWeights()[0], graph[2].getName() / "Weights");
    EXPECT_EQ(graph[6].getLayer()->getSharedWeights()[1], graph[2].getName() / "Biases");
    EXPECT_EQ(graph[7].getLayer()->getSharedWeights()[0], graph[3].getName() / "Weights");
    EXPECT_EQ(graph[7].getLayer()->getSharedWeights()[1], graph[3].getName() / "Biases");

    EXPECT_EQ(graph[8].getLayer()->getSharedWeights()[0], graph[2].getName() / "Weights");
    EXPECT_EQ(graph[8].getLayer()->getSharedWeights()[1], graph[2].getName() / "Biases");
    EXPECT_EQ(graph[9].getLayer()->getSharedWeights()[0], graph[3].getName() / "Weights");
    EXPECT_EQ(graph[9].getLayer()->getSharedWeights()[1], graph[3].getName() / "Biases");
    */
}

TEST(TestRecurrentMetaLayer, SimpleForwardSeq2Unit)
{
    PROFILE_TEST

    // d.polubotko(TODO): implement
    /*
    // Port of TestLSTM.SimpleForwardSeq2Unit
    const auto eps_rel = 1e-4_dt;
    const size_t input_size = 4U;
    const size_t hidden_size = 3U;
    const size_t sequence_length = 2U;
    const size_t batch_size = 2U;


    raul::NetDef netdef;
    netdef.add<raul::DataLayer>("data", raul::DataParams{ { "in" }, sequence_length, 1, input_size });
    netdef.add<raul::DataLayer>("state", raul::DataParams{ { "hidden_in", "cell_in" }, 1, 1, hidden_size });

    raul::LSTMCellParams cellParams{ "in", "hidden_in", "cell_in", "hidden_out", "cell_out", { "linear_ih::Weights", "linear_ih::Biases", "linear_hh::Weights", "linear_hh::Biases" } };
    raul::RecurrentParams params{ LSTM_CELL_LAYER,
                                  createParam(cellParams),
                                  "in",
                                  "hidden_out",
                                  sequence_length,
                                  raul::Dimension::Depth,
                                  std::map<raul::Name, raul::Name>{ { "hidden_in", "hidden_out" }, { "cell_in", "cell_out" } } };

    netdef.addOp("lstm", RECURRENT_META_LAYER, createParam(params));

    raul::Graph graph(std::move(netdef), batch_size);

    EXPECT_EQ(graph.getNodes().size(), sequence_length + 2U + 2U + 1U);

    EXPECT_FALSE(graph[0].getLayer()->isShared());
    EXPECT_FALSE(graph[1].getLayer()->isShared());
    EXPECT_FALSE(graph[2].getLayer()->isShared());
    EXPECT_FALSE(graph[3].getLayer()->isShared());
    EXPECT_FALSE(graph[4].getLayer()->isShared());
    EXPECT_TRUE(graph[5].getLayer()->isShared());
    EXPECT_FALSE(graph[6].getLayer()->isShared());

    const raul::Tensor input_init{ -8.024929e-01_dt, -1.295186e+00_dt, -7.501815e-01_dt, -1.311966e+00_dt, -2.188337e-01_dt, -2.435065e+00_dt, -7.291476e-02_dt, -3.398641e-02_dt,
                                   7.968872e-01_dt,  -1.848416e-01_dt, -3.701473e-01_dt, -1.210281e+00_dt, -6.226985e-01_dt, -4.637222e-01_dt, 1.921782e+00_dt,  -4.025455e-01_dt };
    graph.getMemoryManager()["in"] = TORANGE(input_init);

    for (auto& [param, grad] : graph.getTrainableParameters())
    {
        std::fill(param.begin(), param.end(), 1.0_dt);
    }

    // Apply
    graph.forward(raul::NetworkMode::Test);

    // Checks
    const raul::Tensor output_golden{ -1.037908e-02_dt, -1.037908e-02_dt, -1.037908e-02_dt, -7.253138e-02_dt, -7.253138e-02_dt, -7.253138e-02_dt,
                                      3.804927e-01_dt,  3.804927e-01_dt,  3.804927e-01_dt,  8.850382e-01_dt,  8.850382e-01_dt,  8.850382e-01_dt };
    const raul::Tensor cell_golden{ -2.369964e-01_dt, -2.369964e-01_dt, -2.369964e-01_dt, 1.526655e+00_dt, 1.526655e+00_dt, 1.526655e+00_dt };
    const auto& outputTensor = graph.getMemoryManager()["hidden_out"];
    const auto& cellTensor = graph.getMemoryManager()["cell_out"];

    EXPECT_EQ(outputTensor.size(), batch_size * hidden_size * sequence_length);
    EXPECT_EQ(cellTensor.size(), batch_size * hidden_size);

    for (size_t i = 0; i < outputTensor.size(); ++i)
    {
        const auto val = outputTensor[i];
        const auto golden_val = output_golden[i];
        ASSERT_TRUE(tools::expect_near_relative(val, golden_val, eps_rel)) << "at " << i << ", expected: " << golden_val << ", got: " << val;
    }

    for (size_t i = 0; i < cellTensor.size(); ++i)
    {
        const auto val = cellTensor[i];
        const auto golden_val = cell_golden[i];
        ASSERT_TRUE(tools::expect_near_relative(val, golden_val, eps_rel)) << "at " << i << ", expected: " << golden_val << ", got: " << val;
    }*/
}

} // UT namespace
