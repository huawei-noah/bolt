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

#include <training/base/layers/basic/DataLayer.h>
#include <training/base/layers/basic/ElementWiseMulLayer.h>
#include <training/base/layers/basic/ElementWiseSumLayer.h>
#include <training/base/layers/basic/LogLayer.h>

namespace UT
{

TEST(TestNetworkParameters, CallbackUnit)
{
    PROFILE_TEST
    const size_t batch = 1u;
    MANAGERS_DEFINE
    NETWORK_PARAMS_DEFINE(networkParameters);

    memory_manager.createTensor("x", batch, 1u, 1u, 1u, 2.7_dt);
    memory_manager.createTensor("y", batch, 1u, 1u, 1u, 2.0_dt);
    memory_manager.createTensor("z", batch, 1u, 1u, 1u, 3.0_dt);

    raul::DataParams dataParams{ { "x", "y", "z" }, 1u, 1u, 1u };
    raul::BasicParams logParams{ { "x" }, { "out1" } };
    raul::ElementWiseLayerParams mulParams{ { "out1", "y" }, { "out2" } };
    raul::ElementWiseLayerParams sumParams{ { "out2", "z" }, { "out3" } };

    // Apply function
    const raul::Names lNames = { "data", "log", "mul", "sum" };
    work.add<raul::DataLayer>(lNames[0], dataParams);
    work.add<raul::LogLayer>(lNames[1], logParams);
    work.add<raul::ElementWiseMulLayer>(lNames[2], mulParams);
    work.add<raul::ElementWiseSumLayer>(lNames[3], sumParams);

    TENSORS_CREATE(batch)
    memory_manager["x"] = 2.7_dt;
    memory_manager["y"] = 2.0_dt;
    memory_manager["z"] = 3.0_dt;

    auto first = [](raul::BasicLayer* layer, const raul::MemoryManager& mem)
    {
        std::cout << "Before forward of " << layer->getName() << std::endl;

        // Check that everything allocated
        for (size_t i = 0; i < layer->getOutputs().size(); ++i)
        {
            EXPECT_EQ(mem[layer->getOutputs()[i]].size(), 1u);
        }
        for (size_t i = 0; i < layer->getInputs().size(); ++i)
        {
            std::cout << "Input[" + layer->getInputs()[i] + "]:" << std::endl;
            for (size_t j = 0; j < mem[layer->getInputs()[i]].size(); ++j)
            {
                std::cout << mem[layer->getInputs()[i]][j] << std::endl;
            }
        }
    };

    auto second = [](raul::BasicLayer* layer, const raul::MemoryManager& mem)
    {
        std::cout << "After forward of " << layer->getName() << std::endl;
        for (size_t i = 0; i < layer->getOutputs().size(); ++i)
        {
            std::cout << "Output[" + layer->getOutputs()[i] + "]:" << std::endl;
            for (size_t j = 0; j < mem[layer->getOutputs()[i]].size(); ++j)
            {
                std::cout << mem[layer->getOutputs()[i]][j] << std::endl;
            }
        }
    };

    auto third = [](raul::BasicLayer* layer, const raul::MemoryManager& mem)
    {
        std::cout << "Before backward of " << layer->getName() << std::endl;

        // Check that gradients allocated
        for (size_t i = 0; i < layer->getInputs().size(); ++i)
        {
            EXPECT_EQ(mem[layer->getInputs()[i].grad()].size(), 1u);
        }
        for (size_t i = 0; i < layer->getOutputs().size(); ++i)
        {
            std::cout << "Incoming deltas[" + layer->getOutputs()[i].grad() + "]:" << std::endl;
            for (size_t j = 0; j < mem[layer->getOutputs()[i].grad()].size(); ++j)
            {
                std::cout << mem[layer->getOutputs()[i].grad()][j] << std::endl;
            }
        }
    };

    auto fourth = [](raul::BasicLayer* layer, const raul::MemoryManager& mem)
    {
        std::cout << "After backward of " << layer->getName() << std::endl;

        // Check that gradients still exist (Eager execution)
        for (size_t i = 0; i < layer->getOutputs().size(); ++i)
        {
            EXPECT_EQ(mem[layer->getOutputs()[i]].size(), 1u);
            EXPECT_EQ(mem[layer->getOutputs()[i].grad()].size(), 1u);
        }
    };

    networkParameters.mCallback = raul::CallbackHelper(first, second, third, fourth);

    // Apply function
    for (size_t i = 0; i < 4; ++i)
    {
        networkParameters.mCallback(work[lNames[i]], memory_manager, raul::NetworkParameters::CallbackPlace::Before_Forward);
        work[lNames[i]]->forwardComputeImpl(raul::NetworkMode::Train);
        networkParameters.mCallback(work[lNames[i]], memory_manager, raul::NetworkParameters::CallbackPlace::After_Forward);
    }

    for (size_t i = 4; i > 0; --i)
    {
        networkParameters.mCallback(work[lNames[i - 1]], memory_manager, raul::NetworkParameters::CallbackPlace::Before_Backward);
        work[lNames[i - 1]]->backwardComputeImpl();
        networkParameters.mCallback(work[lNames[i - 1]], memory_manager, raul::NetworkParameters::CallbackPlace::After_Backward);
    }
}

TEST(TestNetworkParameters, DummyClassOperatorUnit)
{
    PROFILE_TEST
    const size_t batch = 1u;
    MANAGERS_DEFINE
    NETWORK_PARAMS_DEFINE(networkParameters);

    class Dummy
    {
      public:
        void operator()(raul::BasicLayer* layer, const raul::MemoryManager& mem, raul::NetworkParameters::CallbackPlace place)
        {
            EXPECT_EQ(layer->getName(), raul::Name("log"));
            EXPECT_EQ(mem[layer->getInputs()[0]].size(), 1u);
            EXPECT_EQ(place, raul::NetworkParameters::CallbackPlace::Before_Forward);
        }
    };

    ASSERT_NO_THROW(networkParameters.mCallback = Dummy());

    work.add<raul::DataLayer>("data", raul::DataParams{ { "x" }, 1u, 1u, 1u });
    work.add<raul::LogLayer>("log", raul::BasicParams{ { "x" }, { "out1" } });
    TENSORS_CREATE(batch)

    networkParameters.mCallback(work["log"], memory_manager, raul::NetworkParameters::CallbackPlace::Before_Forward);
}

}