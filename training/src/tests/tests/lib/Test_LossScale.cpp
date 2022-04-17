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

#include <training/base/layers/BasicLayer.h>
#include <training/base/initializers/ConstantInitializer.h>
#include <training/compiler/Layers.h>
#include <training/base/optimizers/Adam.h>

namespace UT
{

template<typename MM>
void init_grad(MM& memoryManager, const raul::Name& tensorName, typename MM::type value)
{
    auto& grad = memoryManager[tensorName.grad()];
    grad.memAllocate(nullptr);
    grad = value;
}

TEST(TestLossScale, NoThrowUnit)
{
    PROFILE_TEST

    const auto batch_size = 1;

    auto network = raul::Workflow();

    network.add<raul::DataLayer>("data", raul::DataParams{ { "data" }, 1, 1, 1 });
    network.add<raul::LinearLayer>("fc_in", raul::LinearParams{ { "data" }, { "fc_in" }, 1 });
    network.add<raul::ReLUActivation>("nl", raul::BasicParams{ { "fc_in" }, { "nl" } });
    network.add<raul::LinearLayer>("fc_out", raul::LinearParams{ { "nl" }, { "fc_out" }, 1 });

    network.setScaling("fc_in", raul::ScalingStrategy(2.0_dt));

    network.preparePipelines();
    network.setBatchSize(batch_size);
    network.prepareMemoryForTraining();

    ASSERT_NO_THROW(network.forwardPassTraining());
    ASSERT_NO_THROW(network.backwardPassTraining());
}

TEST(TestLossScale, OneLayerUnit)
{
    PROFILE_TEST

    const auto abs_err = 1e-3;
    const auto batch_size = 1;
    const auto scale_factor = 300.0_dt;

    auto network = raul::Workflow();

    network.add<raul::DataLayer>("data", raul::DataParams{ { "data" }, 1, 1, 1 });
    network.add<raul::LinearLayer>("fc_in", raul::LinearParams{ { "data" }, { "fc_in" }, 1 });
    network.add<raul::ReLUActivation>("relu", raul::BasicParams{ { "fc_in" }, { "relu" } });
    network.add<raul::LinearLayer>("fc_out", raul::LinearParams{ { "relu" }, { "fc_out" }, 1 });

    network.setScaling("fc_in", raul::ScalingStrategy(scale_factor));

    network.preparePipelines();
    network.setBatchSize(batch_size);
    network.prepareMemoryForTraining();

    auto init_ones = raul::initializers::ConstantInitializer{ 1.0 };
    auto& memoryManagerFP32 = network.getMemoryManager();
    for (auto& paramName : network.getTrainableParameterNames())
    {
        auto& tensor = memoryManagerFP32[paramName];
        init_ones(tensor);
    }

    network.getNetworkParameters().mCallback = [&](raul::BasicLayer* layer, raul::MemoryManager& memoryManager, raul::NetworkParameters::CallbackPlace place)
    {
        if (layer->getName() == "fc_in" && place == raul::NetworkParameters::CallbackPlace::Before_Backward)
        {
            auto& tensorValue = memoryManager["fc_inGradient"];
            ASSERT_EQ(tensorValue.size(), 1);
            ASSERT_NEAR(tensorValue[0], scale_factor, abs_err);
        }
    };

    init_ones(memoryManagerFP32["data"]);
    init_grad(memoryManagerFP32, "fc_out", 1.0_dt);

    network.forwardPassTraining();
    network.backwardPassTraining();
}

TEST(TestLossScale, TwoLayersUnit)
{
    PROFILE_TEST

    const auto abs_err = 1e-3;
    const auto batch_size = 1;
    const auto scale_factor_1 = 300.0_dt;
    const auto scale_factor_2 = 70.0_dt;

    auto network = raul::Workflow();

    network.add<raul::DataLayer>("data", raul::DataParams{ { "data" }, 1, 1, 1 });
    network.add<raul::LinearLayer>("fc_in", raul::LinearParams{ { "data" }, { "fc_in" }, 1 });
    network.add<raul::ReLUActivation>("relu", raul::BasicParams{ { "fc_in" }, { "relu" } });
    network.add<raul::LinearLayer>("fc_out", raul::LinearParams{ { "relu" }, { "fc_out" }, 1 });

    network.setScaling("fc_out", raul::ScalingStrategy(scale_factor_1));
    network.setScaling("fc_in", raul::ScalingStrategy(scale_factor_2));

    network.preparePipelines();
    network.setBatchSize(batch_size);
    network.prepareMemoryForTraining();

    auto init_ones = raul::initializers::ConstantInitializer{ 1.0 };
    auto& memoryManagerFP32 = network.getMemoryManager();
    for (auto& paramName : network.getTrainableParameterNames())
    {
        auto& tensor = memoryManagerFP32[paramName];
        init_ones(tensor);
    }

    network.getNetworkParameters().mCallback = [&](raul::BasicLayer* layer, raul::MemoryManager& memoryManager, raul::NetworkParameters::CallbackPlace place)
    {
        if (layer->getName() == "fc_in" && place == raul::NetworkParameters::CallbackPlace::Before_Backward)
        {
            auto& tensorValue = memoryManager["fc_inGradient"];
            ASSERT_EQ(tensorValue.size(), 1);
            ASSERT_NEAR(tensorValue[0], scale_factor_2, abs_err);
        }

        if (layer->getName() == "fc_out" && place == raul::NetworkParameters::CallbackPlace::Before_Backward)
        {
            auto& tensorValue = memoryManager["fc_outGradient"];
            ASSERT_EQ(tensorValue.size(), 1);
            ASSERT_NEAR(tensorValue[0], scale_factor_1, abs_err);
        }
    };

    init_ones(memoryManagerFP32["data"]);
    init_grad(memoryManagerFP32, "fc_out", 1.0_dt);

    network.forwardPassTraining();
    network.backwardPassTraining();
}

TEST(TestLossScale, ThreeLayersUnit)
{
    PROFILE_TEST

    const auto abs_err = 1e-3;
    const auto batch_size = 1;
    const auto scale_factor_1 = 300.0_dt;
    const auto scale_factor_2 = 5.0_dt;
    const auto scale_factor_3 = 70.0_dt;

    auto network = raul::Workflow();

    network.add<raul::DataLayer>("data", raul::DataParams{ { "data" }, 1, 1, 1 });
    network.add<raul::LinearLayer>("fc_in", raul::LinearParams{ { "data" }, { "fc_in" }, 1 });
    network.add<raul::ReLUActivation>("relu", raul::BasicParams{ { "fc_in" }, { "relu" } });
    network.add<raul::LinearLayer>("fc_out", raul::LinearParams{ { "relu" }, { "fc_out" }, 1 });

    network.setScaling("fc_out", raul::ScalingStrategy(scale_factor_1));
    network.setScaling("relu", raul::ScalingStrategy(scale_factor_2));
    network.setScaling("fc_in", raul::ScalingStrategy(scale_factor_3));

    network.preparePipelines();
    network.setBatchSize(batch_size);
    network.prepareMemoryForTraining();

    auto init_ones = raul::initializers::ConstantInitializer{ 1.0 };
    auto& memoryManagerFP32 = network.getMemoryManager();
    for (auto& paramName : network.getTrainableParameterNames())
    {
        auto& tensor = memoryManagerFP32[paramName];
        init_ones(tensor);
    }

    network.getNetworkParameters().mCallback = [&](raul::BasicLayer* layer, raul::MemoryManager& memoryManager, raul::NetworkParameters::CallbackPlace place)
    {
        if (layer->getName() == "fc_in" && place == raul::NetworkParameters::CallbackPlace::Before_Backward)
        {
            auto& tensorValue = memoryManager["fc_inGradient"];
            ASSERT_EQ(tensorValue.size(), 1);
            ASSERT_NEAR(tensorValue[0], scale_factor_3, abs_err);
        }

        if (layer->getName() == "relu" && place == raul::NetworkParameters::CallbackPlace::Before_Backward)
        {
            auto& tensorValue = memoryManager["reluGradient"];
            ASSERT_EQ(tensorValue.size(), 1);
            ASSERT_NEAR(tensorValue[0], scale_factor_2, abs_err);
        }

        if (layer->getName() == "fc_out" && place == raul::NetworkParameters::CallbackPlace::Before_Backward)
        {
            auto& tensorValue = memoryManager["fc_outGradient"];
            ASSERT_EQ(tensorValue.size(), 1);
            ASSERT_NEAR(tensorValue[0], scale_factor_1, abs_err);
        }
    };

    init_ones(memoryManagerFP32["data"]);
    init_grad(memoryManagerFP32, "fc_out", 1.0_dt);

    network.forwardPassTraining();
    network.backwardPassTraining();
}

TEST(TestLossScale, OneLayerFP16Unit)
{
    PROFILE_TEST

    const auto abs_err = 1e-3;
    const auto batch_size = 1;
    const auto scale_factor = 300.0_dt;

    auto network = raul::Workflow(raul::CompressionMode::NONE, raul::CalculationMode::DETERMINISTIC, raul::AllocationMode::STANDARD, raul::ExecutionTarget::CPUFP16);

    network.add<raul::DataLayer>("data", raul::DataParams{ { "data" }, 1, 1, 1 });
    network.add<raul::LinearLayer>("fc_in", raul::LinearParams{ { "data" }, { "fc_in" }, 1 });
    network.add<raul::ReLUActivation>("relu", raul::BasicParams{ { "fc_in" }, { "relu" } });
    network.add<raul::LinearLayer>("fc_out", raul::LinearParams{ { "relu" }, { "fc_out" }, 1 });

    network.setScaling("fc_in", raul::ScalingStrategy(scale_factor));

    network.preparePipelines();
    network.setBatchSize(batch_size);
    network.prepareMemoryForTraining();

    auto init_ones = raul::initializers::ConstantInitializer{ 1.0 };
    auto& memoryManagerFP16 = network.getMemoryManager<raul::MemoryManagerFP16>();
    for (auto& paramName : network.getTrainableParameterNames())
    {
        auto& tensor = memoryManagerFP16[paramName];
        init_ones(tensor);
    }

    network.getNetworkParameters().mCallbackFP16 = [&](raul::BasicLayer* layer, raul::MemoryManagerFP16& memoryManager, raul::NetworkParameters::CallbackPlace place)
    {
        if (layer->getName() == "fc_in" && place == raul::NetworkParameters::CallbackPlace::Before_Backward)
        {
            auto& tensorValue = memoryManager["fc_inGradient"];
            ASSERT_EQ(tensorValue.size(), 1);
            ASSERT_NEAR(tensorValue[0], scale_factor, abs_err);
        }
    };

    init_ones(memoryManagerFP16["data"]);
    init_grad(memoryManagerFP16, "fc_out", 1.0_hf);

    network.forwardPassTraining();
    network.backwardPassTraining();
}

TEST(TestLossScale, TwoLayersFP16Unit)
{
    PROFILE_TEST

    const auto abs_err = 1e-3;
    const auto batch_size = 1;
    const auto scale_factor_1 = 300.0_dt;
    const auto scale_factor_2 = 70.0_dt;

    auto network = raul::Workflow(raul::CompressionMode::NONE, raul::CalculationMode::DETERMINISTIC, raul::AllocationMode::STANDARD, raul::ExecutionTarget::CPUFP16);

    network.add<raul::DataLayer>("data", raul::DataParams{ { "data" }, 1, 1, 1 });
    network.add<raul::LinearLayer>("fc_in", raul::LinearParams{ { "data" }, { "fc_in" }, 1 });
    network.add<raul::ReLUActivation>("relu", raul::BasicParams{ { "fc_in" }, { "relu" } });
    network.add<raul::LinearLayer>("fc_out", raul::LinearParams{ { "relu" }, { "fc_out" }, 1 });

    network.setScaling("fc_out", raul::ScalingStrategy(scale_factor_1));
    network.setScaling("fc_in", raul::ScalingStrategy(scale_factor_2));

    network.preparePipelines();
    network.setBatchSize(batch_size);
    network.prepareMemoryForTraining();

    auto init_ones = raul::initializers::ConstantInitializer{ 1.0 };
    auto& memoryManagerFP16 = network.getMemoryManager<raul::MemoryManagerFP16>();
    for (auto& paramName : network.getTrainableParameterNames())
    {
        auto& tensor = memoryManagerFP16[paramName];
        init_ones(tensor);
    }

    network.getNetworkParameters().mCallbackFP16 = [&](raul::BasicLayer* layer, raul::MemoryManagerFP16& memoryManager, raul::NetworkParameters::CallbackPlace place)
    {
        if (layer->getName() == "fc_in" && place == raul::NetworkParameters::CallbackPlace::Before_Backward)
        {
            auto& tensorValue = memoryManager["fc_inGradient"];
            ASSERT_EQ(tensorValue.size(), 1);
            ASSERT_NEAR(tensorValue[0], scale_factor_2, abs_err);
        }

        if (layer->getName() == "fc_out" && place == raul::NetworkParameters::CallbackPlace::Before_Backward)
        {
            auto& tensorValue = memoryManager["fc_outGradient"];
            ASSERT_EQ(tensorValue.size(), 1);
            ASSERT_NEAR(tensorValue[0], scale_factor_1, abs_err);
        }
    };

    init_ones(memoryManagerFP16["data"]);
    init_grad(memoryManagerFP16, "fc_out", 1.0_hf);

    network.forwardPassTraining();
    network.backwardPassTraining();
}

TEST(TestLossScale, ThreeLayersFP16Unit)
{
    PROFILE_TEST

    const auto abs_err = 1e-3;
    const auto batch_size = 1;
    const auto scale_factor_1 = 300.0_dt;
    const auto scale_factor_2 = 5.0_dt;
    const auto scale_factor_3 = 70.0_dt;

    auto network = raul::Workflow(raul::CompressionMode::NONE, raul::CalculationMode::DETERMINISTIC, raul::AllocationMode::STANDARD, raul::ExecutionTarget::CPUFP16);

    network.add<raul::DataLayer>("data", raul::DataParams{ { "data" }, 1, 1, 1 });
    network.add<raul::LinearLayer>("fc_in", raul::LinearParams{ { "data" }, { "fc_in" }, 1 });
    network.add<raul::ReLUActivation>("relu", raul::BasicParams{ { "fc_in" }, { "relu" } });
    network.add<raul::LinearLayer>("fc_out", raul::LinearParams{ { "relu" }, { "fc_out" }, 1 });

    network.setScaling("fc_out", raul::ScalingStrategy(scale_factor_1));
    network.setScaling("relu", raul::ScalingStrategy(scale_factor_2));
    network.setScaling("fc_in", raul::ScalingStrategy(scale_factor_3));

    network.preparePipelines();
    network.setBatchSize(batch_size);
    network.prepareMemoryForTraining();

    auto init_ones = raul::initializers::ConstantInitializer{ 1.0 };
    auto& memoryManagerFP16 = network.getMemoryManager<raul::MemoryManagerFP16>();
    for (auto& paramName : network.getTrainableParameterNames())
    {
        auto& tensor = memoryManagerFP16[paramName];
        init_ones(tensor);
    }

    network.getNetworkParameters().mCallbackFP16 = [&](raul::BasicLayer* layer, raul::MemoryManagerFP16& memoryManager, raul::NetworkParameters::CallbackPlace place)
    {
        if (layer->getName() == "fc_in" && place == raul::NetworkParameters::CallbackPlace::Before_Backward)
        {
            auto& tensorValue = memoryManager["fc_inGradient"];
            ASSERT_EQ(tensorValue.size(), 1);
            ASSERT_NEAR(tensorValue[0], scale_factor_3, abs_err);
        }

        if (layer->getName() == "relu" && place == raul::NetworkParameters::CallbackPlace::Before_Backward)
        {
            auto& tensorValue = memoryManager["reluGradient"];
            ASSERT_EQ(tensorValue.size(), 1);
            ASSERT_NEAR(tensorValue[0], scale_factor_2, abs_err);
        }

        if (layer->getName() == "fc_out" && place == raul::NetworkParameters::CallbackPlace::Before_Backward)
        {
            auto& tensorValue = memoryManager["fc_outGradient"];
            ASSERT_EQ(tensorValue.size(), 1);
            ASSERT_NEAR(tensorValue[0], scale_factor_1, abs_err);
        }
    };

    init_ones(memoryManagerFP16["data"]);
    init_grad(memoryManagerFP16, "fc_out", 1.0_hf);

    network.forwardPassTraining();
    network.backwardPassTraining();
}

} // UT namespace