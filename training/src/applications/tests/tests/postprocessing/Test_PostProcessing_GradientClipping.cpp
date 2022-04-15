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
#include <iostream>
#include <training/common/MemoryManager.h>
#include <training/network/Workflow.h>
#include <training/postprocessing/GradientClipping.h>
#include <training/layers/basic/DataLayer.h>
#include <tests/tools/TestTools.h>

namespace UT
{

TEST(TestGradientClipping, IncorrectGlobalNormUnit)
{
    EXPECT_THROW(raul::postprocessing::GradientClipping clip(0.1_dt, -1.0_dt), raul::Exception);
}

TEST(TestGradientClipping, IncorrectClipNormUnit)
{
    EXPECT_THROW(raul::postprocessing::GradientClipping clip(0.0_dt), raul::Exception);
    EXPECT_THROW(raul::postprocessing::GradientClipping clip(-1.0_dt), raul::Exception);
}

TEST(TestGradientClipping, StreamUnit)
{
    std::ostringstream stream;
    raul::postprocessing::GradientClipping clip(0.1_dt);
    stream << clip;
    ASSERT_STREQ(stream.str().c_str(), "GradientClipping(clip norm = 0.1)");
}

TEST(TestGradientClipping, ProcessGradientsInfClipUnit)
{
    MANAGERS_DEFINE
    NETWORK_PARAMS_DEFINE(networkParameters);
    constexpr raul::dtype clipNorm = std::numeric_limits<raul::dtype>::infinity();
    raul::Tensor fictiveParams{ 0.0_dt, 0.0_dt, 0.0_dt };
    raul::Tensor grads{ 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt };

    std::vector<raul::ParamAndGradImpl<Tensor>> trainableParams;
    trainableParams.emplace_back(raul::ParamAndGradImpl<Tensor>{ fictiveParams, grads });

    raul::postprocessing::GradientClipping clip(clipNorm);
    ASSERT_NO_THROW(clip.processGradients(trainableParams, networkParameters));

    // Checks
    for (auto& [param, grad] : trainableParams)
    {
        for (size_t i = 0; i < grad.size(); ++i)
        {
            EXPECT_TRUE(std::isnan(grad[i]));
        }
    }
}

TEST(TestGradientClipping, ProcessGradientsUnit)
{
    MANAGERS_DEFINE
    NETWORK_PARAMS_DEFINE(networkParameters);
    const raul::dtype clipNorms[] = { 100.0_dt, 0.5_dt };
    const raul::dtype eps = TODTYPE(1e-5);

    raul::Tensor fictiveParams{ 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt };
    raul::Tensor grads[]{ { 0.29197514_dt, 0.20656645_dt, 0.53539073_dt, 0.5612575_dt, 0.4166745_dt, 0.80782795_dt, 0.4932251_dt, 0.99812925_dt },
                          { 0.5554141_dt, 0.22129297_dt, 0.8649249_dt, 0.77728355_dt, 0.6451167_dt, 0.53036225_dt, 0.01444101_dt, 0.87350917_dt },
                          { 0.1952138_dt, 0.7401732_dt, 0.4878018_dt, 0.8753203_dt, 0.4071133_dt, 0.01454818_dt, 0.7095418_dt, 0.36551023_dt } };

    std::vector<raul::ParamAndGradImpl<Tensor>> trainableParams;
    trainableParams.emplace_back(raul::ParamAndGradImpl<Tensor>{ fictiveParams, grads[0] });
    trainableParams.emplace_back(raul::ParamAndGradImpl<Tensor>{ fictiveParams, grads[1] });
    trainableParams.emplace_back(raul::ParamAndGradImpl<Tensor>{ fictiveParams, grads[2] });

    constexpr raul::dtype realGlobalNorm = 2.891162_dt;
    const raul::Tensor realOutput[]{ { 0.05049443_dt, 0.03572378_dt, 0.09259093_dt, 0.09706435_dt, 0.07206004_dt, 0.13970645_dt, 0.08529877_dt, 0.17261732_dt },
                                     { 0.09605379_dt, 0.0382706_dt, 0.14958085_dt, 0.13442408_dt, 0.11156703_dt, 0.0917213_dt, 0.00249744_dt, 0.15106542_dt },
                                     { 0.03376044_dt, 0.12800619_dt, 0.08436085_dt, 0.15137865_dt, 0.07040653_dt, 0.00251598_dt, 0.12270876_dt, 0.06321165_dt } };

    for (size_t q = 0; q < std::size(clipNorms); ++q)
    {
        raul::postprocessing::GradientClipping clip(clipNorms[q]);
        ASSERT_NO_THROW(clip.processGradients(trainableParams, networkParameters));

        // Checks
        ASSERT_TRUE(tools::expect_near_relative(clip.getGlobalNorm(networkParameters), realGlobalNorm, eps));

        size_t j = 0;
        // No changes expected on 1st iteration
        if (q == 0)
        {
            for (auto& [param, grad] : trainableParams)
            {
                for (size_t i = 0; i < grad.size(); ++i)
                {
                    EXPECT_EQ(grad[i], grads[j][i]);
                }
                j++;
            }
        }
        else
        {
            for (auto& [param, grad] : trainableParams)
            {
                for (size_t i = 0; i < grad.size(); ++i)
                {
                    ASSERT_TRUE(tools::expect_near_relative(grad[i], realOutput[j][i], eps));
                }
                j++;
            }
        }
    }
}

TEST(TestGradientClipping, ProcessGradientsGpuUnit)
{
    PROFILE_TEST

    GPU_ONLY_TEST
    
    const raul::dtype clipNorms[] = { 100.0_dt, 0.5_dt };
    const raul::dtype eps = TODTYPE(1e-5);
    
    raul::WorkflowEager work(raul::CompressionMode::NONE, raul::CalculationMode::DETERMINISTIC, raul::AllocationMode::STANDARD, raul::ExecutionTarget::GPU);
    work.add<raul::DataLayer>("data", raul::DataParams{ { "fictiveParams", "grads1", "grads2", "grads3" }, 1, 1, 8 });

    raul::Tensor fictiveParams{ 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt };
    raul::Tensor grads[]{ { 0.29197514_dt, 0.20656645_dt, 0.53539073_dt, 0.5612575_dt, 0.4166745_dt, 0.80782795_dt, 0.4932251_dt, 0.99812925_dt },
                          { 0.5554141_dt, 0.22129297_dt, 0.8649249_dt, 0.77728355_dt, 0.6451167_dt, 0.53036225_dt, 0.01444101_dt, 0.87350917_dt },
                          { 0.1952138_dt, 0.7401732_dt, 0.4878018_dt, 0.8753203_dt, 0.4071133_dt, 0.01454818_dt, 0.7095418_dt, 0.36551023_dt } };
    
    TENSORS_CREATE(1)
    auto& memory_manager = work.getMemoryManager<raul::MemoryManagerGPU>();
    memory_manager["fictiveParams"] = TORANGE(fictiveParams);
    for (size_t i = 0; i < std::size(grads); ++i)
    {
        memory_manager["grads" + std::to_string(i + 1)] = TORANGE(grads[i]);
    }
    
    auto& networkParameters = work.getNetworkParameters();

    std::vector<raul::ParamAndGradImpl<raul::TensorGPU>> trainableParams;
    trainableParams.push_back(raul::ParamAndGradImpl<raul::TensorGPU>{ memory_manager("fictiveParams"), memory_manager("grads1") });
    trainableParams.push_back(raul::ParamAndGradImpl<raul::TensorGPU>{ memory_manager("fictiveParams"), memory_manager("grads2") });
    trainableParams.push_back(raul::ParamAndGradImpl<raul::TensorGPU>{ memory_manager("fictiveParams"), memory_manager("grads3") });

    constexpr raul::dtype realGlobalNorm = 2.891162_dt;
    const raul::Tensor realOutput[]{ { 0.05049443_dt, 0.03572378_dt, 0.09259093_dt, 0.09706435_dt, 0.07206004_dt, 0.13970645_dt, 0.08529877_dt, 0.17261732_dt },
                                     { 0.09605379_dt, 0.0382706_dt, 0.14958085_dt, 0.13442408_dt, 0.11156703_dt, 0.0917213_dt, 0.00249744_dt, 0.15106542_dt },
                                     { 0.03376044_dt, 0.12800619_dt, 0.08436085_dt, 0.15137865_dt, 0.07040653_dt, 0.00251598_dt, 0.12270876_dt, 0.06321165_dt } };

    for (size_t q = 0; q < std::size(clipNorms); ++q)
    {
        raul::postprocessing::GradientClipping clip(clipNorms[q]);
        ASSERT_NO_THROW(clip.processGradients(trainableParams, networkParameters));

        ASSERT_TRUE(tools::expect_near_relative(clip.getGlobalNorm(networkParameters), realGlobalNorm, eps));

        for (size_t j = 0; j < std::size(grads); ++j)
        {
            const raul::Tensor& grad = memory_manager["grads" + std::to_string(j + 1)]; 
            for (size_t i = 0; i < grad.size(); ++i)
            {
                // No changes expected on 1st iteration
                if (q == 0)
                {
                    EXPECT_EQ(grad[i], grads[j][i]);
                }
                else
                {
                    ASSERT_TRUE(tools::expect_near_relative(grad[i], realOutput[j][i], eps));
                }
            }
        }
    }
}

}
