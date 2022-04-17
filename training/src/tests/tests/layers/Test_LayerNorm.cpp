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

#include <training/compiler/Layers.h>
#include <training/compiler/Workflow.h>

namespace UT
{

using namespace raul;

// layernorm.py
TEST(TestLayerNorm, Unit)
{
    PROFILE_TEST
    MANAGERS_DEFINE
    NETWORK_PARAMS_DEFINE(networkParameters);

    Tensor raw = { -1._dt, 0._dt, 1._dt, 4._dt, 5._dt, 6._dt, 0._dt, 4._dt, 7._dt, -1._dt, 2._dt, 5._dt };
    size_t BATCH_SIZE = 2;
    size_t HEIGHT = 2;
    size_t WIDTH = 3;
    constexpr dtype eps = 1e-4_dt;

    Tensor realLayerNorm = { -1._dt, 0._dt, 1._dt, -1._dt, 0._dt, 1._dt, -1.0441_dt, 0.0949_dt, 0.9492_dt, -1._dt, 0._dt, 1._dt };
    Tensor realLayerNormWeightsGrad = { 0.044343_dt, 0.002493_dt, -0.015904_dt };
    Tensor realLayerNormBiasesGrad = { -0.044287_dt, -0.015210_dt, -0.012081_dt };
    Tensor realInputGrad = { -0.005130_dt, 0.010260_dt, -0.005130_dt, -0.009077_dt, 0.018154_dt, -0.009077_dt, 0.001306_dt, -0.003048_dt, 0.001742_dt, 0.002105_dt, -0.004209_dt, 0.002105_dt };

    work.add<raul::DataLayer>("data", DataParams{ { "in" }, 1, HEIGHT, WIDTH });
    LayerNormLayer layer("ln", LayerNormParams{ "in", "ln", 0. }, networkParameters);
    TENSORS_CREATE(BATCH_SIZE)
    layer.initNotBSTensors();
    memory_manager["in"] = TORANGE(raw);
    // Forward pass
    layer.forwardCompute(NetworkMode::Train);

    const Tensor& ln = memory_manager["ln"];
    ASSERT_EQ(ln.size(), realLayerNorm.size());
    for (size_t i = 0; i < ln.size(); ++i)
    {
        EXPECT_NEAR(ln[i], realLayerNorm[i], eps);
    }
    printf(" - LayerNorm forward is Ok.\n");

    // Backward pass
    memory_manager[Name("ln").grad()].memAllocate(nullptr);
    memory_manager[Name("ln").grad()] =
        TORANGE((Tensor({ 0.001815_dt, -0.037754_dt, -0.108103_dt, -0.147300_dt, -0.012211_dt, 0.068416_dt, -0.001262_dt, 0.026261_dt, 0.075196_dt, 0.102461_dt, 0.008494_dt, -0.047590_dt })));
    layer.backwardCompute();
    // lnLayer.backwardCompute();
    auto& inputGrad = memory_manager[Name("in").grad()];
    auto& lnWeightGrad = memory_manager["ln::WeightsGradient"];
    auto& lnBiasGrad = memory_manager["ln::BiasesGradient"];

    for (size_t i = 0; i < lnWeightGrad.size(); ++i)
    {
        EXPECT_NEAR(lnWeightGrad[i], realLayerNormWeightsGrad[i], eps);
    }
    for (size_t i = 0; i < lnBiasGrad.size(); ++i)
    {
        EXPECT_NEAR(lnBiasGrad[i], realLayerNormBiasesGrad[i], eps);
    }
    for (size_t i = 0; i < inputGrad.size(); ++i)
    {
        EXPECT_NEAR(inputGrad[i], realInputGrad[i], eps);
    }
    printf(" - LayerNorm backward is Ok.\n");
}

// layernorm_tf.py
TEST(TestLayerNorm, TFStyleUnit)
{
    PROFILE_TEST
    MANAGERS_DEFINE
    NETWORK_PARAMS_DEFINE(networkParameters);

    Tensor raw = { -1._dt, 0._dt, 1._dt, 4._dt, 5._dt, 6._dt, 0._dt, 4._dt, 7._dt, -1._dt, 2._dt, 5._dt };
    size_t BATCH_SIZE = 2;
    size_t HEIGHT = 2;
    size_t WIDTH = 3;
    constexpr dtype eps = 1e-4_dt;
    constexpr dtype lnEps = 1e-12_dt;

    Tensor realLayerNorm = { -1.224653_dt, 0.000000_dt, 1.224653_dt, -1.224653_dt, 0.000000_dt, 1.224653_dt, -1.278716_dt, 0.116247_dt, 1.162469_dt, -1.224735_dt, 0.000000_dt, 1.224735_dt };
    Tensor realLayerNormWeightsGrad = { 0.066124_dt, 0.002906_dt, -0.022477_dt };
    Tensor realLayerNormBiasesGrad = { -0.053948_dt, -0.018527_dt, -0.014716_dt };
    Tensor realInputGrad = { -0.006479_dt, 0.012979_dt, -0.006500_dt, -0.011503_dt, 0.022965_dt, -0.011462_dt, 0.001523_dt, -0.003554_dt, 0.002031_dt, 0.002454_dt, -0.004907_dt, 0.002454_dt };

    work.add<DataLayer>("data", DataParams{ { "in" }, 1, HEIGHT, WIDTH });
    LayerNormLayer layer("ln", LayerNormParams{ "in", "ln", lnEps, true }, networkParameters);
    TENSORS_CREATE(BATCH_SIZE)
    layer.initNotBSTensors();
    memory_manager["in"] = TORANGE(raw);
    // Forward pass

    layer.forwardCompute(NetworkMode::Train);

    const Tensor& ln = memory_manager["ln"];
    ASSERT_EQ(ln.size(), realLayerNorm.size());
    for (size_t i = 0; i < ln.size(); ++i)
    {
        EXPECT_NEAR(ln[i], realLayerNorm[i], eps);
    }
    printf(" - LayerNorm forward is Ok.\n");

    // Backward pass
    memory_manager[Name("ln").grad()].memAllocate(nullptr);
    memory_manager[Name("ln").grad()] =
        TORANGE((Tensor({ 0.001874_dt, -0.038998_dt, -0.111665_dt, -0.152153_dt, -0.012613_dt, 0.070670_dt, -0.001202_dt, 0.024997_dt, 0.071577_dt, 0.097530_dt, 0.008085_dt, -0.045300_dt })));

    layer.backwardCompute();
    auto& inputGrad = memory_manager[Name("in").grad()];
    auto& lnWeightGrad = memory_manager["ln::WeightsGradient"];
    auto& lnBiasGrad = memory_manager["ln::BiasesGradient"];

    for (size_t i = 0; i < lnWeightGrad.size(); ++i)
    {
        EXPECT_NEAR(lnWeightGrad[i], realLayerNormWeightsGrad[i], eps);
    }
    for (size_t i = 0; i < lnBiasGrad.size(); ++i)
    {
        EXPECT_NEAR(lnBiasGrad[i], realLayerNormBiasesGrad[i], eps);
    }
    for (size_t i = 0; i < inputGrad.size(); ++i)
    {
        EXPECT_NEAR(inputGrad[i], realInputGrad[i], eps);
    }
    printf(" - LayerNorm backward is Ok.\n");
}

}