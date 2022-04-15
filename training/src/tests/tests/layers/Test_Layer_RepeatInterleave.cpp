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

#include <training/base/common/MemoryManager.h>
#include <training/base/layers/basic/RepeatInterleaveLayer.h>
#include <training/compiler/Workflow.h>

namespace UT
{

TEST(TestLayerRepeatInterleave, IncorrectBatchDimensionUnit)
{
    PROFILE_TEST

    // Initialization
    MANAGERS_DEFINE
    NETWORK_PARAMS_DEFINE(networkParameters);

    // Apply function
    ASSERT_THROW(raul::RepeatInterleaveLayer("repeat", raul::RepeatInterleaveParams{ { "x" }, { "out" }, { 1 }, raul::Dimension::Batch }, networkParameters), raul::Exception);
}

TEST(TestLayerRepeatInterleave, ForwardExampleUnit)
{
    PROFILE_TEST
    // Test parameters
    const auto batch = 1;
    const auto depth = 2;
    const auto height = 2;
    const auto width = 1;

    const raul::Tensor x{ 1.0_dt, 2.0_dt, 3.0_dt, 4.0_dt };

    std::initializer_list<size_t> repeats[] = { { 2 }, { 3 }, { 1, 2 } };

    std::string dimensions[] = { "default", "height", "depth" };

    const raul::Tensor realOuts[] = { { 1.0_dt, 1.0_dt, 2.0_dt, 2.0_dt, 3.0_dt, 3.0_dt, 4.0_dt, 4.0_dt },
                                      { 1.0_dt, 1.0_dt, 1.0_dt, 2.0_dt, 2.0_dt, 2.0_dt, 3.0_dt, 3.0_dt, 3.0_dt, 4.0_dt, 4.0_dt, 4.0_dt },
                                      { 1.0_dt, 2.0_dt, 3.0_dt, 4.0_dt, 3.0_dt, 4.0_dt } };

    yato::dimensionality<4> expectedShapes[] = { yato::dims(1, 8, 1, 1), yato::dims(1, 2, 6, 1), yato::dims(1, 3, 2, 1) };

    for (size_t iter = 0; iter < std::size(dimensions); ++iter)
    {
        MANAGERS_DEFINE
        NETWORK_PARAMS_DEFINE(networkParameters);

        work.tensorNeeded("x", "x", raul::WShape{ raul::BS(), depth, height, width }, DEC_FORW_READ_NOMEMOPT);

        raul::RepeatInterleaveLayer repeat("repeat", raul::RepeatInterleaveParams{ { "x" }, { "out" }, repeats[iter], dimensions[iter] }, networkParameters);
        TENSORS_CREATE(batch);
        memory_manager["x"] = TORANGE(x);
        repeat.forwardCompute(raul::NetworkMode::Test);

        // Checks
        const auto& outTensor = memory_manager["out"];
        EXPECT_EQ(outTensor.getShape(), expectedShapes[iter]);
        for (size_t i = 0; i < outTensor.size(); ++i)
        {
            EXPECT_EQ(outTensor[i], realOuts[iter][i]);
        }
    }
}

TEST(TestLayerRepeatInterleave, BackwardExampleUnit)
{
    PROFILE_TEST
    // Test parameters
    const auto batch = 1;
    const auto depth = 2;
    const auto height = 2;
    const auto width = 1;

    const raul::Tensor x{ 1.0_dt, 2.0_dt, 3.0_dt, 4.0_dt };

    std::initializer_list<size_t> repeats[] = { { 2 }, { 3 }, { 1, 2 } };

    std::string dimensions[] = { "default", "height", "depth" };

    const raul::Tensor realGrads[] = { { 2.0_dt, 2.0_dt, 2.0_dt, 2.0_dt }, { 3.0_dt, 3.0_dt, 3.0_dt, 3.0_dt }, { 1.0_dt, 1.0_dt, 2.0_dt, 2.0_dt } };

    yato::dimensionality<4> expectedShapes[] = { yato::dims(1, 8, 1, 1), yato::dims(1, 2, 6, 1), yato::dims(1, 3, 2, 1) };

    for (size_t iter = 0; iter < std::size(dimensions); ++iter)
    {
        MANAGERS_DEFINE
        NETWORK_PARAMS_DEFINE(networkParameters);

        work.tensorNeeded("x", "x", raul::WShape{ raul::BS(), depth, height, width }, DEC_FORW_READ_NOMEMOPT);

        raul::RepeatInterleaveLayer repeat("repeat", raul::RepeatInterleaveParams{ { "x" }, { "out" }, repeats[iter], dimensions[iter] }, networkParameters);
        TENSORS_CREATE(batch);
        memory_manager["x"] = TORANGE(x);
        memory_manager[raul::Name("out").grad()] = TORANGE(*memory_manager.createTensor("gradient", expectedShapes[iter], 1.0_dt));
        repeat.forwardCompute(raul::NetworkMode::Test);
        repeat.backwardCompute();

        // Checks
        const auto& xTensorGrad = memory_manager[raul::Name("x").grad()];
        EXPECT_EQ(xTensorGrad.getShape(), memory_manager["x"].getShape());
        for (size_t i = 0; i < xTensorGrad.size(); ++i)
        {
            EXPECT_EQ(xTensorGrad[i], realGrads[iter][i]);
        }
    }
}

TEST(TestLayerRepeatInterleave, ForwardUnit)
{
    PROFILE_TEST
    // Test parameters
    const auto batch = 1;
    const auto depth = 2;
    const auto height = 3;
    const auto width = 4;

    const raul::Tensor x{ 0.47356850_dt, 0.61359876_dt, 0.01207304_dt, 0.03959680_dt, 0.56738567_dt, 0.25380611_dt, 0.53375959_dt, 0.60933894_dt,
                          0.71622825_dt, 0.60428441_dt, 0.99795526_dt, 0.14445406_dt, 0.38160384_dt, 0.62407351_dt, 0.96103561_dt, 0.12649149_dt,
                          0.87728381_dt, 0.03422910_dt, 0.39936811_dt, 0.71827561_dt, 0.81016523_dt, 0.50096071_dt, 0.05554926_dt, 0.23256207_dt };

    std::initializer_list<size_t> repeats[] = { { 2 }, { 2, 3 }, { 2, 2, 3 }, { 3, 4, 5, 6 }, { 3 } };

    std::string dimensions[] = { "default", "depth", "height", "width", "width" };

    const raul::Tensor realOuts[] = {
        { 0.47356850_dt, 0.47356850_dt, 0.61359876_dt, 0.61359876_dt, 0.01207304_dt, 0.01207304_dt, 0.03959680_dt, 0.03959680_dt, 0.56738567_dt, 0.56738567_dt, 0.25380611_dt, 0.25380611_dt,
          0.53375959_dt, 0.53375959_dt, 0.60933894_dt, 0.60933894_dt, 0.71622825_dt, 0.71622825_dt, 0.60428441_dt, 0.60428441_dt, 0.99795526_dt, 0.99795526_dt, 0.14445406_dt, 0.14445406_dt,
          0.38160384_dt, 0.38160384_dt, 0.62407351_dt, 0.62407351_dt, 0.96103561_dt, 0.96103561_dt, 0.12649149_dt, 0.12649149_dt, 0.87728381_dt, 0.87728381_dt, 0.03422910_dt, 0.03422910_dt,
          0.39936811_dt, 0.39936811_dt, 0.71827561_dt, 0.71827561_dt, 0.81016523_dt, 0.81016523_dt, 0.50096071_dt, 0.50096071_dt, 0.05554926_dt, 0.05554926_dt, 0.23256207_dt, 0.23256207_dt },
        { 0.47356850_dt, 0.61359876_dt, 0.01207304_dt, 0.03959680_dt, 0.56738567_dt, 0.25380611_dt, 0.53375959_dt, 0.60933894_dt, 0.71622825_dt, 0.60428441_dt, 0.99795526_dt, 0.14445406_dt,
          0.47356850_dt, 0.61359876_dt, 0.01207304_dt, 0.03959680_dt, 0.56738567_dt, 0.25380611_dt, 0.53375959_dt, 0.60933894_dt, 0.71622825_dt, 0.60428441_dt, 0.99795526_dt, 0.14445406_dt,
          0.38160384_dt, 0.62407351_dt, 0.96103561_dt, 0.12649149_dt, 0.87728381_dt, 0.03422910_dt, 0.39936811_dt, 0.71827561_dt, 0.81016523_dt, 0.50096071_dt, 0.05554926_dt, 0.23256207_dt,
          0.38160384_dt, 0.62407351_dt, 0.96103561_dt, 0.12649149_dt, 0.87728381_dt, 0.03422910_dt, 0.39936811_dt, 0.71827561_dt, 0.81016523_dt, 0.50096071_dt, 0.05554926_dt, 0.23256207_dt,
          0.38160384_dt, 0.62407351_dt, 0.96103561_dt, 0.12649149_dt, 0.87728381_dt, 0.03422910_dt, 0.39936811_dt, 0.71827561_dt, 0.81016523_dt, 0.50096071_dt, 0.05554926_dt, 0.23256207_dt },
        { 0.47356850_dt, 0.61359876_dt, 0.01207304_dt, 0.03959680_dt, 0.47356850_dt, 0.61359876_dt, 0.01207304_dt, 0.03959680_dt, 0.56738567_dt, 0.25380611_dt, 0.53375959_dt, 0.60933894_dt,
          0.56738567_dt, 0.25380611_dt, 0.53375959_dt, 0.60933894_dt, 0.71622825_dt, 0.60428441_dt, 0.99795526_dt, 0.14445406_dt, 0.71622825_dt, 0.60428441_dt, 0.99795526_dt, 0.14445406_dt,
          0.71622825_dt, 0.60428441_dt, 0.99795526_dt, 0.14445406_dt, 0.38160384_dt, 0.62407351_dt, 0.96103561_dt, 0.12649149_dt, 0.38160384_dt, 0.62407351_dt, 0.96103561_dt, 0.12649149_dt,
          0.87728381_dt, 0.03422910_dt, 0.39936811_dt, 0.71827561_dt, 0.87728381_dt, 0.03422910_dt, 0.39936811_dt, 0.71827561_dt, 0.81016523_dt, 0.50096071_dt, 0.05554926_dt, 0.23256207_dt,
          0.81016523_dt, 0.50096071_dt, 0.05554926_dt, 0.23256207_dt, 0.81016523_dt, 0.50096071_dt, 0.05554926_dt, 0.23256207_dt },
        { 0.47356850_dt, 0.47356850_dt, 0.47356850_dt, 0.61359876_dt, 0.61359876_dt, 0.61359876_dt, 0.61359876_dt, 0.01207304_dt, 0.01207304_dt, 0.01207304_dt, 0.01207304_dt, 0.01207304_dt,
          0.03959680_dt, 0.03959680_dt, 0.03959680_dt, 0.03959680_dt, 0.03959680_dt, 0.03959680_dt, 0.56738567_dt, 0.56738567_dt, 0.56738567_dt, 0.25380611_dt, 0.25380611_dt, 0.25380611_dt,
          0.25380611_dt, 0.53375959_dt, 0.53375959_dt, 0.53375959_dt, 0.53375959_dt, 0.53375959_dt, 0.60933894_dt, 0.60933894_dt, 0.60933894_dt, 0.60933894_dt, 0.60933894_dt, 0.60933894_dt,
          0.71622825_dt, 0.71622825_dt, 0.71622825_dt, 0.60428441_dt, 0.60428441_dt, 0.60428441_dt, 0.60428441_dt, 0.99795526_dt, 0.99795526_dt, 0.99795526_dt, 0.99795526_dt, 0.99795526_dt,
          0.14445406_dt, 0.14445406_dt, 0.14445406_dt, 0.14445406_dt, 0.14445406_dt, 0.14445406_dt, 0.38160384_dt, 0.38160384_dt, 0.38160384_dt, 0.62407351_dt, 0.62407351_dt, 0.62407351_dt,
          0.62407351_dt, 0.96103561_dt, 0.96103561_dt, 0.96103561_dt, 0.96103561_dt, 0.96103561_dt, 0.12649149_dt, 0.12649149_dt, 0.12649149_dt, 0.12649149_dt, 0.12649149_dt, 0.12649149_dt,
          0.87728381_dt, 0.87728381_dt, 0.87728381_dt, 0.03422910_dt, 0.03422910_dt, 0.03422910_dt, 0.03422910_dt, 0.39936811_dt, 0.39936811_dt, 0.39936811_dt, 0.39936811_dt, 0.39936811_dt,
          0.71827561_dt, 0.71827561_dt, 0.71827561_dt, 0.71827561_dt, 0.71827561_dt, 0.71827561_dt, 0.81016523_dt, 0.81016523_dt, 0.81016523_dt, 0.50096071_dt, 0.50096071_dt, 0.50096071_dt,
          0.50096071_dt, 0.05554926_dt, 0.05554926_dt, 0.05554926_dt, 0.05554926_dt, 0.05554926_dt, 0.23256207_dt, 0.23256207_dt, 0.23256207_dt, 0.23256207_dt, 0.23256207_dt, 0.23256207_dt },
        { 0.47356850_dt, 0.47356850_dt, 0.47356850_dt, 0.61359876_dt, 0.61359876_dt, 0.61359876_dt, 0.01207304_dt, 0.01207304_dt, 0.01207304_dt, 0.03959680_dt, 0.03959680_dt, 0.03959680_dt,
          0.56738567_dt, 0.56738567_dt, 0.56738567_dt, 0.25380611_dt, 0.25380611_dt, 0.25380611_dt, 0.53375959_dt, 0.53375959_dt, 0.53375959_dt, 0.60933894_dt, 0.60933894_dt, 0.60933894_dt,
          0.71622825_dt, 0.71622825_dt, 0.71622825_dt, 0.60428441_dt, 0.60428441_dt, 0.60428441_dt, 0.99795526_dt, 0.99795526_dt, 0.99795526_dt, 0.14445406_dt, 0.14445406_dt, 0.14445406_dt,
          0.38160384_dt, 0.38160384_dt, 0.38160384_dt, 0.62407351_dt, 0.62407351_dt, 0.62407351_dt, 0.96103561_dt, 0.96103561_dt, 0.96103561_dt, 0.12649149_dt, 0.12649149_dt, 0.12649149_dt,
          0.87728381_dt, 0.87728381_dt, 0.87728381_dt, 0.03422910_dt, 0.03422910_dt, 0.03422910_dt, 0.39936811_dt, 0.39936811_dt, 0.39936811_dt, 0.71827561_dt, 0.71827561_dt, 0.71827561_dt,
          0.81016523_dt, 0.81016523_dt, 0.81016523_dt, 0.50096071_dt, 0.50096071_dt, 0.50096071_dt, 0.05554926_dt, 0.05554926_dt, 0.05554926_dt, 0.23256207_dt, 0.23256207_dt, 0.23256207_dt }
    };

    yato::dimensionality<4> expectedShapes[] = { yato::dims(1, 48, 1, 1), yato::dims(1, 5, 3, 4), yato::dims(1, 2, 7, 4), yato::dims(1, 2, 3, 18), yato::dims(1, 2, 3, 12) };

    for (size_t iter = 0; iter < std::size(dimensions); ++iter)
    {
        MANAGERS_DEFINE
        NETWORK_PARAMS_DEFINE(networkParameters);

        work.tensorNeeded("x", "x", raul::WShape{ raul::BS(), depth, height, width }, DEC_FORW_READ_NOMEMOPT);

        raul::RepeatInterleaveLayer repeat("repeat", raul::RepeatInterleaveParams{ { "x" }, { "out" }, repeats[iter], dimensions[iter] }, networkParameters);
        TENSORS_CREATE(batch);
        memory_manager["x"] = TORANGE(x);
        repeat.forwardCompute(raul::NetworkMode::Test);
        // Checks
        const auto& outTensor = memory_manager["out"];
        EXPECT_EQ(outTensor.getShape(), expectedShapes[iter]);
        for (size_t i = 0; i < outTensor.size(); ++i)
        {
            EXPECT_EQ(outTensor[i], realOuts[iter][i]);
        }
    }
}

TEST(TestLayerRepeatInterleave, BackwardUnit)
{
    PROFILE_TEST
    // Test parameters
    const auto batch = 1;
    const auto depth = 2;
    const auto height = 3;
    const auto width = 4;
    const auto eps = TODTYPE(1e-6);

    const raul::Tensor x{ 0.47356850_dt, 0.61359876_dt, 0.01207304_dt, 0.03959680_dt, 0.56738567_dt, 0.25380611_dt, 0.53375959_dt, 0.60933894_dt,
                          0.71622825_dt, 0.60428441_dt, 0.99795526_dt, 0.14445406_dt, 0.38160384_dt, 0.62407351_dt, 0.96103561_dt, 0.12649149_dt,
                          0.87728381_dt, 0.03422910_dt, 0.39936811_dt, 0.71827561_dt, 0.81016523_dt, 0.50096071_dt, 0.05554926_dt, 0.23256207_dt };

    std::initializer_list<size_t> repeats[] = { { 2 }, { 2, 3 }, { 2, 2, 3 }, { 3, 4, 5, 6 }, { 3 } };

    std::string dimensions[] = { "default", "depth", "height", "width", "width" };

    const raul::Tensor outNablas[] = {
        { 0.47356850_dt, 0.61359876_dt, 0.01207304_dt, 0.03959680_dt, 0.56738567_dt, 0.25380611_dt, 0.53375959_dt, 0.60933894_dt, 0.71622825_dt, 0.60428441_dt, 0.99795526_dt, 0.14445406_dt,
          0.38160384_dt, 0.62407351_dt, 0.96103561_dt, 0.12649149_dt, 0.87728381_dt, 0.03422910_dt, 0.39936811_dt, 0.71827561_dt, 0.81016523_dt, 0.50096071_dt, 0.05554926_dt, 0.23256207_dt,
          0.10957229_dt, 0.50652146_dt, 0.59991300_dt, 0.07291150_dt, 0.95796555_dt, 0.98673368_dt, 0.10710853_dt, 0.55515742_dt, 0.28962278_dt, 0.20267731_dt, 0.44419128_dt, 0.03274381_dt,
          0.38145286_dt, 0.86610520_dt, 0.36800802_dt, 0.51245415_dt, 0.63189268_dt, 0.89268374_dt, 0.62026799_dt, 0.19522381_dt, 0.29146129_dt, 0.17491180_dt, 0.65033436_dt, 0.97998011_dt },
        { 0.47356850_dt, 0.61359876_dt, 0.01207304_dt, 0.03959680_dt, 0.56738567_dt, 0.25380611_dt, 0.53375959_dt, 0.60933894_dt, 0.71622825_dt, 0.60428441_dt, 0.99795526_dt, 0.14445406_dt,
          0.38160384_dt, 0.62407351_dt, 0.96103561_dt, 0.12649149_dt, 0.87728381_dt, 0.03422910_dt, 0.39936811_dt, 0.71827561_dt, 0.81016523_dt, 0.50096071_dt, 0.05554926_dt, 0.23256207_dt,
          0.10957229_dt, 0.50652146_dt, 0.59991300_dt, 0.07291150_dt, 0.95796555_dt, 0.98673368_dt, 0.10710853_dt, 0.55515742_dt, 0.28962278_dt, 0.20267731_dt, 0.44419128_dt, 0.03274381_dt,
          0.38145286_dt, 0.86610520_dt, 0.36800802_dt, 0.51245415_dt, 0.63189268_dt, 0.89268374_dt, 0.62026799_dt, 0.19522381_dt, 0.29146129_dt, 0.17491180_dt, 0.65033436_dt, 0.97998011_dt,
          0.76997584_dt, 0.94594479_dt, 0.75355482_dt, 0.54385155_dt, 0.26714522_dt, 0.34428477_dt, 0.96595669_dt, 0.80406678_dt, 0.46522611_dt, 0.21898687_dt, 0.68941998_dt, 0.66582263_dt },
        { 0.47356850_dt, 0.61359876_dt, 0.01207304_dt, 0.03959680_dt, 0.56738567_dt, 0.25380611_dt, 0.53375959_dt, 0.60933894_dt, 0.71622825_dt, 0.60428441_dt, 0.99795526_dt, 0.14445406_dt,
          0.38160384_dt, 0.62407351_dt, 0.96103561_dt, 0.12649149_dt, 0.87728381_dt, 0.03422910_dt, 0.39936811_dt, 0.71827561_dt, 0.81016523_dt, 0.50096071_dt, 0.05554926_dt, 0.23256207_dt,
          0.10957229_dt, 0.50652146_dt, 0.59991300_dt, 0.07291150_dt, 0.95796555_dt, 0.98673368_dt, 0.10710853_dt, 0.55515742_dt, 0.28962278_dt, 0.20267731_dt, 0.44419128_dt, 0.03274381_dt,
          0.38145286_dt, 0.86610520_dt, 0.36800802_dt, 0.51245415_dt, 0.63189268_dt, 0.89268374_dt, 0.62026799_dt, 0.19522381_dt, 0.29146129_dt, 0.17491180_dt, 0.65033436_dt, 0.97998011_dt,
          0.76997584_dt, 0.94594479_dt, 0.75355482_dt, 0.54385155_dt, 0.26714522_dt, 0.34428477_dt, 0.96595669_dt, 0.80406678_dt },
        { 0.47356850_dt, 0.61359876_dt, 0.01207304_dt, 0.03959680_dt, 0.56738567_dt, 0.25380611_dt, 0.53375959_dt, 0.60933894_dt, 0.71622825_dt, 0.60428441_dt, 0.99795526_dt, 0.14445406_dt,
          0.38160384_dt, 0.62407351_dt, 0.96103561_dt, 0.12649149_dt, 0.87728381_dt, 0.03422910_dt, 0.39936811_dt, 0.71827561_dt, 0.81016523_dt, 0.50096071_dt, 0.05554926_dt, 0.23256207_dt,
          0.10957229_dt, 0.50652146_dt, 0.59991300_dt, 0.07291150_dt, 0.95796555_dt, 0.98673368_dt, 0.10710853_dt, 0.55515742_dt, 0.28962278_dt, 0.20267731_dt, 0.44419128_dt, 0.03274381_dt,
          0.38145286_dt, 0.86610520_dt, 0.36800802_dt, 0.51245415_dt, 0.63189268_dt, 0.89268374_dt, 0.62026799_dt, 0.19522381_dt, 0.29146129_dt, 0.17491180_dt, 0.65033436_dt, 0.97998011_dt,
          0.76997584_dt, 0.94594479_dt, 0.75355482_dt, 0.54385155_dt, 0.26714522_dt, 0.34428477_dt, 0.96595669_dt, 0.80406678_dt, 0.46522611_dt, 0.21898687_dt, 0.68941998_dt, 0.66582263_dt,
          0.76642013_dt, 0.30439049_dt, 0.79592609_dt, 0.69084817_dt, 0.35829926_dt, 0.78180724_dt, 0.65462142_dt, 0.91969174_dt, 0.76629215_dt, 0.47312105_dt, 0.03121775_dt, 0.92311883_dt,
          0.41893458_dt, 0.97102535_dt, 0.48558223_dt, 0.17522883_dt, 0.39512479_dt, 0.24138790_dt, 0.10555458_dt, 0.06852686_dt, 0.02500689_dt, 0.29676020_dt, 0.05066872_dt, 0.97760677_dt,
          0.40033334_dt, 0.42702633_dt, 0.69453138_dt, 0.71851665_dt, 0.97512805_dt, 0.19063199_dt, 0.44824940_dt, 0.57315487_dt, 0.04944044_dt, 0.42187548_dt, 0.87570536_dt, 0.37057030_dt,
          0.84195560_dt, 0.02813518_dt, 0.23979330_dt, 0.37214810_dt, 0.90836322_dt, 0.06156033_dt, 0.32391739_dt, 0.46039730_dt, 0.31428313_dt, 0.94868171_dt, 0.94160742_dt, 0.93181843_dt },
        { 0.47356850_dt, 0.61359876_dt, 0.01207304_dt, 0.03959680_dt, 0.56738567_dt, 0.25380611_dt, 0.53375959_dt, 0.60933894_dt, 0.71622825_dt, 0.60428441_dt, 0.99795526_dt, 0.14445406_dt,
          0.38160384_dt, 0.62407351_dt, 0.96103561_dt, 0.12649149_dt, 0.87728381_dt, 0.03422910_dt, 0.39936811_dt, 0.71827561_dt, 0.81016523_dt, 0.50096071_dt, 0.05554926_dt, 0.23256207_dt,
          0.10957229_dt, 0.50652146_dt, 0.59991300_dt, 0.07291150_dt, 0.95796555_dt, 0.98673368_dt, 0.10710853_dt, 0.55515742_dt, 0.28962278_dt, 0.20267731_dt, 0.44419128_dt, 0.03274381_dt,
          0.38145286_dt, 0.86610520_dt, 0.36800802_dt, 0.51245415_dt, 0.63189268_dt, 0.89268374_dt, 0.62026799_dt, 0.19522381_dt, 0.29146129_dt, 0.17491180_dt, 0.65033436_dt, 0.97998011_dt,
          0.76997584_dt, 0.94594479_dt, 0.75355482_dt, 0.54385155_dt, 0.26714522_dt, 0.34428477_dt, 0.96595669_dt, 0.80406678_dt, 0.46522611_dt, 0.21898687_dt, 0.68941998_dt, 0.66582263_dt,
          0.76642013_dt, 0.30439049_dt, 0.79592609_dt, 0.69084817_dt, 0.35829926_dt, 0.78180724_dt, 0.65462142_dt, 0.91969174_dt, 0.76629215_dt, 0.47312105_dt, 0.03121775_dt, 0.92311883_dt }
    };

    const raul::Tensor realGrads[] = {
        { 1.08716726_dt, 0.05166984_dt, 0.82119179_dt, 1.14309859_dt, 1.32051265_dt, 1.14240932_dt, 1.00567734_dt, 1.08752704_dt, 0.91151291_dt, 1.11764371_dt, 1.31112599_dt, 0.28811133_dt,
          0.61609375_dt, 0.67282450_dt, 1.94469929_dt, 0.66226596_dt, 0.49230009_dt, 0.47693509_dt, 1.24755812_dt, 0.88046217_dt, 1.52457643_dt, 0.81549180_dt, 0.46637309_dt, 1.63031447_dt },
        { 0.85517234_dt, 1.23767233_dt, 0.97310865_dt, 0.16608828_dt, 1.44466949_dt, 0.28803521_dt, 0.93312770_dt, 1.32761455_dt, 1.52639341_dt, 1.10524511_dt, 1.05350447_dt, 0.37701613_dt,
          1.26100099_dt, 2.31857157_dt, 1.72147584_dt, 1.12921715_dt, 1.85700345_dt, 2.22370219_dt, 1.69333315_dt, 1.55444801_dt, 1.04631019_dt, 0.59657598_dt, 1.78394556_dt, 1.67854655_dt },
        { 1.04095411_dt, 0.86740488_dt, 0.54583263_dt, 0.64893574_dt, 1.09783208_dt, 1.22835791_dt, 1.95899081_dt, 0.27094555_dt, 1.79702127_dt, 1.04171133_dt, 1.05483031_dt, 1.02374911_dt,
          1.24758840_dt, 1.18941092_dt, 0.55129981_dt, 0.58790123_dt, 1.01334548_dt, 1.75878894_dt, 0.98827600_dt, 0.70767796_dt, 1.32858229_dt, 1.46514130_dt, 2.36984587_dt, 2.32789850_dt },
        { 1.09924030_dt, 1.39454818_dt, 3.07226086_dt, 3.00471735_dt, 1.92780900_dt, 0.89864433_dt, 3.12404513_dt, 1.63150108_dt, 1.61556613_dt, 2.65729856_dt, 2.29191136_dt, 3.62475705_dt,
          2.23524952_dt, 2.34064960_dt, 2.93127108_dt, 3.76806307_dt, 1.87554216_dt, 0.91729611_dt, 1.41856945_dt, 3.40616751_dt, 1.07084465_dt, 2.51010680_dt, 1.61000013_dt, 3.92070532_dt },
        { 1.09924030_dt, 0.86078858_dt, 1.85932684_dt, 1.74669361_dt, 1.96671295_dt, 1.03800440_dt, 1.92780900_dt, 0.78907204_dt, 1.21600676_dt, 2.01761079_dt, 0.95188874_dt, 0.67961240_dt,
          1.61556613_dt, 2.03703070_dt, 1.10695314_dt, 1.80522633_dt, 2.46947551_dt, 1.15528154_dt, 2.23524952_dt, 1.57422948_dt, 1.86673665_dt, 1.83095455_dt, 2.34060526_dt, 1.42745757_dt }
    };

    yato::dimensionality<4> expectedShapes[] = { yato::dims(1, 48, 1, 1), yato::dims(1, 5, 3, 4), yato::dims(1, 2, 7, 4), yato::dims(1, 2, 3, 18), yato::dims(1, 2, 3, 12) };

    for (size_t iter = 0; iter < std::size(expectedShapes); ++iter)
    {
        MANAGERS_DEFINE
        NETWORK_PARAMS_DEFINE(networkParameters);

        work.tensorNeeded("x", "x", raul::WShape{ raul::BS(), depth, height, width }, DEC_FORW_READ_NOMEMOPT);

        raul::RepeatInterleaveLayer repeat("repeat", raul::RepeatInterleaveParams{ { "x" }, { "out" }, repeats[iter], dimensions[iter] }, networkParameters);
        TENSORS_CREATE(batch);
        memory_manager["x"] = TORANGE(x);
        memory_manager[raul::Name("out").grad()] = TORANGE(outNablas[iter]);
        repeat.forwardCompute(raul::NetworkMode::Test);
        repeat.backwardCompute();

        // Checks
        const auto& xTensorGrad = memory_manager[raul::Name("x").grad()];
        EXPECT_EQ(xTensorGrad.getShape(), memory_manager["x"].getShape());
        for (size_t i = 0; i < xTensorGrad.size(); ++i)
        {
            ASSERT_TRUE(tools::expect_near_relative(xTensorGrad[i], realGrads[iter][i], eps));
        }
    }
}

}