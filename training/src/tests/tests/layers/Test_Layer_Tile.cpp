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
#include <training/base/layers/basic/DataLayer.h>
#include <training/base/layers/basic/TileLayer.h>
#include <training/compiler/Workflow.h>

namespace UT
{

TEST(TestLayerTile, IncorrectDimensionUnit)
{
    PROFILE_TEST

    // Initialization
    MANAGERS_DEFINE
    NETWORK_PARAMS_DEFINE(networkParameters);

    // Apply function
    ASSERT_THROW(raul::TileLayer("tile", raul::TilingParams{ { "x" }, { "out" }, 1, raul::Dimension::Batch }, networkParameters), raul::Exception);
}

TEST(TestLayerTile, ForwardUnit)
{
    PROFILE_TEST
    // Test parameters
    const size_t batch = 1;
    const size_t depth = 2;
    const size_t height = 3;
    const size_t width = 4;

    const raul::Tensor x{ 0.29197514_dt, 0.20656645_dt, 0.53539073_dt, 0.5612575_dt,  0.4166745_dt,  0.80782795_dt,  0.4932251_dt,  0.99812925_dt,
                          0.69673514_dt, 0.1253736_dt,  0.7098167_dt,  0.6624156_dt,  0.57225657_dt, 0.36475348_dt,  0.42051828_dt, 0.630057_dt,
                          0.913813_dt,   0.6616472_dt,  0.83347356_dt, 0.08395803_dt, 0.2797594_dt,  0.015523195_dt, 0.72637355_dt, 0.7655387_dt };

    std::vector<size_t> repeats{ 3, 3, 2, 2 };

    std::string dimensions[] = { "depth", "height", "width", "default" };

    const raul::Tensor realOuts[] = {
        { 0.29197514_dt, 0.20656645_dt, 0.53539073_dt, 0.5612575_dt, 0.4166745_dt, 0.80782795_dt, 0.4932251_dt,  0.99812925_dt, 0.69673514_dt, 0.1253736_dt,   0.7098167_dt,  0.6624156_dt,
          0.57225657_dt, 0.36475348_dt, 0.42051828_dt, 0.630057_dt,  0.913813_dt,  0.6616472_dt,  0.83347356_dt, 0.08395803_dt, 0.2797594_dt,  0.015523195_dt, 0.72637355_dt, 0.7655387_dt,
          0.29197514_dt, 0.20656645_dt, 0.53539073_dt, 0.5612575_dt, 0.4166745_dt, 0.80782795_dt, 0.4932251_dt,  0.99812925_dt, 0.69673514_dt, 0.1253736_dt,   0.7098167_dt,  0.6624156_dt,
          0.57225657_dt, 0.36475348_dt, 0.42051828_dt, 0.630057_dt,  0.913813_dt,  0.6616472_dt,  0.83347356_dt, 0.08395803_dt, 0.2797594_dt,  0.015523195_dt, 0.72637355_dt, 0.7655387_dt,
          0.29197514_dt, 0.20656645_dt, 0.53539073_dt, 0.5612575_dt, 0.4166745_dt, 0.80782795_dt, 0.4932251_dt,  0.99812925_dt, 0.69673514_dt, 0.1253736_dt,   0.7098167_dt,  0.6624156_dt,
          0.57225657_dt, 0.36475348_dt, 0.42051828_dt, 0.630057_dt,  0.913813_dt,  0.6616472_dt,  0.83347356_dt, 0.08395803_dt, 0.2797594_dt,  0.015523195_dt, 0.72637355_dt, 0.7655387_dt },
        { 0.29197514_dt, 0.20656645_dt, 0.53539073_dt, 0.5612575_dt, 0.4166745_dt, 0.80782795_dt, 0.4932251_dt,  0.99812925_dt, 0.69673514_dt, 0.1253736_dt,   0.7098167_dt,  0.6624156_dt,
          0.29197514_dt, 0.20656645_dt, 0.53539073_dt, 0.5612575_dt, 0.4166745_dt, 0.80782795_dt, 0.4932251_dt,  0.99812925_dt, 0.69673514_dt, 0.1253736_dt,   0.7098167_dt,  0.6624156_dt,
          0.29197514_dt, 0.20656645_dt, 0.53539073_dt, 0.5612575_dt, 0.4166745_dt, 0.80782795_dt, 0.4932251_dt,  0.99812925_dt, 0.69673514_dt, 0.1253736_dt,   0.7098167_dt,  0.6624156_dt,
          0.57225657_dt, 0.36475348_dt, 0.42051828_dt, 0.630057_dt,  0.913813_dt,  0.6616472_dt,  0.83347356_dt, 0.08395803_dt, 0.2797594_dt,  0.015523195_dt, 0.72637355_dt, 0.7655387_dt,
          0.57225657_dt, 0.36475348_dt, 0.42051828_dt, 0.630057_dt,  0.913813_dt,  0.6616472_dt,  0.83347356_dt, 0.08395803_dt, 0.2797594_dt,  0.015523195_dt, 0.72637355_dt, 0.7655387_dt,
          0.57225657_dt, 0.36475348_dt, 0.42051828_dt, 0.630057_dt,  0.913813_dt,  0.6616472_dt,  0.83347356_dt, 0.08395803_dt, 0.2797594_dt,  0.015523195_dt, 0.72637355_dt, 0.7655387_dt },
        { 0.29197514_dt, 0.20656645_dt, 0.53539073_dt, 0.5612575_dt,  0.29197514_dt, 0.20656645_dt,  0.53539073_dt, 0.5612575_dt, 0.4166745_dt,  0.80782795_dt,  0.4932251_dt,  0.99812925_dt,
          0.4166745_dt,  0.80782795_dt, 0.4932251_dt,  0.99812925_dt, 0.69673514_dt, 0.1253736_dt,   0.7098167_dt,  0.6624156_dt, 0.69673514_dt, 0.1253736_dt,   0.7098167_dt,  0.6624156_dt,
          0.57225657_dt, 0.36475348_dt, 0.42051828_dt, 0.630057_dt,   0.57225657_dt, 0.36475348_dt,  0.42051828_dt, 0.630057_dt,  0.913813_dt,   0.6616472_dt,   0.83347356_dt, 0.08395803_dt,
          0.913813_dt,   0.6616472_dt,  0.83347356_dt, 0.08395803_dt, 0.2797594_dt,  0.015523195_dt, 0.72637355_dt, 0.7655387_dt, 0.2797594_dt,  0.015523195_dt, 0.72637355_dt, 0.7655387_dt },
        { 0.29197514_dt, 0.20656645_dt, 0.53539073_dt, 0.5612575_dt,  0.29197514_dt, 0.20656645_dt,  0.53539073_dt, 0.5612575_dt, 0.4166745_dt,  0.80782795_dt,  0.4932251_dt,  0.99812925_dt,
          0.4166745_dt,  0.80782795_dt, 0.4932251_dt,  0.99812925_dt, 0.69673514_dt, 0.1253736_dt,   0.7098167_dt,  0.6624156_dt, 0.69673514_dt, 0.1253736_dt,   0.7098167_dt,  0.6624156_dt,
          0.29197514_dt, 0.20656645_dt, 0.53539073_dt, 0.5612575_dt,  0.29197514_dt, 0.20656645_dt,  0.53539073_dt, 0.5612575_dt, 0.4166745_dt,  0.80782795_dt,  0.4932251_dt,  0.99812925_dt,
          0.4166745_dt,  0.80782795_dt, 0.4932251_dt,  0.99812925_dt, 0.69673514_dt, 0.1253736_dt,   0.7098167_dt,  0.6624156_dt, 0.69673514_dt, 0.1253736_dt,   0.7098167_dt,  0.6624156_dt,
          0.57225657_dt, 0.36475348_dt, 0.42051828_dt, 0.630057_dt,   0.57225657_dt, 0.36475348_dt,  0.42051828_dt, 0.630057_dt,  0.913813_dt,   0.6616472_dt,   0.83347356_dt, 0.08395803_dt,
          0.913813_dt,   0.6616472_dt,  0.83347356_dt, 0.08395803_dt, 0.2797594_dt,  0.015523195_dt, 0.72637355_dt, 0.7655387_dt, 0.2797594_dt,  0.015523195_dt, 0.72637355_dt, 0.7655387_dt,
          0.57225657_dt, 0.36475348_dt, 0.42051828_dt, 0.630057_dt,   0.57225657_dt, 0.36475348_dt,  0.42051828_dt, 0.630057_dt,  0.913813_dt,   0.6616472_dt,   0.83347356_dt, 0.08395803_dt,
          0.913813_dt,   0.6616472_dt,  0.83347356_dt, 0.08395803_dt, 0.2797594_dt,  0.015523195_dt, 0.72637355_dt, 0.7655387_dt, 0.2797594_dt,  0.015523195_dt, 0.72637355_dt, 0.7655387_dt,
          0.29197514_dt, 0.20656645_dt, 0.53539073_dt, 0.5612575_dt,  0.29197514_dt, 0.20656645_dt,  0.53539073_dt, 0.5612575_dt, 0.4166745_dt,  0.80782795_dt,  0.4932251_dt,  0.99812925_dt,
          0.4166745_dt,  0.80782795_dt, 0.4932251_dt,  0.99812925_dt, 0.69673514_dt, 0.1253736_dt,   0.7098167_dt,  0.6624156_dt, 0.69673514_dt, 0.1253736_dt,   0.7098167_dt,  0.6624156_dt,
          0.29197514_dt, 0.20656645_dt, 0.53539073_dt, 0.5612575_dt,  0.29197514_dt, 0.20656645_dt,  0.53539073_dt, 0.5612575_dt, 0.4166745_dt,  0.80782795_dt,  0.4932251_dt,  0.99812925_dt,
          0.4166745_dt,  0.80782795_dt, 0.4932251_dt,  0.99812925_dt, 0.69673514_dt, 0.1253736_dt,   0.7098167_dt,  0.6624156_dt, 0.69673514_dt, 0.1253736_dt,   0.7098167_dt,  0.6624156_dt,
          0.57225657_dt, 0.36475348_dt, 0.42051828_dt, 0.630057_dt,   0.57225657_dt, 0.36475348_dt,  0.42051828_dt, 0.630057_dt,  0.913813_dt,   0.6616472_dt,   0.83347356_dt, 0.08395803_dt,
          0.913813_dt,   0.6616472_dt,  0.83347356_dt, 0.08395803_dt, 0.2797594_dt,  0.015523195_dt, 0.72637355_dt, 0.7655387_dt, 0.2797594_dt,  0.015523195_dt, 0.72637355_dt, 0.7655387_dt,
          0.57225657_dt, 0.36475348_dt, 0.42051828_dt, 0.630057_dt,   0.57225657_dt, 0.36475348_dt,  0.42051828_dt, 0.630057_dt,  0.913813_dt,   0.6616472_dt,   0.83347356_dt, 0.08395803_dt,
          0.913813_dt,   0.6616472_dt,  0.83347356_dt, 0.08395803_dt, 0.2797594_dt,  0.015523195_dt, 0.72637355_dt, 0.7655387_dt, 0.2797594_dt,  0.015523195_dt, 0.72637355_dt, 0.7655387_dt },
        { 0.29197514_dt, 0.20656645_dt, 0.53539073_dt, 0.5612575_dt, 0.4166745_dt, 0.80782795_dt, 0.4932251_dt,  0.99812925_dt, 0.69673514_dt, 0.1253736_dt,   0.7098167_dt,  0.6624156_dt,
          0.29197514_dt, 0.20656645_dt, 0.53539073_dt, 0.5612575_dt, 0.4166745_dt, 0.80782795_dt, 0.4932251_dt,  0.99812925_dt, 0.69673514_dt, 0.1253736_dt,   0.7098167_dt,  0.6624156_dt,
          0.57225657_dt, 0.36475348_dt, 0.42051828_dt, 0.630057_dt,  0.913813_dt,  0.6616472_dt,  0.83347356_dt, 0.08395803_dt, 0.2797594_dt,  0.015523195_dt, 0.72637355_dt, 0.7655387_dt,
          0.57225657_dt, 0.36475348_dt, 0.42051828_dt, 0.630057_dt,  0.913813_dt,  0.6616472_dt,  0.83347356_dt, 0.08395803_dt, 0.2797594_dt,  0.015523195_dt, 0.72637355_dt, 0.7655387_dt,
          0.29197514_dt, 0.20656645_dt, 0.53539073_dt, 0.5612575_dt, 0.4166745_dt, 0.80782795_dt, 0.4932251_dt,  0.99812925_dt, 0.69673514_dt, 0.1253736_dt,   0.7098167_dt,  0.6624156_dt,
          0.29197514_dt, 0.20656645_dt, 0.53539073_dt, 0.5612575_dt, 0.4166745_dt, 0.80782795_dt, 0.4932251_dt,  0.99812925_dt, 0.69673514_dt, 0.1253736_dt,   0.7098167_dt,  0.6624156_dt,
          0.57225657_dt, 0.36475348_dt, 0.42051828_dt, 0.630057_dt,  0.913813_dt,  0.6616472_dt,  0.83347356_dt, 0.08395803_dt, 0.2797594_dt,  0.015523195_dt, 0.72637355_dt, 0.7655387_dt,
          0.57225657_dt, 0.36475348_dt, 0.42051828_dt, 0.630057_dt,  0.913813_dt,  0.6616472_dt,  0.83347356_dt, 0.08395803_dt, 0.2797594_dt,  0.015523195_dt, 0.72637355_dt, 0.7655387_dt,
          0.29197514_dt, 0.20656645_dt, 0.53539073_dt, 0.5612575_dt, 0.4166745_dt, 0.80782795_dt, 0.4932251_dt,  0.99812925_dt, 0.69673514_dt, 0.1253736_dt,   0.7098167_dt,  0.6624156_dt,
          0.29197514_dt, 0.20656645_dt, 0.53539073_dt, 0.5612575_dt, 0.4166745_dt, 0.80782795_dt, 0.4932251_dt,  0.99812925_dt, 0.69673514_dt, 0.1253736_dt,   0.7098167_dt,  0.6624156_dt,
          0.57225657_dt, 0.36475348_dt, 0.42051828_dt, 0.630057_dt,  0.913813_dt,  0.6616472_dt,  0.83347356_dt, 0.08395803_dt, 0.2797594_dt,  0.015523195_dt, 0.72637355_dt, 0.7655387_dt,
          0.57225657_dt, 0.36475348_dt, 0.42051828_dt, 0.630057_dt,  0.913813_dt,  0.6616472_dt,  0.83347356_dt, 0.08395803_dt, 0.2797594_dt,  0.015523195_dt, 0.72637355_dt, 0.7655387_dt }
    };

    yato::dimensionality<4> expectedShapes[] = { yato::dims(1, 6, 3, 4), yato::dims(1, 2, 9, 4), yato::dims(1, 2, 3, 8), yato::dims(1, 4, 6, 8), yato::dims(1, 6, 6, 4) };

    for (size_t iter = 0; iter < std::size(dimensions); ++iter)
    {
        MANAGERS_DEFINE
        NETWORK_PARAMS_DEFINE(networkParameters);
        work.tensorNeeded("x", "x", raul::WShape{ raul::BS(), depth, height, width }, DEC_FORW_READ_NOMEMOPT);
        std::shared_ptr<raul::TileLayer> tile;
        if (iter < 4)
        {
            raul::TilingParams params = raul::TilingParams{ { "x" }, { "out" }, repeats[iter], dimensions[iter] };
            tile = std::make_shared<raul::TileLayer>("tile", params, networkParameters);
        }
        else
        {
            raul::TilingParams params = raul::TilingParams{ { "x" }, { "out" }, { 3, 2, 1 } };
            tile = std::make_shared<raul::TileLayer>("tile", params, networkParameters);
        }

        TENSORS_CREATE(batch);
        memory_manager["x"] = TORANGE(x);
        tile->forwardCompute(raul::NetworkMode::Test);

        // Checks
        const auto& outTensor = memory_manager["out"];
        EXPECT_EQ(outTensor.getShape(), expectedShapes[iter]);
        for (size_t i = 0; i < outTensor.size(); ++i)
        {
            EXPECT_EQ(outTensor[i], realOuts[iter][i]);
        }
    }
}

TEST(TestLayerTile, BackwardUnit)
{
    PROFILE_TEST
    // Test parameters
    const size_t batch = 1;
    const size_t depth = 2;
    const size_t height = 3;
    const size_t width = 4;
    const auto eps = TODTYPE(1e-6);

    const raul::Tensor x{ 0.29197514_dt, 0.20656645_dt, 0.53539073_dt, 0.5612575_dt,  0.4166745_dt,  0.80782795_dt,  0.4932251_dt,  0.99812925_dt,
                          0.69673514_dt, 0.1253736_dt,  0.7098167_dt,  0.6624156_dt,  0.57225657_dt, 0.36475348_dt,  0.42051828_dt, 0.630057_dt,
                          0.913813_dt,   0.6616472_dt,  0.83347356_dt, 0.08395803_dt, 0.2797594_dt,  0.015523195_dt, 0.72637355_dt, 0.7655387_dt };

    std::vector<size_t> repeats{ 3, 3, 2, 2 };
    std::string dimensions[] = { "depth", "height", "width", "default" };

    const raul::Tensor realGrads[] = {
        { 1.053449_dt,  2.374159_dt,  1.6732247_dt, 0.70244753_dt, 1.536461_dt,  1.5391887_dt, 2.374594_dt,  1.4916971_dt, 1.2150586_dt, 2.096571_dt,  1.5140249_dt, 1.9761292_dt,
          1.8315855_dt, 2.2907639_dt, 1.8733183_dt, 0.7677839_dt,  1.4215688_dt, 0.9596634_dt, 1.4550278_dt, 1.1843588_dt, 1.9347681_dt, 0.6152824_dt, 0.9150052_dt, 1.2870691_dt },
        { 1.3056484_dt, 2.4377136_dt, 1.6344289_dt, 0.70290947_dt, 1.365818_dt,  1.4812039_dt, 2.7720208_dt, 1.8925366_dt,  1.9438181_dt, 1.2583629_dt, 1.2491584_dt, 1.3657807_dt,
          1.5793861_dt, 2.2272096_dt, 1.9121141_dt, 0.76732194_dt, 1.5922117_dt, 1.0176482_dt, 1.0576011_dt, 0.78351927_dt, 1.2060086_dt, 1.4534905_dt, 1.1798717_dt, 1.8974175_dt },
        { 0.45706987_dt, 1.5439833_dt, 1.6067597_dt, 0.8696456_dt,  1.5494368_dt,  1.2326576_dt, 1.1686065_dt, 0.640427_dt,  1.2281567_dt, 0.31222272_dt, 0.957317_dt,  1.1957082_dt,
          0.8841866_dt,  1.4945016_dt, 1.3814917_dt, 0.81252885_dt, 0.78891516_dt, 1.3278598_dt, 1.4611369_dt, 1.1114358_dt, 0.6816741_dt, 0.48272014_dt, 1.1572896_dt, 0.531559_dt },
        { 2.8751132_dt, 4.4563737_dt, 4.323347_dt, 3.5126996_dt, 3.9233608_dt, 3.6477427_dt, 4.7230253_dt, 3.2775667_dt, 4.0725403_dt, 2.4545398_dt, 3.6378407_dt, 3.4821033_dt,
          4.8579206_dt, 3.920012_dt,  5.172658_dt, 3.1395707_dt, 2.5730727_dt, 5.196271_dt,  3.7009196_dt, 5.7858143_dt, 4.1964903_dt, 4.246865_dt,  1.5605282_dt, 3.057518_dt },
        { 3.01969481_dt, 4.01116037_dt, 2.63166189_dt, 1.18747461_dt, 3.21782255_dt, 2.73187304_dt, 3.62101221_dt, 3.91345072_dt, 2.62518144_dt, 2.09885907_dt, 1.66223443_dt, 2.72998357_dt,
          2.61920929_dt, 3.12346983_dt, 3.83695722_dt, 3.02822185_dt, 2.19693327_dt, 3.14736152_dt, 3.00416517_dt, 1.82628059_dt, 2.78075862_dt, 2.08497286_dt, 3.07653618_dt, 3.58166742_dt }
    };

    const raul::Tensor outNablas[] = {
        { 0.16513085_dt, 0.9014813_dt,  0.6309742_dt,  0.4345461_dt,   0.29193902_dt, 0.64250207_dt, 0.9757855_dt,  0.43509948_dt,  0.6601019_dt,   0.60489583_dt,  0.6366315_dt,  0.6144488_dt,
          0.8893349_dt,  0.6277617_dt,  0.53197503_dt, 0.025978208_dt, 0.44087505_dt, 0.25267076_dt, 0.8862232_dt,  0.88729346_dt,  0.78728163_dt,  0.059551954_dt, 0.0710938_dt,  0.3084147_dt,
          0.25118268_dt, 0.9084705_dt,  0.47147965_dt, 0.24238515_dt,  0.63300395_dt, 0.5860311_dt,  0.910012_dt,   0.5701437_dt,   0.49643457_dt,  0.5939151_dt,   0.5414331_dt,  0.44291723_dt,
          0.2924806_dt,  0.73394465_dt, 0.91970384_dt, 0.66851854_dt,  0.21609557_dt, 0.18653381_dt, 0.40716708_dt, 0.009662032_dt, 0.46557856_dt,  0.29618633_dt,  0.75012255_dt, 0.52189696_dt,
          0.6371355_dt,  0.5642073_dt,  0.57077086_dt, 0.025516272_dt, 0.611518_dt,   0.3106556_dt,  0.48879647_dt, 0.4864539_dt,   0.058522105_dt, 0.89776003_dt,  0.33596027_dt, 0.91876316_dt,
          0.64977_dt,    0.9290575_dt,  0.42163944_dt, 0.07328713_dt,  0.76459813_dt, 0.5204588_dt,  0.16163754_dt, 0.28740335_dt,  0.6819079_dt,   0.25954413_dt,  0.09378886_dt, 0.45675743_dt },
        { 0.16513085_dt, 0.9014813_dt,  0.6309742_dt,  0.4345461_dt,   0.29193902_dt, 0.64250207_dt, 0.9757855_dt,  0.43509948_dt,  0.6601019_dt,   0.60489583_dt,  0.6366315_dt,  0.6144488_dt,
          0.8893349_dt,  0.6277617_dt,  0.53197503_dt, 0.025978208_dt, 0.44087505_dt, 0.25267076_dt, 0.8862232_dt,  0.88729346_dt,  0.78728163_dt,  0.059551954_dt, 0.0710938_dt,  0.3084147_dt,
          0.25118268_dt, 0.9084705_dt,  0.47147965_dt, 0.24238515_dt,  0.63300395_dt, 0.5860311_dt,  0.910012_dt,   0.5701437_dt,   0.49643457_dt,  0.5939151_dt,   0.5414331_dt,  0.44291723_dt,
          0.2924806_dt,  0.73394465_dt, 0.91970384_dt, 0.66851854_dt,  0.21609557_dt, 0.18653381_dt, 0.40716708_dt, 0.009662032_dt, 0.46557856_dt,  0.29618633_dt,  0.75012255_dt, 0.52189696_dt,
          0.6371355_dt,  0.5642073_dt,  0.57077086_dt, 0.025516272_dt, 0.611518_dt,   0.3106556_dt,  0.48879647_dt, 0.4864539_dt,   0.058522105_dt, 0.89776003_dt,  0.33596027_dt, 0.91876316_dt,
          0.64977_dt,    0.9290575_dt,  0.42163944_dt, 0.07328713_dt,  0.76459813_dt, 0.5204588_dt,  0.16163754_dt, 0.28740335_dt,  0.6819079_dt,   0.25954413_dt,  0.09378886_dt, 0.45675743_dt },
        { 0.16513085_dt, 0.9014813_dt,  0.6309742_dt,  0.4345461_dt,   0.29193902_dt, 0.64250207_dt, 0.9757855_dt,  0.43509948_dt,  0.6601019_dt,  0.60489583_dt,  0.6366315_dt,  0.6144488_dt,
          0.8893349_dt,  0.6277617_dt,  0.53197503_dt, 0.025978208_dt, 0.44087505_dt, 0.25267076_dt, 0.8862232_dt,  0.88729346_dt,  0.78728163_dt, 0.059551954_dt, 0.0710938_dt,  0.3084147_dt,
          0.25118268_dt, 0.9084705_dt,  0.47147965_dt, 0.24238515_dt,  0.63300395_dt, 0.5860311_dt,  0.910012_dt,   0.5701437_dt,   0.49643457_dt, 0.5939151_dt,   0.5414331_dt,  0.44291723_dt,
          0.2924806_dt,  0.73394465_dt, 0.91970384_dt, 0.66851854_dt,  0.21609557_dt, 0.18653381_dt, 0.40716708_dt, 0.009662032_dt, 0.46557856_dt, 0.29618633_dt,  0.75012255_dt, 0.52189696_dt },
        { 0.16513085_dt,  0.9014813_dt,   0.6309742_dt,  0.4345461_dt,   0.29193902_dt, 0.64250207_dt, 0.9757855_dt,   0.43509948_dt,  0.6601019_dt,   0.60489583_dt,  0.6366315_dt,  0.6144488_dt,
          0.8893349_dt,   0.6277617_dt,   0.53197503_dt, 0.025978208_dt, 0.44087505_dt, 0.25267076_dt, 0.8862232_dt,   0.88729346_dt,  0.78728163_dt,  0.059551954_dt, 0.0710938_dt,  0.3084147_dt,
          0.25118268_dt,  0.9084705_dt,   0.47147965_dt, 0.24238515_dt,  0.63300395_dt, 0.5860311_dt,  0.910012_dt,    0.5701437_dt,   0.49643457_dt,  0.5939151_dt,   0.5414331_dt,  0.44291723_dt,
          0.2924806_dt,   0.73394465_dt,  0.91970384_dt, 0.66851854_dt,  0.21609557_dt, 0.18653381_dt, 0.40716708_dt,  0.009662032_dt, 0.46557856_dt,  0.29618633_dt,  0.75012255_dt, 0.52189696_dt,
          0.6371355_dt,   0.5642073_dt,   0.57077086_dt, 0.025516272_dt, 0.611518_dt,   0.3106556_dt,  0.48879647_dt,  0.4864539_dt,   0.058522105_dt, 0.89776003_dt,  0.33596027_dt, 0.91876316_dt,
          0.64977_dt,     0.9290575_dt,   0.42163944_dt, 0.07328713_dt,  0.76459813_dt, 0.5204588_dt,  0.16163754_dt,  0.28740335_dt,  0.6819079_dt,   0.25954413_dt,  0.09378886_dt, 0.45675743_dt,
          0.80607176_dt,  0.6874014_dt,   0.7146628_dt,  0.7172555_dt,   0.19159257_dt, 0.674997_dt,   0.84543407_dt,  0.2922609_dt,   0.092414856_dt, 0.71844184_dt,  0.29468465_dt, 0.8941331_dt,
          0.7456336_dt,   0.20546448_dt,  0.78517365_dt, 0.9046478_dt,   0.21869493_dt, 0.71179736_dt, 0.2794757_dt,   0.093688846_dt, 0.1307261_dt,   0.15925503_dt,  0.15632963_dt, 0.84454226_dt,
          0.5565096_dt,   0.5307274_dt,   0.16428733_dt, 0.16310573_dt,  0.4557836_dt,  0.34946883_dt, 0.6564821_dt,   0.9208337_dt,   0.10741806_dt,  0.05706513_dt,  0.2794541_dt,  0.0661242_dt,
          0.121813655_dt, 0.45792544_dt,  0.31201506_dt, 0.46504116_dt,  0.65310884_dt, 0.65611696_dt, 0.45208728_dt,  0.8963671_dt,   0.32994986_dt,  0.22004199_dt,  0.2453059_dt,  0.36547518_dt,
          0.04364562_dt,  0.27318442_dt,  0.19908202_dt, 0.33521748_dt,  0.4779179_dt,  0.26450837_dt, 0.31524432_dt,  0.41136813_dt,  0.87558246_dt,  0.2572304_dt,   0.7549573_dt,  0.83434105_dt,
          0.48019493_dt,  0.31500447_dt,  0.746855_dt,   0.1601975_dt,   0.45962846_dt, 0.7234938_dt,  0.24683177_dt,  0.449157_dt,    0.72002196_dt,  0.059944034_dt, 0.57900906_dt, 0.04383695_dt,
          0.85585284_dt,  0.027230382_dt, 0.3232242_dt,  0.83496463_dt,  0.71301806_dt, 0.33625674_dt, 0.82607245_dt,  0.03485012_dt,  0.052514672_dt, 0.64586926_dt,  0.49611855_dt, 0.96355164_dt,
          0.74480903_dt,  0.7778795_dt,   0.42672014_dt, 0.95287013_dt,  0.9313873_dt,  0.7965926_dt,  0.113613844_dt, 0.42684054_dt,  0.4257661_dt,   0.8725982_dt,   0.6143615_dt,  0.4515493_dt,
          0.43694246_dt,  0.40165102_dt,  0.74053156_dt, 0.55630386_dt,  0.60578966_dt, 0.91761243_dt, 0.6631657_dt,   0.19196546_dt,  0.010732412_dt, 0.44978714_dt,  0.24843419_dt, 0.76548266_dt,
          0.21867585_dt,  0.57201135_dt,  0.69218874_dt, 0.31307876_dt,  0.72334254_dt, 0.4814824_dt,  0.029916286_dt, 0.12069726_dt,  0.3200673_dt,   0.44513643_dt,  0.11140478_dt, 0.37603903_dt },
        { 0.16513085_dt,  0.9014813_dt,  0.6309742_dt,  0.4345461_dt,   0.29193902_dt, 0.64250207_dt, 0.9757855_dt,  0.43509948_dt,  0.6601019_dt,   0.60489583_dt,  0.6366315_dt,  0.6144488_dt,
          0.8893349_dt,   0.6277617_dt,  0.53197503_dt, 0.025978208_dt, 0.44087505_dt, 0.25267076_dt, 0.8862232_dt,  0.88729346_dt,  0.78728163_dt,  0.059551954_dt, 0.0710938_dt,  0.3084147_dt,
          0.25118268_dt,  0.9084705_dt,  0.47147965_dt, 0.24238515_dt,  0.63300395_dt, 0.5860311_dt,  0.910012_dt,   0.5701437_dt,   0.49643457_dt,  0.5939151_dt,   0.5414331_dt,  0.44291723_dt,
          0.2924806_dt,   0.73394465_dt, 0.91970384_dt, 0.66851854_dt,  0.21609557_dt, 0.18653381_dt, 0.40716708_dt, 0.009662032_dt, 0.46557856_dt,  0.29618633_dt,  0.75012255_dt, 0.52189696_dt,
          0.6371355_dt,   0.5642073_dt,  0.57077086_dt, 0.025516272_dt, 0.611518_dt,   0.3106556_dt,  0.48879647_dt, 0.4864539_dt,   0.058522105_dt, 0.89776003_dt,  0.33596027_dt, 0.91876316_dt,
          0.64977_dt,     0.9290575_dt,  0.42163944_dt, 0.07328713_dt,  0.76459813_dt, 0.5204588_dt,  0.16163754_dt, 0.28740335_dt,  0.6819079_dt,   0.25954413_dt,  0.09378886_dt, 0.45675743_dt,
          0.80607176_dt,  0.6874014_dt,  0.7146628_dt,  0.7172555_dt,   0.19159257_dt, 0.674997_dt,   0.84543407_dt, 0.2922609_dt,   0.092414856_dt, 0.71844184_dt,  0.29468465_dt, 0.8941331_dt,
          0.7456336_dt,   0.20546448_dt, 0.78517365_dt, 0.9046478_dt,   0.21869493_dt, 0.71179736_dt, 0.2794757_dt,  0.093688846_dt, 0.1307261_dt,   0.15925503_dt,  0.15632963_dt, 0.84454226_dt,
          0.5565096_dt,   0.5307274_dt,  0.16428733_dt, 0.16310573_dt,  0.4557836_dt,  0.34946883_dt, 0.6564821_dt,  0.9208337_dt,   0.10741806_dt,  0.05706513_dt,  0.2794541_dt,  0.0661242_dt,
          0.121813655_dt, 0.45792544_dt, 0.31201506_dt, 0.46504116_dt,  0.65310884_dt, 0.65611696_dt, 0.45208728_dt, 0.8963671_dt,   0.32994986_dt,  0.22004199_dt,  0.2453059_dt,  0.36547518_dt,
          0.04364562_dt,  0.27318442_dt, 0.19908202_dt, 0.33521748_dt,  0.4779179_dt,  0.26450837_dt, 0.31524432_dt, 0.41136813_dt,  0.87558246_dt,  0.2572304_dt,   0.7549573_dt,  0.83434105_dt,
          0.48019493_dt,  0.31500447_dt, 0.746855_dt,   0.1601975_dt,   0.45962846_dt, 0.7234938_dt,  0.24683177_dt, 0.449157_dt,    0.72002196_dt,  0.059944034_dt, 0.57900906_dt, 0.04383695_dt }
    };

    yato::dimensionality<4> expectedShapes[] = { yato::dims(1, 6, 3, 4), yato::dims(1, 2, 9, 4), yato::dims(1, 2, 3, 8), yato::dims(1, 4, 6, 8), yato::dims(1, 6, 6, 4) };

    for (size_t iter = 0; iter < std::size(expectedShapes); ++iter)
    {
        MANAGERS_DEFINE
        NETWORK_PARAMS_DEFINE(networkParameters);
        work.tensorNeeded("x", "x", raul::WShape{ raul::BS(), depth, height, width }, DEC_FORW_READ_NOMEMOPT);
        std::shared_ptr<raul::TileLayer> tile;
        if (iter < 4)
        {
            raul::TilingParams params = raul::TilingParams{ { "x" }, { "out" }, repeats[iter], dimensions[iter] };
            tile = std::make_shared<raul::TileLayer>("tile", params, networkParameters);
        }
        else
        {
            raul::TilingParams params = raul::TilingParams{ { "x" }, { "out" }, { 3, 2, 1 } };
            tile = std::make_shared<raul::TileLayer>("tile", params, networkParameters);
        }

        // Apply function
        TENSORS_CREATE(batch);
        memory_manager["x"] = TORANGE(x);
        memory_manager[raul::Name("out").grad()] = TORANGE(outNablas[iter]);
        tile->forwardCompute(raul::NetworkMode::Test);
        tile->backwardCompute();

        // Checks
        const auto& xTensorGrad = memory_manager[raul::Name("x").grad()];
        EXPECT_EQ(xTensorGrad.getShape(), memory_manager["x"].getShape());
        for (size_t i = 0; i < xTensorGrad.size(); ++i)
        {
            ASSERT_TRUE(tools::expect_near_relative(xTensorGrad[i], realGrads[iter][i], eps));
        }
    }
}

TEST(TestLayerTile, BigBackwardRandUnit)
{
    PROFILE_TEST
    // Test parameters
    const size_t tensor_size = 100;
    const size_t depth = 20;
    const size_t height = 30;
    const size_t width = 40;
    const auto random_range1 = std::make_pair(1.0_dt, 100.0_dt);
    const auto random_range2 = std::make_pair(1.0_dt, 1.0_dt);
    const auto eps = TODTYPE(1e-6);

    // Initialization
    MANAGERS_DEFINE
    NETWORK_PARAMS_DEFINE(networkParameters);

    work.tensorNeeded("x", "x", raul::WShape{ raul::BS(), depth, height, width }, DEC_FORW_READ_NOMEMOPT);

    // Apply function
    raul::TileLayer tile("tile", raul::TilingParams{ { "x" }, { "out" }, { 3, 2, 1 } }, networkParameters);
    TENSORS_CREATE(tensor_size);
    tools::init_rand_tensor("x", random_range1, memory_manager);
    tools::init_rand_tensor(raul::Name("out").grad(), random_range2, memory_manager);

    tile.forwardCompute(raul::NetworkMode::Test);
    tile.backwardCompute();

    // Checks
    const auto& xTensorGrad = memory_manager[raul::Name("x").grad()];
    EXPECT_EQ(xTensorGrad.getShape(), memory_manager["x"].getShape());
    for (auto& i : xTensorGrad)
    {
        ASSERT_TRUE(tools::expect_near_relative(i, 6.0_dt, eps));
    }
}

TEST(TestLayerTile, SimpleWidthUnit) 
{
    PROFILE_TEST

    raul::WorkflowEager work;
    auto& mm = work.getMemoryManager();
    size_t width = 2;
    size_t depth = 1;
    size_t height = 2;
    size_t batch = 1;

    const raul::Tensor in{1_dt, 0_dt};
    const raul::Tensor golden(batch, depth, height, width, {1_dt, 1_dt, 0_dt, 0_dt});

    work.add<raul::DataLayer>("in", raul::DataParams{ {"in"}, depth, height, 1 });
    work.add<raul::TileLayer>("tile", raul::TilingParams{ "in", "tile", width, raul::Dimension::Width });
    TENSORS_CREATE(batch);
    mm["in"] = TORANGE(in);

    ASSERT_NO_THROW(work.forwardPassTraining());
    const auto& outTensor = mm["tile"];
    EXPECT_EQ(outTensor.getShape(), golden.getShape());
    for (size_t i = 0; i < outTensor.size(); ++i)
    {
        EXPECT_EQ(outTensor[i], golden[i]);
    }
}

}