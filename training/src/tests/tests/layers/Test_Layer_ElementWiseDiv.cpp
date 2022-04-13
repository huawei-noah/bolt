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

#include <training/base/common/Common.h>
#include <training/base/common/Conversions.h>
#include <training/base/common/MemoryManager.h>
#include <training/base/layers/basic/DataLayer.h>
#include <training/base/layers/basic/TensorLayer.h>
#include <training/base/layers/basic/ElementWiseDivLayer.h>
#include <training/compiler/Workflow.h>
#include <training/system/NameGenerator.h>

namespace UT
{

namespace
{
raul::dtype golden_div_layer(const raul::dtype x, const raul::dtype y)
{
    return x / y;
}

std::pair<raul::dtype, raul::dtype> golden_div_layer_grad(const raul::dtype x, const raul::dtype y, const raul::dtype grad)
{
    const auto x_grad = grad / y;
    const auto y_grad = grad * x / y / y * (-1);
    return std::make_pair(x_grad, y_grad);
}
}

TEST(TestLayerElementWiseDiv, OutputNumExceedsUnit)
{
    PROFILE_TEST

    // Initialization
    MANAGERS_DEFINE
    NETWORK_PARAMS_DEFINE(networkParameters);

    // Apply function
    ASSERT_THROW(raul::ElementWiseDivLayer("div", raul::ElementWiseLayerParams{ { { "x", "y" }, { "x_out", "y_out" } } }, networkParameters), raul::Exception);
}

TEST(TestLayerElementWiseDiv, InputNumExceedsUnit)
{
    PROFILE_TEST

    // Initialization
    MANAGERS_DEFINE
    NETWORK_PARAMS_DEFINE(networkParameters);

    // Apply function
    ASSERT_THROW(raul::ElementWiseDivLayer("div", raul::ElementWiseLayerParams{ { "x", "y", "z" }, { "out" } }, networkParameters), raul::Exception);
}

TEST(TestLayerElementWiseDiv, ForwardZeroDivisionUnit)
{
    PROFILE_TEST
    // Test parameters
    const auto eps_rel = TODTYPE(1e-6);

    const raul::Tensor x{ 3.0_dt, 2.0_dt, 1.0_dt };
    const raul::Tensor y{ 2.0_dt, 1.0_dt, 0.0_dt };
    const raul::Tensor z{ 1.5_dt, 2.0_dt, std::numeric_limits<raul::dtype>::infinity() };

    // Initialization
    MANAGERS_DEFINE
    NETWORK_PARAMS_DEFINE(networkParameters);

    work.tensorNeeded("x", "x", raul::WShape{ raul::BS(), 1u, 1u, 3u }, DEC_FORW_READ_NOMEMOPT);
    work.tensorNeeded("y", "y", raul::WShape{ raul::BS(), 1u, 1u, 3u }, DEC_FORW_READ_NOMEMOPT);

    // Apply function
    raul::ElementWiseDivLayer elementwise_div("div", raul::ElementWiseLayerParams{ { "x", "y" }, { "out" } }, networkParameters);
    TENSORS_CREATE(1);
    memory_manager["x"] = TORANGE(x);
    memory_manager["y"] = TORANGE(y);

    elementwise_div.forwardCompute(raul::NetworkMode::Test);

    // Checks
    const auto& out_tensor = memory_manager["out"];
    for (size_t i = 0; i < out_tensor.size(); ++i)
    {
        ASSERT_TRUE(tools::expect_near_relative(out_tensor[i], z[i], eps_rel));
    }
}

TEST(TestLayerElementWiseDiv, BackwardZeroDivisionUnit)
{
    PROFILE_TEST
    // Test parameters
    const auto eps_rel = TODTYPE(1e-6);

    const raul::Tensor x{ 3.0_dt, 2.0_dt, 1.0_dt };
    const raul::Tensor y{ 2.0_dt, 1.0_dt, 0.0_dt };

    const raul::Tensor deltas{ 1.0_dt, 1.0_dt, 1.0_dt };
    const raul::Tensor x_grad{ 0.5_dt, 1.0_dt, std::numeric_limits<raul::dtype>::infinity() };
    const raul::Tensor y_grad{ -0.75_dt, -2.0_dt, -std::numeric_limits<raul::dtype>::infinity() };

    // Initialization
    MANAGERS_DEFINE
    NETWORK_PARAMS_DEFINE(networkParameters);

    work.tensorNeeded("x", "x", raul::WShape{ raul::BS(), 1u, 1u, 3u }, DEC_FORW_READ_NOMEMOPT);
    work.tensorNeeded("y", "y", raul::WShape{ raul::BS(), 1u, 1u, 3u }, DEC_FORW_READ_NOMEMOPT);

    // Apply function
    raul::ElementWiseDivLayer elementwise_div("div", raul::ElementWiseLayerParams{ { "x", "y" }, { "out" } }, networkParameters);
    TENSORS_CREATE(1);
    memory_manager["x"] = TORANGE(x);
    memory_manager["y"] = TORANGE(y);
    memory_manager[raul::Name("out").grad()] = TORANGE(deltas);

    elementwise_div.forwardCompute(raul::NetworkMode::Test);
    elementwise_div.backwardCompute();

    // Checks
    const auto& x_tensor_grad = memory_manager[raul::Name("x").grad()];
    const auto& y_tensor_grad = memory_manager[raul::Name("y").grad()];
    EXPECT_EQ(x_tensor_grad.getShape(), memory_manager["x"].getShape());
    EXPECT_EQ(y_tensor_grad.getShape(), memory_manager["y"].getShape());

    for (size_t i = 0; i < x_tensor_grad.size(); ++i)
    {
        ASSERT_TRUE(tools::expect_near_relative(x_tensor_grad[i], x_grad[i], eps_rel)) << "expected: " << x_grad[i] << ", got: " << x_tensor_grad[i];
    }

    for (size_t i = 0; i < y_tensor_grad.size(); ++i)
    {
        ASSERT_TRUE(tools::expect_near_relative(y_tensor_grad[i], y_grad[i], eps_rel)) << "expected: " << y_grad[i] << ", got: " << y_tensor_grad[i];
    }
}

TEST(TestLayerElementWiseDiv, ForwardRandUnit)
{
    PROFILE_TEST
    // Test parameters
    const auto eps_rel = TODTYPE(1e-6);
    const auto tensor_size = 1000U;
    const auto random_range = std::make_pair(-100.0_dt, 100.0_dt);

    // Initialization
    MANAGERS_DEFINE
    NETWORK_PARAMS_DEFINE(networkParameters);

    work.tensorNeeded("x", "x", raul::WShape{ raul::BS(), 1u, 1u, 1u }, DEC_FORW_READ_NOMEMOPT);
    work.tensorNeeded("y", "y", raul::WShape{ raul::BS(), 1u, 1u, 1u }, DEC_FORW_READ_NOMEMOPT);

    // Apply function
    raul::ElementWiseDivLayer elementwise_div("div", raul::ElementWiseLayerParams{ { "x", "y" }, { "out" } }, networkParameters);
    TENSORS_CREATE(tensor_size);
    tools::init_rand_tensor("x", random_range, memory_manager);
    tools::init_rand_tensor("y", random_range, memory_manager);

    elementwise_div.forwardCompute(raul::NetworkMode::Test);

    // Checks
    const auto& x_tensor = memory_manager["x"];
    const auto& y_tensor = memory_manager["y"];
    const auto& out_tensor = memory_manager["out"];

    EXPECT_EQ(out_tensor.size(), x_tensor.size());
    EXPECT_EQ(out_tensor.size(), y_tensor.size());

    for (size_t i = 0; i < out_tensor.size(); ++i)
    {
        const auto x_value = x_tensor[i];
        const auto y_value = y_tensor[i];
        const auto out_value = out_tensor[i];
        const auto golden_out_value = golden_div_layer(x_value, y_value);
        ASSERT_TRUE(tools::expect_near_relative(out_value, golden_out_value, eps_rel));
    }
}

TEST(TestLayerElementWiseDiv, BackwardRandUnit)
{
    PROFILE_TEST
    // Test parameters
    const auto eps_rel = TODTYPE(1e-6);
    const auto tensor_size = 1000U;
    const auto random_range = std::make_pair(-100.0_dt, 100.0_dt);

    // Initialization
    MANAGERS_DEFINE
    NETWORK_PARAMS_DEFINE(networkParameters);

    work.tensorNeeded("x", "x", raul::WShape{ raul::BS(), 1u, 1u, 1u }, DEC_FORW_READ_NOMEMOPT);
    work.tensorNeeded("y", "y", raul::WShape{ raul::BS(), 1u, 1u, 1u }, DEC_FORW_READ_NOMEMOPT);

    // Apply function
    raul::ElementWiseDivLayer elementwise_div("div", raul::ElementWiseLayerParams{ { "x", "y" }, { "out" } }, networkParameters);
    TENSORS_CREATE(tensor_size);
    tools::init_rand_tensor("x", random_range, memory_manager);
    tools::init_rand_tensor("y", random_range, memory_manager);
    tools::init_rand_tensor(raul::Name("out").grad(), random_range, memory_manager);

    elementwise_div.forwardCompute(raul::NetworkMode::Train);
    elementwise_div.backwardCompute();

    // Checks
    const auto& x_tensor = memory_manager["x"];
    const auto& y_tensor = memory_manager["y"];
    const auto& x_tensor_grad = memory_manager[raul::Name("x").grad()];
    const auto& y_tensor_grad = memory_manager[raul::Name("y").grad()];
    const auto& out_tensor_grad = memory_manager[raul::Name("out").grad()];

    EXPECT_EQ(out_tensor_grad.size(), x_tensor_grad.size());
    EXPECT_EQ(out_tensor_grad.size(), y_tensor_grad.size());

    for (size_t i = 0; i < out_tensor_grad.size(); ++i)
    {
        const auto x_value = x_tensor[i];
        const auto y_value = y_tensor[i];
        const auto out_grad_value = out_tensor_grad[i];
        const auto x_grad_value = x_tensor_grad[i];
        const auto y_grad_value = y_tensor_grad[i];
        const auto [golden_out_value_x, golden_out_value_y] = golden_div_layer_grad(x_value, y_value, out_grad_value);
        ASSERT_TRUE(tools::expect_near_relative(x_grad_value, golden_out_value_x, eps_rel));
        ASSERT_TRUE(tools::expect_near_relative(y_grad_value, golden_out_value_y, eps_rel));
    }
}

TEST(TestLayerElementWiseDiv, NoBroadcastForwarFailUnit)
{
    PROFILE_TEST
    // Test parameters
    const auto random_range = std::make_pair(-100.0_dt, 100.0_dt);
    const auto enable_broadcast = false;

    // Initialization
    MANAGERS_DEFINE
    NETWORK_PARAMS_DEFINE(networkParameters);

    work.tensorNeeded("x", "x", raul::WShape{ raul::BS(), 1u, 1u, 1u }, DEC_FORW_READ_NOMEMOPT);
    work.tensorNeeded("y", "y", raul::WShape{ raul::BS(), 3u, 1u, 3u }, DEC_FORW_READ_NOMEMOPT);

    // Apply function
    raul::ElementWiseDivLayer elementwise_div("div", raul::ElementWiseLayerParams{ { "x", "y" }, { "out" }, enable_broadcast }, networkParameters);
    TENSORS_CREATE(1);
    tools::init_rand_tensor("x", random_range, memory_manager);
    tools::init_rand_tensor("y", random_range, memory_manager);

    ASSERT_THROW(elementwise_div.forwardCompute(raul::NetworkMode::Test), raul::Exception);
}

TEST(TestLayerElementWiseDiv, BroadcastForwardRandUnit)
{
    PROFILE_TEST
    // Test parameters
    const auto eps_rel = TODTYPE(1e-6);
    const auto random_range = std::make_pair(-100.0_dt, 100.0_dt);
    const auto enable_broadcast = true;

    // Initialization
    MANAGERS_DEFINE
    NETWORK_PARAMS_DEFINE(networkParameters);

    work.tensorNeeded("x", "x", raul::WShape{ raul::BS(), 1u, 1u, 1u }, DEC_FORW_READ_NOMEMOPT);
    work.tensorNeeded("y", "y", raul::WShape{ raul::BS(), 3u, 1u, 3u }, DEC_FORW_READ_NOMEMOPT);

    // Apply function
    raul::ElementWiseDivLayer elementwise_div("div", raul::ElementWiseLayerParams{ { "x", "y" }, { "out" }, enable_broadcast }, networkParameters);
    TENSORS_CREATE(1);
    tools::init_rand_tensor("x", random_range, memory_manager);
    tools::init_rand_tensor("y", random_range, memory_manager);

    elementwise_div.forwardCompute(raul::NetworkMode::Test);

    // Checks
    const auto& x_tensor = memory_manager["x"];
    const auto& y_tensor = memory_manager["y"];
    const auto& out_tensor = memory_manager["out"];

    EXPECT_EQ(out_tensor.size(), y_tensor.size());

    for (size_t i = 0; i < out_tensor.size(); ++i)
    {
        // We have x tensor with shape [1,1] so there is only 1 value in x
        const auto x_value = x_tensor[0];
        const auto y_value = y_tensor[i];
        const auto out_value = out_tensor[i];
        const auto golden_out_value = golden_div_layer(x_value, y_value);
        ASSERT_TRUE(tools::expect_near_relative(out_value, golden_out_value, eps_rel));
    }
}

TEST(TestLayerElementWiseDiv, BroadcastBackwardRandUnit)
{
    PROFILE_TEST
    // Test parameters
    const auto eps_rel = TODTYPE(1e-6);
    const auto random_range = std::make_pair(1.0_dt, 100.0_dt);
    const auto enable_broadcast = true;

    // Initialization
    MANAGERS_DEFINE
    NETWORK_PARAMS_DEFINE(networkParameters);

    work.tensorNeeded("x", "x", raul::WShape{ raul::BS(), 2u, 3u, 4u }, DEC_FORW_READ_NOMEMOPT);
    work.tensorNeeded("y", "y", raul::WShape{ raul::BS(), 1u, 1u, 1u }, DEC_FORW_READ_NOMEMOPT);

    // Apply function
    raul::ElementWiseDivLayer elementwise_div("div", raul::ElementWiseLayerParams{ { "x", "y" }, { "out" }, enable_broadcast }, networkParameters);
    TENSORS_CREATE(1);
    tools::init_rand_tensor("x", random_range, memory_manager);
    tools::init_rand_tensor("y", random_range, memory_manager);
    tools::init_rand_tensor(raul::Name("out").grad(), random_range, memory_manager);

    elementwise_div.forwardCompute(raul::NetworkMode::Train);
    elementwise_div.backwardCompute();

    // Checks
    const auto& x_tensor = memory_manager["x"];
    const auto& y_tensor = memory_manager["y"];

    const auto& out_tensor_grad = memory_manager[raul::Name("out").grad()];
    const auto& x_tensor_grad = memory_manager[raul::Name("x").grad()];
    const auto& y_tensor_grad = memory_manager[raul::Name("y").grad()];

    EXPECT_EQ(out_tensor_grad.size(), x_tensor.size());

    for (size_t i = 0; i < out_tensor_grad.size(); ++i)
    {
        const auto x_value = x_tensor[i];
        // We have y tensor with shape [1,1] so there is only 1 value in y
        const auto y_value = y_tensor[0];
        const auto out_grad_value = out_tensor_grad[i];
        const auto x_grad_value = x_tensor_grad[i];
        const auto [golden_out_value_x, golden_out_value_y] = golden_div_layer_grad(x_value, y_value, out_grad_value);
        ASSERT_TRUE(tools::expect_near_relative(x_grad_value, golden_out_value_x, eps_rel));
    }

    auto golden_out_value = 0.0_dt;
    for (size_t i = 0; i < out_tensor_grad.size(); ++i)
    {
        const auto x_value = x_tensor[i];
        // We have y tensor with shape [1,1] so there is only 1 value in y
        const auto y_value = y_tensor[0];
        const auto out_grad_value = out_tensor_grad[i];
        const auto [golden_out_value_x, golden_out_value_y] = golden_div_layer_grad(x_value, y_value, out_grad_value);
        golden_out_value += golden_out_value_y;
    }
    // We have y tensor with shape [1,1] so there is only 1 value in grad of y
    ASSERT_TRUE(tools::expect_near_relative(y_tensor_grad[0], golden_out_value, eps_rel));
}

TEST(TestLayerElementWiseDiv, BroadcastForwardFunc)
{
    PROFILE_TEST
    // Test parameters
    const auto eps_rel = TODTYPE(1e-6);
    const auto enable_broadcast = true;
    const auto shape = yato::dims(5, 4, 3, 2);

    // See broadcasting.ipynb (seed 0)
    const raul::Tensor x{ 0.1033359326064693_dt,  0.48656171252681413_dt, 0.26826552260919523_dt, 0.77789329039684629_dt, 0.46992440350274567_dt, 0.99579112335869169_dt,
                          0.7922695881242352_dt,  0.93204329767498884_dt, 0.21747891739260439_dt, 0.4719288679087088_dt,  0.99415248552170699_dt, 0.0057258796833245_dt,
                          0.26544109756295542_dt, 0.42194057001397134_dt, 0.04668182220684081_dt, 0.95499018226495858_dt, 0.32216882381204004_dt, 0.33478528995344214_dt,
                          0.12209337140187504_dt, 0.55431150968945719_dt, 0.36458131861678045_dt, 0.86235267336661425_dt, 0.95191105862818748_dt, 0.43618645497613173_dt,
                          0.14027554365465433_dt, 0.81191335595594505_dt, 0.19024521112905723_dt, 0.78636232005979567_dt, 0.57441018559926216_dt, 0.96256729778404049_dt };
    const raul::Tensor y{ 0.35537000495037541_dt, 0.73477384868738793_dt, 0.72838401246540652_dt, 0.49430872789854596_dt, 0.7289773666006969_dt,  0.44661990703196996_dt, 0.39500410903829208_dt,
                          0.89862335756822109_dt, 0.45748917853917637_dt, 0.09670656422271051_dt, 0.65839241759395906_dt, 0.24527510459208224_dt, 0.13743707385911919_dt, 0.22845330022296384_dt,
                          0.48653049268892479_dt, 0.28265795763984414_dt, 0.68238250997643435_dt, 0.43896126229277976_dt, 0.59572414410976138_dt, 0.27390336240389146_dt };
    const raul::Tensor z{ 0.290784059338095757_dt, 1.36916933266429908_dt,  0.754890730428012247_dt,  2.18896721603015676_dt,  1.32235246913528015_dt,  2.8021248543408892_dt,
                          0.140636377833901344_dt, 0.662192473774095181_dt, 0.365099442622283277_dt,  1.05868396348956573_dt,  0.639549712258031922_dt, 1.3552348455753962_dt,
                          0.141870127347663494_dt, 0.668001636773875029_dt, 0.3683023213279768_dt,    1.06797139569807786_dt,  0.645160238913212059_dt, 1.36712380601020578_dt,
                          0.209051402037308237_dt, 0.984327577211378202_dt, 0.542708448118389697_dt,  1.57369928243812929_dt,  0.950669848579317356_dt, 2.01451252457567787_dt,
                          1.08682330127569937_dt,  1.27856273785455254_dt,  0.298334252003917388_dt,  0.647384801683714817_dt, 1.36376317162980154_dt,  0.00785467415816339457_dt,
                          1.77392358838031572_dt,  2.08688256613754408_dt,  0.486944074745501221_dt,  1.05666778501865388_dt,  2.22594754481139478_dt,  0.0128204757404927483_dt,
                          2.00572492790810886_dt,  2.35957873943188678_dt,  0.550573810287937504_dt,  1.1947441991368235_dt,   2.51681555400056078_dt,  0.0144957471385934993_dt,
                          0.881648113697165114_dt, 1.03719015294372707_dt,  0.242013425937566917_dt,  0.525168708262606199_dt, 1.10630608157348465_dt,  0.00637183491292658052_dt,
                          0.580212844401137628_dt, 0.922296285479983435_dt, 0.102039183431402808_dt,  2.08745961011442693_dt,  0.704210807435419195_dt, 0.731788434914364516_dt,
                          2.7448095141882769_dt,   4.363101650910302_dt,    0.482716169083774294_dt,  9.87513298544721962_dt,  3.33140595368584203_dt,  3.46186727492921698_dt,
                          0.4031654837900292_dt,   0.640864868334781956_dt, 0.0709027336272123054_dt, 1.45048781964241269_dt,  0.489326449094568128_dt, 0.50848898165760692_dt,
                          1.0822178549445991_dt,   1.72027475318256196_dt,  0.190324339212809557_dt,  3.89354714108961675_dt,  1.31349989371257214_dt,  1.36493791536741793_dt,
                          0.888358344466993577_dt, 4.03320220756197134_dt,  2.65271449965892758_dt,   6.27452731022616739_dt,  6.92615923709166292_dt,  3.17371756199675525_dt,
                          0.53443470189625375_dt,  2.42636682923146729_dt,  1.59586803193895466_dt,   3.77474377706508468_dt,  4.16676431331545505_dt,  1.90930249005125474_dt,
                          0.250947007919477771_dt, 1.13931504400870076_dt,  0.749349370893150679_dt,  1.77245349741723301_dt,  1.95652908282732274_dt,  0.896524393703352573_dt,
                          0.431947405342266832_dt, 1.96106812034546496_dt,  1.28983214080008701_dt,   3.05086996512372366_dt,  3.36771363727565509_dt,  1.5431600037664952_dt,
                          0.205567319800589066_dt, 1.18982146242872489_dt,  0.278795555788244387_dt,  1.15237760136459566_dt,  0.841771553639466874_dt, 1.41059784462717563_dt,
                          0.319562466450838911_dt, 1.8496241598066403_dt,   0.433398633253808341_dt,  1.79141620823776759_dt,  1.30856691681404058_dt,  2.19282971065912502_dt,
                          0.235470636940994543_dt, 1.36290154425292376_dt,  0.319351184621459283_dt,  1.32001082688182847_dt,  0.964221764853344321_dt, 1.61579366440230898_dt,
                          0.512135164838930734_dt, 2.96423289159450576_dt,  0.694570557511178355_dt,  2.87094803495053208_dt,  2.09712717857128883_dt,  3.51425878578540907_dt };

    // Initialization
    MANAGERS_DEFINE
    NETWORK_PARAMS_DEFINE(networkParameters);

    work.tensorNeeded("x", "x", raul::WShape{ raul::BS(), 1u, 3u, 2u }, DEC_FORW_READ_NOMEMOPT);
    work.tensorNeeded("y", "y", raul::WShape{ raul::BS(), 4u, 1u, 1u }, DEC_FORW_READ_NOMEMOPT);

    // Apply function
    raul::ElementWiseDivLayer elementwise_div("div", raul::ElementWiseLayerParams{ { "x", "y" }, { "out" }, enable_broadcast }, networkParameters);
    TENSORS_CREATE(5);
    memory_manager["x"] = TORANGE(x);
    memory_manager["y"] = TORANGE(y);

    elementwise_div.forwardCompute(raul::NetworkMode::Test);

    // Checks
    const auto& out_tensor = memory_manager["out"];
    EXPECT_EQ(out_tensor.getShape(), shape);
    for (size_t i = 0; i < out_tensor.size(); ++i)
    {
        ASSERT_TRUE(tools::expect_near_relative(out_tensor[i], z[i], eps_rel));
    }
}

TEST(TestLayerElementWiseDiv, BroadcastBackwardFunc)
{
    PROFILE_TEST
    // Test parameters
    const auto eps_rel = TODTYPE(1e-6);
    const auto enable_broadcast = true;
    const auto shape = yato::dims(1, 2, 2, 2);

    // See broadcasting.ipynb (seed 0)
    const raul::Tensor x{ 0.40343133345893134_dt, 0.54692206441810598_dt };
    const raul::Tensor y{ 0.92483389330772503_dt, 0.47847837523050574_dt, 0.91370178736457419_dt, 0.90764699003836147_dt };
    const raul::Tensor z{ 0.43622031629489105_dt, 0.84315478889632023_dt, 0.59137329241038705_dt, 1.14304447751610017_dt,
                          0.44153501617038982_dt, 0.4444804399581388_dt,  0.59857830200334261_dt, 0.60257134152451763_dt };

    const raul::Tensor deltas{ 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt };
    const raul::Tensor x_grad{ 5.36743277413323039_dt, 5.36743277413323039_dt };
    const raul::Tensor y_grad{ -1.11111153704588683_dt, -4.15107425796531349_dt, -1.13835097244777406_dt, -1.15358921802671643_dt };

    // Initialization
    MANAGERS_DEFINE
    NETWORK_PARAMS_DEFINE(networkParameters);

    work.tensorNeeded("x", "x", raul::WShape{ raul::BS(), 1u, 2u, 1u }, DEC_FORW_READ_NOMEMOPT);
    work.tensorNeeded("y", "y", raul::WShape{ raul::BS(), 2u, 1u, 2u }, DEC_FORW_READ_NOMEMOPT);

    // Apply function
    raul::ElementWiseDivLayer elementwise_div("div", raul::ElementWiseLayerParams{ { "x", "y" }, { "out" }, enable_broadcast }, networkParameters);
    TENSORS_CREATE(1);
    memory_manager["x"] = TORANGE(x);
    memory_manager["y"] = TORANGE(y);
    memory_manager[raul::Name("out").grad()] = TORANGE(deltas);

    elementwise_div.forwardCompute(raul::NetworkMode::Train);
    elementwise_div.backwardCompute();

    // Checks
    const auto& x_tensor_grad = memory_manager[raul::Name("x").grad()];
    const auto& y_tensor_grad = memory_manager[raul::Name("y").grad()];
    const auto& out_tensor = memory_manager["out"];

    EXPECT_EQ(out_tensor.getShape(), shape);
    EXPECT_EQ(x_tensor_grad.getShape(), memory_manager["x"].getShape());
    EXPECT_EQ(y_tensor_grad.getShape(), memory_manager["y"].getShape());

    for (size_t i = 0; i < out_tensor.size(); ++i)
    {
        ASSERT_TRUE(tools::expect_near_relative(out_tensor[i], z[i], eps_rel)) << "expected: " << z[i] << ", got: " << out_tensor[i];
    }

    for (size_t i = 0; i < x_tensor_grad.size(); ++i)
    {
        ASSERT_TRUE(tools::expect_near_relative(x_tensor_grad[i], x_grad[i], eps_rel)) << "expected: " << x_grad[i] << ", got: " << x_tensor_grad[i];
    }

    for (size_t i = 0; i < y_tensor_grad.size(); ++i)
    {
        ASSERT_TRUE(tools::expect_near_relative(y_tensor_grad[i], y_grad[i], eps_rel)) << "expected: " << y_grad[i] << ", got: " << y_tensor_grad[i];
    }
}

}