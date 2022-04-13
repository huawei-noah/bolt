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
#include <training/base/layers/basic/TensorLayer.h>
#include <training/base/layers/basic/ElementWiseMulLayer.h>

namespace
{

template<typename T>
T golden_mul_layer(const T x, const T y)
{
    return x * y;
}

template<typename T>
std::pair<T, T> golden_mul_layer_grad(const T x, const T y, const T grad)
{
    T x_grad = grad * y;
    T y_grad = grad * x;
    return std::make_pair(x_grad, y_grad);
}

}

namespace UT
{

TEST(TestLayerElementWiseMul, ForwardRandUnit)
{
    PROFILE_TEST
    // Test parameters
    const auto eps_rel = 1e-6_dt;
    const auto tensor_size = 1000U;
    const auto random_range = std::make_pair(-100.0_dt, 100.0_dt);

    // Initialization
    MANAGERS_DEFINE
    NETWORK_PARAMS_DEFINE(networkParameters);

    work.tensorNeeded("x", "x", raul::WShape{ raul::BS(), 1u, 1u, 1u }, DEC_FORW_READ_NOMEMOPT);
    work.tensorNeeded("y", "y", raul::WShape{ raul::BS(), 1u, 1u, 1u }, DEC_FORW_READ_NOMEMOPT);

    // Apply function
    raul::ElementWiseMulLayer elementwise_mul("mul", raul::ElementWiseLayerParams{ { "x", "y" }, { "out" } }, networkParameters);
    TENSORS_CREATE(tensor_size);
    tools::init_rand_tensor("x", random_range, memory_manager);
    tools::init_rand_tensor("y", random_range, memory_manager);

    elementwise_mul.forwardCompute(raul::NetworkMode::Test);

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
        const auto golden_out_value = golden_mul_layer(x_value, y_value);
        ASSERT_TRUE(tools::expect_near_relative(out_value, golden_out_value, eps_rel));
    }
}

TEST(TestLayerElementWiseMul, BackwardRandUnit)
{
    PROFILE_TEST
    // Test parameters
    const auto eps_rel = 1e-6_dt;
    const auto tensor_size = 1000U;
    const auto random_range = std::make_pair(-100.0_dt, 100.0_dt);

    // Initialization
    MANAGERS_DEFINE
    NETWORK_PARAMS_DEFINE(networkParameters);

    work.tensorNeeded("x", "x", raul::WShape{ raul::BS(), 1u, 1u, 1u }, DEC_FORW_READ_NOMEMOPT);
    work.tensorNeeded("y", "y", raul::WShape{ raul::BS(), 1u, 1u, 1u }, DEC_FORW_READ_NOMEMOPT);

    // Apply function
    raul::ElementWiseMulLayer elementwise_mul("mul", raul::ElementWiseLayerParams{ { "x", "y" }, { "out" } }, networkParameters);
    TENSORS_CREATE(tensor_size);
    tools::init_rand_tensor("x", random_range, memory_manager);
    tools::init_rand_tensor("y", random_range, memory_manager);
    tools::init_rand_tensor(raul::Name("out").grad(), random_range, memory_manager);

    elementwise_mul.forwardCompute(raul::NetworkMode::Train);
    elementwise_mul.backwardCompute();

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
        const auto [golden_out_value_x, golden_out_value_y] = golden_mul_layer_grad(x_value, y_value, out_grad_value);
        ASSERT_TRUE(tools::expect_near_relative(x_grad_value, golden_out_value_x, eps_rel));
        ASSERT_TRUE(tools::expect_near_relative(y_grad_value, golden_out_value_y, eps_rel));
    }
}

TEST(TestLayerElementWiseMul, MultipleRandUnit)
{
    PROFILE_TEST
    // Test parameters
    const auto eps_rel = 1e-6_dt;
    const auto tensor_amount = 10U;
    const auto tensor_size = 10U;
    const auto random_range = std::make_pair(-10.0_dt, 10.0_dt);

    // Initialization
    MANAGERS_DEFINE
    NETWORK_PARAMS_DEFINE(networkParameters);

    // Generate tensor names
    auto name_gen = raul::NameGenerator("factor");
    raul::Names factor_tensor_names(tensor_amount);
    std::generate(factor_tensor_names.begin(), factor_tensor_names.end(), [&]() { return name_gen.generate(); });

    // Create and initialize tensors with random data
    for (const auto& tensor_name : factor_tensor_names)
    {
        work.tensorNeeded("x", tensor_name, raul::WShape{ raul::BS(), 1u, 1u, 1u }, DEC_FORW_READ_NOMEMOPT);
    }

    const auto params = raul::ElementWiseLayerParams{ factor_tensor_names, { "out" } };
    memory_manager.createTensor(raul::Name("out").grad(), tensor_size, 1, 1, 1);
    tools::init_rand_tensor(raul::Name("out").grad(), random_range, memory_manager);

    // Apply function
    raul::ElementWiseMulLayer elementwise_mul("mul", raul::ElementWiseLayerParams{ factor_tensor_names, { "out" } }, networkParameters);
    TENSORS_CREATE(tensor_size);
    for (const auto& tensor_name : factor_tensor_names)
    {
        tools::init_rand_tensor(tensor_name, random_range, memory_manager);
    }
    tools::init_rand_tensor(raul::Name("out").grad(), random_range, memory_manager);

    elementwise_mul.forwardCompute(raul::NetworkMode::Train);
    const auto& out_tensor = memory_manager["out"];

    const auto skip_and_mul = [&](const std::optional<size_t> skip_idx, const size_t axis) {
        auto out = 1.0_dt;
        for (size_t i = 0; i < factor_tensor_names.size(); ++i)
        {
            if (skip_idx && i == skip_idx.value()) continue;
            const auto tensor_name = factor_tensor_names[i];
            const auto& tensor = memory_manager[tensor_name];
            out = golden_mul_layer(out, tensor[axis]);
        }
        return out;
    };
    // Check forward
    for (size_t i = 0; i < out_tensor.size(); ++i)
    {
        const auto golden_out = skip_and_mul(std::nullopt, i);
        ASSERT_TRUE(tools::expect_near_relative(out_tensor[i], golden_out, eps_rel));
    }

    elementwise_mul.backwardCompute();

    // Check sizes
    const auto& out_tensor_grad = memory_manager[raul::Name("out").grad()];
    for (const auto& tensor_name : factor_tensor_names)
    {
        const auto& tensor = memory_manager[tensor_name];
        const auto& tensor_grad = memory_manager[tensor_name.grad()];
        EXPECT_EQ(out_tensor.getShape(), tensor.getShape());
        EXPECT_EQ(out_tensor_grad.getShape(), tensor_grad.getShape());
    }

    // Check backward
    for (size_t i = 0; i < out_tensor_grad.size(); ++i)
    {
        for (size_t j = 0; j < factor_tensor_names.size(); ++j)
        {
            const auto tensor_name = factor_tensor_names[j];
            const auto& tensor = memory_manager[tensor_name];
            const auto& tensor_grad = memory_manager[tensor_name.grad()];
            const auto skip_mul_out_i = skip_and_mul(j, i);
            const auto [_, tensor_golden_grad] = golden_mul_layer_grad(skip_mul_out_i, tensor[i], out_tensor_grad[i]);
            ASSERT_TRUE(tools::expect_near_relative(tensor_grad[i], tensor_golden_grad, eps_rel));
        }
    }
}

TEST(TestLayerElementWiseMul, BroadcastForwardRandUnit)
{
    PROFILE_TEST
    // Test parameters
    const auto eps_rel = 1e-6_dt;
    const auto random_range = std::make_pair(-100.0_dt, 100.0_dt);
    const auto enable_broadcast = true;

    // Initialization
    MANAGERS_DEFINE
    NETWORK_PARAMS_DEFINE(networkParameters);

    work.tensorNeeded("x", "x", raul::WShape{ raul::BS(), 1u, 1u, 1u }, DEC_FORW_READ_NOMEMOPT);
    work.tensorNeeded("y", "y", raul::WShape{ raul::BS(), 2u, 2u, 1u }, DEC_FORW_READ_NOMEMOPT);

    // Apply function
    raul::ElementWiseMulLayer elementwise_mul("mul", raul::ElementWiseLayerParams{ { "x", "y" }, { "out" }, enable_broadcast }, networkParameters);
    TENSORS_CREATE(1);
    tools::init_rand_tensor("x", random_range, memory_manager);
    tools::init_rand_tensor("y", random_range, memory_manager);

    elementwise_mul.forwardCompute(raul::NetworkMode::Test);

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
        const auto golden_out_value = golden_mul_layer(x_value, y_value);
        ASSERT_TRUE(tools::expect_near_relative(out_value, golden_out_value, eps_rel));
    }
}

TEST(TestLayerElementWiseMul, BroadcastBackwardRandUnit)
{
    PROFILE_TEST
    // Test parameters
    const auto eps_rel = 1e-6_dt;
    const auto random_range = std::make_pair(-100.0_dt, 100.0_dt);
    const auto enable_broadcast = true;

    // Initialization
    MANAGERS_DEFINE
    NETWORK_PARAMS_DEFINE(networkParameters);

    work.tensorNeeded("x", "x", raul::WShape{ raul::BS(), 2u, 2u, 1u }, DEC_FORW_READ_NOMEMOPT);
    work.tensorNeeded("y", "y", raul::WShape{ raul::BS(), 1u, 1u, 1u }, DEC_FORW_READ_NOMEMOPT);

    // Apply function
    raul::ElementWiseMulLayer elementwise_mul("mul", raul::ElementWiseLayerParams{ { "x", "y" }, { "out" }, enable_broadcast }, networkParameters);
    TENSORS_CREATE(1);
    tools::init_rand_tensor("x", random_range, memory_manager);
    tools::init_rand_tensor("y", random_range, memory_manager);

    elementwise_mul.forwardCompute(raul::NetworkMode::Train);
    elementwise_mul.backwardCompute();

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
        const auto [golden_out_value_x, golden_out_value_y] = golden_mul_layer_grad(x_value, y_value, out_grad_value);
        ASSERT_TRUE(tools::expect_near_relative(x_grad_value, golden_out_value_x, eps_rel));
    }

    auto golden_out_value = 0.0_dt;
    for (size_t i = 0; i < out_tensor_grad.size(); ++i)
    {
        const auto x_value = x_tensor[i];
        // We have y tensor with shape [1,1] so there is only 1 value in y
        const auto y_value = y_tensor[0];
        const auto out_grad_value = out_tensor_grad[i];
        const auto [golden_out_value_x, golden_out_value_y] = golden_mul_layer_grad(x_value, y_value, out_grad_value);
        golden_out_value += golden_out_value_y;
    }
    // We have y tensor with shape [1,1] so there is only 1 value in grad of y
    ASSERT_TRUE(tools::expect_near_relative(y_tensor_grad[0], golden_out_value, eps_rel));
}

TEST(TestLayerElementWiseMul, BroadcastForwardUnit)
{
    PROFILE_TEST
    // Test parameters
    const auto eps_rel = 1e-6_dt;
    const auto enable_broadcast = true;
    const auto shape = yato::dims(5, 3, 2, 3);

    // See broadcasting.ipynb (seed 0)
    const raul::Tensor x{ 0.49625658988952637_dt,  0.7682217955589294_dt,  0.08847743272781372_dt, 0.13203048706054688_dt,  0.30742281675338745_dt, 0.6340786814689636_dt,
                          0.4900934100151062_dt,   0.8964447379112244_dt,  0.455627977848053_dt,   0.6323062777519226_dt,   0.3488934636116028_dt,  0.40171730518341064_dt,
                          0.022325754165649414_dt, 0.16885894536972046_dt, 0.2938884496688843_dt,  0.518521785736084_dt,    0.6976675987243652_dt,  0.800011396408081_dt,
                          0.16102945804595947_dt,  0.28226858377456665_dt, 0.6816085577011108_dt,  0.9151939749717712_dt,   0.39709991216659546_dt, 0.8741558790206909_dt,
                          0.41940832138061523_dt,  0.5529070496559143_dt,  0.9527381062507629_dt,  0.036164820194244385_dt, 0.1852310299873352_dt,  0.37341737747192383_dt };
    const raul::Tensor y{ 0.3051000237464905_dt,  0.9320003986358643_dt,  0.17591017484664917_dt, 0.2698335647583008_dt, 0.15067976713180542_dt, 0.03171950578689575_dt, 0.20812976360321045_dt,
                          0.9297990202903748_dt,  0.7231091856956482_dt,  0.7423362731933594_dt,  0.5262957811355591_dt, 0.24365824460983276_dt, 0.584592342376709_dt,   0.033152639865875244_dt,
                          0.13871687650680542_dt, 0.242235004901886_dt,   0.815468966960907_dt,   0.793160617351532_dt,  0.2782524824142456_dt,  0.48195880651474_dt,    0.8197803497314453_dt,
                          0.9970665574073792_dt,  0.6984410881996155_dt,  0.5675464272499084_dt,  0.8352431654930115_dt, 0.2055988311767578_dt,  0.593172013759613_dt,   0.11234724521636963_dt,
                          0.1534569263458252_dt,  0.24170821905136108_dt, 0.7262365221977234_dt,  0.7010802030563354_dt, 0.2038237452507019_dt,  0.6510535478591919_dt,  0.7744860053062439_dt,
                          0.4368913173675537_dt,  0.5190907716751099_dt,  0.6158523559570312_dt,  0.8101882934570312_dt, 0.9800970554351807_dt,  0.1146882176399231_dt,  0.3167651295661926_dt,
                          0.6965049505233765_dt,  0.9142746925354004_dt,  0.9351036548614502_dt };
    const raul::Tensor z{ 0.1514078974723816_dt,   0.7159830331802368_dt,   0.015564080327749252_dt,  0.04028250649571419_dt,  0.2865181863307953_dt,   0.11154089123010635_dt,
                          0.13390667736530304_dt,  0.11575548350811005_dt,  0.0028064604848623276_dt, 0.035626258701086044_dt, 0.04632239788770676_dt,  0.0201126616448164_dt,
                          0.1032857671380043_dt,   0.7142918705940247_dt,   0.06397884339094162_dt,   0.027479473501443863_dt, 0.2858414351940155_dt,   0.45850813388824463_dt,
                          0.363814115524292_dt,    0.47179508209228516_dt,  0.11101751029491425_dt,   0.4693838953971863_dt,   0.18362115323543549_dt,  0.0978817343711853_dt,
                          0.286504864692688_dt,    0.029719509184360504_dt, 0.06320329010486603_dt,   0.36964139342308044_dt,  0.011566739529371262_dt, 0.05572497099637985_dt,
                          0.11871778219938278_dt,  0.731022834777832_dt,    0.36138617992401123_dt,   0.15316671133041382_dt,  0.2845118045806885_dt,   0.318626344203949_dt,
                          0.006212196312844753_dt, 0.08138305693864822_dt,  0.24092397093772888_dt,   0.14427997171878815_dt,  0.336247056722641_dt,    0.6558336019515991_dt,
                          0.022260263562202454_dt, 0.11793802678585052_dt,  0.16679534316062927_dt,   0.5170007348060608_dt,   0.4872797131538391_dt,   0.45404359698295593_dt,
                          0.018647434189915657_dt, 0.03471720218658447_dt,  0.17432640492916107_dt,   0.43309178948402405_dt,  0.1434396356344223_dt,   0.47454437613487244_dt,
                          0.018091216683387756_dt, 0.0433160699903965_dt,   0.16475039720535278_dt,   0.10281952470541_dt,     0.060937732458114624_dt, 0.2112906575202942_dt,
                          0.11694547533988953_dt,  0.1978929191827774_dt,   0.13892801105976105_dt,   0.6646472811698914_dt,   0.27839890122413635_dt,  0.1781737208366394_dt,
                          0.10483880341053009_dt,  0.21861307322978973_dt,  0.2977888584136963_dt,    0.5958402752876282_dt,   0.30754831433296204_dt,  0.38191109895706177_dt,
                          0.21771098673343658_dt,  0.3405091166496277_dt,   0.7718972563743591_dt,    0.018772823736071587_dt, 0.11407496780157089_dt,  0.3025383949279785_dt,
                          0.4110608696937561_dt,   0.06341192126274109_dt,  0.3017942011356354_dt,    0.03544503450393677_dt,  0.02124381624162197_dt,  0.11828560382127762_dt,
                          0.29211997985839844_dt,  0.5055088996887207_dt,   0.8909088969230652_dt,    0.025188976898789406_dt, 0.16935203969478607_dt,  0.34918394684791565_dt };

    // Initialization
    MANAGERS_DEFINE
    NETWORK_PARAMS_DEFINE(networkParameters);

    work.tensorNeeded("x", "x", raul::WShape{ raul::BS(), 1u, 2u, 3u }, DEC_FORW_READ_NOMEMOPT);
    work.tensorNeeded("y", "y", raul::WShape{ raul::BS(), 3u, 1u, 3u }, DEC_FORW_READ_NOMEMOPT);

    // Apply function
    raul::ElementWiseMulLayer elementwise_mul("mul", raul::ElementWiseLayerParams{ { "x", "y" }, { "out" }, enable_broadcast }, networkParameters);
    TENSORS_CREATE(5);
    memory_manager["x"] = TORANGE(x);
    memory_manager["y"] = TORANGE(y);

    elementwise_mul.forwardCompute(raul::NetworkMode::Test);

    // Checks
    const auto& out_tensor = memory_manager["out"];
    EXPECT_EQ(out_tensor.getShape(), shape);
    for (size_t i = 0; i < out_tensor.size(); ++i)
    {
        ASSERT_TRUE(tools::expect_near_relative(out_tensor[i], z[i], eps_rel));
    }
}

TEST(TestLayerElementWiseMul, BroadcastBackwardUnit)
{
    PROFILE_TEST
    // Test parameters
    const auto eps_rel = 1e-6_dt;
    const auto enable_broadcast = true;
    const auto shape = yato::dims(1, 2, 1, 3);

    // See broadcasting.ipynb (seed 0)
    const raul::Tensor x{ 0.49625658988952637_dt, 0.7682217955589294_dt };
    const raul::Tensor y{ 0.08847743272781372_dt, 0.13203048706054688_dt, 0.30742281675338745_dt };
    const raul::Tensor z{ 0.04390750825405121_dt, 0.0655210018157959_dt, 0.15256059169769287_dt, 0.06797029078006744_dt, 0.10142869502305984_dt, 0.23616890609264374_dt };

    const raul::Tensor deltas{ 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt };
    const raul::Tensor x_grad{ 0.527930736541748_dt, 0.527930736541748_dt };
    const raul::Tensor y_grad{ 1.2644784450531006_dt, 1.2644784450531006_dt, 1.2644784450531006_dt };

    // Initialization
    MANAGERS_DEFINE
    NETWORK_PARAMS_DEFINE(networkParameters);

    work.tensorNeeded("x", "x", raul::WShape{ raul::BS(), 2u, 1u, 1u }, DEC_FORW_READ_NOMEMOPT);
    work.tensorNeeded("y", "y", raul::WShape{ raul::BS(), 1u, 1u, 3u }, DEC_FORW_READ_NOMEMOPT);

    // Apply function
    raul::ElementWiseMulLayer elementwise_mul("mul", raul::ElementWiseLayerParams{ { "x", "y" }, { "out" }, enable_broadcast }, networkParameters);
    TENSORS_CREATE(1);
    memory_manager["x"] = TORANGE(x);
    memory_manager["y"] = TORANGE(y);
    memory_manager[raul::Name("out").grad()] = TORANGE(deltas);

    elementwise_mul.forwardCompute(raul::NetworkMode::Train);
    elementwise_mul.backwardCompute();

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

#ifdef ANDROID
TEST(TestLayerElementWiseMul, ForwardRandFP16Unit)
{
    PROFILE_TEST
    // Test parameters
    const auto eps_rel = 1e-6_hf;
    const auto tensor_size = 1000U;
    const auto random_range = std::make_pair(0.0_hf, 100.0_hf);

    // Initialization

    raul::WorkflowEager work{ raul::CompressionMode::NONE, raul::CalculationMode::DETERMINISTIC, raul::AllocationMode::STANDARD, raul::ExecutionTarget::CPUFP16 };
    auto& memory_manager = work.getMemoryManager<raul::MemoryManagerFP16>();
    NETWORK_PARAMS_DEFINE(networkParameters);

    work.tensorNeeded("x", "x", raul::WShape{ raul::BS(), 1u, 1u, 1u }, DEC_FORW_READ_NOMEMOPT);
    work.tensorNeeded("y", "y", raul::WShape{ raul::BS(), 1u, 1u, 1u }, DEC_FORW_READ_NOMEMOPT);

    // Apply function
    raul::ElementWiseMulLayer elementwise_mul("mul", raul::ElementWiseLayerParams{ { "x", "y" }, { "out" } }, networkParameters);
    TENSORS_CREATE(tensor_size);

    tools::init_rand_tensor<raul::MemoryManagerFP16>("x", random_range, memory_manager);
    tools::init_rand_tensor<raul::MemoryManagerFP16>("y", random_range, memory_manager);

    elementwise_mul.forwardCompute(raul::NetworkMode::Test);

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
        const auto golden_out_value = golden_mul_layer(x_value, y_value);
        ASSERT_TRUE(tools::expect_near_relative(out_value, golden_out_value, eps_rel));
    }
}

TEST(TestLayerElementWiseMul, BroadcastBackwardRandFP16Unit)
{
    PROFILE_TEST
    // Test parameters
    const auto eps_rel = 1e-6_hf;
    const auto random_range = std::make_pair(0.0_hf, 100.0_hf);
    const auto enable_broadcast = true;

    // Initialization
    raul::WorkflowEager work{ raul::CompressionMode::NONE, raul::CalculationMode::DETERMINISTIC, raul::AllocationMode::STANDARD, raul::ExecutionTarget::CPUFP16 };
    auto& memory_manager = work.getMemoryManager<raul::MemoryManagerFP16>();
    NETWORK_PARAMS_DEFINE(networkParameters);

    work.tensorNeeded("x", "x", raul::WShape{ raul::BS(), 2u, 2u, 1u }, DEC_FORW_READ_NOMEMOPT);
    work.tensorNeeded("y", "y", raul::WShape{ raul::BS(), 1u, 1u, 1u }, DEC_FORW_READ_NOMEMOPT);

    // Apply function
    raul::ElementWiseMulLayer elementwise_mul("mul", raul::ElementWiseLayerParams{ { "x", "y" }, { "out" }, enable_broadcast }, networkParameters);
    TENSORS_CREATE(1);
    tools::init_rand_tensor("x", random_range, memory_manager);
    tools::init_rand_tensor("y", random_range, memory_manager);

    elementwise_mul.forwardCompute(raul::NetworkMode::Train);
    elementwise_mul.backwardCompute();

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
        const auto [golden_out_value_x, golden_out_value_y] = golden_mul_layer_grad(x_value, y_value, out_grad_value);
        ASSERT_TRUE(tools::expect_near_relative(x_grad_value, golden_out_value_x, eps_rel));
    }

    auto golden_out_value = 0.0_hf;
    for (size_t i = 0; i < out_tensor_grad.size(); ++i)
    {
        const auto x_value = x_tensor[i];
        // We have y tensor with shape [1,1] so there is only 1 value in y
        const auto y_value = y_tensor[0];
        const auto out_grad_value = out_tensor_grad[i];
        const auto [golden_out_value_x, golden_out_value_y] = golden_mul_layer_grad(x_value, y_value, out_grad_value);
        golden_out_value += golden_out_value_y;
    }
    // We have y tensor with shape [1,1] so there is only 1 value in grad of y
    ASSERT_TRUE(tools::expect_near_relative(y_tensor_grad[0], golden_out_value, eps_rel));
}
#endif // ANDROID

} // UT namespace