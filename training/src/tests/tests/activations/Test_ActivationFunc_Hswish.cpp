// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include <chrono>
#include <cstdio>
#include <tests/tools/TestTools.h>

#include <training/base/layers/activations/HSwishActivation.h>
#include <training/compiler/Layers.h>
#include <training/compiler/Workflow.h>
#include <training/base/optimizers/SGD.h>

namespace UT
{

namespace
{
const auto indeterminate_value_m3 = 0.0_dt;
const auto indeterminate_value_p3 = 1.0_dt;

raul::dtype golden_hswish(const raul::dtype x)
{
    return x / 6.0_dt * std::min(std::max(x + 3.0_dt, .0_dt), 6.0_dt);
}

raul::dtype golden_hswish_grad(raul::dtype x, raul::dtype grad)
{
    if (x == -3.0_dt)
    {
        return indeterminate_value_m3 * grad;
    }
    if (x == 3.0_dt)
    {
        return indeterminate_value_p3 * grad;
    }
    if (x > 3.0_dt)
    {
        return grad;
    }
    if (x > -3.0_dt && x < 3.0_dt)
    {
        return 1.0_dt / 6.0_dt * (2.0_dt * x + 3.0_dt) * grad;
    }
    return 0.0_dt;
}
}

TEST(TestActivationFuncHSwish, Unit)
{
    PROFILE_TEST
    auto eps = 1e-4_dt;

    {
        MANAGERS_DEFINE
        NETWORK_PARAMS_DEFINE(networkParameters);

        raul::Tensor raw{ 0.37479503f, 0.74949179f, 0.92839539f, 0.40420301f, 0.9961107f,  0.66630404f, 0.82047117f, 0.99906097f, 0.48825795f, 0.6590186f,  0.14482136f, 0.6286064f,  0.54629563f,
                          0.18286226f, 0.73805489f, 0.50778409f, 0.76171372f, 0.74803807f, 0.22766416f, 0.80420801f, 0.76574229f, 0.795332f,   0.00788858f, 0.21592507f, 0.66575409f, 0.91054865f,
                          0.07787714f, 0.93616972f, 0.78133592f, 0.89633024f, 0.96769797f, 0.57117974f, 0.58974003f, 0.6582992f,  0.47186511f, 0.03024406f, 0.86331904f, 0.4101691f,  0.74629559f,
                          0.1786914f,  0.26763513f, 0.68119195f, 0.41871249f, 0.41247165f, 0.48607614f, 0.65607838f, 0.47523406f, 0.45516007f, 0.39255511f, 0.43910054f, 0.34682715f, 0.08459205f,
                          0.68237903f, 0.84103279f, 0.54420833f, 0.15065369f, 0.72408225f, 0.20047205f, 0.26391343f, 0.51592856f, 0.09968541f, 0.90827312f, 0.85315302f, 0.98048446f, 0.59192641f,
                          0.54249449f, 0.51840919f, 0.36040054f, 0.3678461f,  0.68468829f, 0.41993762f, 0.20087144f, 0.21774159f, 0.58968185f, 0.05935208f, 0.22305229f, 0.50138518f, 0.65757976f,
                          0.14150964f, 0.07537156f, 0.34186466f, 0.34769964f, 0.89359716f, 0.97368842f, 0.55870338f, 0.75948831f, 0.23420801f, 0.06065636f, 0.10140947f, 0.59413656f, 0.88430418f,
                          0.29935133f, 0.36099395f, 0.59542665f, 0.25830884f, 0.07518427f, 0.09374659f, 0.2718588f,  0.32238792f, 0.65717813f, 0.17969255f, 0.79472564f, 0.54135358f, 0.02137773f,
                          0.70453999f, 0.1567105f,  0.30232926f, 0.76933107f, 0.13083392f, 0.24884672f, 0.94565419f, 0.18572746f, 0.59809335f, 0.81106049f, 0.43165358f, 0.4994478f,  0.10799541f,
                          0.83711806f, 0.32547974f, 0.5652622f,  0.70515491f, 0.9385492f,  0.8703726f,  0.05524975f, 0.543256f,   0.33594836f, 0.90279934f, 0.00769521f, 0.22831708f, 0.55580578f,
                          0.57967145f, 0.47428567f, 0.98923259f, 0.01861445f, 0.08208355f, 0.53100689f, 0.27314956f, 0.03744627f, 0.70743921f, 0.48315441f, 0.89972966f, 0.82618481f, 0.79273583f,
                          0.71205182f, 0.70734133f, 0.22427137f, 0.94323804f, 0.45126228f, 0.72933639f, 0.7159068f,  0.1407426f,  0.39469651f, 0.25006817f, 0.81597341f, 0.94659794f, 0.96640427f,
                          0.0459238f,  0.83164318f, 0.31587163f, 0.09632278f, 0.50825189f, 0.89314523f, 0.05260463f, 0.97404436f, 0.33124845f, 0.49051957f, 0.35728925f, 0.08471391f, 0.54142684f,
                          0.93035542f, 0.55526997f, 0.34721205f, 0.67594097f, 0.19324233f, 0.91126217f, 0.93079399f, 0.31820165f, 0.63886702f, 0.7115078f,  0.04140918f, 0.42797978f, 0.31839019f,
                          0.15497529f, 0.9243903f,  0.10203447f, 0.02429839f, 0.79189611f, 0.77230195f, 0.5449423f,  0.40429956f, 0.00810897f, 0.69576717f, 0.81711272f, 0.1335145f,  0.84167646f,
                          0.70258132f, 0.81158035f, 0.2585105f,  0.09862106f, 0.37432664f, 0.74594114f, 0.84899346f, 0.8828963f,  0.37912551f, 0.34303555f, 0.88711791f, 0.40549404f, 0.70713591f,
                          0.5982242f,  0.3516502f,  0.72905793f, 0.9808251f,  0.36860929f, 0.90998312f, 0.36086885f, 0.88600347f, 0.94211039f, 0.85760511f, 0.12342954f, 0.14555429f, 0.80846367f,
                          0.67246539f, 0.2034387f,  0.76852984f, 0.92605775f, 0.26831058f, 0.67321516f, 0.94508101f, 0.56843361f, 0.3550119f,  0.99057556f, 0.43329525f, 0.35586554f, 0.00825131f,
                          0.98605478f, 0.21982703f, 0.89359794f, 0.21367511f, 0.11302765f, 0.05410334f, 0.55786171f, 0.47368395f, 0.85497701f, 0.9911735f,  0.41700463f, 0.90840312f, 0.07303944f,
                          0.84749951f, 0.71403399f, 0.03476528f, 0.61093322f, 0.05770533f, 0.33471661f, 0.04379381f, 0.10857736f, 0.87996621f, 0.90394243f, 0.73068102f, 0.29301801f, 0.19324834f,
                          0.5182907f,  0.92317623f, 0.3433689f,  0.70590362f, 0.85885382f, 0.08763445f, 0.34156856f, 0.31501855f, 0.99025243f, 0.28233952f, 0.99291062f, 0.11555683f, 0.13840601f,
                          0.18735525f, 0.50227961f, 0.53595563f, 0.70512296f, 0.41612818f, 0.14544152f, 0.99252427f, 0.88588884f, 0.86861712f, 0.22435276f, 0.81870535f, 0.86298226f, 0.21892615f,
                          0.89298307f, 0.56381208f, 0.37950085f, 0.61620922f, 0.73818377f, 0.70532032f, 0.91082064f, 0.27172544f, 0.55463835f, 0.62264405f, 0.71936221f, 0.71908206f, 0.12704653f,
                          0.53684262f, 0.71610209f, 0.28590477f, 0.55936588f, 0.62242997f, 0.879633f,   0.32378584f, 0.75152276f, 0.17728816f, 0.18134275f, 0.14376092f, 0.80051592f, 0.55731164f,
                          0.58153594f, 0.93270095f, 0.34074676f, 0.42614444f, 0.10055618f, 0.82608708f, 0.28488983f, 0.49550837f, 0.79601586f, 0.32068777f, 0.57682384f, 0.6719388f,  0.38668566f,
                          0.19383373f, 0.70447052f, 0.9298161f,  0.18070789f, 0.93694095f, 0.33000843f, 0.58258544f, 0.07584233f, 0.43948981f, 0.15010104f, 0.55299119f, 0.76852814f, 0.09187515f,
                          0.79264914f, 0.37365331f, 0.31016106f, 0.2233624f,  0.88615481f, 0.37971135f, 0.41666305f, 0.94860019f, 0.69704601f, 0.36850484f, 0.89811771f, 0.3676741f,  0.44550689f,
                          0.48621984f, 0.23688324f, 0.99989737f, 0.99483116f, 0.1974391f,  0.6843357f,  0.31812494f, 0.19924475f, 0.77234562f };
        size_t batch = 3;
        size_t in_w = 6;
        size_t in_h = 5;
        size_t depth = 4;

        std::vector<raul::dtype> realOut{ 0.2108f, 0.4684f, 0.6079f, 0.2293f, 0.6634f, 0.4071f, 0.5224f, 0.6659f, 0.2839f, 0.4019f, 0.0759f, 0.3802f, 0.3229f, 0.0970f, 0.4598f,
                                          0.2969f, 0.4776f, 0.4673f, 0.1225f, 0.5099f, 0.4806f, 0.5031f, 0.0040f, 0.1157f, 0.4067f, 0.5935f, 0.0399f, 0.6142f, 0.4924f, 0.5821f,

                                          0.6399f, 0.3400f, 0.3528f, 0.4014f, 0.2730f, 0.0153f, 0.5559f, 0.2331f, 0.4660f, 0.0947f, 0.1458f, 0.4179f, 0.2386f, 0.2346f, 0.2824f,
                                          0.3998f, 0.2753f, 0.2621f, 0.2220f, 0.2517f, 0.1935f, 0.0435f, 0.4188f, 0.5384f, 0.3215f, 0.0791f, 0.4494f, 0.1069f, 0.1436f, 0.3023f,

                                          0.0515f, 0.5916f, 0.5479f, 0.6505f, 0.3544f, 0.3203f, 0.3040f, 0.2018f, 0.2065f, 0.4205f, 0.2394f, 0.1072f, 0.1168f, 0.3528f, 0.0303f,
                                          0.1198f, 0.2926f, 0.4009f, 0.0741f, 0.0386f, 0.1904f, 0.1940f, 0.5799f, 0.6449f, 0.3314f, 0.4759f, 0.1262f, 0.0309f, 0.0524f, 0.3559f,

                                          0.5725f, 0.1646f, 0.2022f, 0.3568f, 0.1403f, 0.0385f, 0.0483f, 0.1482f, 0.1785f, 0.4006f, 0.0952f, 0.5026f, 0.3195f, 0.0108f, 0.4350f,
                                          0.0824f, 0.1664f, 0.4833f, 0.0683f, 0.1347f, 0.6219f, 0.0986f, 0.3587f, 0.5152f, 0.2469f, 0.2913f, 0.0559f, 0.5354f, 0.1804f, 0.3359f,

                                          0.4355f, 0.6161f, 0.5614f, 0.0281f, 0.3208f, 0.1868f, 0.5872f, 0.0039f, 0.1228f, 0.3294f, 0.3458f, 0.2746f, 0.6577f, 0.0094f, 0.0422f,
                                          0.3125f, 0.1490f, 0.0190f, 0.4371f, 0.2805f, 0.5848f, 0.5269f, 0.5011f, 0.4405f, 0.4371f, 0.1205f, 0.6199f, 0.2596f, 0.4533f, 0.4434f,

                                          0.0737f, 0.2233f, 0.1355f, 0.5190f, 0.6226f, 0.6389f, 0.0233f, 0.5311f, 0.1746f, 0.0497f, 0.2972f, 0.5795f, 0.0268f, 0.6451f, 0.1839f,
                                          0.2854f, 0.1999f, 0.0436f, 0.3196f, 0.6094f, 0.3290f, 0.1937f, 0.4141f, 0.1028f, 0.5940f, 0.6098f, 0.1760f, 0.3875f, 0.4401f, 0.0210f,

                                          0.2445f, 0.1761f, 0.0815f, 0.6046f, 0.0528f, 0.0122f, 0.5005f, 0.4856f, 0.3220f, 0.2294f, 0.0041f, 0.4286f, 0.5198f, 0.0697f, 0.5389f,
                                          0.4336f, 0.5156f, 0.1404f, 0.0509f, 0.2105f, 0.4657f, 0.5446f, 0.5714f, 0.2135f, 0.1911f, 0.5747f, 0.2302f, 0.4369f, 0.3588f, 0.1964f,

                                          0.4531f, 0.6507f, 0.2070f, 0.5930f, 0.2021f, 0.5738f, 0.6190f, 0.5514f, 0.0643f, 0.0763f, 0.5132f, 0.4116f, 0.1086f, 0.4827f, 0.6060f,
                                          0.1462f, 0.4121f, 0.6214f, 0.3381f, 0.1985f, 0.6588f, 0.2479f, 0.1990f, 0.0041f, 0.6551f, 0.1180f, 0.5799f, 0.1144f, 0.0586f, 0.0275f,

                                          0.3308f, 0.2742f, 0.5493f, 0.6593f, 0.2375f, 0.5917f, 0.0374f, 0.5435f, 0.4420f, 0.0176f, 0.3677f, 0.0294f, 0.1860f, 0.0222f, 0.0563f,
                                          0.5690f, 0.5882f, 0.4543f, 0.1608f, 0.1028f, 0.3039f, 0.6036f, 0.1913f, 0.4360f, 0.5524f, 0.0451f, 0.1902f, 0.1740f, 0.6586f, 0.1545f,

                                          0.6608f, 0.0600f, 0.0724f, 0.0995f, 0.2932f, 0.3159f, 0.4354f, 0.2369f, 0.0762f, 0.6604f, 0.5737f, 0.5601f, 0.1206f, 0.5211f, 0.5556f,
                                          0.1175f, 0.5794f, 0.3349f, 0.2138f, 0.3714f, 0.4599f, 0.4356f, 0.5937f, 0.1482f, 0.3286f, 0.3759f, 0.4459f, 0.4457f, 0.0662f, 0.3165f,

                                          0.4435f, 0.1566f, 0.3318f, 0.3758f, 0.5688f, 0.1794f, 0.4699f, 0.0939f, 0.0962f, 0.0753f, 0.5071f, 0.3304f, 0.3471f, 0.6113f, 0.1897f,
                                          0.2433f, 0.0520f, 0.5268f, 0.1560f, 0.2887f, 0.5036f, 0.1775f, 0.3439f, 0.4112f, 0.2183f, 0.1032f, 0.4349f, 0.6090f, 0.0958f, 0.6148f,

                                          0.1832f, 0.3479f, 0.0389f, 0.2519f, 0.0788f, 0.3275f, 0.4827f, 0.0473f, 0.5010f, 0.2101f, 0.1711f, 0.1200f, 0.5740f, 0.2139f, 0.2373f,
                                          0.6243f, 0.4295f, 0.2069f, 0.5835f, 0.2064f, 0.2558f, 0.2825f, 0.1278f, 0.6666f, 0.6624f, 0.1052f, 0.4202f, 0.1759f, 0.1062f, 0.4856f };

        work.add<raul::DataLayer>("data", raul::DataParams{ { "in" }, depth, in_h, in_w });
        auto params = raul::HSwishActivationParams{ { "in" }, { "out" } };
        raul::HSwishActivation hswish("hswish", params, networkParameters);
        TENSORS_CREATE(batch);
        memory_manager["in"] = TORANGE(raw);

        const raul::Tensor& out = memory_manager["out"];
        hswish.forwardCompute(raul::NetworkMode::Test);

        EXPECT_EQ(out.size(), realOut.size());
        for (size_t i = 0; i < out.size(); ++i)
            EXPECT_NEAR(TODTYPE(out[i]), TODTYPE(realOut[i]), eps);
    }
}

TEST(TestActivationFuncHSwish, ForwardRandUnit)
{
    PROFILE_TEST
    // Test parameters
    const auto eps = TODTYPE(1e-4);
    const auto tensor_size = 1000U;
    // We have a significant change in the range [-4,4]
    const auto random_range = std::pair<raul::dtype, raul::dtype>(-4.0f, 4.0f);

    // Random generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(random_range.first, std::nextafter(random_range.second, std::numeric_limits<raul::dtype>::max()));

    // Initialization
    MANAGERS_DEFINE
    NETWORK_PARAMS_DEFINE(networkParameters);

    work.add<raul::DataLayer>("data", raul::DataParams{ { "in" }, 1, 1, 1 });
    const auto params = raul::HSwishActivationParams{ { "in" }, { "out" } };
    // Apply function
    raul::HSwishActivation hswish_activation("hswish", params, networkParameters);
    TENSORS_CREATE(tensor_size);

    auto& in_tensor = memory_manager["in"];
    for (auto& val : in_tensor)
    {
        val = static_cast<raul::dtype>(dis(gen));
    }

    hswish_activation.forwardCompute(raul::NetworkMode::Test);

    // Checks
    const auto& out_tensor = memory_manager["out"];

    EXPECT_EQ(out_tensor.size(), in_tensor.size());

    for (size_t i = 0; i < out_tensor.size(); ++i)
    {
        const auto in_value = in_tensor[i];
        const auto out_value = out_tensor[i];
        const auto golden_out_value = golden_hswish(in_value);
        EXPECT_NEAR(out_value, golden_out_value, eps);
    }
}

TEST(TestActivationFuncHSwish, BackwardRandUnit)
{
    PROFILE_TEST
    // Test parameters
    const auto eps = 1e-6_dt;
    const auto tensor_size = 1000U;
    // We have a significant change in the range [-5,5]
    const auto random_range = std::pair<raul::dtype, raul::dtype>(-4.0f, 4.0f);

    // Random generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(random_range.first, std::nextafter(random_range.second, std::numeric_limits<raul::dtype>::max()));

    // Initialization
    MANAGERS_DEFINE
    NETWORK_PARAMS_DEFINE(networkParameters);

    work.add<raul::DataLayer>("data", raul::DataParams{ { "in" }, 1, 1, 1 });

    const auto params = raul::HSwishActivationParams{ { "in" }, { "out" } };

    // Apply function
    raul::HSwishActivation hswish_activation("hswish", params, networkParameters);
    TENSORS_CREATE(tensor_size);

    auto& in_tensor = memory_manager["in"];
    for (auto& val : in_tensor)
    {
        val = static_cast<raul::dtype>(dis(gen));
    }

    auto& grad_tensor = memory_manager[raul::Name("out").grad()];
    for (auto& val : grad_tensor)
    {
        val = static_cast<raul::dtype>(dis(gen));
    }

    hswish_activation.forwardCompute(raul::NetworkMode::Train);
    hswish_activation.backwardCompute();

    // Checks
    const auto& out_tensor_grad = memory_manager[raul::Name("in").grad()];

    EXPECT_EQ(out_tensor_grad.size(), grad_tensor.size());

    for (size_t i = 0; i < out_tensor_grad.size(); ++i)
    {
        const auto in_value = in_tensor[i];
        const auto grad_value = grad_tensor[i];
        const auto out_value = out_tensor_grad[i];
        const auto golden_out_value = golden_hswish_grad(in_value, grad_value);
        EXPECT_NEAR(out_value, golden_out_value, eps);
    }
}

} // UT namespace
