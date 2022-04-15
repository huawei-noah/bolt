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
#include <training/base/layers/basic/MaxPoolLayer.h>
#include <training/base/layers/parameters/LayerParameters.h>
#include <training/compiler/Layers.h>
#include <training/compiler/Workflow.h>
#include <training/base/optimizers/SGD.h>

namespace UT
{
using namespace raul;

TEST(TestCNNMaxPool, Unit)
{
    PROFILE_TEST
    dtype eps = TODTYPE(1e-4);
    {
        MANAGERS_DEFINE
        NETWORK_PARAMS_DEFINE(networkParameters);

        Tensor raw = { 1, 2, 3, 4, 5, 6, 7, 8, 9 };
        size_t batch = 1;
        size_t stride_w = 1;
        size_t stride_h = 1;
        size_t in_w = 3;
        size_t in_h = 3;
        size_t padding_w = 0;
        size_t padding_h = 0;
        size_t depth = 1;
        size_t kernel_height = 3;
        size_t kernel_width = 3;

        work.add<DataLayer>("data", DataParams{ { "in" }, depth, in_h, in_w });
        auto params = Pool2DParams{ { "in" }, { "mp" }, kernel_width, kernel_height, stride_w, stride_h, padding_w, padding_h };
        MaxPoolLayer2D maxpool("mp1", params, networkParameters);
        TENSORS_CREATE(batch);
        memory_manager["in"] = TORANGE(raw);
        const Tensor& out = memory_manager["mp"];
        maxpool.forwardCompute(NetworkMode::Train);
        EXPECT_EQ(TODTYPE(9.f), out[0]);

        memory_manager.clear();
    }
    {
        MANAGERS_DEFINE
        NETWORK_PARAMS_DEFINE(networkParameters);

        Tensor raw = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
        size_t batch = 1;
        size_t in_w = 4;
        size_t in_h = 3;
        size_t depth = 1;
        size_t kernel_height = 3;
        size_t kernel_width = 2;
        size_t stride_w = 1;
        size_t stride_h = 1;
        size_t padding_w = 0;
        size_t padding_h = 0;

        work.add<DataLayer>("data", DataParams{ { "in" }, depth, in_h, in_w });
        auto params = Pool2DParams{ { "in" }, { "mp" }, kernel_width, kernel_height, stride_w, stride_h, padding_w, padding_h };
        MaxPoolLayer2D maxpool("mp1", params, networkParameters);
        TENSORS_CREATE(batch);
        memory_manager["in"] = TORANGE(raw);
        const Tensor& out = memory_manager["mp"];

        maxpool.forwardCompute(NetworkMode::Train);

        EXPECT_EQ(TODTYPE(10.f), out[0]);
        EXPECT_EQ(TODTYPE(11.f), out[1]);
        EXPECT_EQ(TODTYPE(12.f), out[2]);

        memory_manager.clear();
    }
    {
        MANAGERS_DEFINE
        NETWORK_PARAMS_DEFINE(networkParameters);

        Tensor raw = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 };
        size_t batch = 1;
        size_t in_w = 5;
        size_t in_h = 3;
        size_t depth = 1;
        size_t kernel_height = 2;
        size_t kernel_width = 3;
        size_t stride_w = 3;
        size_t stride_h = 1;
        size_t padding_w = 1;
        size_t padding_h = 0;

        work.add<DataLayer>("data", DataParams{ { "in" }, depth, in_h, in_w });
        auto params = Pool2DParams{ { "in" }, { "mp" }, kernel_width, kernel_height, stride_w, stride_h, padding_w, padding_h };
        MaxPoolLayer2D maxpool("mp1", params, networkParameters);
        TENSORS_CREATE(batch);
        memory_manager["in"] = TORANGE(raw);
        const Tensor& out = memory_manager["mp"];

        maxpool.forwardCompute(NetworkMode::Train);

        EXPECT_EQ(TODTYPE(7.f), out[0]);
        EXPECT_EQ(TODTYPE(10.f), out[1]);
        EXPECT_EQ(TODTYPE(12.f), out[2]);
        EXPECT_EQ(TODTYPE(15.f), out[3]);

        memory_manager.clear();
    }
    {
        MANAGERS_DEFINE
        NETWORK_PARAMS_DEFINE(networkParameters);

        Tensor raw = { 0.37479503f, 0.74949179f, 0.92839539f, 0.40420301f, 0.9961107f,  0.66630404f, 0.82047117f, 0.99906097f, 0.48825795f, 0.6590186f,  0.14482136f, 0.6286064f,  0.54629563f,
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
        size_t kernel_height = 3;
        size_t kernel_width = 4;
        size_t stride_w = 2;
        size_t stride_h = 1;
        size_t padding_w = 0;
        size_t padding_h = 1;

        work.add<DataLayer>("data", DataParams{ { "in" }, depth, in_h, in_w });
        auto params = Pool2DParams{ { "in" }, { "mp" }, kernel_width, kernel_height, stride_w, stride_h, padding_w, padding_h };
        MaxPoolLayer2D maxpool("mp1", params, networkParameters);
        TENSORS_CREATE(batch);
        memory_manager["in"] = TORANGE(raw);
        const Tensor& out = memory_manager["mp"];
        maxpool.forwardCompute(NetworkMode::Train);

        Tensor realOut{ 0.9991f, 0.9961f, 0.9991f, 0.9961f, 0.9991f, 0.7953f, 0.9362f, 0.9362f, 0.9362f, 0.9362f,

                        0.9677f, 0.7463f, 0.9677f, 0.7463f, 0.8633f, 0.8410f, 0.7241f, 0.8410f, 0.7241f, 0.8410f,

                        0.9805f, 0.9805f, 0.9805f, 0.9805f, 0.6847f, 0.9737f, 0.7595f, 0.9737f, 0.7595f, 0.9737f,

                        0.8843f, 0.7947f, 0.8843f, 0.7947f, 0.9457f, 0.9457f, 0.9457f, 0.9457f, 0.9457f, 0.9457f,

                        0.9385f, 0.8704f, 0.9892f, 0.8704f, 0.9892f, 0.8997f, 0.9892f, 0.9432f, 0.9432f, 0.9432f,

                        0.8316f, 0.9664f, 0.9740f, 0.9664f, 0.9740f, 0.8931f, 0.9740f, 0.7115f, 0.9308f, 0.7115f,

                        0.9244f, 0.9244f, 0.9244f, 0.9244f, 0.8490f, 0.8829f, 0.8871f, 0.8829f, 0.8871f, 0.8829f,

                        0.9808f, 0.9100f, 0.9808f, 0.9451f, 0.9906f, 0.9906f, 0.9906f, 0.9906f, 0.9906f, 0.9906f,

                        0.9912f, 0.9912f, 0.9912f, 0.9912f, 0.9232f, 0.9232f, 0.9232f, 0.9903f, 0.9232f, 0.9903f,

                        0.9929f, 0.9925f, 0.9929f, 0.9925f, 0.9925f, 0.9925f, 0.8630f, 0.9108f, 0.7382f, 0.9108f,

                        0.7515f, 0.8796f, 0.9327f, 0.8796f, 0.9327f, 0.8261f, 0.9327f, 0.9369f, 0.9298f, 0.9369f,

                        0.7926f, 0.7926f, 0.9486f, 0.9486f, 0.9486f, 0.9999f, 0.9948f, 0.9999f, 0.9948f, 0.9999f };
        EXPECT_EQ(out.size(), realOut.size());
        for (size_t i = 0; i < out.size(); ++i)
            EXPECT_NEAR(out[i], realOut[i], eps);

        memory_manager.clear();
    }
    {
        MANAGERS_DEFINE
        NETWORK_PARAMS_DEFINE(networkParameters);

        Tensor raw = { 0.96348341f, 0.15586064f, 0.85801312f, 0.90698303f, 0.90365074f, 0.94244516f, 0.2677996f,  0.80804349f, 0.74314807f, 0.87085426f, 0.86303181f, 0.78441631f, 0.47724223f,
                       0.87729077f, 0.6889961f,  0.10920712f, 0.502198f,   0.12459541f, 0.12104428f, 0.47144409f, 0.36667848f, 0.04591243f, 0.95481916f, 0.96010846f, 0.04127819f, 0.20963143f,
                       0.74559766f, 0.45135642f, 0.19580783f, 0.83668515f, 0.16764521f, 0.9361491f,  0.85118285f, 0.29937364f, 0.16381705f, 0.62936431f, 0.31180585f, 0.43994714f, 0.88493073f,
                       0.85579459f, 0.968812f,   0.09730115f, 0.06377765f, 0.28767497f, 0.0827621f,  0.72560324f, 0.47989796f, 0.22874052f, 0.84744393f, 0.984151f,   0.45700374f, 0.43884011f,
                       0.72322093f, 0.40692036f, 0.62789155f, 0.67226483f, 0.28128765f, 0.84102396f, 0.04455707f, 0.80358821f, 0.81809951f, 0.33479248f, 0.81593632f, 0.24718449f, 0.43681173f,
                       0.96602754f, 0.47296632f, 0.40400648f, 0.9589902f,  0.75884039f, 0.10686399f, 0.0876774f,  0.24735809f, 0.91264605f, 0.95306229f, 0.19035099f, 0.66844836f, 0.92064013f,
                       0.74866233f, 0.63805628f, 0.43843872f, 0.30932513f, 0.52097359f, 0.47408321f, 0.30412759f, 0.62930547f, 0.69347172f, 0.90716773f, 0.01624499f, 0.40114112f, 0.20085816f,
                       0.64728273f, 0.36900157f, 0.73347583f, 0.58660132f, 0.13134383f, 0.65294528f, 0.20504019f, 0.32097811f, 0.64777429f, 0.99471421f, 0.90513771f, 0.54918972f, 0.57589054f,
                       0.0360899f,  0.14018544f, 0.16177198f, 0.65594011f, 0.57178114f, 0.19571533f, 0.82147836f, 0.51522836f, 0.99352937f, 0.75577097f, 0.1459991f,  0.73017348f, 0.13535563f,
                       0.98249258f, 0.0399929f,  0.29204748f, 0.19490314f, 0.8181297f,  0.59210481f, 0.01795114f, 0.56539541f, 0.80806186f, 0.11588852f, 0.04440137f, 0.5408287f,  0.03401919f,
                       0.6801882f,  0.52087458f, 0.58138926f, 0.15212478f, 0.40800414f, 0.25477715f, 0.09736978f, 0.37100523f, 0.03739581f, 0.19658006f, 0.04517923f, 0.91809436f, 0.1281456f,
                       0.48575601f, 0.73038811f, 0.8124972f,  0.67501298f, 0.84129705f, 0.6783478f,  0.24936025f, 0.52186499f, 0.94674669f, 0.51971989f, 0.51616039f, 0.91161219f, 0.97109265f,
                       0.83201668f, 0.79153971f, 0.3663171f,  0.99702797f, 0.2939686f,  0.67578681f, 0.57569441f, 0.49055591f, 0.34768672f, 0.5827828f,  0.85240913f, 0.76038318f, 0.2918782f,
                       0.36743325f, 0.54590447f, 0.23366542f, 0.02593257f, 0.77679126f, 0.43271272f, 0.63130965f, 0.58377302f, 0.41068779f, 0.38479488f, 0.92633595f, 0.17081529f, 0.89736346f,
                       0.25978647f, 0.93597133f, 0.60374193f, 0.42577718f, 0.12651146f, 0.13197426f, 0.31386783f, 0.82884385f, 0.79896298f, 0.29060389f, 0.83111172f, 0.49998233f, 0.39149675f,
                       0.2833226f,  0.07848344f, 0.53238081f, 0.77137435f, 0.9494026f,  0.26211563f, 0.42989275f, 0.82957998f, 0.14372078f, 0.75329345f, 0.42269241f, 0.58228735f, 0.49442186f,
                       0.9368008f,  0.54981332f, 0.85718476f, 0.79682913f, 0.76635431f, 0.64293444f, 0.76701953f, 0.07457774f, 0.05627946f, 0.45719259f, 0.44601716f, 0.59308929f, 0.03508906f,
                       0.48355423f, 0.03819983f, 0.59721818f, 0.09493453f, 0.71203695f, 0.69309733f, 0.55120406f, 0.97079439f, 0.92886076f, 0.01575748f, 0.79944308f, 0.17436161f, 0.2199184f,
                       0.91300355f, 0.33894585f, 0.65701187f, 0.3347884f,  0.94329126f, 0.80632752f, 0.2943788f,  0.53829543f, 0.20138331f, 0.37048269f, 0.71986792f, 0.00633717f, 0.75972272f,
                       0.45640044f, 0.50374774f, 0.14231403f, 0.03533724f, 0.15261205f, 0.61933864f, 0.65028447f, 0.08359025f, 0.83675682f, 0.24401583f, 0.97756609f, 0.39470918f, 0.63805997f,
                       0.24649591f, 0.28133793f, 0.66552684f, 0.30532626f, 0.43491843f, 0.12394513f, 0.99693077f, 0.86963143f, 0.9114928f,  0.68174528f, 0.32254159f, 0.86191744f, 0.40482803f,
                       0.9846427f,  0.01945828f, 0.84647737f, 0.48184591f, 0.05604898f, 0.72827427f, 0.94253392f, 0.40046416f, 0.59490486f, 0.95523345f, 0.44728608f, 0.0415417f,  0.58937309f,
                       0.72378481f, 0.25805044f, 0.93793498f, 0.23147678f, 0.53864162f, 0.13551794f, 0.88318072f, 0.54479799f, 0.06905515f, 0.11122608f, 0.36390078f, 0.77017189f, 0.33376034f,
                       0.56859618f, 0.34538924f, 0.69229637f, 0.50362595f, 0.32238991f, 0.593114f,   0.5276401f,  0.70368907f, 0.16368335f, 0.5398101f,  0.61860845f, 0.28642138f, 0.11475666f,
                       0.24564118f, 0.50615165f, 0.32953329f, 0.04286386f, 0.54089183f, 0.66019695f, 0.55185473f, 0.57669059f, 0.02843475f, 0.61376359f, 0.85501606f, 0.0516355f,  0.88004504f,
                       0.35604976f, 0.68486446f, 0.36970255f, 0.67258104f, 0.72354148f, 0.93205347f, 0.61432838f, 0.4404678f,  0.64398353f, 0.11808112f, 0.70818733f, 0.4822714f,  0.27013521f,
                       0.22835508f, 0.54965191f, 0.82348522f, 0.39969524f, 0.55174928f, 0.25527024f, 0.36193352f, 0.06758679f, 0.63336846f, 0.23996246f, 0.92850972f, 0.01016019f, 0.40463047f,
                       0.78515765f, 0.36628844f, 0.83752248f, 0.10819647f, 0.19302783f, 0.62098004f, 0.02504087f, 0.5898777f,  0.52094242f };
        size_t batch = 4;
        size_t in_w = 5;
        size_t in_h = 6;
        size_t depth = 3;
        size_t kernel_height = 4;
        size_t kernel_width = 3;
        size_t stride_w = 1;
        size_t stride_h = 2;
        size_t padding_w = 1;
        size_t padding_h = 0;

        work.add<DataLayer>("data", DataParams{ { "in" }, depth, in_h, in_w });
        auto params = Pool2DParams{ { "in" }, { "mp" }, kernel_width, kernel_height, stride_w, stride_h, padding_w, padding_h };
        MaxPoolLayer2D maxpool("mp1", params, networkParameters);
        TENSORS_CREATE(batch);
        memory_manager["in"] = TORANGE(raw);
        const Tensor& out = memory_manager["mp"];
        maxpool.forwardCompute(NetworkMode::Train);

        Tensor realOut{ 0.9635f, 0.9635f, 0.9070f, 0.9070f, 0.9070f, 0.8630f, 0.9548f, 0.9601f, 0.9601f, 0.9601f,

                        0.9688f, 0.9688f, 0.9361f, 0.9842f, 0.9842f, 0.9688f, 0.9688f, 0.8474f, 0.9842f, 0.9842f,

                        0.9660f, 0.9660f, 0.9590f, 0.9590f, 0.9590f, 0.6935f, 0.9206f, 0.9206f, 0.9531f, 0.9531f,

                        0.9947f, 0.9947f, 0.9051f, 0.7335f, 0.7335f, 0.9947f, 0.9947f, 0.9935f, 0.9935f, 0.7558f,

                        0.8181f, 0.8181f, 0.8181f, 0.5921f, 0.5654f, 0.9181f, 0.9181f, 0.9181f, 0.8413f, 0.7304f,

                        0.9711f, 0.9711f, 0.9467f, 0.9970f, 0.9970f, 0.8524f, 0.8524f, 0.8524f, 0.9263f, 0.9263f,

                        0.8974f, 0.8974f, 0.9360f, 0.9494f, 0.9494f, 0.7990f, 0.8311f, 0.9368f, 0.9494f, 0.9494f,

                        0.8572f, 0.8572f, 0.9708f, 0.9708f, 0.9708f, 0.7994f, 0.7994f, 0.9708f, 0.9708f, 0.9708f,

                        0.8368f, 0.9776f, 0.9776f, 0.9776f, 0.7199f, 0.9969f, 0.9969f, 0.9969f, 0.9776f, 0.9115f,

                        0.8619f, 0.9552f, 0.9846f, 0.9846f, 0.9846f, 0.7238f, 0.9552f, 0.9552f, 0.9552f, 0.9379f,

                        0.7037f, 0.7037f, 0.7037f, 0.6602f, 0.6186f, 0.6849f, 0.8550f, 0.8550f, 0.8800f, 0.8800f,

                        0.9321f, 0.9321f, 0.9285f, 0.9285f, 0.9285f, 0.8235f, 0.8235f, 0.9285f, 0.9285f, 0.9285f };
        EXPECT_EQ(out.size(), realOut.size());
        for (size_t i = 0; i < out.size(); ++i)
            EXPECT_NEAR(out[i], realOut[i], eps);

        memory_manager.clear();
    }
    {
        MANAGERS_DEFINE
        NETWORK_PARAMS_DEFINE(networkParameters);

        Tensor raw = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 };
        size_t in_w = 5;
        size_t in_h = 3;
        size_t depth = 1;
        size_t kernel_height = 2;
        size_t kernel_width = 3;
        size_t stride_w = 3;
        size_t stride_h = 1;
        size_t padding_w = 5;
        size_t padding_h = 0;

        work.add<DataLayer>("data", DataParams{ { "in" }, depth, in_h, in_w });
        auto params = Pool2DParams{ { "in" }, { "mp" }, kernel_width, kernel_height, stride_w, stride_h, padding_w, padding_h };
        EXPECT_THROW(MaxPoolLayer2D maxpool4("mp1", params, networkParameters), raul::Exception);

        memory_manager.clear();
    }
}

} // UT namespace