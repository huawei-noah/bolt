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

#include <training/base/initializers/RandomUniformInitializer.h>
#include <training/base/layers/composite/rnn/GRUCellLayer.h>
#include <training/compiler/Layers.h>

namespace UT
{

TEST(TestGRUCell, BuildUnit)
{
    PROFILE_TEST
    raul::Workflow netdef;
    netdef.add<raul::DataLayer>("fake_data_in", raul::DataParams{ { "in", "labels" }, 1, 1, 1, 1 });
    netdef.add<raul::DataLayer>("fake_data_state", raul::DataParams{ { "hidden" }, 1, 1, 1, 1 });
    raul::GRUCellLayer("GRU", raul::GRUCellParams{ "in", "hidden", "new_hidden", {} }, netdef.getNetworkParameters());
    netdef.preparePipelines();
    netdef.setBatchSize(1u);
    netdef.prepareMemoryForTraining();
    netdef.printInfo(std::cout);
}

// see gru_cell.py
TEST(TestGRUCell, SimpleZeroHiddenUnit)
{
    PROFILE_TEST
    // Test parameters
    const auto eps_rel = 1e-5_dt;
    const auto input_size = 9U;
    const auto hidden_size = 5U;
    const auto batch_size = 3U;

    const raul::Tensor input_init{ -1.47774279_dt, -1.75567114_dt, 0.07616609_dt, -1.07860339_dt, 1.44034219_dt, -0.11059419_dt,
        0.57686025_dt, -0.16917409_dt, -0.06402487_dt, 1.03842556_dt, 0.90682352_dt, 1.36152399_dt, 2.03717399_dt, 0.64304245_dt,
        -0.73256963_dt, -0.48771340_dt, -0.23395768_dt, 0.70731831_dt, 0.58004808_dt, 0.48257831_dt, -0.82978928_dt, 1.26783395_dt,
        0.27356258_dt, -0.61465430_dt, -0.02349441_dt, 1.17166984_dt, 0.39868718_dt };
    
    const raul::Tensor hidden_golden{ -0.53094757_dt, -0.53094757_dt, -0.53094757_dt, -0.53094757_dt, -0.53094757_dt, 0.00071675_dt,
        0.00071675_dt, 0.00071675_dt, 0.00071675_dt, 0.00071675_dt, 0.00895447_dt, 0.00895447_dt, 0.00895447_dt, 0.00895447_dt,
        0.00895447_dt };

    const raul::Tensor inputs_grad_golden{ 2.16085649_dt, 2.16085649_dt, 2.16085649_dt, 2.16085649_dt, 2.16085649_dt, 2.16085649_dt,
        2.16085649_dt, 2.16085649_dt, 2.16085649_dt, -0.00358115_dt, -0.00358115_dt, -0.00358115_dt, -0.00358115_dt, -0.00358115_dt,
        -0.00358115_dt, -0.00358115_dt, -0.00358115_dt, -0.00358115_dt, -0.04435632_dt, -0.04435632_dt, -0.04435632_dt, -0.04435632_dt,
        -0.04435632_dt, -0.04435632_dt, -0.04435632_dt, -0.04435632_dt, -0.04435632_dt };
    const raul::Tensor hidden_grad_golden{ 1.90440905_dt, 1.90440905_dt, 1.90440905_dt, 1.90440905_dt, 1.90440905_dt, 0.99570209_dt,
        0.99570209_dt, 0.99570209_dt, 0.99570209_dt, 0.99570209_dt, 0.94668758_dt, 0.94668758_dt, 0.94668758_dt, 0.94668758_dt,
        0.94668758_dt };

    // Initialization
    MANAGERS_DEFINE
    NETWORK_PARAMS_DEFINE(networkParameters);

    work.add<raul::DataLayer>("data1", raul::DataParams{ { "in" }, 1, 1, input_size });
    work.add<raul::DataLayer>("data2", raul::DataParams{ { "hidden" }, 1, 1, hidden_size });

    // Network
    const auto params = raul::GRUCellParams{ "in", "hidden", "new_hidden", {} };
    raul::GRUCellLayer("gru_cell", params, networkParameters);
    TENSORS_CREATE(batch_size)

    memory_manager["in"] = TORANGE(input_init);
    memory_manager["hidden"] = 0.0_dt;

    for (auto& [param, grad] : work.getTrainableParameters())
    {
        param = 1.0_dt;
    }

    // Forward
    ASSERT_NO_THROW(work.forwardPassTraining());

    // Checks
    const auto& hidden_input_tensor = memory_manager["hidden"];
    const auto& hidden_new_tensor = memory_manager["new_hidden"];

    EXPECT_EQ(hidden_input_tensor.size(), hidden_new_tensor.size());

    for (size_t i = 0; i < hidden_input_tensor.size(); ++i)
    {
        const auto hidden_val = hidden_new_tensor[i];
        const auto golden_hidden_val = hidden_golden[i];
        ASSERT_TRUE(tools::expect_near_relative(hidden_val, golden_hidden_val, eps_rel)) << "at " << i << ", expected: " << golden_hidden_val << ", got: " << hidden_val;
    }

    // Backward
    memory_manager[raul::Name("new_hidden").grad()] = 1.0_dt;
    ASSERT_NO_THROW(work.backwardPassTraining());

    // Checks
    const auto& inputs_grad = memory_manager[raul::Name("in").grad()];
    const auto& inputs = memory_manager["in"];

    EXPECT_EQ(inputs.size(), inputs_grad.size());

    for (size_t i = 0; i < inputs_grad.size(); ++i)
    {
        const auto val = inputs_grad[i];
        const auto golden_val = inputs_grad_golden[i];
        ASSERT_TRUE(tools::expect_near_relative(val, golden_val, eps_rel)) << "at " << i << ", expected: " << golden_val << ", got: " << val;
    }

    const auto& hidden_grad = memory_manager[raul::Name("hidden").grad()];

    EXPECT_EQ(hidden_grad.size(), hidden_input_tensor.size());

    for (size_t i = 0; i < hidden_grad.size(); ++i)
    {
        const auto val = hidden_grad[i];
        const auto golden_val = hidden_grad_golden[i];
        ASSERT_TRUE(tools::expect_near_relative(val, golden_val, eps_rel)) << "at " << i << ", expected: " << golden_val << ", got: " << val;
    }
}

TEST(TestGRUCell, SimpleRandomHiddenUnit)
{
    PROFILE_TEST
    // Test parameters
    const auto eps_rel = 1e-5_dt;
    const auto input_size = 4U;
    const auto hidden_size = 3U;
    const auto batch_size = 2U;

    const raul::Tensor input_init{ -0.13295190_dt, -0.04689599_dt, -0.28016463_dt, 0.54008639_dt, -0.14635307_dt, -0.15740128_dt, -1.01898205_dt, 0.02789201_dt };
    const raul::Tensor hidden_init{ 0.80468506_dt, -1.31859577_dt, -1.06609106_dt, -2.97711873_dt, 1.99564660_dt, -0.96829116_dt };
    
    const raul::Tensor hidden_golden{ 0.73356980_dt, -0.58812207_dt, -0.43094379_dt, -1.02896798_dt, 0.08311620_dt, -0.57972389_dt };

    const raul::Tensor inputs_grad_golden{ -0.19915806_dt, -0.19915806_dt, -0.19915806_dt, -0.19915806_dt, 1.42452824_dt, 1.42452824_dt, 1.42452824_dt, 1.42452824_dt };
    const raul::Tensor hidden_grad_golden{ 0.15815496_dt, 0.15815496_dt, 0.15815496_dt, 0.23563468_dt, 0.23563468_dt, 0.23563468_dt };

    // Initialization
    MANAGERS_DEFINE
    NETWORK_PARAMS_DEFINE(networkParameters);

    work.add<raul::DataLayer>("data1", raul::DataParams{ { "in" }, 1, 1, input_size });
    work.add<raul::DataLayer>("data2", raul::DataParams{ { "hidden" }, 1, 1, hidden_size });

    // Network
    const auto params = raul::GRUCellParams{ "in", "hidden", "new_hidden", {} };
    raul::GRUCellLayer("gru_cell", params, networkParameters);
    TENSORS_CREATE(batch_size)

    memory_manager["in"] = TORANGE(input_init);
    memory_manager["hidden"] = TORANGE(hidden_init);

    for (auto& [param, grad] : work.getTrainableParameters())
    {
        param = 1.0_dt;
    }

    // Forward
    ASSERT_NO_THROW(work.forwardPassTraining());

    // Checks
    const auto& hidden_input_tensor = memory_manager["hidden"];
    const auto& hidden_new_tensor = memory_manager["new_hidden"];

    EXPECT_EQ(hidden_input_tensor.size(), hidden_new_tensor.size());

    for (size_t i = 0; i < hidden_input_tensor.size(); ++i)
    {
        const auto hidden_val = hidden_new_tensor[i];
        const auto golden_hidden_val = hidden_golden[i];
        ASSERT_TRUE(tools::expect_near_relative(hidden_val, golden_hidden_val, eps_rel)) << "at " << i << ", expected: " << golden_hidden_val << ", got: " << hidden_val;
    }

    // Backward
    memory_manager[raul::Name("new_hidden").grad()] = 1.0_dt;
    ASSERT_NO_THROW(work.backwardPassTraining());

    // Checks
    const auto& inputs_grad = memory_manager[raul::Name("in").grad()];
    const auto& inputs = memory_manager["in"];

    EXPECT_EQ(inputs.size(), inputs_grad.size());

    for (size_t i = 0; i < inputs_grad.size(); ++i)
    {
        const auto val = inputs_grad[i];
        const auto golden_val = inputs_grad_golden[i];
        ASSERT_TRUE(tools::expect_near_relative(val, golden_val, eps_rel)) << "at " << i << ", expected: " << golden_val << ", got: " << val;
    }

    const auto& hidden_grad = memory_manager[raul::Name("hidden").grad()];

    EXPECT_EQ(hidden_grad.size(), hidden_input_tensor.size());

    for (size_t i = 0; i < hidden_grad.size(); ++i)
    {
        const auto val = hidden_grad[i];
        const auto golden_val = hidden_grad_golden[i];
        ASSERT_TRUE(tools::expect_near_relative(val, golden_val, eps_rel)) << "at " << i << ", expected: " << golden_val << ", got: " << val;
    }
}

TEST(TestGRUCell, RandomWeightsFusionOffAndOnUnit)
{
    PROFILE_TEST
    // Test parameters
    const auto eps_rel = 1e-5_dt;
    const auto input_size = 7U;
    const auto hidden_size = 4U;
    const auto batch_size = 3U;

    const raul::Tensor input_init{ -0.68704545_dt, -2.33485818_dt, 0.09404085_dt, -0.20208217_dt, 3.12574148_dt, 1.62500739_dt,
        -0.63396734_dt, 1.75850010_dt, -0.57933402_dt, 0.52056360_dt, 1.06761038_dt, -1.56271243_dt, -0.87330669_dt, 0.37735614_dt,
        -2.18030119_dt, -0.14689700_dt, 0.72165412_dt, 0.31223938_dt, 0.32089281_dt, -1.56860936_dt, 0.21175182_dt };
    const raul::Tensor hidden_init{ 1.20043528_dt, -0.15125844_dt, 0.89194465_dt, 0.69662333_dt, -0.15776138_dt, -1.64236963_dt,
        -0.39137629_dt, -1.31812286_dt, -1.75797606_dt, 0.10276648_dt, 0.26235840_dt, 0.50530910_dt };

    const raul::Tensor ih_weights{ -0.28185493_dt, -0.28055829_dt, -0.38474041_dt, 0.33566517_dt, 0.35546559_dt, -0.05690634_dt,
        -0.28934300_dt, 0.38645273_dt, 0.31974447_dt, 0.03716701_dt, -0.23606765_dt, 0.45952392_dt, 0.20447034_dt, -0.37957269_dt,
        0.47854143_dt, 0.37968689_dt, -0.18224543_dt, 0.28107727_dt, -0.28409451_dt, -0.07835716_dt, 0.42455059_dt, 0.02065957_dt,
        -0.35360909_dt, -0.16711646_dt, -0.13572258_dt, -0.09646440_dt, 0.04785031_dt, 0.46241480_dt, 0.02677017_dt, -0.30871999_dt,
        0.02562714_dt, 0.23974359_dt, 0.24802011_dt, -0.45696926_dt, -0.08947122_dt, -0.37157226_dt, -0.21334279_dt, 0.18014669_dt,
        -0.35506511_dt, 0.18586344_dt, 0.42438906_dt, 0.03279418_dt, -0.33324385_dt, -0.17914248_dt, 0.10918206_dt, -0.38115901_dt,
        0.24840516_dt, -0.45393479_dt, -0.48064667_dt, -0.48583031_dt, -0.10143167_dt, 0.33621645_dt, -0.47323948_dt, 0.41559356_dt,
        -0.20001143_dt, 0.14644206_dt, 0.02280146_dt, -0.45085955_dt, 0.41466451_dt, 0.26922172_dt, 0.49699783_dt, 0.25260609_dt,
        -0.33003449_dt, 0.41729188_dt, 0.02687222_dt, 0.23710823_dt, -0.40091455_dt, -0.14381325_dt, -0.49093878_dt, -0.19474626_dt,
        0.10786557_dt, -0.39258087_dt, 0.15938210_dt, 0.26840341_dt, 0.06965464_dt, -0.33454168_dt, -0.38765985_dt, -0.15425831_dt,
        0.21947908_dt, 0.49319822_dt, 0.28751451_dt, -0.05630463_dt, 0.17530823_dt, -0.49053144_dt };
    const raul::Tensor hh_weights{ -0.42705065_dt, 0.23330396_dt, -0.28320760_dt, 0.24054784_dt, -0.35296607_dt, -0.24765545_dt,
        -0.41184449_dt,0.26092035_dt, -0.05094755_dt, 0.38480055_dt, 0.30943608_dt, 0.27667129_dt, 0.01607805_dt, -0.15458900_dt,
        -0.10871583_dt, 0.06645030_dt, 0.24785477_dt, -0.35029495_dt, 0.41963893_dt, -0.05436504_dt, -0.41897279_dt, -0.27052891_dt,
        0.44240886_dt, 0.45726359_dt, -0.46313983_dt, 0.35264915_dt, 0.25057960_dt, 0.29595923_dt, 0.42326462_dt, -0.26947516_dt,
        0.15788788_dt, 0.20461661_dt, -0.14774668_dt, 0.16732657_dt, -0.14385670_dt, 0.30913067_dt, -0.13872731_dt, -0.18639785_dt,
        0.12587452_dt, 0.17734683_dt, -0.24428582_dt, 0.04419917_dt, 0.28976786_dt, -0.04974836_dt, 0.15216696_dt, -0.12059349_dt,
        0.17524981_dt, -0.36219710_dt };
    const raul::Tensor ih_biases{ -0.29401439_dt, -0.25379527_dt, 0.45950544_dt, -0.13454205_dt, -0.00136518_dt, -0.24224776_dt,
        0.49914503_dt, 0.48833507_dt, -0.37709332_dt, -0.40533495_dt, -0.37899649_dt, -0.00241137_dt };
    const raul::Tensor hh_biases{ -0.12745196_dt, -0.32727283_dt, -0.17933607_dt, 0.09446543_dt, -0.26124537_dt, 0.11079127_dt,
        -0.11465794_dt, -0.24228168_dt, 0.06869274_dt, 0.41112912_dt, -0.33803964_dt, 0.02321720_dt };

    const raul::Tensor hidden_golden{ 1.15116501_dt, -0.22627246_dt, 0.76003265_dt, 0.65564865_dt, -0.32599488_dt, 0.30409038_dt,
        0.07302547_dt, -0.07713708_dt, -0.94339150_dt, -0.01185670_dt, 0.25705278_dt, 0.43279916_dt };

    const raul::Tensor inputs_grad_golden{ -0.06205156_dt, -0.11707603_dt, 0.10811222_dt, -0.00926699_dt, 0.07639554_dt, -0.10442813_dt,
        -0.16265041_dt, 0.45669836_dt, -0.21845661_dt, 0.55357927_dt, 0.48653150_dt, -0.09047136_dt, -0.41427892_dt, -0.85915112_dt,
        -0.00800398_dt, -0.05152406_dt, 0.35865912_dt, -0.04100856_dt, 0.14345874_dt, 0.26601300_dt, -0.29337540_dt };
    const raul::Tensor hidden_grad_golden{ 0.71030843_dt, 0.88720351_dt, 0.87098521_dt, 0.98754054_dt, 0.62120128_dt, 0.10333990_dt,
        0.22900221_dt, -0.14070934_dt, 0.35126179_dt, 0.92843390_dt, 0.76738483_dt, 0.94897246_dt };
    const raul::Tensor ih_weights_grad_golden{ -0.11158133_dt, 0.00196455_dt, 0.02207936_dt, -0.00178930_dt, 0.03388711_dt,
        -0.05153515_dt, 0.00347505_dt, 0.02151464_dt, -0.03634772_dt, 0.04463237_dt, 0.05855070_dt, -0.05827445_dt, -0.08093993_dt,
        0.02229045_dt, -0.13203172_dt, 0.06505504_dt, -0.03921376_dt, -0.07903744_dt, 0.09153767_dt, 0.04898011_dt, -0.02265527_dt,
        0.22539201_dt, -0.06163184_dt, 0.03907458_dt, 0.10346460_dt, -0.16171342_dt, -0.05449111_dt, 0.03331917_dt, 1.14799678_dt,
        -0.08594241_dt, -0.26294976_dt, -0.02966374_dt, -0.19954802_dt, 0.66892284_dt, -0.07717165_dt, -0.69866252_dt, -0.01895342_dt,
        -0.06668114_dt, -0.26599085_dt, 0.64897746_dt, 0.20096782_dt, -0.12314733_dt, -0.21734332_dt, -0.19725603_dt, -0.02676402_dt,
        -0.10153750_dt, 0.44480217_dt, 0.22722758_dt, -0.09347940_dt, -0.63375926_dt, 0.06178473_dt, -0.09889315_dt, -0.28399488_dt,
        0.56805682_dt, 0.21391854_dt, -0.11618468_dt, -0.62966585_dt, -0.18863520_dt, 0.42599195_dt, 0.35003769_dt, -0.15434620_dt,
        -0.86832333_dt, 0.16704439_dt, 0.51606363_dt, -0.38505769_dt, 0.42970464_dt, 0.64234579_dt, -0.73804444_dt, -0.76220328_dt,
        0.23778228_dt, 1.17398190_dt, -0.94368589_dt, 0.47939503_dt, 0.84242839_dt, -0.62719125_dt, -0.46752310_dt, 0.19090149_dt,
        0.75417721_dt, -0.59985816_dt, 0.53401029_dt, 0.82055140_dt, -0.85075861_dt, -0.85590893_dt, 0.27495939_dt };
    const raul::Tensor hh_weights_grad_golden{ -0.06879749_dt, 0.02627936_dt, 0.01582104_dt, 0.03811130_dt, -0.05328522_dt,
        -0.07500345_dt, -0.00935309_dt, -0.04681793_dt, -0.00212387_dt, 0.12686367_dt, 0.02227901_dt, 0.09520691_dt, 0.02250028_dt,
        -0.17216827_dt, -0.04403837_dt, -0.14523764_dt, 0.81720275_dt, -0.23362128_dt, -0.12760787_dt, -0.34663787_dt, -0.03185679_dt,
        0.42914343_dt, 0.18589064_dt, 0.43685034_dt, 0.12731318_dt, 0.11139744_dt, 0.12358128_dt, 0.17601015_dt, -0.01004303_dt,
        0.45217478_dt, 0.15674014_dt, 0.41814232_dt, -0.58024323_dt, -0.01532696_dt, 0.07699983_dt, 0.12894915_dt, -0.09628728_dt,
        -0.22395962_dt, -0.03463840_dt, -0.15305349_dt, -0.09230665_dt, -0.88297045_dt, -0.18214068_dt, -0.67916596_dt, -0.18996060_dt,
        -0.71913588_dt, -0.11495291_dt, -0.50746447_dt };
    const raul::Tensor ih_biases_grad_golden{ 0.02681053_dt, 0.07684670_dt, -0.08346894_dt, 0.08473006_dt, -0.30082783_dt,
        -0.10661648_dt, 0.03127038_dt, -0.18045160_dt, 0.64852643_dt, 0.76234984_dt, 1.06390333_dt, 0.99692702_dt };
    const raul::Tensor hh_biases_grad_golden{ 0.02681053_dt, 0.07684670_dt, -0.08346894_dt, 0.08473006_dt, -0.30082783_dt,
        -0.10661648_dt, 0.03127038_dt, -0.18045160_dt, 0.36137265_dt, 0.19431981_dt, 0.58304089_dt, 0.57078493_dt };

    // Initialization
    for (size_t q = 0; q < 2; ++q)
    {
        MANAGERS_DEFINE
        NETWORK_PARAMS_DEFINE(networkParameters);

        work.add<raul::DataLayer>("data1", raul::DataParams{ { "in" }, 1, 1, input_size });
        work.add<raul::DataLayer>("data2", raul::DataParams{ { "hidden" }, 1, 1, hidden_size });

        // Network
        bool useFusion = (q == 1 ? true : false);
        auto params = raul::GRUCellParams{ "in", "hidden", "new_hidden", {}, true, true, false, useFusion };
        raul::GRUCellLayer("gru_cell", params, networkParameters);
        TENSORS_CREATE(batch_size)

        memory_manager["in"] = TORANGE(input_init);
        memory_manager["hidden"] = TORANGE(hidden_init);
        memory_manager[raul::Name("gru_cell") / "linear_ih" / "Weights"] = TORANGE(ih_weights);
        memory_manager[raul::Name("gru_cell") / "linear_hh" / "Weights"] = TORANGE(hh_weights);
        memory_manager[raul::Name("gru_cell") / "linear_ih" / "Biases"] = TORANGE(ih_biases);
        memory_manager[raul::Name("gru_cell") / "linear_hh" / "Biases"] = TORANGE(hh_biases);

        // Forward
        ASSERT_NO_THROW(work.forwardPassTraining());

        // Checks
        const auto& hidden_input_tensor = memory_manager["hidden"];
        const auto& hidden_new_tensor = memory_manager["new_hidden"];

        EXPECT_EQ(hidden_input_tensor.size(), hidden_new_tensor.size());

        for (size_t i = 0; i < hidden_input_tensor.size(); ++i)
        {
            const auto hidden_val = hidden_new_tensor[i];
            const auto golden_hidden_val = hidden_golden[i];
            ASSERT_TRUE(tools::expect_near_relative(hidden_val, golden_hidden_val, eps_rel)) << "at " << i << ", expected: " << golden_hidden_val << ", got: " << hidden_val;
        }

        // Backward
        memory_manager[raul::Name("new_hidden").grad()] = 1.0_dt;
        ASSERT_NO_THROW(work.backwardPassTraining());

        // Checks
        const auto& inputs_grad = memory_manager[raul::Name("in").grad()];
        const auto& inputs = memory_manager["in"];

        EXPECT_EQ(inputs.size(), inputs_grad.size());

        for (size_t i = 0; i < inputs_grad.size(); ++i)
        {
            const auto val = inputs_grad[i];
            const auto golden_val = inputs_grad_golden[i];
            ASSERT_TRUE(tools::expect_near_relative(val, golden_val, eps_rel)) << "at " << i << ", expected: " << golden_val << ", got: " << val;
        }

        const auto& hidden_grad = memory_manager[raul::Name("hidden").grad()];

        EXPECT_EQ(hidden_grad.size(), hidden_input_tensor.size());

        for (size_t i = 0; i < hidden_grad.size(); ++i)
        {
            const auto val = hidden_grad[i];
            const auto golden_val = hidden_grad_golden[i];
            ASSERT_TRUE(tools::expect_near_relative(val, golden_val, eps_rel)) << "at " << i << ", expected: " << golden_val << ", got: " << val;
        }

        const auto ih_weights_name = raul::Name("gru_cell") / raul::Name("linear_ih") / "Weights";
        const auto& ih_weights_grad = memory_manager[ih_weights_name.grad()];

        EXPECT_EQ(ih_weights_grad.size(), ih_weights.size());

        for (size_t i = 0; i < ih_weights_grad.size(); ++i)
        {
            const auto val = ih_weights_grad[i];
            const auto golden_val = ih_weights_grad_golden[i];
            ASSERT_TRUE(tools::expect_near_relative(val, golden_val, eps_rel)) << "at " << i << ", expected: " << golden_val << ", got: " << val;
        }

        const auto hh_weights_name = raul::Name("gru_cell") / raul::Name("linear_hh") / "Weights";
        const auto& hh_weights_grad = memory_manager[hh_weights_name.grad()];

        EXPECT_EQ(hh_weights_grad.size(), hh_weights.size());

        for (size_t i = 0; i < hh_weights_grad.size(); ++i)
        {
            const auto val = hh_weights_grad[i];
            const auto golden_val = hh_weights_grad_golden[i];
            ASSERT_TRUE(tools::expect_near_relative(val, golden_val, eps_rel)) << "at " << i << ", expected: " << golden_val << ", got: " << val;
        }

        const auto ih_biases_name = raul::Name("gru_cell") / raul::Name("linear_ih") / "Biases";
        const auto& ih_biases_grad = memory_manager[ih_biases_name.grad()];

        EXPECT_EQ(ih_biases_grad.size(), ih_biases.size());

        for (size_t i = 0; i < ih_biases_grad.size(); ++i)
        {
            const auto val = ih_biases_grad[i];
            const auto golden_val = ih_biases_grad_golden[i];
            ASSERT_TRUE(tools::expect_near_relative(val, golden_val, eps_rel)) << "at " << i << ", expected: " << golden_val << ", got: " << val;
        }

        const auto hh_biases_name = raul::Name("gru_cell") / raul::Name("linear_hh") / "Biases";
        const auto& hh_biases_grad = memory_manager[hh_biases_name.grad()];

        EXPECT_EQ(hh_biases_grad.size(), hh_biases.size());

        for (size_t i = 0; i < hh_biases_grad.size(); ++i)
        {
            const auto val = hh_biases_grad[i];
            const auto golden_val = hh_biases_grad_golden[i];
            ASSERT_TRUE(tools::expect_near_relative(val, golden_val, eps_rel)) << "at " << i << ", expected: " << golden_val << ", got: " << val;
        }
    }
}

#ifdef ANDROID
TEST(TestGRUCell, RandomWeightsFusionOnFP16Unit)
{
    PROFILE_TEST
    // Test parameters
    const auto eps_rel = 0.1_hf;
    const auto input_size = 7U;
    const auto hidden_size = 4U;
    const auto batch_size = 3U;

    const raul::TensorFP16 input_init{ -0.68704545_hf, -2.33485818_hf, 0.09404085_hf, -0.20208217_hf, 3.12574148_hf, 1.62500739_hf,
        -0.63396734_hf, 1.75850010_hf, -0.57933402_hf, 0.52056360_hf, 1.06761038_hf, -1.56271243_hf, -0.87330669_hf, 0.37735614_hf,
        -2.18030119_hf, -0.14689700_hf, 0.72165412_hf, 0.31223938_hf, 0.32089281_hf, -1.56860936_hf, 0.21175182_hf };
    const raul::TensorFP16 hidden_init{ 1.20043528_hf, -0.15125844_hf, 0.89194465_hf, 0.69662333_hf, -0.15776138_hf, -1.64236963_hf,
        -0.39137629_hf, -1.31812286_hf, -1.75797606_hf, 0.10276648_hf, 0.26235840_hf, 0.50530910_hf };

    const raul::TensorFP16 ih_weights{ -0.28185493_hf, -0.28055829_hf, -0.38474041_hf, 0.33566517_hf, 0.35546559_hf, -0.05690634_hf,
        -0.28934300_hf, 0.38645273_hf, 0.31974447_hf, 0.03716701_hf, -0.23606765_hf, 0.45952392_hf, 0.20447034_hf, -0.37957269_hf,
        0.47854143_hf, 0.37968689_hf, -0.18224543_hf, 0.28107727_hf, -0.28409451_hf, -0.07835716_hf, 0.42455059_hf, 0.02065957_hf,
        -0.35360909_hf, -0.16711646_hf, -0.13572258_hf, -0.09646440_hf, 0.04785031_hf, 0.46241480_hf, 0.02677017_hf, -0.30871999_hf,
        0.02562714_hf, 0.23974359_hf, 0.24802011_hf, -0.45696926_hf, -0.08947122_hf, -0.37157226_hf, -0.21334279_hf, 0.18014669_hf,
        -0.35506511_hf, 0.18586344_hf, 0.42438906_hf, 0.03279418_hf, -0.33324385_hf, -0.17914248_hf, 0.10918206_hf, -0.38115901_hf,
        0.24840516_hf, -0.45393479_hf, -0.48064667_hf, -0.48583031_hf, -0.10143167_hf, 0.33621645_hf, -0.47323948_hf, 0.41559356_hf,
        -0.20001143_hf, 0.14644206_hf, 0.02280146_hf, -0.45085955_hf, 0.41466451_hf, 0.26922172_hf, 0.49699783_hf, 0.25260609_hf,
        -0.33003449_hf, 0.41729188_hf, 0.02687222_hf, 0.23710823_hf, -0.40091455_hf, -0.14381325_hf, -0.49093878_hf, -0.19474626_hf,
        0.10786557_hf, -0.39258087_hf, 0.15938210_hf, 0.26840341_hf, 0.06965464_hf, -0.33454168_hf, -0.38765985_hf, -0.15425831_hf,
        0.21947908_hf, 0.49319822_hf, 0.28751451_hf, -0.05630463_hf, 0.17530823_hf, -0.49053144_hf };
    const raul::TensorFP16 hh_weights{ -0.42705065_hf, 0.23330396_hf, -0.28320760_hf, 0.24054784_hf, -0.35296607_hf, -0.24765545_hf,
        -0.41184449_hf,0.26092035_hf, -0.05094755_hf, 0.38480055_hf, 0.30943608_hf, 0.27667129_hf, 0.01607805_hf, -0.15458900_hf,
        -0.10871583_hf, 0.06645030_hf, 0.24785477_hf, -0.35029495_hf, 0.41963893_hf, -0.05436504_hf, -0.41897279_hf, -0.27052891_hf,
        0.44240886_hf, 0.45726359_hf, -0.46313983_hf, 0.35264915_hf, 0.25057960_hf, 0.29595923_hf, 0.42326462_hf, -0.26947516_hf,
        0.15788788_hf, 0.20461661_hf, -0.14774668_hf, 0.16732657_hf, -0.14385670_hf, 0.30913067_hf, -0.13872731_hf, -0.18639785_hf,
        0.12587452_hf, 0.17734683_hf, -0.24428582_hf, 0.04419917_hf, 0.28976786_hf, -0.04974836_hf, 0.15216696_hf, -0.12059349_hf,
        0.17524981_hf, -0.36219710_hf };
    const raul::TensorFP16 ih_biases{ -0.29401439_hf, -0.25379527_hf, 0.45950544_hf, -0.13454205_hf, -0.00136518_hf, -0.24224776_hf,
        0.49914503_hf, 0.48833507_hf, -0.37709332_hf, -0.40533495_hf, -0.37899649_hf, -0.00241137_hf };
    const raul::TensorFP16 hh_biases{ -0.12745196_hf, -0.32727283_hf, -0.17933607_hf, 0.09446543_hf, -0.26124537_hf, 0.11079127_hf,
        -0.11465794_hf, -0.24228168_hf, 0.06869274_hf, 0.41112912_hf, -0.33803964_hf, 0.02321720_hf };

    const raul::TensorFP16 hidden_golden{ 1.15116501_hf, -0.22627246_hf, 0.76003265_hf, 0.65564865_hf, -0.32599488_hf, 0.30409038_hf,
        0.07302547_hf, -0.07713708_hf, -0.94339150_hf, -0.01185670_hf, 0.25705278_hf, 0.43279916_hf };

    const raul::TensorFP16 inputs_grad_golden{ -0.06205156_hf, -0.11707603_hf, 0.10811222_hf, -0.00926699_hf, 0.07639554_hf, -0.10442813_hf,
        -0.16265041_hf, 0.45669836_hf, -0.21845661_hf, 0.55357927_hf, 0.48653150_hf, -0.09047136_hf, -0.41427892_hf, -0.85915112_hf,
        -0.00800398_hf, -0.05152406_hf, 0.35865912_hf, -0.04100856_hf, 0.14345874_hf, 0.26601300_hf, -0.29337540_hf };
    const raul::TensorFP16 hidden_grad_golden{ 0.71030843_hf, 0.88720351_hf, 0.87098521_hf, 0.98754054_hf, 0.62120128_hf, 0.10333990_hf,
        0.22900221_hf, -0.14070934_hf, 0.35126179_hf, 0.92843390_hf, 0.76738483_hf, 0.94897246_hf };
    const raul::TensorFP16 ih_weights_grad_golden{ -0.11158133_hf, 0.00196455_hf, 0.02207936_hf, -0.00178930_hf, 0.03388711_hf,
        -0.05153515_hf, 0.00347505_hf, 0.02151464_hf, -0.03634772_hf, 0.04463237_hf, 0.05855070_hf, -0.05827445_hf, -0.08093993_hf,
        0.02229045_hf, -0.13203172_hf, 0.06505504_hf, -0.03921376_hf, -0.07903744_hf, 0.09153767_hf, 0.04898011_hf, -0.02265527_hf,
        0.22539201_hf, -0.06163184_hf, 0.03907458_hf, 0.10346460_hf, -0.16171342_hf, -0.05449111_hf, 0.03331917_hf, 1.14799678_hf,
        -0.08594241_hf, -0.26294976_hf, -0.02966374_hf, -0.19954802_hf, 0.66892284_hf, -0.07717165_hf, -0.69866252_hf, -0.01895342_hf,
        -0.06668114_hf, -0.26599085_hf, 0.64897746_hf, 0.20096782_hf, -0.12314733_hf, -0.21734332_hf, -0.19725603_hf, -0.02676402_hf,
        -0.10153750_hf, 0.44480217_hf, 0.22722758_hf, -0.09347940_hf, -0.63375926_hf, 0.06178473_hf, -0.09889315_hf, -0.28399488_hf,
        0.56805682_hf, 0.21391854_hf, -0.11618468_hf, -0.62966585_hf, -0.18863520_hf, 0.42599195_hf, 0.35003769_hf, -0.15434620_hf,
        -0.86832333_hf, 0.16704439_hf, 0.51606363_hf, -0.38505769_hf, 0.42970464_hf, 0.64234579_hf, -0.73804444_hf, -0.76220328_hf,
        0.23778228_hf, 1.17398190_hf, -0.94368589_hf, 0.47939503_hf, 0.84242839_hf, -0.62719125_hf, -0.46752310_hf, 0.19090149_hf,
        0.75417721_hf, -0.59985816_hf, 0.53401029_hf, 0.82055140_hf, -0.85075861_hf, -0.85590893_hf, 0.27495939_hf };
    const raul::TensorFP16 hh_weights_grad_golden{ -0.06879749_hf, 0.02627936_hf, 0.01582104_hf, 0.03811130_hf, -0.05328522_hf,
        -0.07500345_hf, -0.00935309_hf, -0.04681793_hf, -0.00212387_hf, 0.12686367_hf, 0.02227901_hf, 0.09520691_hf, 0.02250028_hf,
        -0.17216827_hf, -0.04403837_hf, -0.14523764_hf, 0.81720275_hf, -0.23362128_hf, -0.12760787_hf, -0.34663787_hf, -0.03185679_hf,
        0.42914343_hf, 0.18589064_hf, 0.43685034_hf, 0.12731318_hf, 0.11139744_hf, 0.12358128_hf, 0.17601015_hf, -0.01004303_hf,
        0.45217478_hf, 0.15674014_hf, 0.41814232_hf, -0.58024323_hf, -0.01532696_hf, 0.07699983_hf, 0.12894915_hf, -0.09628728_hf,
        -0.22395962_hf, -0.03463840_hf, -0.15305349_hf, -0.09230665_hf, -0.88297045_hf, -0.18214068_hf, -0.67916596_hf, -0.18996060_hf,
        -0.71913588_hf, -0.11495291_hf, -0.50746447_hf };
    const raul::TensorFP16 ih_biases_grad_golden{ 0.02681053_hf, 0.07684670_hf, -0.08346894_hf, 0.08473006_hf, -0.30082783_hf,
        -0.10661648_hf, 0.03127038_hf, -0.18045160_hf, 0.64852643_hf, 0.76234984_hf, 1.06390333_hf, 0.99692702_hf };
    const raul::TensorFP16 hh_biases_grad_golden{ 0.02681053_hf, 0.07684670_hf, -0.08346894_hf, 0.08473006_hf, -0.30082783_hf,
        -0.10661648_hf, 0.03127038_hf, -0.18045160_hf, 0.36137265_hf, 0.19431981_hf, 0.58304089_hf, 0.57078493_hf };

    // Initialization
    for (size_t q = 0; q < 2; ++q)
    {
        raul::WorkflowEager work{ raul::CompressionMode::NONE, raul::CalculationMode::DETERMINISTIC, raul::AllocationMode::STANDARD, raul::ExecutionTarget::CPUFP16 };
        auto& memory_manager = work.getMemoryManager<raul::MemoryManagerFP16>();
        NETWORK_PARAMS_DEFINE(networkParameters);

        work.add<raul::DataLayer>("data1", raul::DataParams{ { "in" }, 1, 1, input_size });
        work.add<raul::DataLayer>("data2", raul::DataParams{ { "hidden" }, 1, 1, hidden_size });

        // Network
        bool useFusion = (q == 1 ? true : false);
        auto params = raul::GRUCellParams{ "in", "hidden", "new_hidden", {}, true, true, false, useFusion };
        raul::GRUCellLayer("gru_cell", params, networkParameters);
        TENSORS_CREATE(batch_size)

        memory_manager["in"] = TORANGE_FP16(input_init);
        memory_manager["hidden"] = TORANGE_FP16(hidden_init);
        memory_manager[raul::Name("gru_cell") / "linear_ih" / "Weights"] = TORANGE_FP16(ih_weights);
        memory_manager[raul::Name("gru_cell") / "linear_hh" / "Weights"] = TORANGE_FP16(hh_weights);
        memory_manager[raul::Name("gru_cell") / "linear_ih" / "Biases"] = TORANGE_FP16(ih_biases);
        memory_manager[raul::Name("gru_cell") / "linear_hh" / "Biases"] = TORANGE_FP16(hh_biases);

        // Forward
        ASSERT_NO_THROW(work.forwardPassTraining());

        // Checks
        const auto& hidden_input_tensor = memory_manager["hidden"];
        const auto& hidden_new_tensor = memory_manager["new_hidden"];

        EXPECT_EQ(hidden_input_tensor.size(), hidden_new_tensor.size());

        for (size_t i = 0; i < hidden_input_tensor.size(); ++i)
        {
            const auto hidden_val = hidden_new_tensor[i];
            const auto golden_hidden_val = hidden_golden[i];
            ASSERT_TRUE(tools::expect_near_relative(hidden_val, golden_hidden_val, eps_rel)) << "at " << i << ", expected: " << TODTYPE(golden_hidden_val) << ", got: " << TODTYPE(hidden_val);
        }

        // Backward
        memory_manager[raul::Name("new_hidden").grad()] = 1.0_hf;
        ASSERT_NO_THROW(work.backwardPassTraining());

        // Checks
        const auto& inputs_grad = memory_manager[raul::Name("in").grad()];
        const auto& inputs = memory_manager["in"];

        EXPECT_EQ(inputs.size(), inputs_grad.size());

        for (size_t i = 0; i < inputs_grad.size(); ++i)
        {
            const auto val = inputs_grad[i];
            const auto golden_val = inputs_grad_golden[i];
            ASSERT_TRUE(tools::expect_near_relative(val, golden_val, eps_rel)) << "at " << i << ", expected: " << TODTYPE(golden_val) << ", got: " << TODTYPE(val);
        }

        const auto& hidden_grad = memory_manager[raul::Name("hidden").grad()];

        EXPECT_EQ(hidden_grad.size(), hidden_input_tensor.size());

        for (size_t i = 0; i < hidden_grad.size(); ++i)
        {
            const auto val = hidden_grad[i];
            const auto golden_val = hidden_grad_golden[i];
            ASSERT_TRUE(tools::expect_near_relative(val, golden_val, eps_rel)) << "at " << i << ", expected: " << TODTYPE(golden_val) << ", got: " << TODTYPE(val);
        }

        const auto& ih_weights_grad = memory_manager[raul::Name(raul::Name("gru_cell") / "linear_ih" / "Weights").grad()];

        EXPECT_EQ(ih_weights_grad.size(), ih_weights.size());

        for (size_t i = 0; i < ih_weights_grad.size(); ++i)
        {
            const auto val = ih_weights_grad[i];
            const auto golden_val = ih_weights_grad_golden[i];
            ASSERT_TRUE(tools::expect_near_relative(val, golden_val, eps_rel)) << "at " << i << ", expected: " << TODTYPE(golden_val) << ", got: " << TODTYPE(val);
        }

        const auto& hh_weights_grad = memory_manager[raul::Name(raul::Name("gru_cell") / "linear_hh" / "Weights").grad()];

        EXPECT_EQ(hh_weights_grad.size(), hh_weights.size());

        for (size_t i = 0; i < hh_weights_grad.size(); ++i)
        {
            const auto val = hh_weights_grad[i];
            const auto golden_val = hh_weights_grad_golden[i];
            ASSERT_TRUE(tools::expect_near_relative(val, golden_val, eps_rel)) << "at " << i << ", expected: " << TODTYPE(golden_val) << ", got: " << TODTYPE(val);
        }

        const auto& ih_biases_grad = memory_manager[raul::Name(raul::Name("gru_cell") / "linear_ih" / "Biases").grad()];

        EXPECT_EQ(ih_biases_grad.size(), ih_biases.size());

        for (size_t i = 0; i < ih_biases_grad.size(); ++i)
        {
            const auto val = ih_biases_grad[i];
            const auto golden_val = ih_biases_grad_golden[i];
            ASSERT_TRUE(tools::expect_near_relative(val, golden_val, eps_rel)) << "at " << i << ", expected: " << TODTYPE(golden_val) << ", got: " << TODTYPE(val);
        }

        const auto& hh_biases_grad = memory_manager[raul::Name(raul::Name("gru_cell") / "linear_hh" / "Biases").grad()];

        EXPECT_EQ(hh_biases_grad.size(), hh_biases.size());

        for (size_t i = 0; i < hh_biases_grad.size(); ++i)
        {
            const auto val = hh_biases_grad[i];
            const auto golden_val = hh_biases_grad_golden[i];
            ASSERT_TRUE(tools::expect_near_relative(val, golden_val, eps_rel)) << "at " << i << ", expected: " << TODTYPE(golden_val) << ", got: " << TODTYPE(val);
        }
    }
}
#endif // ANDROID

}