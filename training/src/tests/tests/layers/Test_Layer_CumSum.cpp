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

#include <training/base/layers/basic/CumSumLayer.h>
#include <training/base/layers/basic/DataLayer.h>
#include <training/compiler/Workflow.h>

namespace UT
{

TEST(TestLayerCumSum, InputNumExceedsUnit)
{
    PROFILE_TEST

    // Initialization
    MANAGERS_DEFINE
    NETWORK_PARAMS_DEFINE(networkParameters);

    // Apply function
    ASSERT_THROW(raul::CumSumLayer("csum", raul::BasicParamsWithDim{ { "x", "y" }, { "x_out" }, raul::Dimension::Width }, networkParameters), raul::Exception);
}

TEST(TestLayerCumSum, OutputNumExceedsUnit)
{
    PROFILE_TEST

    // Initialization
    MANAGERS_DEFINE
    NETWORK_PARAMS_DEFINE(networkParameters);

    // Apply function
    ASSERT_THROW(raul::CumSumLayer("csum", raul::BasicParamsWithDim{ { "x" }, { "x_out", "y_out" }, raul::Dimension::Width }, networkParameters), raul::Exception);
}

TEST(TestLayerCumSum, IncorrectDimensionUnit)
{
    PROFILE_TEST

    // Initialization
    MANAGERS_DEFINE
    NETWORK_PARAMS_DEFINE(networkParameters);

    // Apply function
    ASSERT_THROW(raul::CumSumLayer("csum", raul::BasicParamsWithDim{ { "x" }, { "out" }, raul::Dimension::Default }, networkParameters), raul::Exception);
}

TEST(TestLayerCumSum, ForwardSimpleUnit)
{
    PROFILE_TEST
    // Test parameters
    const auto eps_rel = TODTYPE(1e-3);

    const raul::Tensor x{ -0.8286_dt, -0.4890_dt, 0.5155_dt, 0.8443_dt, 0.1865_dt, -0.1752_dt, -2.0595_dt, 0.1850_dt, -1.1571_dt, -0.4243_dt };
    const raul::Tensor realOut{ -0.8286_dt, -1.3176_dt, -0.8020_dt, 0.0422_dt, 0.2289_dt, 0.0535_dt, -2.0058_dt, -1.8209_dt, -2.9780_dt, -3.4022_dt };

    // Initialization
    MANAGERS_DEFINE
    NETWORK_PARAMS_DEFINE(networkParameters);

    work.tensorNeeded("x", "x", raul::WShape{ raul::BS(), 1u, 1u, 1u }, DEC_FORW_READ_NOMEMOPT);

    // Apply function
    raul::CumSumLayer cumsum("csum", raul::BasicParamsWithDim{ { "x" }, { "out" }, raul::Dimension::Batch }, networkParameters);
    TENSORS_CREATE(10);
    memory_manager["x"] = TORANGE(x);

    cumsum.forwardCompute(raul::NetworkMode::Test);

    // Checks
    const auto& xTensor = memory_manager["x"];
    const auto& outTensor = memory_manager["out"];

    EXPECT_EQ(outTensor.size(), xTensor.size());

    for (size_t i = 0; i < outTensor.size(); ++i)
    {
        ASSERT_TRUE(tools::expect_near_relative(outTensor[i], realOut[i], eps_rel));
    }
}

TEST(TestLayerCumSum, BackwardSimpleUnit)
{
    PROFILE_TEST
    // Test parameters
    const auto eps_rel = TODTYPE(1e-3);

    const raul::Tensor x{ -0.8286_dt, -0.4890_dt, 0.5155_dt, 0.8443_dt, 0.1865_dt, -0.1752_dt, -2.0595_dt, 0.1850_dt, -1.1571_dt, -0.4243_dt };
    const raul::Tensor deltas{ 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt };
    const raul::Tensor realGrad{ 10.0_dt, 9.0_dt, 8.0_dt, 7.0_dt, 6.0_dt, 5.0_dt, 4.0_dt, 3.0_dt, 2.0_dt, 1.0_dt };

    // Initialization
    MANAGERS_DEFINE
    NETWORK_PARAMS_DEFINE(networkParameters);

    work.tensorNeeded("x", "x", raul::WShape{ raul::BS(), 1u, 1u, 1u }, DEC_FORW_READ_NOMEMOPT);

    // Apply function
    raul::CumSumLayer cumsum("csum", raul::BasicParamsWithDim{ { "x" }, { "out" }, raul::Dimension::Batch }, networkParameters);
    TENSORS_CREATE(10);
    memory_manager["x"] = TORANGE(x);
    memory_manager[raul::Name("out").grad()] = TORANGE(deltas);

    cumsum.forwardCompute(raul::NetworkMode::Test);
    cumsum.backwardCompute();

    // Checks
    const auto& xNablaTensor = memory_manager[raul::Name("x").grad()];
    const auto& xTensor = memory_manager["x"];

    EXPECT_EQ(xTensor.size(), xNablaTensor.size());

    for (size_t i = 0; i < xNablaTensor.size(); ++i)
    {
        ASSERT_TRUE(tools::expect_near_relative(xNablaTensor[i], realGrad[i], eps_rel));
    }
}

TEST(TestLayerCumSum, ForwardUnit)
{
    PROFILE_TEST
    // Test parameters
    const auto eps_rel = TODTYPE(1e-6);
    const auto batch = 2;
    const auto depth = 3;
    const auto height = 4;
    const auto width = 5;

    const raul::Tensor x{ 0.1033096910_dt, 0.9701558948_dt, 0.9480701089_dt, 0.9787087440_dt, 0.3267636895_dt, 0.9200272560_dt, 0.0502749681_dt, 0.4583170414_dt, 0.8607905507_dt, 0.4881546497_dt,
                          0.7980576158_dt, 0.2011045814_dt, 0.1913603544_dt, 0.8979360461_dt, 0.9541048408_dt, 0.5241690278_dt, 0.6006127000_dt, 0.9887800217_dt, 0.7473886609_dt, 0.5498082042_dt,
                          0.0670150518_dt, 0.1167818904_dt, 0.1723778248_dt, 0.9939703345_dt, 0.6243668795_dt, 0.3656120300_dt, 0.5017478466_dt, 0.2137093544_dt, 0.8107876778_dt, 0.7783825397_dt,
                          0.2362361550_dt, 0.2898594737_dt, 0.3328117728_dt, 0.9092149138_dt, 0.2501674891_dt, 0.6224393249_dt, 0.9649521708_dt, 0.5299566984_dt, 0.2069533467_dt, 0.6873005629_dt,
                          0.1918165684_dt, 0.8134448528_dt, 0.9125209451_dt, 0.9396399260_dt, 0.8208933473_dt, 0.4034467340_dt, 0.9324436188_dt, 0.2018597722_dt, 0.9788960814_dt, 0.4333596826_dt,
                          0.7238065600_dt, 0.8973705173_dt, 0.0776838064_dt, 0.6971374750_dt, 0.3664962053_dt, 0.0779988170_dt, 0.3857882619_dt, 0.3668601513_dt, 0.7975063324_dt, 0.9332120419_dt,
                          0.8668168187_dt, 0.5823799968_dt, 0.3222199082_dt, 0.5328013897_dt, 0.0239760280_dt, 0.6003485918_dt, 0.8691412807_dt, 0.3132150769_dt, 0.1712092757_dt, 0.2083655000_dt,
                          0.6775689721_dt, 0.6496044993_dt, 0.0529118180_dt, 0.7317054272_dt, 0.3720138669_dt, 0.3189361095_dt, 0.8919355273_dt, 0.7041678429_dt, 0.7928147316_dt, 0.6565824151_dt,
                          0.7744513750_dt, 0.8949885964_dt, 0.6901841164_dt, 0.9020239711_dt, 0.3684692383_dt, 0.5173735023_dt, 0.8764913678_dt, 0.2990424037_dt, 0.9684888721_dt, 0.0940009356_dt,
                          0.7392263412_dt, 0.6003669500_dt, 0.6738508344_dt, 0.3602285385_dt, 0.8780175447_dt, 0.6230656505_dt, 0.3569628000_dt, 0.8145191073_dt, 0.6073390245_dt, 0.5124547482_dt,
                          0.6408753395_dt, 0.1860215068_dt, 0.5974498987_dt, 0.1584112048_dt, 0.1544559598_dt, 0.8474228978_dt, 0.3584001660_dt, 0.6629422307_dt, 0.4294191003_dt, 0.4718081951_dt,
                          0.3983595371_dt, 0.7621403337_dt, 0.7940700650_dt, 0.6270959973_dt, 0.3249167800_dt, 0.9852560759_dt, 0.9440631270_dt, 0.6515852809_dt, 0.2359522581_dt, 0.1550757289_dt };

    raul::Dimension dimensions[]{ raul::Dimension::Batch, raul::Dimension::Depth, raul::Dimension::Height, raul::Dimension::Width };
    const raul::Tensor realOut[]{
        { 0.1033096910_dt, 0.9701558948_dt, 0.9480701089_dt, 0.9787087440_dt, 0.3267636895_dt, 0.9200272560_dt, 0.0502749681_dt, 0.4583170414_dt, 0.8607905507_dt, 0.4881546497_dt, 0.7980576158_dt,
          0.2011045814_dt, 0.1913603544_dt, 0.8979360461_dt, 0.9541048408_dt, 0.5241690278_dt, 0.6006127000_dt, 0.9887800217_dt, 0.7473886609_dt, 0.5498082042_dt, 0.0670150518_dt, 0.1167818904_dt,
          0.1723778248_dt, 0.9939703345_dt, 0.6243668795_dt, 0.3656120300_dt, 0.5017478466_dt, 0.2137093544_dt, 0.8107876778_dt, 0.7783825397_dt, 0.2362361550_dt, 0.2898594737_dt, 0.3328117728_dt,
          0.9092149138_dt, 0.2501674891_dt, 0.6224393249_dt, 0.9649521708_dt, 0.5299566984_dt, 0.2069533467_dt, 0.6873005629_dt, 0.1918165684_dt, 0.8134448528_dt, 0.9125209451_dt, 0.9396399260_dt,
          0.8208933473_dt, 0.4034467340_dt, 0.9324436188_dt, 0.2018597722_dt, 0.9788960814_dt, 0.4333596826_dt, 0.7238065600_dt, 0.8973705173_dt, 0.0776838064_dt, 0.6971374750_dt, 0.3664962053_dt,
          0.0779988170_dt, 0.3857882619_dt, 0.3668601513_dt, 0.7975063324_dt, 0.9332120419_dt, 0.9701265097_dt, 1.5525358915_dt, 1.2702900171_dt, 1.5115101337_dt, 0.3507397175_dt, 1.5203758478_dt,
          0.9194162488_dt, 0.7715321183_dt, 1.0319998264_dt, 0.6965201497_dt, 1.4756265879_dt, 0.8507090807_dt, 0.2442721725_dt, 1.6296415329_dt, 1.3261187077_dt, 0.8431051373_dt, 1.4925482273_dt,
          1.6929478645_dt, 1.5402033329_dt, 1.2063906193_dt, 0.8414664268_dt, 1.0117704868_dt, 0.8625619411_dt, 1.8959943056_dt, 0.9928361177_dt, 0.8829855323_dt, 1.3782391548_dt, 0.5127517581_dt,
          1.7792766094_dt, 0.8723834753_dt, 0.9754624963_dt, 0.8902264237_dt, 1.0066626072_dt, 1.2694435120_dt, 1.1281850338_dt, 1.2455049753_dt, 1.3219149113_dt, 1.3444757462_dt, 0.8142923713_dt,
          1.1997553110_dt, 0.8326919079_dt, 0.9994663596_dt, 1.5099709034_dt, 1.0980510712_dt, 0.9753493071_dt, 1.2508696318_dt, 1.2908437252_dt, 0.8648020029_dt, 1.4083151817_dt, 0.9051678777_dt,
          1.1221661568_dt, 1.6595108509_dt, 0.8717538714_dt, 1.3242335320_dt, 0.6914129853_dt, 1.0632548332_dt, 1.3298513889_dt, 1.0184454918_dt, 1.0334585905_dt, 1.0882878304_dt },
        { 0.1033096910_dt, 0.9701558948_dt, 0.9480701089_dt, 0.9787087440_dt, 0.3267636895_dt, 0.9200272560_dt, 0.0502749681_dt, 0.4583170414_dt, 0.8607905507_dt, 0.4881546497_dt, 0.7980576158_dt,
          0.2011045814_dt, 0.1913603544_dt, 0.8979360461_dt, 0.9541048408_dt, 0.5241690278_dt, 0.6006127000_dt, 0.9887800217_dt, 0.7473886609_dt, 0.5498082042_dt, 0.1703247428_dt, 1.0869377851_dt,
          1.1204478741_dt, 1.9726791382_dt, 0.9511305690_dt, 1.2856392860_dt, 0.5520228148_dt, 0.6720263958_dt, 1.6715781689_dt, 1.2665371895_dt, 1.0342937708_dt, 0.4909640551_dt, 0.5241721272_dt,
          1.8071509600_dt, 1.2042722702_dt, 1.1466083527_dt, 1.5655648708_dt, 1.5187367201_dt, 0.9543420076_dt, 1.2371087074_dt, 0.3621413112_dt, 1.9003826380_dt, 2.0329689980_dt, 2.9123189449_dt,
          1.7720239162_dt, 1.6890859604_dt, 1.4844664335_dt, 0.8738861680_dt, 2.6504743099_dt, 1.6998968124_dt, 1.7581002712_dt, 1.3883345127_dt, 0.6018559337_dt, 2.5042884350_dt, 1.5707685947_dt,
          1.2246072292_dt, 1.9513530731_dt, 1.8855968714_dt, 1.7518483400_dt, 2.1703207493_dt, 0.8668168187_dt, 0.5823799968_dt, 0.3222199082_dt, 0.5328013897_dt, 0.0239760280_dt, 0.6003485918_dt,
          0.8691412807_dt, 0.3132150769_dt, 0.1712092757_dt, 0.2083655000_dt, 0.6775689721_dt, 0.6496044993_dt, 0.0529118180_dt, 0.7317054272_dt, 0.3720138669_dt, 0.3189361095_dt, 0.8919355273_dt,
          0.7041678429_dt, 0.7928147316_dt, 0.6565824151_dt, 1.6412682533_dt, 1.4773685932_dt, 1.0124039650_dt, 1.4348254204_dt, 0.3924452662_dt, 1.1177220345_dt, 1.7456326485_dt, 0.6122574806_dt,
          1.1396981478_dt, 0.3023664355_dt, 1.4167952538_dt, 1.2499713898_dt, 0.7267626524_dt, 1.0919339657_dt, 1.2500314713_dt, 0.9420017600_dt, 1.2488982677_dt, 1.5186870098_dt, 1.4001537561_dt,
          1.1690371037_dt, 2.2821435928_dt, 1.6633901596_dt, 1.6098539829_dt, 1.5932365656_dt, 0.5469012260_dt, 1.9651449919_dt, 2.1040327549_dt, 1.2751996517_dt, 1.5691173077_dt, 0.7741746306_dt,
          1.8151547909_dt, 2.0121116638_dt, 1.5208327770_dt, 1.7190299034_dt, 1.5749481916_dt, 1.9272577763_dt, 2.1929614544_dt, 2.1702723503_dt, 1.6361060143_dt, 1.3241128922_dt },
        { 0.1033096910_dt, 0.9701558948_dt, 0.9480701089_dt, 0.9787087440_dt, 0.3267636895_dt, 1.0233368874_dt, 1.0204308033_dt, 1.4063870907_dt, 1.8394992352_dt, 0.8149183393_dt, 1.8213945627_dt,
          1.2215354443_dt, 1.5977475643_dt, 2.7374353409_dt, 1.7690231800_dt, 2.3455636501_dt, 1.8221480846_dt, 2.5865275860_dt, 3.4848239422_dt, 2.3188314438_dt, 0.0670150518_dt, 0.1167818904_dt,
          0.1723778248_dt, 0.9939703345_dt, 0.6243668795_dt, 0.4326270819_dt, 0.6185297370_dt, 0.3860871792_dt, 1.8047580719_dt, 1.4027494192_dt, 0.6688632369_dt, 0.9083892107_dt, 0.7188989520_dt,
          2.7139730453_dt, 1.6529169083_dt, 1.2913025618_dt, 1.8733413219_dt, 1.2488555908_dt, 2.9209263325_dt, 2.3402175903_dt, 0.1918165684_dt, 0.8134448528_dt, 0.9125209451_dt, 0.9396399260_dt,
          0.8208933473_dt, 0.5952633023_dt, 1.7458884716_dt, 1.1143807173_dt, 1.9185359478_dt, 1.2542530298_dt, 1.3190698624_dt, 2.6432590485_dt, 1.1920645237_dt, 2.6156735420_dt, 1.6207492352_dt,
          1.3970687389_dt, 3.0290472507_dt, 1.5589246750_dt, 3.4131798744_dt, 2.5539612770_dt, 0.8668168187_dt, 0.5823799968_dt, 0.3222199082_dt, 0.5328013897_dt, 0.0239760280_dt, 1.4671654701_dt,
          1.4515212774_dt, 0.6354349852_dt, 0.7040106654_dt, 0.2323415279_dt, 2.1447343826_dt, 2.1011257172_dt, 0.6883468032_dt, 1.4357161522_dt, 0.6043553948_dt, 2.4636704922_dt, 2.9930613041_dt,
          1.3925147057_dt, 2.2285308838_dt, 1.2609378099_dt, 0.7744513750_dt, 0.8949885964_dt, 0.6901841164_dt, 0.9020239711_dt, 0.3684692383_dt, 1.2918248177_dt, 1.7714799643_dt, 0.9892265201_dt,
          1.8705128431_dt, 0.4624701738_dt, 2.0310511589_dt, 2.3718469143_dt, 1.6630773544_dt, 2.2307415009_dt, 1.3404877186_dt, 2.6541168690_dt, 2.7288098335_dt, 2.4775965214_dt, 2.8380804062_dt,
          1.8529424667_dt, 0.6408753395_dt, 0.1860215068_dt, 0.5974498987_dt, 0.1584112048_dt, 0.1544559598_dt, 1.4882981777_dt, 0.5444216728_dt, 1.2603921890_dt, 0.5878303051_dt, 0.6262641549_dt,
          1.8866577148_dt, 1.3065619469_dt, 2.0544621944_dt, 1.2149262428_dt, 0.9511809349_dt, 2.8719139099_dt, 2.2506251335_dt, 2.7060475349_dt, 1.4508786201_dt, 1.1062567234_dt },
        { 0.1033096910_dt, 1.0734655857_dt, 2.0215356350_dt, 3.0002443790_dt, 3.3270082474_dt, 0.9200272560_dt, 0.9703022242_dt, 1.4286192656_dt, 2.2894098759_dt, 2.7775645256_dt, 0.7980576158_dt,
          0.9991621971_dt, 1.1905225515_dt, 2.0884585381_dt, 3.0425634384_dt, 0.5241690278_dt, 1.1247817278_dt, 2.1135616302_dt, 2.8609504700_dt, 3.4107584953_dt, 0.0670150518_dt, 0.1837969422_dt,
          0.3561747670_dt, 1.3501451015_dt, 1.9745119810_dt, 0.3656120300_dt, 0.8673598766_dt, 1.0810692310_dt, 1.8918569088_dt, 2.6702394485_dt, 0.2362361550_dt, 0.5260956287_dt, 0.8589074016_dt,
          1.7681223154_dt, 2.0182898045_dt, 0.6224393249_dt, 1.5873914957_dt, 2.1173481941_dt, 2.3243014812_dt, 3.0116021633_dt, 0.1918165684_dt, 1.0052614212_dt, 1.9177823067_dt, 2.8574223518_dt,
          3.6783156395_dt, 0.4034467340_dt, 1.3358902931_dt, 1.5377501249_dt, 2.5166461468_dt, 2.9500060081_dt, 0.7238065600_dt, 1.6211770773_dt, 1.6988608837_dt, 2.3959984779_dt, 2.7624945641_dt,
          0.0779988170_dt, 0.4637870789_dt, 0.8306472301_dt, 1.6281535625_dt, 2.5613656044_dt, 0.8668168187_dt, 1.4491968155_dt, 1.7714166641_dt, 2.3042180538_dt, 2.3281941414_dt, 0.6003485918_dt,
          1.4694898129_dt, 1.7827049494_dt, 1.9539141655_dt, 2.1622796059_dt, 0.6775689721_dt, 1.3271734715_dt, 1.3800852299_dt, 2.1117906570_dt, 2.4838047028_dt, 0.3189361095_dt, 1.2108716965_dt,
          1.9150395393_dt, 2.7078542709_dt, 3.3644366264_dt, 0.7744513750_dt, 1.6694400311_dt, 2.3596241474_dt, 3.2616481781_dt, 3.6301174164_dt, 0.5173735023_dt, 1.3938648701_dt, 1.6929073334_dt,
          2.6613960266_dt, 2.7553970814_dt, 0.7392263412_dt, 1.3395932913_dt, 2.0134441853_dt, 2.3736727238_dt, 3.2516901493_dt, 0.6230656505_dt, 0.9800284505_dt, 1.7945475578_dt, 2.4018864632_dt,
          2.9143414497_dt, 0.6408753395_dt, 0.8268968463_dt, 1.4243466854_dt, 1.5827579498_dt, 1.7372138500_dt, 0.8474228978_dt, 1.2058230639_dt, 1.8687653542_dt, 2.2981843948_dt, 2.7699925900_dt,
          0.3983595371_dt, 1.1604998112_dt, 1.9545699358_dt, 2.5816659927_dt, 2.9065828323_dt, 0.9852560759_dt, 1.9293191433_dt, 2.5809044838_dt, 2.8168568611_dt, 2.9719324112_dt }
    };

    // Initialization

    for (size_t iter = 0; iter < std::size(dimensions); iter++)
    {
        MANAGERS_DEFINE
        NETWORK_PARAMS_DEFINE(networkParameters);

        work.tensorNeeded("x", "x", raul::WShape{ raul::BS(), depth, height, width }, DEC_FORW_READ_NOMEMOPT);

        // Apply function
        raul::CumSumLayer cumsum("csum", raul::BasicParamsWithDim{ { "x" }, { "out" }, dimensions[iter] }, networkParameters);
        TENSORS_CREATE(batch);
        memory_manager["x"] = TORANGE(x);

        cumsum.forwardCompute(raul::NetworkMode::Test);

        // Checks
        const auto& xTensor = memory_manager["x"];
        const auto& outTensor = memory_manager["out"];

        EXPECT_EQ(outTensor.size(), xTensor.size());
        EXPECT_EQ(outTensor.size(), realOut[iter].size());
        for (size_t i = 0; i < outTensor.size(); ++i)
        {
            ASSERT_TRUE(tools::expect_near_relative(outTensor[i], realOut[iter][i], eps_rel));
        }
    }
}

TEST(TestLayerCumSum, BackwardUnit)
{
    PROFILE_TEST
    // Test parameters
    const auto eps_rel = TODTYPE(1e-6);
    const auto batch = 2;
    const auto depth = 3;
    const auto height = 4;
    const auto width = 5;

    const raul::Tensor x{ 0.1033096910_dt, 0.9701558948_dt, 0.9480701089_dt, 0.9787087440_dt, 0.3267636895_dt, 0.9200272560_dt, 0.0502749681_dt, 0.4583170414_dt, 0.8607905507_dt, 0.4881546497_dt,
                          0.7980576158_dt, 0.2011045814_dt, 0.1913603544_dt, 0.8979360461_dt, 0.9541048408_dt, 0.5241690278_dt, 0.6006127000_dt, 0.9887800217_dt, 0.7473886609_dt, 0.5498082042_dt,
                          0.0670150518_dt, 0.1167818904_dt, 0.1723778248_dt, 0.9939703345_dt, 0.6243668795_dt, 0.3656120300_dt, 0.5017478466_dt, 0.2137093544_dt, 0.8107876778_dt, 0.7783825397_dt,
                          0.2362361550_dt, 0.2898594737_dt, 0.3328117728_dt, 0.9092149138_dt, 0.2501674891_dt, 0.6224393249_dt, 0.9649521708_dt, 0.5299566984_dt, 0.2069533467_dt, 0.6873005629_dt,
                          0.1918165684_dt, 0.8134448528_dt, 0.9125209451_dt, 0.9396399260_dt, 0.8208933473_dt, 0.4034467340_dt, 0.9324436188_dt, 0.2018597722_dt, 0.9788960814_dt, 0.4333596826_dt,
                          0.7238065600_dt, 0.8973705173_dt, 0.0776838064_dt, 0.6971374750_dt, 0.3664962053_dt, 0.0779988170_dt, 0.3857882619_dt, 0.3668601513_dt, 0.7975063324_dt, 0.9332120419_dt,
                          0.8668168187_dt, 0.5823799968_dt, 0.3222199082_dt, 0.5328013897_dt, 0.0239760280_dt, 0.6003485918_dt, 0.8691412807_dt, 0.3132150769_dt, 0.1712092757_dt, 0.2083655000_dt,
                          0.6775689721_dt, 0.6496044993_dt, 0.0529118180_dt, 0.7317054272_dt, 0.3720138669_dt, 0.3189361095_dt, 0.8919355273_dt, 0.7041678429_dt, 0.7928147316_dt, 0.6565824151_dt,
                          0.7744513750_dt, 0.8949885964_dt, 0.6901841164_dt, 0.9020239711_dt, 0.3684692383_dt, 0.5173735023_dt, 0.8764913678_dt, 0.2990424037_dt, 0.9684888721_dt, 0.0940009356_dt,
                          0.7392263412_dt, 0.6003669500_dt, 0.6738508344_dt, 0.3602285385_dt, 0.8780175447_dt, 0.6230656505_dt, 0.3569628000_dt, 0.8145191073_dt, 0.6073390245_dt, 0.5124547482_dt,
                          0.6408753395_dt, 0.1860215068_dt, 0.5974498987_dt, 0.1584112048_dt, 0.1544559598_dt, 0.8474228978_dt, 0.3584001660_dt, 0.6629422307_dt, 0.4294191003_dt, 0.4718081951_dt,
                          0.3983595371_dt, 0.7621403337_dt, 0.7940700650_dt, 0.6270959973_dt, 0.3249167800_dt, 0.9852560759_dt, 0.9440631270_dt, 0.6515852809_dt, 0.2359522581_dt, 0.1550757289_dt };

    raul::Dimension dimensions[]{ raul::Dimension::Batch, raul::Dimension::Depth, raul::Dimension::Height, raul::Dimension::Width };
    const raul::Tensor deltas{ 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt,
                               1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt,
                               1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt,
                               1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt,
                               1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt,
                               1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt };
    const raul::Tensor realGrad[]{ { 2.0_dt, 2.0_dt, 2.0_dt, 2.0_dt, 2.0_dt, 2.0_dt, 2.0_dt, 2.0_dt, 2.0_dt, 2.0_dt, 2.0_dt, 2.0_dt, 2.0_dt, 2.0_dt, 2.0_dt, 2.0_dt, 2.0_dt, 2.0_dt, 2.0_dt, 2.0_dt,
                                     2.0_dt, 2.0_dt, 2.0_dt, 2.0_dt, 2.0_dt, 2.0_dt, 2.0_dt, 2.0_dt, 2.0_dt, 2.0_dt, 2.0_dt, 2.0_dt, 2.0_dt, 2.0_dt, 2.0_dt, 2.0_dt, 2.0_dt, 2.0_dt, 2.0_dt, 2.0_dt,
                                     2.0_dt, 2.0_dt, 2.0_dt, 2.0_dt, 2.0_dt, 2.0_dt, 2.0_dt, 2.0_dt, 2.0_dt, 2.0_dt, 2.0_dt, 2.0_dt, 2.0_dt, 2.0_dt, 2.0_dt, 2.0_dt, 2.0_dt, 2.0_dt, 2.0_dt, 2.0_dt,
                                     1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt,
                                     1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt,
                                     1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt },
                                   { 3.0_dt, 3.0_dt, 3.0_dt, 3.0_dt, 3.0_dt, 3.0_dt, 3.0_dt, 3.0_dt, 3.0_dt, 3.0_dt, 3.0_dt, 3.0_dt, 3.0_dt, 3.0_dt, 3.0_dt, 3.0_dt, 3.0_dt, 3.0_dt, 3.0_dt, 3.0_dt,
                                     2.0_dt, 2.0_dt, 2.0_dt, 2.0_dt, 2.0_dt, 2.0_dt, 2.0_dt, 2.0_dt, 2.0_dt, 2.0_dt, 2.0_dt, 2.0_dt, 2.0_dt, 2.0_dt, 2.0_dt, 2.0_dt, 2.0_dt, 2.0_dt, 2.0_dt, 2.0_dt,
                                     1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt,
                                     3.0_dt, 3.0_dt, 3.0_dt, 3.0_dt, 3.0_dt, 3.0_dt, 3.0_dt, 3.0_dt, 3.0_dt, 3.0_dt, 3.0_dt, 3.0_dt, 3.0_dt, 3.0_dt, 3.0_dt, 3.0_dt, 3.0_dt, 3.0_dt, 3.0_dt, 3.0_dt,
                                     2.0_dt, 2.0_dt, 2.0_dt, 2.0_dt, 2.0_dt, 2.0_dt, 2.0_dt, 2.0_dt, 2.0_dt, 2.0_dt, 2.0_dt, 2.0_dt, 2.0_dt, 2.0_dt, 2.0_dt, 2.0_dt, 2.0_dt, 2.0_dt, 2.0_dt, 2.0_dt,
                                     1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt },
                                   { 4.0_dt, 4.0_dt, 4.0_dt, 4.0_dt, 4.0_dt, 3.0_dt, 3.0_dt, 3.0_dt, 3.0_dt, 3.0_dt, 2.0_dt, 2.0_dt, 2.0_dt, 2.0_dt, 2.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt,
                                     4.0_dt, 4.0_dt, 4.0_dt, 4.0_dt, 4.0_dt, 3.0_dt, 3.0_dt, 3.0_dt, 3.0_dt, 3.0_dt, 2.0_dt, 2.0_dt, 2.0_dt, 2.0_dt, 2.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt,
                                     4.0_dt, 4.0_dt, 4.0_dt, 4.0_dt, 4.0_dt, 3.0_dt, 3.0_dt, 3.0_dt, 3.0_dt, 3.0_dt, 2.0_dt, 2.0_dt, 2.0_dt, 2.0_dt, 2.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt,
                                     4.0_dt, 4.0_dt, 4.0_dt, 4.0_dt, 4.0_dt, 3.0_dt, 3.0_dt, 3.0_dt, 3.0_dt, 3.0_dt, 2.0_dt, 2.0_dt, 2.0_dt, 2.0_dt, 2.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt,
                                     4.0_dt, 4.0_dt, 4.0_dt, 4.0_dt, 4.0_dt, 3.0_dt, 3.0_dt, 3.0_dt, 3.0_dt, 3.0_dt, 2.0_dt, 2.0_dt, 2.0_dt, 2.0_dt, 2.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt,
                                     4.0_dt, 4.0_dt, 4.0_dt, 4.0_dt, 4.0_dt, 3.0_dt, 3.0_dt, 3.0_dt, 3.0_dt, 3.0_dt, 2.0_dt, 2.0_dt, 2.0_dt, 2.0_dt, 2.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt },
                                   { 5.0_dt, 4.0_dt, 3.0_dt, 2.0_dt, 1.0_dt, 5.0_dt, 4.0_dt, 3.0_dt, 2.0_dt, 1.0_dt, 5.0_dt, 4.0_dt, 3.0_dt, 2.0_dt, 1.0_dt, 5.0_dt, 4.0_dt, 3.0_dt, 2.0_dt, 1.0_dt,
                                     5.0_dt, 4.0_dt, 3.0_dt, 2.0_dt, 1.0_dt, 5.0_dt, 4.0_dt, 3.0_dt, 2.0_dt, 1.0_dt, 5.0_dt, 4.0_dt, 3.0_dt, 2.0_dt, 1.0_dt, 5.0_dt, 4.0_dt, 3.0_dt, 2.0_dt, 1.0_dt,
                                     5.0_dt, 4.0_dt, 3.0_dt, 2.0_dt, 1.0_dt, 5.0_dt, 4.0_dt, 3.0_dt, 2.0_dt, 1.0_dt, 5.0_dt, 4.0_dt, 3.0_dt, 2.0_dt, 1.0_dt, 5.0_dt, 4.0_dt, 3.0_dt, 2.0_dt, 1.0_dt,
                                     5.0_dt, 4.0_dt, 3.0_dt, 2.0_dt, 1.0_dt, 5.0_dt, 4.0_dt, 3.0_dt, 2.0_dt, 1.0_dt, 5.0_dt, 4.0_dt, 3.0_dt, 2.0_dt, 1.0_dt, 5.0_dt, 4.0_dt, 3.0_dt, 2.0_dt, 1.0_dt,
                                     5.0_dt, 4.0_dt, 3.0_dt, 2.0_dt, 1.0_dt, 5.0_dt, 4.0_dt, 3.0_dt, 2.0_dt, 1.0_dt, 5.0_dt, 4.0_dt, 3.0_dt, 2.0_dt, 1.0_dt, 5.0_dt, 4.0_dt, 3.0_dt, 2.0_dt, 1.0_dt,
                                     5.0_dt, 4.0_dt, 3.0_dt, 2.0_dt, 1.0_dt, 5.0_dt, 4.0_dt, 3.0_dt, 2.0_dt, 1.0_dt, 5.0_dt, 4.0_dt, 3.0_dt, 2.0_dt, 1.0_dt, 5.0_dt, 4.0_dt, 3.0_dt, 2.0_dt, 1.0_dt } };

    for (size_t iter = 0; iter < std::size(dimensions); iter++)
    {
        MANAGERS_DEFINE
        NETWORK_PARAMS_DEFINE(networkParameters);

        work.tensorNeeded("x", "x", raul::WShape{ raul::BS(), depth, height, width }, DEC_FORW_READ_NOMEMOPT);

        // Apply function
        raul::CumSumLayer cumsum("csum", raul::BasicParamsWithDim{ { "x" }, { "out" }, dimensions[iter] }, networkParameters);
        TENSORS_CREATE(batch);
        memory_manager["x"] = TORANGE(x);
        memory_manager[raul::Name("out").grad()] = TORANGE(deltas);

        cumsum.forwardCompute(raul::NetworkMode::Test);
        cumsum.backwardCompute();

        // Checks
        const auto& xTensor = memory_manager["x"];
        const auto& xNablaTensor = memory_manager[raul::Name("x").grad()];

        EXPECT_EQ(xNablaTensor.size(), xTensor.size());
        EXPECT_EQ(xNablaTensor.size(), realGrad[iter].size());
        for (size_t i = 0; i < xNablaTensor.size(); ++i)
        {
            ASSERT_TRUE(tools::expect_near_relative(xNablaTensor[i], realGrad[iter][i], eps_rel));
        }
    }
}

}
