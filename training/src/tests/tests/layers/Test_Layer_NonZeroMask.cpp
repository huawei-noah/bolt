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
#include <training/base/layers/basic/NonZeroMaskLayer.h>
#include <training/compiler/Workflow.h>

namespace UT
{

namespace
{

raul::dtype golden_mask_layer(raul::dtype x)
{
    return x == 0.0_dt ? 0.0_dt : 1.0_dt;
}

}

TEST(TestLayerNonZeroMask, CpuUnit)
{
    PROFILE_TEST
    // Test parameters
    const auto batch = 2;
    const auto depth = 3;
    const auto height = 4;
    const auto width = 5;

    const raul::Tensor x{ 0.0_dt,          0.9701558948_dt, 0.9480701089_dt, 0.9787087440_dt, 0.3267636895_dt, 0.9200272560_dt, 0.0_dt,          0.0_dt,          0.8607905507_dt, 0.0_dt,
                          0.7980576158_dt, 0.2011045814_dt, 0.1913603544_dt, 0.8979360461_dt, 0.9541048408_dt, 0.5241690278_dt, 0.6006127000_dt, 0.9887800217_dt, 0.0_dt,          0.5498082042_dt,
                          0.0670150518_dt, 0.1167818904_dt, 0.1723778248_dt, 0.9939703345_dt, 0.6243668795_dt, 0.3656120300_dt, 0.0_dt,          0.2137093544_dt, 0.8107876778_dt, 0.7783825397_dt,
                          0.2362361550_dt, 0.0_dt,          0.3328117728_dt, 0.9092149138_dt, 0.2501674891_dt, 0.6224393249_dt, 0.9649521708_dt, 0.5299566984_dt, 0.2069533467_dt, 0.6873005629_dt,
                          0.1918165684_dt, 0.8134448528_dt, 0.0_dt,          0.9396399260_dt, 0.8208933473_dt, 0.4034467340_dt, 0.0_dt,          0.0_dt,          0.9788960814_dt, 0.4333596826_dt,
                          0.7238065600_dt, 0.8973705173_dt, 0.0_dt,          0.6971374750_dt, 0.3664962053_dt, 0.0779988170_dt, 0.3857882619_dt, 0.3668601513_dt, 0.0_dt,          0.9332120419_dt,
                          0.0_dt,          0.5823799968_dt, 0.3222199082_dt, 0.5328013897_dt, 0.0239760280_dt, 0.6003485918_dt, 0.0_dt,          0.3132150769_dt, 0.1712092757_dt, 0.2083655000_dt,
                          0.6775689721_dt, 0.0_dt,          0.0_dt,          0.7317054272_dt, 0.3720138669_dt, 0.3189361095_dt, 0.0_dt,          0.7041678429_dt, 0.0_dt,          0.6565824151_dt,
                          0.7744513750_dt, 0.8949885964_dt, 0.6901841164_dt, 0.0_dt,          0.3684692383_dt, 0.5173735023_dt, 0.8764913678_dt, 0.2990424037_dt, 0.9684888721_dt, 0.0940009356_dt,
                          0.0_dt,          0.0_dt,          0.6738508344_dt, 0.3602285385_dt, 0.8780175447_dt, 0.0_dt,          0.3569628000_dt, 0.8145191073_dt, 0.6073390245_dt, 0.5124547482_dt,
                          0.6408753395_dt, 0.1860215068_dt, 0.5974498987_dt, 0.1584112048_dt, 0.1544559598_dt, 0.8474228978_dt, 0.3584001660_dt, 0.0_dt,          0.4294191003_dt, 0.4718081951_dt,
                          0.3983595371_dt, 0.7621403337_dt, 0.7940700650_dt, 0.6270959973_dt, 0.0_dt,          0.9852560759_dt, 0.9440631270_dt, 0.6515852809_dt, 0.2359522581_dt, 0.1550757289_dt };

    MANAGERS_DEFINE
    NETWORK_PARAMS_DEFINE(networkParameters);

    work.add<raul::DataLayer>("data", raul::DataParams{ { "x" }, depth, height, width });
    work.add<raul::NonZeroMaskLayer>("mask", raul::BasicParams{ { "x" }, { "out" } });

    TENSORS_CREATE(batch);
    memory_manager["x"] = TORANGE(x);

    ASSERT_NO_THROW(work.forwardPassTraining());

    // Checks
    const auto& xTensor = memory_manager["x"];
    const auto& outTensor = memory_manager["out"];

    EXPECT_EQ(outTensor.size(), xTensor.size());
    for (size_t i = 0; i < outTensor.size(); ++i)
    {
        EXPECT_EQ(outTensor[i], golden_mask_layer(xTensor[i]));
    }

    ASSERT_NO_THROW(work.backwardPassTraining());
}

}