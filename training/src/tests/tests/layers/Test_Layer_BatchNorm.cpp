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
#include <training/base/layers/basic/TransposeLayer.h>
#include <training/base/layers/basic/trainable/Batchnorm.h>
#include <training/compiler/Layers.h>
#include <training/compiler/Workflow.h>

namespace UT
{

TEST(TestBatchNorm, Unit)
{
    PROFILE_TEST
    const raul::dtype eps = TODTYPE(1e-4);
    const size_t batch = 1;
    const raul::Tensor input = { -1.12583983_dt, -1.15236020_dt, -0.25057858_dt, -0.43387881_dt, 0.84871036_dt, 0.69200915_dt,  -0.31601277_dt, -2.11521935_dt,
                                 0.46809641_dt,  -0.15771244_dt, 1.44366014_dt,  0.26604941_dt,  0.16645534_dt, 0.87438184_dt,  -0.14347385_dt, -0.11160933_dt,
                                 0.93182659_dt,  1.25900924_dt,  2.00498056_dt,  0.05373690_dt,  0.61805665_dt, -0.41280222_dt, -0.84106481_dt, -2.31604195_dt };

    const raul::Tensor realOut = { -1.02864230_dt, -1.05667686_dt, -0.10340743_dt, -0.29717332_dt, 1.05864620_dt,  0.89299798_dt,  -0.17257762_dt, -2.07451105_dt,
                                   0.65630078_dt,  -0.00523904_dt, 1.68756485_dt,  0.44271776_dt,  -0.00674019_dt, 0.65912879_dt,  -0.29825664_dt, -0.26828519_dt,
                                   0.71316075_dt,  1.02090561_dt,  1.72255909_dt,  -0.11276208_dt, 0.41803172_dt,  -0.55158430_dt, -0.95440412_dt, -2.34175348_dt };

    std::array<std::string, 3> dimensions{ "depth", "height", "width" };
    yato::dimensionality<4U, size_t> shapes[] = { yato::dims(batch, 2, 3, 4), yato::dims(batch, 3, 2, 4), yato::dims(batch, 4, 2, 3) };

    for (size_t iter = 0; iter < std::size(dimensions); ++iter)
    {
        MANAGERS_DEFINE
        NETWORK_PARAMS_DEFINE(networkParameters);

        work.add<raul::DataLayer>("data1", raul::DataParams{ { "in1" }, shapes[iter][1], shapes[iter][2], shapes[iter][3] });
        work.add<raul::DataLayer>("data2", raul::DataParams{ { "in2" }, shapes[iter][1], shapes[iter][2], shapes[iter][3] });
        // First branch
        raul::BatchNormLayer norm("bn", raul::BatchnormParams{ { "in1" }, { "out1" }, 0.0f, 1e-5f, dimensions[iter] }, networkParameters);

        // Second branch
        raul::TransposeLayer tr1("tr1", raul::TransposingParams{ { "in2" }, { "intermediate1" }, "depth", dimensions[iter] }, networkParameters);
        raul::BatchNormLayer norm_default("bn_def", raul::BatchnormParams{ { "intermediate1" }, { "intermediate2" } }, networkParameters);
        raul::TransposeLayer tr2("tr2", raul::TransposingParams{ { "intermediate2" }, { "out2" }, "depth", dimensions[iter] }, networkParameters);
        TENSORS_CREATE(batch);
        memory_manager["in1"] = TORANGE(input);
        memory_manager["in2"] = TORANGE(input);
        norm.initNotBSTensors();
        norm_default.initNotBSTensors();

        // First
        norm.forwardCompute(raul::NetworkMode::Train);
        if (iter == 0)
        {
            const raul::Tensor& out = memory_manager["out1"];
            for (size_t i = 0; i < out.size(); ++i)
            {
                EXPECT_NEAR(out[i], realOut[i], eps);
            }
        }
        else
        {
            // Second variant
            tr1.forwardCompute(raul::NetworkMode::Train);
            norm_default.forwardCompute(raul::NetworkMode::Train);
            tr2.forwardCompute(raul::NetworkMode::Train);

            // Compare
            const raul::Tensor& out1 = memory_manager["out1"];
            const raul::Tensor& out2 = memory_manager["out2"];
            EXPECT_EQ(out1.getShape(), out2.getShape());
            for (size_t i = 0; i < out1.size(); ++i)
            {
                EXPECT_NEAR(out1[i], out2[i], eps);
            }
        }
    }
}

}