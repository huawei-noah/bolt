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
#include <training/base/layers/basic/AveragePoolLayer.h>
#include <training/base/layers/basic/DataLayer.h>
#include <training/base/layers/parameters/LayerParameters.h>
#include <training/compiler/Layers.h>
#include <training/compiler/Workflow.h>
#include <training/base/optimizers/SGD.h>

namespace UT
{
using namespace raul;
TEST(TestCNNAveragePool, Unit)
{
    PROFILE_TEST
    dtype eps = TODTYPE(1e-6);
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
        auto params = Pool2DParams{ { "in" }, { "avg" }, kernel_width, kernel_height, stride_w, stride_h, padding_w, padding_h };
        AveragePoolLayer avgpool("avg1", params, networkParameters);
        TENSORS_CREATE(batch);
        memory_manager["in"] = TORANGE(raw);
        const Tensor& out = memory_manager["avg"];
        avgpool.forwardCompute(NetworkMode::Train);
        EXPECT_NEAR(TODTYPE(5.f), out[0], eps);
        printf(" - AveragePool with square kernel and stride = 1 is Ok.\n");

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
        auto params = Pool2DParams{ { "in" }, { "avg" }, kernel_width, kernel_height, stride_w, stride_h, padding_w, padding_h };
        AveragePoolLayer avgpool("avg1", params, networkParameters);
        TENSORS_CREATE(batch);
        memory_manager["in"] = TORANGE(raw);
        const Tensor& out = memory_manager["avg"];
        avgpool.forwardCompute(NetworkMode::Train);

        EXPECT_NEAR(TODTYPE(5.5f), out[0], eps);
        EXPECT_NEAR(TODTYPE(6.5f), out[1], eps);
        EXPECT_NEAR(TODTYPE(7.5f), out[2], eps);
        printf(" - AveragePool with kernel = (2,3) and stride = 1 is Ok.\n");

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
        auto params = Pool2DParams{ { "in" }, { "avg" }, kernel_width, kernel_height, stride_w, stride_h, padding_w, padding_h };
        AveragePoolLayer avgpool("avg1", params, networkParameters);
        TENSORS_CREATE(batch);
        memory_manager["in"] = TORANGE(raw);
        const Tensor& out = memory_manager["avg"];
        avgpool.forwardCompute(NetworkMode::Train);

        EXPECT_NEAR(TODTYPE(2.6666667f), out[0], eps);
        EXPECT_NEAR(TODTYPE(6.5f), out[1], eps);
        EXPECT_NEAR(TODTYPE(6.0f), out[2], eps);
        EXPECT_NEAR(TODTYPE(11.5f), out[3], eps);

        printf(" - AveragePool with kernel = (3,2) and stride = (3,1) and padding (1,0) is Ok.\n");
    }
    {
        MANAGERS_DEFINE
        NETWORK_PARAMS_DEFINE(networkParameters);

        size_t in_w = 5;
        size_t in_h = 3;
        size_t depth = 1;
        size_t kernel_height = 2;
        size_t kernel_width = 3;
        size_t stride_w = 3;
        size_t stride_h = 1;
        size_t padding_w = 2;
        size_t padding_h = 0;

        work.add<DataLayer>("data", DataParams{ { "in" }, depth, in_h, in_w });
        auto params = Pool2DParams{ { "in" }, { "out" }, kernel_width, kernel_height, stride_w, stride_h, padding_w, padding_h };
        EXPECT_THROW(AveragePoolLayer averagepool4("avg1", params, networkParameters), raul::Exception);
        printf(" - AveragePool with wrong Padding make throw - Ok.\n");
    }
    {
        MANAGERS_DEFINE
        NETWORK_PARAMS_DEFINE(networkParameters);
        size_t in_w = 0;
        size_t in_h = 0;
        size_t depth = 1;
        size_t kernel_height = 2;
        size_t kernel_width = 3;
        size_t stride_w = 1;
        size_t stride_h = 1;
        size_t padding_w = 2;
        size_t padding_h = 0;

        work.add<DataLayer>("data", DataParams{ { "in" }, depth, in_h, in_w });
        auto params = Pool2DParams{ { "in2" }, { "out2" }, kernel_width, kernel_height, stride_w, stride_h, padding_w, padding_h };
        EXPECT_THROW(AveragePoolLayer averagepool5("avg1", params, networkParameters), raul::Exception);
        printf(" - AveragePool with wrong Input size make throw - Ok.\n");
    }
}

} // UT namespace