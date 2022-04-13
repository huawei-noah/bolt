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
#include <training/base/common/MemoryManager.h>
#include <training/base/layers/basic/DataLayer.h>
#include <training/base/layers/basic/ReshapeLayer.h>
#include <training/compiler/Workflow.h>

namespace UT
{

TEST(TestReshapeLayer, ReshapeLayerUnit)
{
    PROFILE_TEST

    {
        int N = 5, C = 1, H = 4, W = 1;
        MANAGERS_DEFINE
        NETWORK_PARAMS_DEFINE(networkParameters);

        work.add<raul::DataLayer>("data", raul::DataParams{ { "in" }, static_cast<size_t>(C), static_cast<size_t>(H), static_cast<size_t>(W) });
        work.add<raul::ReshapeLayer>("r", raul::ViewParams{ "in", "out", C, -1, W });
        TENSORS_CREATE(N);

        work.forwardPassTraining();
        EXPECT_EQ(memory_manager["out"].getShape(), memory_manager["in"].getShape());

        work.backwardPassTraining();
        EXPECT_EQ(memory_manager[raul::Name("in").grad()].getShape(), memory_manager["in"].getShape());
    }

    {
        int N = 5, C = 1, H = 4, W = 1;
        MANAGERS_DEFINE
        NETWORK_PARAMS_DEFINE(networkParameters);

        work.add<raul::DataLayer>("data", raul::DataParams{ { "in" }, static_cast<size_t>(C), static_cast<size_t>(H), static_cast<size_t>(W) });
        work.add<raul::ReshapeLayer>("r", raul::ViewParams{ "in", "out", C * H * W });
        TENSORS_CREATE(N);

        work.forwardPassTraining();
        EXPECT_EQ(memory_manager["out"].getShape(), yato::dims(N, C * H * W, 1, 1));

        work.backwardPassTraining();
        EXPECT_EQ(memory_manager[raul::Name("in").grad()].getShape(), memory_manager["in"].getShape());
    }

    {
        int N = 5, C = 4, H = 3, W = 2;
        MANAGERS_DEFINE
        NETWORK_PARAMS_DEFINE(networkParameters);

        work.add<raul::DataLayer>("data", raul::DataParams{ { "in" }, static_cast<size_t>(C), static_cast<size_t>(H), static_cast<size_t>(W) });
        work.add<raul::ReshapeLayer>("r", raul::ViewParams{ "in", "out", C, -1, H });
        TENSORS_CREATE(N);

        work.forwardPassTraining();
        EXPECT_EQ(memory_manager["out"].getShape(), yato::dims(N, C, W, H));

        work.backwardPassTraining();
        EXPECT_EQ(memory_manager[raul::Name("in").grad()].getShape(), memory_manager["in"].getShape());
    }

    {
        int N = 5, C = 4, H = 3, W = 2;
        MANAGERS_DEFINE
        NETWORK_PARAMS_DEFINE(networkParameters);

        work.add<raul::DataLayer>("data", raul::DataParams{ { "in" }, static_cast<size_t>(C), static_cast<size_t>(H), static_cast<size_t>(W) });
        work.add<raul::ReshapeLayer>("r", raul::ViewParams{ "in", "out", H, C, W });
        TENSORS_CREATE(N);

        work.forwardPassTraining();
        EXPECT_EQ(memory_manager["out"].getShape(), yato::dims(N, H, C, W));

        work.backwardPassTraining();
        EXPECT_EQ(memory_manager[raul::Name("in").grad()].getShape(), memory_manager["in"].getShape());
    }

    {
        int C = 1, H = 4, W = 1;
        MANAGERS_DEFINE
        NETWORK_PARAMS_DEFINE(networkParameters);

        work.add<raul::DataLayer>("data", raul::DataParams{ { "in" }, static_cast<size_t>(C), static_cast<size_t>(H), static_cast<size_t>(W) });
        EXPECT_THROW(work.add<raul::ReshapeLayer>("r", raul::ViewParams{ "in", "out", H, H, H }), raul::Exception);
    }

    {
        int C = 2, H = 4, W = 1;
        MANAGERS_DEFINE
        NETWORK_PARAMS_DEFINE(networkParameters);

        work.add<raul::DataLayer>("data", raul::DataParams{ { "in" }, static_cast<size_t>(C), static_cast<size_t>(H), static_cast<size_t>(W) });
        EXPECT_THROW(work.add<raul::ReshapeLayer>("r", raul::ViewParams{ "in", "out", -1, -1, W }), raul::Exception);
    }
}

} // UT namespace