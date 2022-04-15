// Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include <tests/tools/TestTools.h>

#include <chrono>
#include <cstdio>
#include <tests/tools/TestTools.h>

#include <training/common/Common.h>
#include <training/common/Conversions.h>
#include <training/common/MemoryManager.h>
#include <training/layers/BasicLayer.h>
#include <training/layers/parameters/Parameters.h>
#include <training/network/Graph.h>
#include <training/tools/Datasets.h>

#include "model_print.h"
#include "model_tools.h"

namespace UT
{

TEST(TestLoadBoltModel, Unit)
{
    PROFILE_TEST("TestLoadBoltModel.Unit")
    std::string model_path = (tools::getTestAssetsDir() / "bolt").string();
    ModelSpec ms;
    ASSERT_EQ(mt_create_model(&ms), SUCCESS);
    ASSERT_EQ(mt_load(tools::getTestAssetsDir().string().c_str(), "bolt/mobilenet_v1_f32.bolt", &ms), SUCCESS);
    print_ms(ms);
    ASSERT_EQ(mt_destroy_model(&ms), SUCCESS);
}

} // UT namespace