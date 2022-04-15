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
#include <utility>

#include <training/common/Common.h>
#include <training/common/TensorGPU.h>
#include <training/opencl/OpenCLKernelManager.h>

namespace UT
{

TEST(TestTensorGPU, ConstructorGpuUnit)
{
    PROFILE_TEST

    GPU_ONLY_TEST

    auto [platform, device, context] = raul::Common::getGpuPlatformDeviceAndContext();

    cl::CommandQueue queue(context, device);
    raul::OpenCLKernelManager manager(queue);

    {
        raul::TensorGPU t(&manager, "", 1, 2, 3, 4);

        EXPECT_EQ(t.size(), 24u);

        EXPECT_EQ(t.getBatchSize(), 1u);
        EXPECT_EQ(t.getDepth(), 2u);
        EXPECT_EQ(t.getHeight(), 3u);
        EXPECT_EQ(t.getWidth(), 4u);
        EXPECT_FALSE(t.empty());
        t.memClear();
        EXPECT_TRUE(t.empty());
    }

    {
        const raul::Tensor t("", 1, 2, 3, 4, 1_dt);
        raul::TensorGPU tg(&manager, t);

        EXPECT_EQ(tg.size(), 24u);
        EXPECT_EQ(tg.getBatchSize(), 1u);
        EXPECT_EQ(tg.getDepth(), 2u);
        EXPECT_EQ(tg.getHeight(), 3u);
        EXPECT_EQ(tg.getWidth(), 4u);
        EXPECT_FALSE(tg.empty());

        const raul::Tensor tt(raul::TensorGPUHelper(tg, &manager));
        EXPECT_EQ(tt.size(), 24u);

        EXPECT_EQ(tt.getBatchSize(), 1u);
        EXPECT_EQ(tt.getDepth(), 2u);
        EXPECT_EQ(tt.getHeight(), 3u);
        EXPECT_EQ(tt.getWidth(), 4u);
        EXPECT_FALSE(tt.empty());

        for (auto val : tt)
        {
            EXPECT_EQ(val, 1_dt);
        }

        tg.memClear();

        raul::Tensor ttt(raul::TensorGPUHelper(tg, &manager)); // allocate shape but copy only buffer (empty)
        EXPECT_EQ(ttt.size(), 24u);

        for (auto val : ttt)
        {
            EXPECT_EQ(val, 0_dt);
        }
    }
}

TEST(TestTensorGPU, MemAllocGpuUnit)
{
    PROFILE_TEST

    GPU_ONLY_TEST

    auto [platform, device, context] = raul::Common::getGpuPlatformDeviceAndContext();

    cl::CommandQueue queue(context, device);
    raul::OpenCLKernelManager manager(queue);
    {
        raul::TensorGPU t(&manager, "", 1, 2, 3, 4);

        EXPECT_EQ(t.size(), 24u);

        EXPECT_FALSE(t.empty());
        t.memClear();
        EXPECT_TRUE(t.empty());
        EXPECT_EQ(t.size(), 0u);

        cl::Buffer buffer = cl::Buffer(context, CL_MEM_READ_WRITE, 24u * sizeof(raul::dtype));
        cl_buffer_region rgn = { 0, 24u * sizeof(raul::dtype) };
        cl::Buffer subBuffer = buffer.createSubBuffer(CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &rgn);
        t.memAllocate(subBuffer);
        EXPECT_FALSE(t.empty());
        EXPECT_EQ(t.size(), 24u);
    }

    {
        size_t alignElem = device.getInfo<CL_DEVICE_MEM_BASE_ADDR_ALIGN>() / 8 / sizeof(raul::dtype);

        raul::Tensor t("", 1, 1, 1, alignElem * 2u, 1_dt);
        std::fill(t.begin() + 12u, t.end(), 3_dt);

        raul::TensorGPU tt(&manager, t);

        EXPECT_EQ(tt.size(), alignElem * 2u);

        raul::TensorGPU tg(&manager, "", 1, 2, 3, 2, 2_dt);

        EXPECT_EQ(tg.size(), 12u);

        EXPECT_EQ(tg.getBatchSize(), 1u);
        EXPECT_EQ(tg.getDepth(), 2u);
        EXPECT_EQ(tg.getHeight(), 3u);
        EXPECT_EQ(tg.getWidth(), 2u);

        {
            const raul::Tensor tmp(raul::TensorGPUHelper(tg, &manager));

            for (auto val : tmp)
            {
                EXPECT_EQ(val, 2_dt);
            }
        }

        {
            cl_buffer_region rgn = { 0, 12u * sizeof(raul::dtype) };
            cl::Buffer subBuffer = tt.getBuffer().createSubBuffer(CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &rgn);

            tg.memAllocate(subBuffer);

            EXPECT_EQ(tg.size(), 12u);
        }

        {
            raul::Tensor tmp(raul::TensorGPUHelper(tg, &manager));

            for (auto val : tmp)
            {
                EXPECT_EQ(val, 1_dt);
            }
        }

        {
            cl_buffer_region rgn = { alignElem * sizeof(raul::dtype), 12u * sizeof(raul::dtype) };
            cl::Buffer subBuffer = tt.getBuffer().createSubBuffer(CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &rgn);

            tg.memAllocate(subBuffer);

            EXPECT_EQ(tg.size(), 12u);
        }

        {
            raul::Tensor tmp(raul::TensorGPUHelper(tg, &manager));

            for (auto val : tmp)
            {
                EXPECT_EQ(val, 3_dt);
            }
        }
    }
}

TEST(TestTensorGPU, CopyTensorGpuUnit)
{
    PROFILE_TEST

    GPU_ONLY_TEST

    auto [platform, device, context] = raul::Common::getGpuPlatformDeviceAndContext();

    cl::CommandQueue queue(context, device);
    raul::OpenCLKernelManager manager(queue);
    raul::TensorGPU tg(&manager, "", 1, 2, 3, 4, 10_dt);
    raul::TensorGPU ttg(&manager, "", 1, 2, 3, 4);

    {
        raul::Tensor tmp(raul::TensorGPUHelper(tg, &manager));

        EXPECT_EQ(tmp.size(), 24u);

        for (auto val : tmp)
        {
            EXPECT_EQ(val, 10_dt);
        }
    }

    ttg.copy(tg);

    {
        raul::Tensor tmp(raul::TensorGPUHelper(ttg, &manager));

        EXPECT_EQ(tmp.size(), 24u);

        for (auto val : tmp)
        {
            EXPECT_EQ(val, 10_dt);
        }
    }
}

TEST(TestTensorGPU, CopyFromToGpuUnit)
{
    PROFILE_TEST

    GPU_ONLY_TEST

    auto [platform, device, context] = raul::Common::getGpuPlatformDeviceAndContext();

    cl::CommandQueue queue(context, device);
    raul::OpenCLKernelManager manager(queue);

    const raul::Tensor t("", 1, 2, 3, 4, 3_dt);
    raul::TensorGPU tg(&manager, "", 1, 2, 3, 4);

    tg.copyFrom(t);

    {
        raul::Tensor tmp(raul::TensorGPUHelper(tg, &manager));

        EXPECT_EQ(tmp.size(), 24u);

        for (auto val : tmp)
        {
            EXPECT_EQ(val, 3_dt);
        }
    }

    {
        raul::Tensor tt("", 1, 2, 3, 4);

        tg.copyTo(tt);

        EXPECT_EQ(tt.size(), 24u);

        for (auto val : tt)
        {
            EXPECT_EQ(val, 3_dt);
        }

        tt.memClear();

        EXPECT_THROW(tg.copyTo(tt), raul::Exception);
    }

    {
        raul::Tensor tt("", 1, 2, 3, 3);

        EXPECT_NO_THROW(tg.copyTo(tt));
    }

    tg.memClear();

    EXPECT_THROW(tg.copyFrom(t), raul::Exception);

    {
        raul::TensorGPU ttg(&manager, "", 1, 2, 3, 3);
        EXPECT_THROW(ttg.copyFrom(t), raul::Exception);
    }

    {
        cl::Buffer clMem = cl::Buffer(context, CL_MEM_READ_WRITE, tg.getShape().total_size() * sizeof(float));
        tg.memAllocate(clMem);
    }

    raul::TensorGPUHelper tgH(tg, &manager);
    tgH = t;

    {
        const raul::Tensor tmp(raul::TensorGPUHelper(tg, &manager));

        EXPECT_EQ(tmp.size(), 24u);

        for (auto val : tmp)
        {
            EXPECT_EQ(val, 3_dt);
        }
    }

    {
        raul::Tensor tt(tgH);

        EXPECT_EQ(tt.size(), 24u);

        for (auto val : tt)
        {
            EXPECT_EQ(val, 3_dt);
        }
    }

    {
        raul::Tensor tt("", 1, 2, 3, 4);

        tt = tgH;

        EXPECT_EQ(tt.size(), 24u);

        for (auto val : tt)
        {
            EXPECT_EQ(val, 3_dt);
        }
    }
}

TEST(TestTensorGPU, PoolAllocationGpuUnit)
{
    PROFILE_TEST

    GPU_ONLY_TEST

    auto [platform, device, context] = raul::Common::getGpuPlatformDeviceAndContext();

    cl::CommandQueue queue(context, device);
    raul::OpenCLKernelManager manager(queue);
    {
        raul::TensorGPU tg(&manager, "", 10u, 1u, 1u, 1u, raul::AllocationMode::POOL, false);

        EXPECT_EQ(tg.size(), 0u);
        EXPECT_EQ(tg.getBatchSize(), 10u);
        EXPECT_EQ(tg.getDepth(), 1u);
        EXPECT_EQ(tg.getHeight(), 1u);
        EXPECT_EQ(tg.getWidth(), 1u);
        EXPECT_TRUE(tg.empty());
    }

    {
        raul::TensorGPU tg(&manager, "", 10u, 1u, 1u, 1u, raul::AllocationMode::POOL, true);

        EXPECT_EQ(tg.size(), 0u);
        EXPECT_EQ(tg.getBatchSize(), 10u);
        EXPECT_EQ(tg.getDepth(), 1u);
        EXPECT_EQ(tg.getHeight(), 1u);
        EXPECT_EQ(tg.getWidth(), 1u);
        EXPECT_TRUE(tg.empty());
    }
}

} // UT namespace