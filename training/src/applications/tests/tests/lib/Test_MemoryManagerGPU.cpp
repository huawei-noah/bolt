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

#include <training/common/MemoryManagerGPU.h>
#include <training/opencl/OpenCLKernelManager.h>

#include <tests/tools/TestTools.h>
namespace UT
{

TEST(TestMemoryManagerGPU, GeneralGpuUnit)
{
    PROFILE_TEST

    GPU_ONLY_TEST

    auto [platform, device, context] = raul::Common::getGpuPlatformDeviceAndContext();

    cl::CommandQueue queue(context, device);
    raul::OpenCLKernelManager manager(queue);

    raul::MemoryManagerGPU memory_manager;

    EXPECT_THROW(memory_manager.createTensor("1", 1, 1, 1, 1), raul::Exception);

    memory_manager.setGpuAttribs(&manager);

    EXPECT_EQ(memory_manager.tensorExists("test"), false);
    EXPECT_EQ(memory_manager.tensorExists(""), false);

    {
        raul::TensorGPU& blob = *memory_manager.createTensor("test", 1, 1, 1, 1);
        EXPECT_EQ(blob.size(), static_cast<size_t>(1));
    }

    {
        raul::TensorGPU& blob = *memory_manager.createTensor("test1", 1, 10, 1, 1);
        EXPECT_EQ(blob.size(), static_cast<size_t>(10));
    }

    {
        raul::TensorGPU& blob = *memory_manager.createTensor("test11", 1, 10, 1, 1, 1.0_dt);
        EXPECT_EQ(blob.size(), static_cast<size_t>(10));

        raul::Tensor blobTmp(raul::TensorGPUHelper(blob, &manager));
        for (raul::dtype d : blobTmp)
            EXPECT_EQ(d, 1.0_dt);
    }

    EXPECT_THROW(memory_manager.createTensor("", 1, 1, 1, 1), raul::Exception);
    EXPECT_THROW(memory_manager.createTensor("test", 1, 1, 1, 1), raul::Exception);

    EXPECT_EQ(memory_manager.tensorExists("test"), true);

    EXPECT_THROW(memory_manager.getTensor("test2"), raul::Exception);

    raul::TensorGPU& blob = memory_manager.getTensor("test");

    EXPECT_EQ(blob.size(), static_cast<size_t>(1));

    EXPECT_THROW(memory_manager.deleteTensor("test2"), raul::Exception);
    memory_manager.deleteTensor("test");
    EXPECT_THROW(memory_manager.deleteTensor("test"), raul::Exception);
    EXPECT_EQ(memory_manager.tensorExists("test"), false);
    EXPECT_THROW(memory_manager.getTensor("test"), raul::Exception);
}

TEST(TestMemoryManagerGPU, ClearGpuUnit)
{
    PROFILE_TEST

    GPU_ONLY_TEST

    auto [platform, device, context] = raul::Common::getGpuPlatformDeviceAndContext();

    cl::CommandQueue queue(context, device);
    raul::OpenCLKernelManager manager(queue);
    raul::MemoryManagerGPU memory_manager;
    memory_manager.setGpuAttribs(&manager);

    EXPECT_EQ(memory_manager.size(), static_cast<size_t>(0));

    memory_manager.createTensor("test", 1, 1, 1, 1);
    memory_manager.createTensor("test2", 1, 1, 1, 1);

    EXPECT_EQ(memory_manager.tensorExists("test"), true);
    EXPECT_EQ(memory_manager.tensorExists("test2"), true);

    EXPECT_EQ(memory_manager.size(), static_cast<size_t>(2));

    memory_manager.clear();

    EXPECT_EQ(memory_manager.tensorExists("test"), false);
    EXPECT_EQ(memory_manager.tensorExists("test2"), false);

    EXPECT_EQ(memory_manager.size(), static_cast<size_t>(0));
}

TEST(TestMemoryManagerGPU, AnonymousGpuUnit)
{
    PROFILE_TEST

    GPU_ONLY_TEST

    auto [platform, device, context] = raul::Common::getGpuPlatformDeviceAndContext();

    cl::CommandQueue queue(context, device);
    raul::OpenCLKernelManager manager(queue);
    raul::MemoryManagerGPU memoryManager;
    memoryManager.setGpuAttribs(&manager);

    raul::TensorGPU& t1 = *memoryManager.createTensor(1, 1, 1, 1);
    raul::TensorGPU& t2 = *memoryManager.createTensor(1, 1, 1, 2);
    raul::TensorGPU& t3 = *memoryManager.createTensor(1, 1, 1, 3, 1.0_dt);

    EXPECT_EQ(memoryManager.size(), 3u);

    EXPECT_EQ(t1.getName(), "TensorGPU_1");
    EXPECT_EQ(t2.getName(), "TensorGPU_2");
    EXPECT_EQ(t3.getName(), "TensorGPU_3");
    EXPECT_EQ(t3.size(), 3u);
    raul::Tensor t3Tmp(raul::TensorGPUHelper(t3, &manager));
    EXPECT_EQ(t3Tmp[0], 1.0_dt);
    EXPECT_EQ(t3Tmp[1], 1.0_dt);
    EXPECT_EQ(t3Tmp[2], 1.0_dt);
}

TEST(TestMemoryManagerGPU, ShapeGpuUnit)
{
    PROFILE_TEST

    GPU_ONLY_TEST

    auto [platform, device, context] = raul::Common::getGpuPlatformDeviceAndContext();

    cl::CommandQueue queue(context, device);
    raul::OpenCLKernelManager manager(queue);
    raul::MemoryManagerGPU memoryManager;

    EXPECT_THROW(memoryManager.createShape("test", 1, 1, 1, 2, raul::AllocationMode::STANDARD), raul::Exception);

    memoryManager.setGpuAttribs(&manager);

    memoryManager.createShape("test", 1, 1, 1, 2, raul::AllocationMode::STANDARD);

    EXPECT_EQ(memoryManager("test").size(), 0u);
    EXPECT_EQ(memoryManager("test").getBatchSize(), 1u);
    EXPECT_EQ(memoryManager("test").getDepth(), 1u);
    EXPECT_EQ(memoryManager("test").getHeight(), 1u);
    EXPECT_EQ(memoryManager("test").getWidth(), 2u);
}

TEST(TestMemoryManagerGPU, CopyGpuUnit)
{
    PROFILE_TEST

    GPU_ONLY_TEST

    auto [platform, device, context] = raul::Common::getGpuPlatformDeviceAndContext();

    cl::CommandQueue queue(context, device);
    raul::OpenCLKernelManager manager(queue);
    raul::MemoryManagerGPU memoryManager;
    memoryManager.setGpuAttribs(&manager);

    memoryManager.createTensor("t1", 1, 2, 3, 4, 10_dt);
    memoryManager.createTensor("t2", 1, 2, 3, 4);

    {
        raul::Tensor tmp(memoryManager["t1"]);
        EXPECT_EQ(tmp.size(), 24u);
        for (auto val : tmp)
        {
            EXPECT_EQ(val, 10_dt);
        }
    }

    memoryManager["t2"] = memoryManager["t1"];

    {
        raul::Tensor tmp(memoryManager["t2"]);
        EXPECT_EQ(tmp.size(), 24u);
        for (auto val : tmp)
        {
            EXPECT_EQ(val, 10_dt);
        }
    }
}

TEST(TestMemoryManagerGPU, CopyFromToGpuUnit)
{
    PROFILE_TEST

    GPU_ONLY_TEST

    auto [platform, device, context] = raul::Common::getGpuPlatformDeviceAndContext();

    cl::CommandQueue queue(context, device);
    raul::OpenCLKernelManager manager(queue);
    raul::MemoryManagerGPU memoryManager;
    memoryManager.setGpuAttribs(&manager);

    raul::TensorGPU& tg = *memoryManager.createTensor(1, 2, 3, 4, 1.0_dt);
    raul::TensorGPU& ttg = *memoryManager.createTensor(1, 2, 3, 4);

    raul::Tensor t(1, 2, 3, 4);

    EXPECT_THROW(memoryManager.copy("1", t), raul::Exception);

    memoryManager.copy(tg.getName(), t);

    EXPECT_EQ(t.size(), 24u);

    for (auto val : t)
    {
        EXPECT_EQ(val, 1_dt);
    }

    EXPECT_THROW(memoryManager.copy(t, "1"), raul::Exception);

    memoryManager.copy(t, ttg.getName());

    {
        raul::Tensor tmp(raul::TensorGPUHelper(ttg, &manager));
        EXPECT_EQ(tmp.size(), 24u);
        for (auto val : tmp)
        {
            EXPECT_EQ(val, 1_dt);
        }
    }

    memoryManager.createTensor("t1", 1, 2, 3, 5);
    memoryManager["t1"] = raul::Tensor("", 1, 2, 3, 5, 2_dt);

    {
        raul::Tensor tmp(memoryManager["t1"]);
        EXPECT_EQ(tmp.size(), 30u);
        for (auto val : tmp)
        {
            EXPECT_EQ(val, 2_dt);
        }
    }

    {
        raul::Tensor tmp(1, 2, 3, 5);
        tmp = memoryManager["t1"];
        EXPECT_EQ(tmp.size(), 30u);
        for (auto val : tmp)
        {
            EXPECT_EQ(val, 2_dt);
        }
    }
}

} // UT namespace
