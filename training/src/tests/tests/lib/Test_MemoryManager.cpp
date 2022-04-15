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

#include <training/base/common/MemoryManager.h>

namespace UT
{

TEST(TestMemoryManager, GeneralUnit)
{
    PROFILE_TEST
    raul::MemoryManager memory_manager;
    EXPECT_EQ(memory_manager.tensorExists("test"), false);
    EXPECT_EQ(memory_manager.tensorExists(""), false);

    {
        raul::Tensor& blob = *memory_manager.createTensor("test", 1, 1, 1, 1);
        EXPECT_EQ(blob.size(), static_cast<size_t>(1));
    }

    {
        raul::Tensor& blob = *memory_manager.createTensor("test1", 1, 10, 1, 1);
        EXPECT_EQ(blob.size(), static_cast<size_t>(10));
    }

    {
        raul::Tensor& blob = *memory_manager.createTensor("test11", 1, 10, 1, 1, 1.0_dt);
        EXPECT_EQ(blob.size(), static_cast<size_t>(10));
        for (raul::dtype d : blob)
            EXPECT_EQ(d, 1.0_dt);
    }

    EXPECT_THROW(memory_manager.createTensor("test", 1, 1, 1, 1), raul::Exception);

    EXPECT_EQ(memory_manager.tensorExists("test"), true);

    EXPECT_THROW(memory_manager.getTensor("test2"), raul::Exception);

    raul::Tensor& blob = memory_manager.getTensor("test");

    EXPECT_EQ(blob.size(), static_cast<size_t>(1));
    // blob.getData().resize(10);
    // EXPECT_EQ(memory_manager.getTensor("test").size(), static_cast<size_t>(10));

    EXPECT_THROW(memory_manager.deleteTensor("test2"), raul::Exception);
    memory_manager.deleteTensor("test");
    EXPECT_THROW(memory_manager.deleteTensor("test"), raul::Exception);
    EXPECT_EQ(memory_manager.tensorExists("test"), false);
    EXPECT_THROW(memory_manager.getTensor("test"), raul::Exception);
}

TEST(TestMemoryManager, ClearUnit)
{
    PROFILE_TEST
    raul::MemoryManager memory_manager;
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

TEST(TestMemoryManager, AnonymousUnit)
{
    PROFILE_TEST
    raul::MemoryManager memoryManager;

    raul::Tensor& t1 = *memoryManager.createTensor(1, 1, 1, 1);
    raul::Tensor& t2 = *memoryManager.createTensor(1, 1, 1, 2);
    raul::Tensor& t3 = *memoryManager.createTensor(1, 1, 1, 3, 1.0_dt);

    EXPECT_EQ(memoryManager.size(), 3u);

    EXPECT_EQ(t1.getName(), "Tensor_1");
    EXPECT_EQ(t2.getName(), "Tensor_2");
    EXPECT_EQ(t3.getName(), "Tensor_3");
    EXPECT_EQ(t3.size(), 3u);
    EXPECT_EQ(t3[0], 1.0_dt);
    EXPECT_EQ(t3[1], 1.0_dt);
    EXPECT_EQ(t3[2], 1.0_dt);
}

#if 0
TEST(TestMemoryManager, ShapeUnit)
{
    PROFILE_TEST
    raul::MemoryManager memoryManager;

    memoryManager.createShape("test", 1, 1, 1, 2);

    EXPECT_EQ(memoryManager["test"].size(), 0u);
    EXPECT_EQ(memoryManager["test"].getBatchSize(), 1u);
    EXPECT_EQ(memoryManager["test"].getDepth(), 1u);
    EXPECT_EQ(memoryManager["test"].getHeight(), 1u);
    EXPECT_EQ(memoryManager["test"].getWidth(), 2u);

    memoryManager.createShape("test2", memoryManager["test"]);
    EXPECT_EQ(memoryManager["test2"].size(), 0u);
    EXPECT_EQ(memoryManager["test2"].getBatchSize(), 1u);
    EXPECT_EQ(memoryManager["test2"].getDepth(), 1u);
    EXPECT_EQ(memoryManager["test2"].getHeight(), 1u);
    EXPECT_EQ(memoryManager["test2"].getWidth(), 2u);

    memoryManager["test"].allocate();
    EXPECT_EQ(memoryManager["test"].size(), 2u);
    EXPECT_EQ(memoryManager["test2"].size(), 0u);

    memoryManager["test2"].allocate();
    EXPECT_EQ(memoryManager["test2"].size(), 2u);

    raul::Tensor* shape = memoryManager.createShape(memoryManager["test"]);
    EXPECT_NE(shape, nullptr);
    EXPECT_EQ(shape, &memoryManager[shape->getName()]);
    EXPECT_EQ(shape->getName(), "Tensor_1");
    EXPECT_EQ(shape->size(), 0u);
    EXPECT_EQ(shape->getBatchSize(), 1u);
    EXPECT_EQ(shape->getDepth(), 1u);
    EXPECT_EQ(shape->getHeight(), 1u);
    EXPECT_EQ(shape->getWidth(), 2u);
}
#endif

TEST(TestMemoryManager, AliasesUnit)
{
    PROFILE_TEST
    raul::MemoryManager memoryManager;
    const raul::MemoryManager& memoryManagerC = memoryManager;

    memoryManager.createTensor("test", 1, 1, 1, 2);
    EXPECT_THROW(memoryManager.createAlias("test", "test"), raul::Exception);
    EXPECT_THROW(memoryManager.createAlias("test2", "test3"), raul::Exception);
    EXPECT_THROW(memoryManager.createAlias("test2", "test"), raul::Exception);

    EXPECT_NO_THROW(memoryManager.createAlias("test", "test2"));
    EXPECT_THROW(memoryManager.createAlias("test2", "test2"), raul::Exception);
    EXPECT_NO_THROW(memoryManager.createAlias("test2", "test3")); // alias on alias possible

    EXPECT_EQ(memoryManager.getTensor("test").getWidth(), 2u);
    EXPECT_EQ(memoryManager.getTensor("test2").getWidth(), 2u);
    EXPECT_EQ(memoryManager.getTensor("test3").getWidth(), 2u);
    EXPECT_EQ(memoryManager.getTensor("test").getName(), "test");
    EXPECT_EQ(memoryManager.getTensor("test2").getName(), "test");
    EXPECT_EQ(memoryManager.getTensor("test3").getName(), "test");

    EXPECT_EQ(memoryManager["test"].getWidth(), 2u);
    EXPECT_EQ(memoryManager["test2"].getWidth(), 2u);
    EXPECT_EQ(memoryManager["test3"].getWidth(), 2u);
    EXPECT_EQ(memoryManager["test"].getName(), "test");
    EXPECT_EQ(memoryManager["test2"].getName(), "test");
    EXPECT_EQ(memoryManager["test3"].getName(), "test");

    EXPECT_EQ(memoryManagerC.getTensor("test").getWidth(), 2u);
    EXPECT_EQ(memoryManagerC.getTensor("test2").getWidth(), 2u);
    EXPECT_EQ(memoryManagerC.getTensor("test3").getWidth(), 2u);
    EXPECT_EQ(memoryManagerC.getTensor("test").getName(), "test");
    EXPECT_EQ(memoryManagerC.getTensor("test2").getName(), "test");
    EXPECT_EQ(memoryManagerC.getTensor("test3").getName(), "test");

    EXPECT_EQ(memoryManagerC["test"].getWidth(), 2u);
    EXPECT_EQ(memoryManagerC["test2"].getWidth(), 2u);
    EXPECT_EQ(memoryManagerC["test3"].getWidth(), 2u);
    EXPECT_EQ(memoryManagerC["test"].getName(), "test");
    EXPECT_EQ(memoryManagerC["test2"].getName(), "test");
    EXPECT_EQ(memoryManagerC["test3"].getName(), "test");

    EXPECT_THROW(memoryManager.createTensor("test", 1, 1, 1, 1), raul::Exception);
    EXPECT_THROW(memoryManager.createTensor("test2", 1, 1, 1, 1), raul::Exception);
    EXPECT_THROW(memoryManager.createTensor("test3", 1, 1, 1, 1), raul::Exception);

    EXPECT_THROW(memoryManager.createTensor("test", 1, 1, 1, 1, 1.0_dt), raul::Exception);
    EXPECT_THROW(memoryManager.createTensor("test2", 1, 1, 1, 1, 1.0_dt), raul::Exception);
    EXPECT_THROW(memoryManager.createTensor("test3", 1, 1, 1, 1, 1.0_dt), raul::Exception);

    EXPECT_THROW(memoryManager.createShape("test", 1, 1, 1, 1, raul::AllocationMode::STANDARD), raul::Exception);
    EXPECT_THROW(memoryManager.createShape("test2", 1, 1, 1, 1, raul::AllocationMode::STANDARD), raul::Exception);
    EXPECT_THROW(memoryManager.createShape("test3", 1, 1, 1, 1, raul::AllocationMode::STANDARD), raul::Exception);

    EXPECT_EQ(memoryManager.size(), 1u);
    EXPECT_EQ(memoryManager.getTotalMemory(), 8u);

    EXPECT_NO_THROW(memoryManager.createShape("shape", 1, 1, 1, 4, raul::AllocationMode::STANDARD));
    EXPECT_NO_THROW(memoryManager.createAlias("shape", "shape2"));
    EXPECT_NO_THROW(memoryManager.createAlias("shape2", "shape3"));

    EXPECT_EQ(memoryManagerC["shape"].getWidth(), 4u);
    EXPECT_EQ(memoryManagerC["shape2"].getWidth(), 4u);
    EXPECT_EQ(memoryManagerC["shape3"].getWidth(), 4u);
    EXPECT_EQ(memoryManagerC["shape"].getName(), "shape");
    EXPECT_EQ(memoryManagerC["shape2"].getName(), "shape");
    EXPECT_EQ(memoryManagerC["shape3"].getName(), "shape");

    EXPECT_EQ(memoryManager.size(), 2u);
    EXPECT_EQ(memoryManager.getTotalMemory(), 8u);

    EXPECT_EQ(memoryManager.tensorExists("test"), true);
    EXPECT_EQ(memoryManager.tensorExists("test2"), true);
    EXPECT_EQ(memoryManager.tensorExists("test3"), true);

    EXPECT_EQ(memoryManager.tensorExists("shape"), true);
    EXPECT_EQ(memoryManager.tensorExists("shape2"), true);
    EXPECT_EQ(memoryManager.tensorExists("shape3"), true);

    EXPECT_NO_THROW(memoryManager.deleteTensor("shape3"));
    EXPECT_EQ(memoryManager.tensorExists("shape"), false);
    EXPECT_EQ(memoryManager.tensorExists("shape2"), false);
    EXPECT_EQ(memoryManager.tensorExists("shape3"), false);

    EXPECT_EQ(memoryManager.size(), 1u);
    EXPECT_EQ(memoryManager.getTotalMemory(), 8u);

    EXPECT_NO_THROW(memoryManager.deleteTensor("test"));
    EXPECT_EQ(memoryManager.tensorExists("test"), false);
    EXPECT_EQ(memoryManager.tensorExists("test2"), false);
    EXPECT_EQ(memoryManager.tensorExists("test3"), false);

    EXPECT_EQ(memoryManager.size(), 0u);
    EXPECT_EQ(memoryManager.getTotalMemory(), 0u);
}

} // UT namespace