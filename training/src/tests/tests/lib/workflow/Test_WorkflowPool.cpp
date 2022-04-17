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

#include <training/compiler/WorkflowPool.h>

namespace UT
{

TEST(TestWorkflowPool, GeneralIncorrectOrderUnit)
{
    PROFILE_TEST

    raul::WorkflowPool<raul::MemoryManager> pool;

    EXPECT_THROW(pool.getOffset(""), raul::Exception);

    raul::MemoryManager manager;
    EXPECT_NO_THROW(pool.createPool(manager));
}

TEST(TestWorkflowPool, CreateIntervalsIncorrectParamsUnit)
{
    PROFILE_TEST

    raul::WorkflowPool<raul::MemoryManager> pool;

    {
        raul::Names layers = {};
        raul::WorkflowPool<raul::MemoryManager>::Timeline timeline = {};

        EXPECT_NO_THROW(pool.createIntervals(layers, timeline));
    }

    {
        raul::Names layers = { "LA", "L2", "L3", "L4" };
        raul::WorkflowPool<raul::MemoryManager>::Timeline timeline = {
            { "t1", { "L1", "L3" } },
            { "t2", { "L2", "L3" } },
            { "t3", { "L2", "L4" } },
        };

        EXPECT_THROW(pool.createIntervals(layers, timeline), raul::Exception);
    }

    {
        raul::Names layers = { "L1", "LA", "L3", "L4" };
        raul::WorkflowPool<raul::MemoryManager>::Timeline timeline = {
            { "t1", { "L1", "L2" } },
            { "t2", { "L1", "L3" } },
            { "t3", { "L1", "L4" } },
        };

        EXPECT_THROW(pool.createIntervals(layers, timeline), raul::Exception);
    }

    {
        raul::Names layers = { "L1", "L2", "L3", "L4" };
        raul::WorkflowPool<raul::MemoryManager>::Timeline timeline = {
            { "t1", { "L2", "L1" } },
            { "t2", { "L1", "L3" } },
            { "t3", { "L1", "L4" } },
        };

        EXPECT_THROW(pool.createIntervals(layers, timeline), raul::Exception);
    }
}

TEST(TestWorkflowPool, CreatePoolIncorrectParamsUnit)
{
    PROFILE_TEST

    raul::WorkflowPool<raul::MemoryManager> pool;

    {
        raul::Names layers = { "L1", "L2", "L3", "L4" };
        raul::WorkflowPool<raul::MemoryManager>::Timeline timeline = {
            { "t1", { "L1", "L2" } },
            { "t2", { "L2", "L3" } },
            { "t3", { "L2", "L4" } },
            { "t4", { "L3", "L4" } },
        };

        pool.createIntervals(layers, timeline);

        raul::MemoryManager manager;

        manager.createShape("t1", 10u, 1u, 1u, 1u, raul::AllocationMode::STANDARD);
        manager.createShape("t2", 10u, 2u, 1u, 1u, raul::AllocationMode::POOL);
        manager.createShape("t3", 10u, 3u, 1u, 1u, raul::AllocationMode::POOL);
        manager.createShape("t4", 10u, 4u, 1u, 1u, raul::AllocationMode::POOL);

        EXPECT_THROW(pool.createPool(manager), raul::Exception);
    }
}

TEST(TestWorkflowPool, CreatePoolUnit)
{
    PROFILE_TEST

    raul::WorkflowPool<raul::MemoryManager> pool;

    {
        /*
        1       2       3       4
        |-t1-10|        |-t4-40-|
                |-t2-20-|
                |-----t3-30-----|
        */
        raul::Names layers = { "L1", "L2", "L3", "L4" };
        raul::WorkflowPool<raul::MemoryManager>::Timeline timeline = {
            { "t1", { "L1", "L1" } },
            { "t2", { "L2", "L3" } },
            { "t3", { "L2", "L4" } },
            { "t4", { "L3", "L4" } },
        };

        pool.createIntervals(layers, timeline);

        raul::MemoryManager manager;

        manager.createShape("t1", 10u, 1u, 1u, 1u, raul::AllocationMode::POOL);
        manager.createShape("t2", 10u, 2u, 1u, 1u, raul::AllocationMode::POOL);
        manager.createShape("t3", 10u, 3u, 1u, 1u, raul::AllocationMode::POOL);
        manager.createShape("t4", 10u, 4u, 1u, 1u, raul::AllocationMode::POOL);

        pool.createPool(manager);

        EXPECT_EQ(pool.getPool().size(), 90u);

        EXPECT_EQ(pool.getOffset("t1"), &pool.getPool()[0]);
        EXPECT_EQ(pool.getOffset("t2"), &pool.getPool()[0]);
        EXPECT_EQ(pool.getOffset("t3"), &pool.getPool()[20]);
        EXPECT_EQ(pool.getOffset("t4"), &pool.getPool()[50]);
    }

    {
        /*
        1       2       3       4
        |-t1-10-|       |-t4-40-|
                |-t2-20-|
                |-----t3-30-----|
        */
        raul::Names layers = { "L1", "L2", "L3", "L4" };
        raul::WorkflowPool<raul::MemoryManager>::Timeline timeline = {
            { "t1", { "L1", "L2" } },
            { "t2", { "L2", "L3" } },
            { "t3", { "L2", "L4" } },
            { "t4", { "L3", "L4" } },
        };

        pool.createIntervals(layers, timeline);

        raul::MemoryManager manager;

        manager.createShape("t1", 10u, 1u, 1u, 1u, raul::AllocationMode::POOL);
        manager.createShape("t2", 10u, 2u, 1u, 1u, raul::AllocationMode::POOL);
        manager.createShape("t3", 10u, 3u, 1u, 1u, raul::AllocationMode::POOL);
        manager.createShape("t4", 10u, 4u, 1u, 1u, raul::AllocationMode::POOL);

        pool.createPool(manager);

        EXPECT_THROW(pool.getOffset(""), raul::Exception);

        EXPECT_EQ(pool.getPool().size(), 100u);

        EXPECT_EQ(pool.getOffset("t1"), &pool.getPool()[0]);
        EXPECT_EQ(pool.getOffset("t2"), &pool.getPool()[10]);
        EXPECT_EQ(pool.getOffset("t3"), &pool.getPool()[30]);
        EXPECT_EQ(pool.getOffset("t4"), &pool.getPool()[60]);
    }

    {
        /*
        1       2       3       4       5       6
        |-t1-100|       |-t2-10-|       |-t4-10-|
                        |---------t3-80---------|
        */
        raul::Names layers = { "L1", "L2", "L3", "L4", "L5", "L6" };
        raul::WorkflowPool<raul::MemoryManager>::Timeline timeline = {
            { "t1", { "L1", "L2" } },
            { "t2", { "L3", "L4" } },
            { "t3", { "L3", "L6" } },
            { "t4", { "L5", "L6" } },
        };

        pool.createIntervals(layers, timeline);

        raul::MemoryManager manager;

        manager.createShape("t1", 10u, 10u, 1u, 1u, raul::AllocationMode::POOL);
        manager.createShape("t2", 10u, 1u, 1u, 1u, raul::AllocationMode::POOL);
        manager.createShape("t3", 10u, 8u, 1u, 1u, raul::AllocationMode::POOL);
        manager.createShape("t4", 10u, 1u, 1u, 1u, raul::AllocationMode::POOL);

        pool.createPool(manager);

        EXPECT_THROW(pool.getOffset(""), raul::Exception);

        EXPECT_EQ(pool.getPool().size(), 100u);

        EXPECT_EQ(pool.getOffset("t1"), &pool.getPool()[0]);
        EXPECT_EQ(pool.getOffset("t2"), &pool.getPool()[0]);
        EXPECT_EQ(pool.getOffset("t3"), &pool.getPool()[10]);
        EXPECT_EQ(pool.getOffset("t4"), &pool.getPool()[0]);
    }

    {
        /*
        1       2       3       4       5       6
        |-t1-100|       |-t2-10-|       |-t4-10-|
                        |---------t3-70---------|
                                        |-t5-10-|
                                        |-t6-10-|
        */
        raul::Names layers = { "L1", "L2", "L3", "L4", "L5", "L6" };
        raul::WorkflowPool<raul::MemoryManager>::Timeline timeline = {
            { "t1", { "L1", "L2" } }, { "t2", { "L3", "L4" } }, { "t3", { "L3", "L6" } }, { "t4", { "L5", "L6" } }, { "t5", { "L5", "L6" } }, { "t6", { "L5", "L6" } },
        };

        pool.createIntervals(layers, timeline);

        raul::MemoryManager manager;

        manager.createShape("t1", 10u, 10u, 1u, 1u, raul::AllocationMode::POOL);
        manager.createShape("t2", 10u, 1u, 1u, 1u, raul::AllocationMode::POOL);
        manager.createShape("t3", 10u, 7u, 1u, 1u, raul::AllocationMode::POOL);
        manager.createShape("t4", 10u, 1u, 1u, 1u, raul::AllocationMode::POOL);
        manager.createShape("t5", 10u, 1u, 1u, 1u, raul::AllocationMode::POOL);
        manager.createShape("t6", 10u, 1u, 1u, 1u, raul::AllocationMode::POOL);

        pool.createPool(manager);

        EXPECT_THROW(pool.getOffset(""), raul::Exception);

        EXPECT_EQ(pool.getPool().size(), 100u);
        EXPECT_EQ(pool.getPoolSize(), 400u);

        EXPECT_EQ(pool.getOffset("t1"), &pool.getPool()[0]);
        EXPECT_EQ(pool.getOffset("t2"), &pool.getPool()[0]);
        EXPECT_EQ(pool.getOffset("t3"), &pool.getPool()[10]);
        EXPECT_EQ(pool.getOffset("t4"), &pool.getPool()[0]);
        EXPECT_EQ(pool.getOffset("t5"), &pool.getPool()[80]);
        EXPECT_EQ(pool.getOffset("t6"), &pool.getPool()[90]);
    }
}

} // UT namespace