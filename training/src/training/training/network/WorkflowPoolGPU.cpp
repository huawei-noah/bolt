// Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "WorkflowPoolGPU.h"

#define PADDING_BYTES(a, b) (((b) - ((a) % (b))) % (b))

namespace raul
{

void WorkflowPoolGPU::createPool(const MemoryManagerGPU& manager)
{
    if (mIntervals.empty())
    {
        return;
    }

    if (!mIntervalsPrepared)
    {
        THROW_NONAME("WorkflowPoolGPU", "intervals hasn`t been prepared");
    }

    if (mPoolPrepared)
    {
        return;
    }

    size_t alignBytes = manager.getAlignment();

    mPool = cl::Buffer();

    size_t poolSize = 0;

    std::vector<Interval*> intervalsOrdered;

    for (auto& interval : mIntervals)
    {
        Name targetName = interval.tName;
        if (!manager.tensorExists(targetName))
        {
            auto itM = mTensorNameMapper.find(targetName);
            if (itM != mTensorNameMapper.end())
            {
                targetName = (*itM).second;
            }
        }
        const TensorGPU& t = manager(targetName);

        if (t.getAllocationMode() != AllocationMode::POOL)
        {
            THROW_NONAME("WorkflowPoolGPU", "tenosor [" + targetName + "] not in POOL allocation mode");
        }

        size_t totalSize = t.getShape().total_size();
        size_t totalSizeBytes = totalSize * sizeof(dtype);

        if (totalSize == 0)
        {
            interval.offset = 0;
            continue;
        }

        totalSizeBytes += PADDING_BYTES(totalSizeBytes, alignBytes);
        totalSize = totalSizeBytes / sizeof(dtype);

        size_t curOffset = 0;

        for (const auto intervalOrdered : intervalsOrdered)
        {
            // time overlap
            if (interval.start <= intervalOrdered->finish && interval.finish >= intervalOrdered->start)
            {
                if (curOffset + totalSize <= intervalOrdered->offset)
                {
                    break;
                }
                curOffset = std::max(curOffset, intervalOrdered->upperBound);
            }
        }

        interval.offset = curOffset;
        interval.upperBound = curOffset + totalSize;

        poolSize = std::max(poolSize, curOffset + totalSize);

        {
            auto cmp = [](Interval* interval, size_t offset) { return interval->offset < offset; };
            auto it = std::lower_bound(intervalsOrdered.begin(), intervalsOrdered.end(), interval.offset, cmp);
            intervalsOrdered.insert(it, &interval);
        }
    }

    mPool = manager.getKernelManager()->createBuffer(poolSize * sizeof(raul::dtype), "WorkflowPoolGPU[createPool]");

    mTotalMemory = poolSize * sizeof(dtype);

    mPoolPrepared = true;
}

void WorkflowPoolGPU::clearPool()
{
    mPoolPrepared = false;
}

cl::Buffer WorkflowPoolGPU::getOffset(const Name& tName)
{
    if (!mPoolPrepared)
    {
        THROW_NONAME("WorkflowPoolGPU", "pool hasn`t been prepared");
    }

    auto it = mNameToInterval.find(tName);

    if (it == mNameToInterval.end())
    {
        THROW_NONAME("WorkflowPoolGPU", "wrong tensor name " + tName);
    }

    size_t size = mIntervals[(*it).second].upperBound - mIntervals[(*it).second].offset;

    cl_buffer_region rgn = { mIntervals[(*it).second].offset * sizeof(raul::dtype), size * sizeof(raul::dtype) };

    return mPool.createSubBuffer(CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &rgn);
}

void WorkflowPoolGPU::setTensorNameMapper(const Name& from, const Name& to)
{
    mTensorNameMapper.insert({ from, to });
}
} // namespace raul
