// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "WorkflowPool.h"

namespace raul
{
WorkflowBasicPool::WorkflowBasicPool()
    : mPoolPrepared(false)
    , mIntervalsPrepared(false)
    , mTotalMemory(0u)
{
}

void WorkflowBasicPool::createIntervals(const Names& lNames, const Timeline& timeline)
{
    if (timeline.empty())
    {
        return;
    }

    mPoolPrepared = false;
    mTotalMemory = 0u;

    std::unordered_map<Name, size_t> layers;

    for (size_t q = 0; q < lNames.size(); ++q)
    {
        layers.insert({ lNames[q], q });
    }

    mIntervals.clear();
    mNameToInterval.clear();

    for (const auto& range : timeline)
    {
        const Name& tName = range.first;
        auto itF = layers.find(range.second.first);
        auto itL = layers.find(range.second.second);

        if (itF == layers.end())
        {
            THROW_NONAME("WorkflowPool", "layers not equal timeline");
        }

        if (itL == layers.end())
        {
            THROW_NONAME("WorkflowPool", "layers not equal timeline");
        }

        size_t fLayerIndex = (*itF).second;
        size_t lLayerIndex = (*itL).second;

        if (mNameToInterval.find(tName) != mNameToInterval.end())
        {
            THROW_NONAME("WorkflowPool", "tensor duplication detected");
        }

        if (lLayerIndex < fLayerIndex)
        {
            THROW_NONAME("WorkflowPool", "wrong layers order");
        }

        mNameToInterval.insert({ tName, mIntervals.size() });
        mIntervals.emplace_back(Interval(fLayerIndex, lLayerIndex, tName));
    }

    if (mIntervals.empty())
    {
        THROW_NONAME("WorkflowPool", "no data");
    }

    mIntervalsPrepared = true;
}

template<typename MM>
void WorkflowPool<MM>::createPool(const MM& manager)
{
    if (mIntervals.empty())
    {
        return;
    }

    if (!mIntervalsPrepared)
    {
        THROW_NONAME("WorkflowPool", "intervals has not been prepared");
    }

    if (mPoolPrepared)
    {
        return;
    }

    mPool.clear();

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
        const auto& t = manager[targetName];

        if (t.getAllocationMode() != AllocationMode::POOL)
        {
            THROW_NONAME("WorkflowPool", "tensor [" + targetName + "] not in POOL allocation mode");
        }

        const size_t totalSize = t.getShape().total_size();

        if (totalSize == 0)
        {
            interval.offset = 0;
            continue;
        }

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

    mPool.resize(poolSize);

    mTotalMemory = poolSize * sizeof(typename MM::type);

    mPoolPrepared = true;
}

template<typename MM>
void WorkflowPool<MM>::clearPool()
{
    mPoolPrepared = false;
}

template<typename MM>
typename WorkflowPool<MM>::type* WorkflowPool<MM>::getOffset(const Name& tName)
{
    if (!mPoolPrepared)
    {
        THROW_NONAME("WorkflowPool", "pool has not been prepared");
    }

    auto it = mNameToInterval.find(tName);

    if (it == mNameToInterval.end())
    {
        THROW_NONAME("WorkflowPool", "wrong tensor name " + tName);
    }

    return &mPool[mIntervals[(*it).second].offset];
}

template<typename MM>
void WorkflowPool<MM>::setTensorNameMapper(const Name& from, const Name& to)
{
    mTensorNameMapper.insert({ from, to });
}

template class WorkflowPool<MemoryManager>;
template class WorkflowPool<MemoryManagerFP16>;
} // namespace raul
