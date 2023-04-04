// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef WORKFLOWPOOL_H
#define WORKFLOWPOOL_H

#include <training/system/Name.h>

#include "Workflow.h"

namespace raul
{
class WorkflowBasicPool
{
  public:
    WorkflowBasicPool();

    ~WorkflowBasicPool(){}

    // Tensor name vs pair - Layer first, last names in sequence
    typedef std::map<Name, std::pair<Name, Name>> Timeline;

    void createIntervals(const Names& lNames, const Timeline& timeline);

    struct Interval
    {
        Interval(size_t s, size_t f, const raul::Name& name)
            : start(s)
            , finish(f)
            , tName(name)
            , offset(0)
            , upperBound(0)
        {
        }

        size_t start, finish;
        raul::Name tName;
        size_t offset;
        size_t upperBound;
    };

  protected:
    bool mPoolPrepared;

    bool mIntervalsPrepared;

    std::vector<Interval> mIntervals;

    std::unordered_map<Name, size_t> mNameToInterval;

    size_t mTotalMemory;
};

/**
 * @brief Class to calculate one large pool for tensors
 *
 */
template<typename MM>
class WorkflowPool : public WorkflowBasicPool
{
  public:
    WorkflowPool() {}
    ~WorkflowPool(){}

    void createPool(const MM& manager);

    void clearPool();

    typedef typename MM::type type;

    type* getOffset(const Name& tName);

    typedef std::vector<type> Pool;
    const Pool& getPool() const { return mPool; } // for tests

    /**
     * @brief Get size of allocated memory by pool in bytes
     */
    size_t getPoolSize() const { return mTotalMemory; }

    void setTensorNameMapper(const Name& from, const Name& to);
    void clearTensorNameMapper() { mTensorNameMapper.clear(); }
  private:
    std::unordered_map<Name, Name> mTensorNameMapper;

    Pool mPool;
};
} // raul namespace

#endif
