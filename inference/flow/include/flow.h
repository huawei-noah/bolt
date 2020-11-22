/**
 * @file
 * @brief Flow API Document
 *
 * @copyright
 * @code
 * Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
 * WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 * @endcode
 */

#ifndef FLOW_INCLUDE_FLOW_H_
#define FLOW_INCLUDE_FLOW_H_

#include <string>
#include <vector>
#include <map>
#include <queue>
#include "flow.pb.h"

#include "node.h"
#include "task.h"
#include "schedule.h"
#include "tensor.hpp"

class Flow {
public:
    Flow();

    ~Flow();

    /**
     * @brief initialize flow
     * @param  graphPaths     predefined flow graph file path array
     * @param  precision      data process precision
     * @param  affinityPolicy CPU affinity setting
     * @param  cpuThreads     the number of CPU cores to use(default is 1)
     * @param  useGPU         whether to use ARM MALI GPU(default is false)
     *
     * @return
     */
    void init(std::vector<std::string> graphPaths,
        DataType precision,
        AffinityPolicy affinityPolicy = AFFINITY_CPU_HIGH_PERFORMANCE,
        int cpuThreads = 1,
        bool useGPU = true);

    /**
     * @brief
     * @param  task           predefined flow task
     *
     * @return
     */
    void enqueue(Task task);

    /** get already finished tasks
     * @brief
     * @param  block          set to blocked until all tasks has finished(default is false)
     *
     * @return finishedTasks: array of already finished tasks
     */
    std::vector<Task> dequeue(bool block = false);

    /**
     * @brief get the current number of unfinished tasks
     *
     * @return size : the number of unfinished tasks
     */
    unsigned int size();

private:
    Schedule<flow::GraphParameter, Node, Tensor> schedule;
    std::queue<std::shared_ptr<Task>> tasks;
};
#endif  // FLOW_INCLUDE_FLOW_H_
