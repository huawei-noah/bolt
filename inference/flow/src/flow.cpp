// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "flow.h"

Flow::Flow()
{}

Flow::~Flow()
{
    this->schedule.end();
}

void Flow::init(std::vector<std::string> graphPath,
    DataType precision,
    AffinityPolicy affinityPolicy,
    int cpuThreads,
    bool useGPU)
{
    UNI_DEBUG_LOG("flow (schedule) init begin\n");
    flowBuildFunctions();
    this->schedule.init(graphPath, precision, affinityPolicy, cpuThreads, useGPU);
    UNI_DEBUG_LOG("flow init end\n");
}

void Flow::enqueue(Task task)
{
    UNI_DEBUG_LOG("user enqueues task: begin\n");
    if (task.status != TASK_READY) {
        UNI_ERROR_LOG("task is not ready to add queue\n");
    }
    std::shared_ptr<Task> taskPtr = std::shared_ptr<Task>(new Task(&task));
    this->tasks.emplace(taskPtr);
    this->schedule.enqueue(taskPtr.get());
    UNI_DEBUG_LOG("user enqueues task: end\n");
}

std::vector<Task> Flow::dequeue(bool block)
{
    std::vector<Task> outputs;
    if (this->tasks.size() == 0) {
        return outputs;
    }
    for (;;) {
        if (this->tasks.size() == 0) {
            break;
        }
        auto task = this->tasks.front();
        if (task->status == TASK_END) {
            outputs.push_back(*task.get());
            this->tasks.pop();
        } else {
            if (!block) {
                break;
            }
        }
    }
    if (outputs.size() > 0) {
        UNI_DEBUG_LOG("user get result (num=%d) end\n", (int)outputs.size());
    }
    return outputs;
}

unsigned int Flow::size()
{
    return this->tasks.size();
}
