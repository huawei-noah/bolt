/**
 * @file
 * @brief Task API Document
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

#ifndef UNI_INCLUDE_TASK_H_
#define UNI_INCLUDE_TASK_H_

#include <string>
#include <map>
#include <memory>
#include <iostream>
#include <sstream>

#include "tensor.hpp"
#include "profiling.h"

/** task status */
typedef enum TaskStatus {
    TASK_CREATE,  ///< task is created
    TASK_READY,   ///< task can be processed
    TASK_RUN,     ///< task is being processed
    TASK_END      ///< task has been finished
} TaskStatus;

class Task {
public:
    /**
     * @brief Task constructor
     *
     * @return
     */
    Task()
    {
        this->status = TASK_CREATE;
    }

    /**
     * @brief Task constructor
     * @param  graphPath      predefined flow graph file path
     * @param  data           graph input data
     *
     * @return
     */
    Task(std::string graphPath, std::map<std::string, std::shared_ptr<Tensor>> data)
    {
        this->set(ut_time_ms(), graphPath, data, TASK_READY);
    }

    /**
     * @brief Task constructor
     * @param  id             time series data stamp
     * @param  graphPath      predefined flow graph file path
     * @param  data           graph input data map
     *
     * @return
     */
    Task(int id, std::string graphPath, std::map<std::string, std::shared_ptr<Tensor>> data)
    {
        this->set(id, graphPath, data, TASK_READY);
    }

    /**
     * @brief Task copy constructor
     * @param  task           copy from task to generate new Task
     *
     * @return
     */
    Task(Task *task)
    {
        this->set(task->id, task->graphPath, task->data, task->status);
    }

    /**
     * @brief Task set function
     * @param  id             time series data stamp
     * @param  graphPath      predefined flow graph file path
     * @param  data           graph input data map
     * @param  status         task status
     *
     * @return
     */
    void set(int id,
        std::string graphPath,
        std::map<std::string, std::shared_ptr<Tensor>> data,
        TaskStatus status)
    {
        this->id = id;
        this->graphPath = graphPath;
        this->data = data;
        this->status = status;
    }

    friend std::ostream &operator<<(std::ostream &os, const Task &task)
    {
        os << "Task " << task.id << "(timestamp " << task.id << ", status " << task.status
           << ", graph " << task.graphPath << ", data " << std::endl;
        for (auto iter : task.data) {
            os << "tensor name " << iter.first << " " << iter.second->string(1) << std::endl;
        }
        os << ")";
        return os;
    }

    /** time stamp */
    int id;
    /** task status */
    TaskStatus status;
    /** predefined flow graph file path */
    std::string graphPath;
    /** graph data */
    std::map<std::string, std::shared_ptr<Tensor>> data;
};
#endif  // UNI_INCLUDE_TASK_H_
