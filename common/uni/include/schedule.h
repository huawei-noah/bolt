// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef FLOW_INCLUDE_SCHEDULE_H_
#define FLOW_INCLUDE_SCHEDULE_H_

#define _USE_WEIGHT_SHARE

#include <queue>
#include <map>
#include <vector>
#include <string>

#include "graph.h"
#include "task.h"

template <class GraphParameter, class ComputeNode, class DataTensor>
class Schedule {
public:
    Schedule()
    {
        pthread_mutex_init(&(this->taskQueueLock), NULL);
        pthread_cond_init(&(this->condition), NULL);
        this->threadNum = 0;
        this->stop = false;
    }

    ~Schedule()
    {
        pthread_mutex_lock(&(this->taskQueueLock));
        pthread_mutex_destroy(&(this->taskQueueLock));
        pthread_cond_destroy(&(this->condition));
#if !defined(__ANDROID_API__) && !defined(__APPLE__)
        pthread_barrier_destroy(&(this->barrier));
#endif
        delete[] this->threads;
    }

    int init(std::vector<std::string> graphPath,
        DataType dataType,
        AffinityPolicy affinityPolicy,
        int threadNum,
        bool useGPU)
    {
        UNI_DEBUG_LOG("schedule init begin\n");
        if (threadNum <= 0) {
            return 1;
        }
        if (pthread_mutex_init(&(this->taskQueueLock), NULL)) {
            return 1;
        }
        if (pthread_cond_init(&(this->condition), NULL)) {
            return 1;
        }
#if !defined(__ANDROID_API__) && !defined(__APPLE__)
        if (pthread_barrier_init(&(this->barrier), NULL, threadNum)) {
            return 1;
        }
#endif
        this->precision = dataType;
        this->deviceInfo = get_cpu_info(affinityPolicy);
        this->graphPath = graphPath;

#ifdef _USE_WEIGHT_SHARE
        for (unsigned int i = 0; i < graphPath.size(); i++) {
            this->graph[graphPath[i]].init(graphPath[i]);
            this->graph[graphPath[i]].ready(this->precision, this->deviceInfo.affinityPolicy, -1);
        }
#endif
#ifndef _USE_OPENMP
        int cpuId;
        if (this->deviceInfo.affinityPolicy == AFFINITY_CPU_LOW_POWER) {
            cpuId = 3;
        } else {
            cpuId = 4;
        }
        set_thread_affinity(0, &cpuId, 1);
#endif
        this->threadNum = threadNum;
        this->threads = new pthread_t[threadNum];
        for (int i = 0; i < threadNum; i++) {
            if (pthread_create(this->threads + i, NULL, worker, reinterpret_cast<void *>(this)) !=
                0) {
                this->end();
                UNI_ERROR_LOG("schedule create thread pool fail\n");
                return 1;
            }
        }
        this->useGPU = useGPU;
        UNI_DEBUG_LOG("schedule init end\n");
        return 0;
    }

    int end()
    {
        UNI_DEBUG_LOG("schedule exit begin\n");
        if (pthread_mutex_lock(&(this->taskQueueLock)) != 0) {
            return 1;
        }

        this->stop = true;

        if ((pthread_cond_broadcast(&(this->condition)) != 0) ||
            (pthread_mutex_unlock(&(this->taskQueueLock)) != 0)) {
            return 1;
        }

        for (int i = 0; i < this->threadNum; i++) {
            if (pthread_join(this->threads[i], NULL) != 0) {
                return 1;
            }
        }
        UNI_DEBUG_LOG("schedule exit end\n");
        return 0;
    }

    int enqueue(Task *task)
    {
        UNI_DEBUG_LOG("schedule enqueue task begin\n");
        if (this->threadNum == 0 || task == nullptr) {
            UNI_WARNING_LOG("schedule enqueue task failed because task or schedule is "
                            "deprecated\n");
            return 1;
        }
        if (pthread_mutex_lock(&(this->taskQueueLock)) != 0) {
            UNI_WARNING_LOG("schedule enqueue task failed because of can not acquire task queue "
                            "lock\n");
            return 1;
        }
        if (this->stop) {
            UNI_WARNING_LOG("schedule enqueue task failed because schedule has end\n");
            return 1;
        }
        this->taskQueue.push(task);
        if (pthread_cond_signal(&(this->condition)) != 0) {
            UNI_WARNING_LOG("schedule enqueue task failed because can not find worker\n");
            return 1;
        }
        pthread_mutex_unlock(&(this->taskQueueLock));
        UNI_DEBUG_LOG("schedule enqueue task end\n");
        return 0;
    }

private:
    int threadNum;
    pthread_mutex_t taskQueueLock;
    std::queue<Task *> taskQueue;
    pthread_cond_t condition;
#if !defined(__ANDROID_API__) && !defined(__APPLE__)
    pthread_barrier_t barrier;
#endif
    pthread_t *threads;
    int stop;

    std::vector<std::string> graphPath;
    std::map<std::string, Graph<GraphParameter, ComputeNode, DataTensor>> graph;

    bool useGPU;
    DeviceInfo deviceInfo;
    DataType precision;

    int getThreadId(pthread_t tid)
    {
        for (int i = 0; i < this->threadNum; i++) {
            if (this->threads[i] == tid) {
                return i;
            }
        }
        return -1;
    }

    static void *worker(void *_schedule)
    {
        Schedule *schedule = reinterpret_cast<Schedule *>(_schedule);
        int threadId = schedule->getThreadId(pthread_self());
        UNI_DEBUG_LOG("worker(%d) begin\n", threadId);
        std::map<std::string, Graph<GraphParameter, ComputeNode, DataTensor>> threadPrivateGraph;
        double timeStart = ut_time_ms();
#ifdef _USE_WEIGHT_SHARE
        int gpuId = -1, cpuId = -1;
        if (schedule->useGPU && threadId == schedule->threadNum - 1) {
            gpuId = 0;
            for (unsigned int i = 0; i < schedule->graphPath.size(); i++) {
                threadPrivateGraph[schedule->graphPath[i]].init(schedule->graphPath[i]);
                threadPrivateGraph[schedule->graphPath[i]].ready(
                    schedule->precision, schedule->deviceInfo.affinityPolicy, gpuId);
            }
        }
        if (gpuId < 0) {
#ifndef _USE_OPENMP
            if (schedule->deviceInfo.affinityPolicy == AFFINITY_CPU_HIGH_PERFORMANCE) {
                cpuId = schedule->deviceInfo.cpuNum - 1 - threadId;
            } else {
                cpuId = threadId;
            }
#endif
#if !defined(__ANDROID_API__) && !defined(__APPLE__)
            if (threadId == 0) {
#else
            if (0) {
#endif
                threadPrivateGraph = schedule->graph;
#ifndef _USE_OPENMP
                for (unsigned int i = 0; i < schedule->graphPath.size(); i++) {
                    threadPrivateGraph[schedule->graphPath[i]].setRuntime(
                        cpuId, schedule->deviceInfo.archs[cpuId]);
                }
#endif
            } else {
                for (unsigned int i = 0; i < schedule->graphPath.size(); i++) {
                    threadPrivateGraph[schedule->graphPath[i]] =
                        schedule->graph[schedule->graphPath[i]].clone();
#ifndef _USE_OPENMP
                    threadPrivateGraph[schedule->graphPath[i]].setRuntime(
                        cpuId, schedule->deviceInfo.archs[cpuId]);
#endif
                }
            }
        }
#if !defined(__ANDROID_API__) && !defined(__APPLE__)
        pthread_barrier_wait(&(schedule->barrier));
#endif
#else
        for (unsigned int i = 0; i < schedule->graphPath.size(); i++) {
            threadPrivateGraph[schedule->graphPath[i]].init(schedule->graphPath[i]);
            threadPrivateGraph[schedule->graphPath[i]].ready(
                schedule->precision, schedule->deviceInfo.affinityPolicy, -1);
#ifndef _USE_OPENMP
            threadPrivateGraph[schedule->graphPath[i]].setRuntime(6, ARM_A76);
#endif
        }
#endif
        UNI_DEBUG_LOG("start to wait task\n");
        double timeEnd = ut_time_ms();
        UNI_PROFILE_INFO("graphs init", "init", timeStart * 1000, (timeEnd - timeStart) * 1000);
        while (1) {
            pthread_mutex_lock(&(schedule->taskQueueLock));
            while (schedule->taskQueue.empty() && !(schedule->stop)) {
                pthread_cond_wait(&(schedule->condition), &(schedule->taskQueueLock));
            }
            if (schedule->stop) {
                break;
            }

            Task *task = nullptr;
            if (!(schedule->taskQueue.empty())) {
                task = schedule->taskQueue.front();
                schedule->taskQueue.pop();
            }
            pthread_mutex_unlock(&(schedule->taskQueueLock));
            if (task != nullptr) {
                threadPrivateGraph[task->graphPath].run(task->data);
                task->status = TASK_END;
            }
        }

        pthread_mutex_unlock(&(schedule->taskQueueLock));
        pthread_exit(NULL);
        UNI_DEBUG_LOG("worker end\n");
        return (NULL);
    }
};
#endif  // UNI_INCLUDE_SCHEDULE_H_
