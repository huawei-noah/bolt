// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _MODEL_H
#define _MODEL_H

#include "thread_affinity.h"
#include "operator.hpp"
#include "algorithm_map.h"
#include "tensor_desc.h"
#ifdef _USE_MALI
#include "gcl.h"
#endif
#include "profiling.h"

class Model {
public:
    Model()
    {}
    Model(AffinityPolicy affinityPolicy, DataType dt, std::string name)
    {
        this->set_device_info(affinityPolicy);
        this->dt = dt;
        this->name = name;
        std::string deviceName = "";
#ifdef _USE_MALI
        if (this->deviceInfo.schedule == MALI) {
            this->handle = OCLContext::getInstance().handle;
            deviceName = this->handle->deviceName;
        }
#endif
        algorithmMap = std::shared_ptr<AlgorithmMap>(
            new AlgorithmMap(this->deviceInfo.schedule, name, deviceName, dt));
    }

    void set_device_info(AffinityPolicy affinityPolicy)
    {
#ifndef _USE_IOS
        this->deviceInfo = get_cpu_info(affinityPolicy);
        this->set_runtime_device_dynamic();
#else
        this->deviceInfo.affinityPolicy = affinityPolicy;
        this->deviceInfo.schedule = ARM_A76;
#endif
    }

    void set_runtime_device(int cpuId, int threadId = 0)
    {
        this->set_runtime_device(cpuId, this->deviceInfo.archs[cpuId], threadId);
    }

    void set_runtime_device(int cpuId, Arch arch, int threadId = 0)
    {
        this->deviceInfo.schedule = arch;
        if (cpuId >= 0 && cpuId < this->deviceInfo.cpuNum) {
            set_thread_affinity(threadId, &cpuId, 1);
            for (auto op : ops) {
                op->set_schedule(this->deviceInfo.schedule);
            }
        }
    }

    void set_runtime_device_dynamic(int threadId = 0)
    {
        set_cpu_dynamic(&this->deviceInfo, threadId);
    }

    Arch get_runtime_device()
    {
        return this->deviceInfo.schedule;
    }

    virtual EE infer_output_tensors_size(std::map<std::string, TensorDesc>) = 0;
    virtual void assign_output_tensor() = 0;
    virtual void infer_tmp_memory_size() = 0;
    virtual void assign_tmp_tensor() = 0;

    virtual void ready(std::map<std::string, TensorDesc> inputDescMap)
    {
        infer_output_tensors_size(inputDescMap);
        assign_output_tensor();

        infer_tmp_memory_size();
        assign_tmp_tensor();
    }

    virtual void run() = 0;

#ifdef _USE_INT8
    virtual U32 find_next_dynamic_scale_op(std::vector<U32> calibratedOpIdx, U32 startIdx)
    {
        CHECK_REQUIREMENT(startIdx < this->ops.size())
        for (U32 i = startIdx; i < this->ops.size();) {
            auto op = this->ops[i];
            if (op->is_dynamic_scale()) {
                bool calibrated = false;
                for (auto idx : calibratedOpIdx) {
                    if (i == idx) {
                        calibrated = true;
                        break;
                    }
                }
                if (!calibrated) {
                    return i;
                }
            }

            if (op->get_type() == OT_Repeat || op->get_type() == OT_Jump) {
                i = op->get_next_operator_index();
            } else {
                i++;
            }
        }

        return 0;  // The first layer should never be quantized
    }

    virtual std::shared_ptr<Operator> get_operator_by_index(U32 index)
    {
        return this->ops[index];
    }

    virtual void run_till_breakpoint(U32 opIdx)
    {
        CHECK_REQUIREMENT(MALI != this->deviceInfo.schedule);
        for (U32 i = 0; i < this->ops.size();) {
            auto op = this->ops[i];
            if (op->get_type() == OT_Repeat || op->get_type() == OT_Jump) {
                if (opIdx == i) {
                    break;
                }
                i = op->get_next_operator_index();
            } else {
                op->run();
                if (opIdx == i) {
                    break;
                }
                i++;
            }
        }
    }
#endif

    virtual bool checkOperator()
    {
        for (auto op : this->ops) {
            if (!op->checkOperator()) {
                return false;
            }
        }
        return true;
    }

    std::string get_name()
    {
        return this->name;
    }

    void loadAlgorithmMapFromFileStream(const char *algoFileStream)
    {
        this->algorithmMap->loadAlgorithmMapFromFileStream(algoFileStream);
    }

    void loadAlgorithmMapFromText(std::string algorithmMapPath)
    {
        this->algorithmMap->loadAlgorithmMapFromText(algorithmMapPath);
    }

    void saveAlgorithmMapToText(std::string algorithmMapPath)
    {
        this->algorithmMap->saveAlgorithmMapToText(algorithmMapPath);
    }

protected:
    std::vector<std::shared_ptr<Operator>> ops;
    DeviceInfo deviceInfo;
    DataType dt;
#ifdef _USE_MALI
    std::shared_ptr<GCLHandle> handle;
#endif
    std::shared_ptr<AlgorithmMap> algorithmMap;

private:
    std::string name;
};
#endif
