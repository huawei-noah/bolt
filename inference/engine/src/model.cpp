// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "model.hpp"
#include "thread_affinity.h"

Model::Model(AffinityPolicy affinityPolicy, DataType dt, std::string name)
{
    this->set_device_info(affinityPolicy);
    this->dt = dt;
    this->name = name;
    std::string deviceName = "";
    if (IS_GPU(this->deviceInfo.schedule)) {
#ifdef _USE_GPU
        if (OCLContext::getInstance().handle->useQualcommDev) {
            this->deviceInfo.schedule = QUALCOMM;
        }
#else
        UNI_ERROR_LOG("This library not support GPU, please rebuild library with --gpu option.\n");
#endif
    }
    algorithmMap = std::shared_ptr<AlgorithmMap>(
        new AlgorithmMap(this->deviceInfo.schedule, name, deviceName, dt));
    UNI_DEBUG_LOG("Inference use %s.\n", ArchName()[this->deviceInfo.schedule])
}

void Model::set_runtime_device(int cpuId, int threadId)
{
    this->set_runtime_device(cpuId, this->deviceInfo.archs[cpuId], threadId);
}

void Model::set_runtime_device(int cpuId, Arch arch, int threadId)
{
    this->deviceInfo.schedule = arch;
    UNI_DEBUG_LOG("Inference use %s.\n", ArchName()[this->deviceInfo.schedule])
    if (cpuId >= 0 && cpuId < this->deviceInfo.cpuNum) {
        set_thread_affinity(threadId, &cpuId, 1);
        for (auto op : ops) {
            op->set_schedule(this->deviceInfo.schedule);
        }
    }
}

void Model::set_runtime_device_dynamic(int threadId)
{
    set_cpu_dynamic(&this->deviceInfo, threadId);
}

Arch Model::get_runtime_device()
{
    return this->deviceInfo.schedule;
}

void Model::ready(std::map<std::string, TensorDesc> inputDescMap)
{
    infer_output_tensors_size(inputDescMap);
    assign_output_tensor();

    infer_tmp_memory_size();
    assign_tmp_tensor();
}

#ifdef _USE_INT8
U32 Model::find_next_dynamic_scale_op(std::vector<U32> calibratedOpIdx, U32 startIdx)
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

std::shared_ptr<Operator> Model::get_operator_by_index(U32 index)
{
    return this->ops[index];
}

void Model::run_till_breakpoint(U32 opIdx)
{
    CHECK_REQUIREMENT(IS_CPU(this->deviceInfo.schedule));
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

std::string Model::get_name()
{
    return this->name;
}

void Model::loadAlgorithmMap(const char *path, bool useFileStream)
{
    UNI_DEBUG_LOG("load algorithm map...\n");
    std::string algoName = this->algorithmMap->getAlgorithmFileName();
    const char *algoInfo = nullptr;
    if (IS_GPU(this->deviceInfo.schedule)) {
#ifdef _USE_GPU
        algoInfo = gcl_get_algorithm_info(OCLContext::getInstance().handle.get(), algoName);
#endif
    }
    if (!algoInfo && useFileStream) {
        algoInfo = path;
    }
    if (algoInfo) {
        this->algorithmMap->loadAlgorithmMapFromFileStream(algoInfo);
    } else if (path) {
        this->algorithmMap->loadAlgorithmMapFromFile(path);
    }
    UNI_DEBUG_LOG("load algorithm map end.\n");
}

void Model::saveAlgorithmMapToFile(std::string algorithmMapPath)
{
    this->algorithmMap->saveAlgorithmMapToFile(algorithmMapPath);
}

void Model::set_device_info(AffinityPolicy affinityPolicy)
{
#ifndef _USE_IOS
    this->deviceInfo = get_cpu_info(affinityPolicy);
    this->set_runtime_device_dynamic();
#else
    this->deviceInfo.affinityPolicy = affinityPolicy;
    this->deviceInfo.schedule = ARM_A76;
#endif
}
