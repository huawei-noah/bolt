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

#include "operator.hpp"
#include "algorithm_map.h"
#include "affinity_policy.h"
#ifdef _USE_GPU
#include "gcl.h"
#endif

class Model {
public:
    Model()
    {}

    explicit Model(AffinityPolicy affinityPolicy, DataType dt, std::string name);

    virtual ~Model() = default;

    virtual void ready(std::map<std::string, TensorDesc> inputDescMap);

    virtual void run() = 0;

#ifdef _USE_INT8
    virtual U32 find_next_dynamic_scale_op(std::vector<U32> calibratedOpIdx, U32 startIdx);

    virtual std::shared_ptr<Operator> get_operator_by_index(U32 index);

    virtual void run_till_breakpoint(U32 opIdx);
#endif

    void loadAlgorithmMap(const char *path, bool useFileStream = false);

    void saveAlgorithmMapToFile(std::string algorithmMapPath);

    void set_runtime_device(int cpuId, int threadId = 0);

    void set_runtime_device(int cpuId, Arch arch, int threadId = 0);

    void set_runtime_device_dynamic(int threadId = 0);

    Arch get_runtime_device();

    std::string get_name();

protected:
    DataType dt;
    std::vector<std::shared_ptr<Operator>> ops;
    DeviceInfo deviceInfo;
    std::shared_ptr<AlgorithmMap> algorithmMap;

    virtual EE infer_output_tensors_size(std::map<std::string, TensorDesc>) = 0;
    virtual void assign_output_tensor() = 0;
    virtual void infer_tmp_memory_size() = 0;
    virtual void assign_tmp_tensor() = 0;

private:
    std::string name;

    void set_device_info(AffinityPolicy affinityPolicy);
};
#endif
