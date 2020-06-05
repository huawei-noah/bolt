// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), 
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, 
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE 
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR 
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


#ifndef _HPP_INFERENCE
#define _HPP_INFERENCE

#include "cnn.hpp"
#ifdef _USE_MALI
#include "gcl.h"
#endif
#ifdef _BUILD_TEST
#include "sequential.hpp"
#endif
#include "thread_affinity.h"
#include "op_type.h"
#include "model_serialize_deserialize.hpp"
typedef enum{
    d_CPU = 0,
    d_GPU = 1
} DeviceTypeIn;

inline HashMap<std::string, TensorDesc> extractInputDims(const ModelSpec *ms) {
    HashMap<std::string, TensorDesc> inputDescMap;
    int inputNum = ms->num_inputs;
    for (int i = 0; i < inputNum; i++) {
        inputDescMap[ms->input_names[i]] = ms->input_dims[i];
    }
    return inputDescMap;
}

inline Arch getCpuArchInfo(const char *cpuAffinityPolicyName){
    int *cpuids;
    Arch *archs;
    int cpuNum;
    thread_affinity_init(&cpuNum, &archs, &cpuids);
    CpuAffinityPolicy affinityPolicy = thread_affinity_get_policy_by_name(cpuAffinityPolicyName);
    Arch arch = thread_affinity_set_by_policy(cpuNum, archs, cpuids, affinityPolicy, 0);
    thread_affinity_destroy(&cpuNum, &archs, &cpuids);
    return arch;
}

inline Arch getArch(const char *cpuAffinityPolicyName, DeviceTypeIn device)
{
    Arch arch = CPU_GENERAL;
    if(device == d_CPU){
        arch = getCpuArchInfo(cpuAffinityPolicyName);
    } else if(device == d_GPU) {
        arch = MALI;
    } else {
        CHECK_STATUS(NOT_SUPPORTED);
    }
    return arch;
}

inline std::shared_ptr<CNN> createPipelinefromMs(Arch arch, ModelSpec* ms, const char *algorithmMapPath)
{
    CNN* cnn;
    cnn = new CNN(arch, ms->dt, ms->model_name);

    cnn->sort_operators_sequential(ms);
    //TODO this function is not tested
    //cnn ->sort_operators_tp(ms);

    // create ops
    cnn->initialize_ops(ms);

    HashMap<std::string, TensorDesc> inputDescMap = extractInputDims(ms);

    cnn->loadAlgorithmMapFromText(algorithmMapPath);

    // assign space for output, tmp, bias, and trans_weight
    cnn->ready(inputDescMap);

    CHECK_STATUS(cnn->mark_input_output(ms));
#ifdef _USE_MALI
    if(arch == MALI) cnn->mali_prepare();
#endif

    cnn->saveAlgorithmMapToText(algorithmMapPath);

    return std::shared_ptr<CNN>(cnn);
}

inline std::shared_ptr<CNN> createPipelineWithConfigure(const char *cpuAffinityPolicyName,
    const char* modelPath,
    DeviceTypeIn device,
    const char *algorithmMapPath)
{
    // set cpu affinity
    //TODO add mali support
    Arch arch = getArch(cpuAffinityPolicyName, device);

    // deserialize model from file
    ModelSpec ms;
    deserialize_model_from_file(modelPath, &ms);

    std::shared_ptr<CNN> pipeline = createPipelinefromMs(arch, &ms, algorithmMapPath);

    CHECK_STATUS(mt_destroy_model(&ms));
    return pipeline;
}

inline std::shared_ptr<CNN> createPipeline(const char *cpuAffinityPolicyName, const char* modelPath, DeviceTypeIn device)
{
    return createPipelineWithConfigure(cpuAffinityPolicyName, modelPath, device, "");
}

#ifdef _BUILD_TEST
inline Sequential createSequentialPipeline(const char *cpuAffinityPolicyName, DataType dt, const char *modelName)
{
    // set cpu affinity
    Arch arch = getCpuArchInfo(cpuAffinityPolicyName);
    auto sequential = Sequential(arch, dt, modelName);
    return sequential;
}
#endif
#endif
