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

#include "cnn.h"
#ifdef _USE_MALI
#include "gcl.h"
#endif
#include "thread_affinity.h"

inline std::map<std::string, TensorDesc> extractInputDims(const ModelSpec *ms)
{
    std::map<std::string, TensorDesc> inputDescMap;
    int inputNum = ms->num_inputs;
    for (int i = 0; i < inputNum; i++) {
        inputDescMap[ms->input_names[i]] = ms->input_dims[i];
    }
    return inputDescMap;
}

inline std::shared_ptr<CNN> createPipelinefromMs(
    const char *affinityPolicyName, ModelSpec *ms, const char *algorithmMapPath)
{
    AffinityPolicy affinityPolicy = thread_affinity_get_policy_by_name(affinityPolicyName);
    CNN *cnn = new CNN(affinityPolicy, ms->dt, ms->model_name);

    cnn->sort_operators_sequential(ms);

    // create ops
    cnn->initialize_ops(ms);

    std::map<std::string, TensorDesc> inputDescMap = extractInputDims(ms);

    cnn->loadAlgorithmMap(algorithmMapPath);

    // assign space for output, tmp, bias, and trans_weight
    cnn->ready(inputDescMap);

    CHECK_STATUS(cnn->mark_input_output());

    return std::shared_ptr<CNN>(cnn);
}

inline std::shared_ptr<CNN> createPipeline(
    const char *affinityPolicyName, const char *modelPath, const char *algorithmMapPath = "")
{
    // deserialize model from file
    ModelSpec ms;
    CHECK_STATUS(deserialize_model_from_file(modelPath, &ms));
    std::shared_ptr<CNN> pipeline = createPipelinefromMs(affinityPolicyName, &ms, algorithmMapPath);
    CHECK_STATUS(mt_destroy_model(&ms));
    return pipeline;
}

#endif
