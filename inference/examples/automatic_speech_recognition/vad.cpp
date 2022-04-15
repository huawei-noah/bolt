// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include <iostream>

#include "inference.hpp"
#include "tensor.hpp"
#include "data_loader.hpp"
#include "profiling.h"
#include "parse_command.h"

int verify(Tensor vad, Tensor eoq)
{
    I32 result = 0;
    U32 num = vad.length();
    CHECK_REQUIREMENT(2 == num);
    if (abs(vad.element(0) - 0.999107) >= 0.0005) {
        result = 1;
    }
    if (abs(vad.element(1) - 0.0009) >= 0.0005) {
        result = 1;
    }

    num = eoq.length();
    CHECK_REQUIREMENT(2 == num);
    if (abs(eoq.element(0) - 1) >= 0.0005) {
        result = 1;
    }
    if (abs(eoq.element(1) - 1.4e-8) >= 0.0005) {
        result = 1;
    }
    return result;
}

int main(int argc, char *argv[])
{
    UNI_TIME_INIT
    ParseRes parse_res;
    parseCommandLine(argc, argv, &parse_res, "examples");

    char *modelPath = (char *)"";
    char *affinityPolicyName = (char *)"";
    if (!parse_res.model.second) {
        exit(-1);
    }
    if (parse_res.model.second) {
        modelPath = parse_res.model.first;
    }
    if (parse_res.archInfo.second) {
        affinityPolicyName = parse_res.archInfo.first;
    }

    auto pipeline = createPipeline(affinityPolicyName, modelPath);

    std::map<std::string, std::shared_ptr<Tensor>> inMap = pipeline->get_input();
    TensorDesc cacheDesc = (*(inMap["input_cache"])).get_desc();

    std::map<std::string, TensorDesc> inputDescMap;
    inputDescMap["input_fea"] = (*(inMap["input_fea"])).get_desc();
    inputDescMap["input_cache"] = cacheDesc;
    pipeline->reready(inputDescMap);

    std::vector<U8> cache;
    cache.resize(tensorNumBytes(cacheDesc), 0);

    double totalTime = 0;
    int loops = 1;
    U32 falseResult = 0;
    std::map<std::string, U8 *> input;
    input["input_cache"] = cache.data();
    input["input_fea"] = cache.data();
    for (int i = 0; i < loops; i++) {
        pipeline->set_input_by_copy(input);

        double timeBegin = ut_time_ms();
        pipeline->run();
        double timeEnd = ut_time_ms();
        totalTime += (timeEnd - timeBegin);
        Tensor vad = pipeline->get_tensor_by_name("output_vad");
        std::cout << "output_vad: " << vad.element(0) << " " << vad.element(1) << std::endl;
        Tensor eoq = pipeline->get_tensor_by_name("output_eoq");
        std::cout << "output_eoq: " << eoq.element(0) << " " << eoq.element(1) << std::endl;
        falseResult += verify(vad, eoq);
        Tensor outCache = pipeline->get_tensor_by_name("output_cache");
        memcpy(cache.data(), (U8 *)((CpuMemory *)(outCache.get_memory()))->get_ptr(),
            tensorNumBytes(cacheDesc));
    }
    UNI_TIME_STATISTICS

    std::cout << "[SUMMARY]:" << std::endl;
    U32 validSequence = loops;
    UNI_CI_LOG("vad rate: %f %%\n", 100.0 * (validSequence - falseResult) / validSequence);
    UNI_CI_LOG("avg_time:%fms/sequence\n", 1.0 * totalTime / validSequence);
    if (falseResult > 0) {
        UNI_ERROR_LOG("verify failed\n");
    }

    return 0;
}
