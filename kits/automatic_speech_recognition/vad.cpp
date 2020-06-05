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
#include "utils.hpp"

void print_help(char* argv[]) {
    std::cout << "usage: " << argv[0] << " modelPath cpuAffinityPolicyName" << std::endl;
}

int verify(Tensor vad, Tensor eoq)
{
    I32 result = 0;
    U32 num = tensorNumElements(vad.get_desc());
    CHECK_REQUIREMENT(2 == num);
    if (abs(vad.getElement(0) - 0.999107) >= 0.0005) {
        result = 1;
    }
    if (abs(vad.getElement(1) - 0.0009) >= 0.0005) {
        result = 1;
    }

    num = tensorNumElements(eoq.get_desc());
    CHECK_REQUIREMENT(2 == num);
    if (abs(eoq.getElement(0) - 1) >= 0.0005) {
        result = 1;
    }
    if (abs(eoq.getElement(1) - 1.4e-8) >= 0.0005) {
        result = 1;
    }
    return result;
}

int main(int argc, char* argv[])
{
    if (argc < 3) {
        print_help(argv);
        return 1;
    }
    UTIL_TIME_INIT

    char* modelPath = argv[1];
    char* cpuAffinityPolicyName = argv[2];
    DeviceTypeIn device = d_CPU;
    auto pipeline = createPipeline(cpuAffinityPolicyName, modelPath, device);

    HashMap<std::string, std::shared_ptr<Tensor>> inMap = pipeline->get_inputs();
    TensorDesc cacheDesc = (*(inMap["input_cache"])).get_desc();
    cacheDesc.df = DF_NCHWC8;

    HashMap<std::string, TensorDesc> inputDescMap;
    inputDescMap["input_fea"] = (*(inMap["input_fea"])).get_desc();
    inputDescMap["input_cache"] = cacheDesc;
    pipeline->reready(inputDescMap);

    Vec<U8> cache;
    cache.resize(tensorNumBytes(cacheDesc), 0);

    double totalTime = 0;
    int loops = 1;
    U32 falseResult = 0;
    for (int i = 0; i < loops; i++) {
        pipeline->copy_to_named_input("input_cache", cache.data());
        pipeline->copy_to_named_input("input_fea", cache.data());

        double timeBegin = ut_time_ms();
        pipeline->run();
        double timeEnd = ut_time_ms();
        totalTime += (timeEnd - timeBegin);
        Tensor vad = pipeline->get_tensor_by_name("output_vad");
        std::cout << "output_vad: " << vad.getElement(0) << " " << vad.getElement(1) << std::endl;
        Tensor eoq = pipeline->get_tensor_by_name("output_eoq");
        std::cout << "output_eoq: " << eoq.getElement(0) << " " << eoq.getElement(1) << std::endl;
        falseResult += verify(vad, eoq);
        Tensor outCache = pipeline->get_tensor_by_name("output_cache");
        memcpy(cache.data(), (U8*)outCache.get_val(), tensorNumBytes(cacheDesc));
    }
    UTIL_TIME_STATISTICS

    std::cout << "[SUMMARY]:" << std::endl;
    U32 validSequence = loops;
    CI_info("vad rate: " << 100.0 * (validSequence - falseResult) / validSequence  << " %");
    CI_info("avg_time:" << 1.0 * totalTime / validSequence << "ms/sequence");
    if (falseResult > 0) {
        std::cerr << "[ERROR] verify failed" << std::endl;
        exit(1);
    }

    return 0;
}
