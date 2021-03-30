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

int main(int argc, char *argv[])
{
    UNI_TIME_INIT
    ParseRes parse_res;
    parseCommandLine(argc, argv, &parse_res, "examples");
    char *modelPath = (char *)"";
    char *sequenceDirectory = (char *)"";
    char *affinityPolicyName = (char *)"CPU_AFFINITY_HIGH_PERFORMANCE";

    if (!parse_res.model.second) {
        exit(-1);
    }
    if (parse_res.model.second) {
        modelPath = parse_res.model.first;
    }
    if (parse_res.inputPath.second) {
        sequenceDirectory = parse_res.inputPath.first;
    }
    if (parse_res.archInfo.second) {
        affinityPolicyName = parse_res.archInfo.first;
    }

    auto pipeline = createPipeline(affinityPolicyName, modelPath);

    // load sequences
    std::map<std::string, std::shared_ptr<Tensor>> inMap = pipeline->get_input();
    std::vector<TensorDesc> sequenceDescs;
    TensorDesc wordInputDesc = (*(inMap["bert_words"])).get_desc();
    wordInputDesc.dt = DT_U32;
    sequenceDescs.push_back(wordInputDesc);
    TensorDesc positionInputDesc = (*(inMap["bert_positions"])).get_desc();
    positionInputDesc.dt = DT_U32;
    sequenceDescs.push_back(positionInputDesc);
    TensorDesc tokenTypeInputDesc = (*(inMap["bert_token_type"])).get_desc();
    tokenTypeInputDesc.dt = DT_U32;
    sequenceDescs.push_back(tokenTypeInputDesc);
    std::vector<std::vector<Tensor>> sequences;
    std::vector<std::string> sequencePaths =
        load_data(sequenceDirectory + std::string("/input"), sequenceDescs, &sequences);

    double totalTime = 0;
    U32 sequenceIndex = 0;
    std::cout << "[RESULT]:" << std::endl;
    for (auto sequence : sequences) {
        std::cout << sequencePaths[sequenceIndex] << std::endl;
        std::map<std::string, TensorDesc> inputDescMap;
        inputDescMap["bert_words"] = sequence[0].get_desc();
        inputDescMap["bert_positions"] = sequence[1].get_desc();
        inputDescMap["bert_token_type"] = sequence[2].get_desc();
        pipeline->reready(inputDescMap);

        std::map<std::string, U8 *> input;
        input["bert_words"] = (U8 *)((CpuMemory *)(sequence[0].get_memory()))->get_ptr();
        input["bert_positions"] = (U8 *)((CpuMemory *)(sequence[1].get_memory()))->get_ptr();
        input["bert_token_type"] = (U8 *)((CpuMemory *)(sequence[2].get_memory()))->get_ptr();

        pipeline->set_input_by_copy(input);

        double timeBegin = ut_time_ms();
        pipeline->run();
        double timeEnd = ut_time_ms();
        totalTime += (timeEnd - timeBegin);

        // stage5: process result
        std::map<std::string, std::shared_ptr<Tensor>> outMap = pipeline->get_output();
        for (auto iter : outMap) {
            std::string key = iter.first;
            std::shared_ptr<Tensor> value = iter.second;
            Tensor result = *value;
            if (key == "other") {
                continue;
            }
            U32 resultElementNum = tensorNumElements(result.get_desc());
            std::cout << "    " << key << ": ";
            std::cout << tensorDesc2Str(result.get_desc());
            std::cout << std::endl;
            std::cout << "        ";
            for (U32 index = 0; index < resultElementNum; index++) {
                std::cout << result.element(index) << " ";
            }
            std::cout << std::endl;
        }

        sequenceIndex++;
    }

    UNI_TIME_STATISTICS

    std::cout << "[SUMMARY]:" << std::endl;
    UNI_CI_LOG("avg_time:%fms/sequence\n", 1.0 * totalTime / sequenceIndex);

    return 0;
}
