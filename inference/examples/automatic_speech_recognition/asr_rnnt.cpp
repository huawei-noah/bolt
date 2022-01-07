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
    char *affinityPolicyName = (char *)"";

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
    TensorDesc soundInputDesc = (*(inMap["sounds"])).get_desc();
    sequenceDescs.push_back(soundInputDesc);

    std::vector<std::vector<Tensor>> sequences, results;
    std::vector<std::string> sequencePaths =
        load_data(sequenceDirectory + std::string("/input"), sequenceDescs, &sequences);
    std::vector<TensorDesc> resultDescs;
    resultDescs.push_back(soundInputDesc);
    std::vector<std::string> resultPaths =
        load_data(sequenceDirectory + std::string("/result"), resultDescs, &results);

    double totalTime = 0;
    U32 sequenceIndex = 0;
    U32 falseResult = 0;
    std::cout << "[RESULT]:" << std::endl;
    for (auto sequence : sequences) {
        std::cout << sequencePaths[sequenceIndex] << ": " << std::endl;
        TensorDesc desc = sequence[0].get_desc();
        TensorDesc inputDesc = tensor3d(soundInputDesc.dt, 1,
            tensorNumElements(desc) / soundInputDesc.dims[0], soundInputDesc.dims[0]);
        std::map<std::string, TensorDesc> inputDescMap;
        inputDescMap["sounds"] = inputDesc;
        pipeline->reready(inputDescMap);

        std::map<std::string, std::shared_ptr<U8>> model_tensors_input;
        model_tensors_input["sounds"] = ((CpuMemory *)(sequence[0].get_memory()))->get_shared_ptr();
        pipeline->set_input_by_assign(model_tensors_input);

        double timeBegin = ut_time_ms();
        pipeline->run();
        double timeEnd = ut_time_ms();
        totalTime += (timeEnd - timeBegin);

        Tensor output = pipeline->get_tensor_by_name("labels");
        std::cout << output.string(32) << std::endl;
        if (resultPaths.size() > sequenceIndex) {
            F32 *result = (F32 *)((CpuMemory *)(results[sequenceIndex][0].get_memory()))->get_ptr();
            U32 inferenceSize = output.length();
            for (U32 i = 0; i < results[sequenceIndex][0].length(); i++) {
                if (i >= inferenceSize || result[i] != output.element(i)) {
                    falseResult++;
                    break;
                }
            }
        }

        sequenceIndex++;
    }

    UNI_TIME_STATISTICS

    std::cout << "[SUMMARY]:" << std::endl;
    UNI_CI_LOG(
        "speech recognition rate: %f %%\n", 100.0 * (sequenceIndex - falseResult) / sequenceIndex);
    UNI_CI_LOG("avg_time:%fms/sequence\n", 1.0 * totalTime / sequenceIndex);

    return 0;
}
