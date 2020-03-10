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
    std::cout << "usage: " << argv[0] << " modelPath sequenceDirectory cpuAffinityPolicyName" << std::endl;
}

int main(int argc, char* argv[]) {
    UTIL_TIME_INIT

    char* modelPath = (char*)"";
    char* sequenceDirectory = (char*)"";
    char* cpuAffinityPolicyName = (char*)"";
    if (argc < 2) {
        print_help(argv);
        return 1;
    }
    modelPath = argv[1];
    if (argc > 2) sequenceDirectory = argv[2];
    if (argc > 3) cpuAffinityPolicyName = argv[3];
    DeviceTypeIn device = d_CPU;
    auto pipeline = createPipeline(cpuAffinityPolicyName, modelPath, device);

    // load sequences
    HashMap<std::string, std::shared_ptr<Tensor>> inMap = pipeline->get_inputs();
    Vec<TensorDesc> sequenceDescs;
    TensorDesc wordInputDesc = (*(inMap["bert_words"])).get_desc();
    wordInputDesc.dt = DT_U32;
    sequenceDescs.push_back(wordInputDesc);
    TensorDesc positionInputDesc = (*(inMap["bert_positions"])).get_desc();
    positionInputDesc.dt = DT_U32;
    sequenceDescs.push_back(positionInputDesc);
    TensorDesc tokenTypeInputDesc = (*(inMap["bert_token_type"])).get_desc();
    tokenTypeInputDesc.dt = DT_U32;
    sequenceDescs.push_back(tokenTypeInputDesc);
    Vec<Vec<Tensor>> sequences;
    Vec<std::string> sequencePaths = load_data(sequenceDirectory, sequenceDescs, &sequences);

    double totalTime = 0;
    U32 sequenceIndex = 0;
    U32 invalidSequence = 0;
    std::cout << "[RESULT]:" << std::endl;
    for (auto sequence: sequences) {
        std::cout << sequencePaths[sequenceIndex] << std::endl;
        // stage3: set input
        Vec<Tensor> input;
        input = sequence;

        Vec<TensorDesc> inputDescs;
        for (Tensor t: sequence) {
            inputDescs.push_back(t.get_desc());
        }
        pipeline->infer_output_tensors_size(inputDescs);

        auto modelInputTensorNames = pipeline->get_model_input_tensor_names();
        HashMap<std::string, std::shared_ptr<U8>> model_tensors_input;
        for (int index = 0; index < (int)modelInputTensorNames.size(); index++) {
            U8* tmp = input[index].get_val();
            std::shared_ptr<U8> tensorPointer(tmp);
            model_tensors_input.insert(std::pair(modelInputTensorNames[index], tensorPointer));
        }
        pipeline->set_input_tensors_value(model_tensors_input);

        // stage4: run
        double timeBegin = ut_time_ms();
        pipeline->run();
        double timeEnd = ut_time_ms();
        totalTime += (timeEnd - timeBegin);

        // stage5: process result
        HashMap<std::string, std::shared_ptr<Tensor>> outMap = pipeline->get_outputs();
        bool invalid = false;
        for (auto iter: outMap) {
            std::string key = iter.first;
            std::shared_ptr<Tensor> value = iter.second;
            Tensor result = *value;
            invalid = result.isInvalid();
            if (key == "other")
                continue;
            U32 resultElementNum = tensorNumElements(result.get_desc());
            std::cout << "    " << key << ": ";
            std::cout << tensorDesc2Str(result.get_desc());
            std::cout << std::endl;
            std::cout << "        ";
            for (U32 index = 0; index < resultElementNum; index++) {
                std::cout << result.getElement(index) << " ";
            }
            std::cout<<std::endl;
        }
        
        if (invalid) {
            totalTime -= (timeEnd - timeBegin);
            std::cout << "nan" << std::endl;
            invalidSequence ++;
        }
        sequenceIndex++;
    }

    UTIL_TIME_STATISTICS

    std::cout << "[SUMMARY]:" << std::endl;
    U32 validSequence = sequenceIndex - invalidSequence;
    std::cout << "time: " << 1.0 * totalTime / validSequence << " ms/sequence" << std::endl;

    return 0;
}
