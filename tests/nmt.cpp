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
    std::cout << "usage: " << argv[0] << " modelPath sequencesDirectory cpuAffinityPolicyName" << std::endl;
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        print_help(argv);
        return 1;
    }
    UTIL_TIME_INIT

    char* modelPath = argv[1];
    char* sequenceDirectory = argv[2];
    char* cpuAffinityPolicyName = (char*)"";
    if (argc > 3) cpuAffinityPolicyName = argv[3];
    DeviceTypeIn device = d_CPU;

    auto pipeline = createPipeline(cpuAffinityPolicyName, modelPath, device);

    // load sequences
    HashMap<std::string, std::shared_ptr<Tensor>> inMap = pipeline->get_inputs();
    Vec<TensorDesc> sequenceDescs;
    TensorDesc wordInputDesc = (*(inMap["nmt_words"])).get_desc();
    wordInputDesc.dt = DT_U32;
    sequenceDescs.push_back(wordInputDesc);
    TensorDesc positionInputDesc = (*(inMap["nmt_positions"])).get_desc();
    positionInputDesc.dt = DT_U32;
    sequenceDescs.push_back(positionInputDesc);

    Vec<Vec<Tensor>> sequences, results;
    Vec<std::string> sequencePaths = load_data(sequenceDirectory+std::string("/input"), sequenceDescs, &sequences);
    Vec<TensorDesc> resultDescs;
    resultDescs.push_back(wordInputDesc);
    Vec<std::string> resultPaths = load_data(sequenceDirectory+std::string("/result"), resultDescs, &results);

    double totalTime = 0;
    U32 sequenceIndex = 0;
    U32 invalidSequence = 0;
    U32 falseResult = 0;
    std::cout << "[RESULT]:" << std::endl;
    for (auto sequence: sequences) {
        std::cout << sequencePaths[sequenceIndex] << ": " << std::endl;
        Vec<TensorDesc> inputDescs;
        for (Tensor t : sequence) {
            inputDescs.push_back(t.get_desc());
        }
        pipeline->infer_output_tensors_size(inputDescs);

        auto modelInputTensorNames = pipeline->get_model_input_tensor_names();
        HashMap<std::string, std::shared_ptr<U8>> model_tensors_input;
        for (int index = 0; index < (int)modelInputTensorNames.size(); index++) {
            U8* tensorPointer = sequence[index].get_val();
            pipeline->copy_to_named_input(modelInputTensorNames[index], tensorPointer);
        }

        double timeBegin = ut_time_ms();
        pipeline->run();
        double timeEnd = ut_time_ms();
        totalTime += (timeEnd - timeBegin);

        Tensor output = pipeline->get_tensor_by_name("decoder_output");
        output.print();
        bool invalid = output.isInvalid();
        if (invalid) {
            totalTime -= (timeEnd - timeBegin);
            std::cout << "nan" << std::endl;
            invalidSequence ++;
        }
        if (resultPaths.size() > sequenceIndex) {
            U32 *result = (U32*)results[sequenceIndex][0].get_val();
            U32 inferenceSize = tensorNumElements(output.get_desc());
            for (U32 i = 0; i < tensorNumElements(results[sequenceIndex][0].get_desc()); i++) {
                if (i >= inferenceSize || result[i] != output.getElement(i)) {
                    falseResult++;
                    break;
                }
            }
        }

        sequenceIndex++;
    }

    UTIL_TIME_STATISTICS

    std::cout << "[SUMMARY]:" << std::endl;
    U32 validSequence = UNI_MAX(1, sequenceIndex - invalidSequence);
    CI_info("translation correct rate: " << 100.0 * (validSequence - falseResult) / validSequence  << " %");
    CI_info("avg_time:" << 1.0 * totalTime / validSequence << "ms/sequence");

    return 0;
}
