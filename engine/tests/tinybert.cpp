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
#include <limits.h>

#include "inference.hpp"
#include "tensor.hpp"
#include "type.h"
#include "tensor_desc.h"
#include "model_serialize_deserialize.hpp"
#include "data_loader.hpp"
#include "result_format.hpp"
#include "utils.hpp"


void print_help(char* argv[]) {
    std::cout << "usage: " << argv[0] << "  model_path  sequences_dir" << std::endl;
}

int main(int argc, char* argv[]) {
    UTIL_TIME_INIT

    char* modelPath = (char*)"";
    char* sequenceDirectory = (char*)"";
    if (argc < 2) {
        print_help(argv);
        return 1;
    }
    modelPath = argv[1];
    if (argc > 2) sequenceDirectory = argv[2];

    const Arch A = ARM_A76;
    ModelSpec ms;
    deserialize_model_from_file(modelPath, &ms);
    auto pipeline = createCNN<A>(&ms);

    // load sequences
    Vec<Vec<Tensor>> sequences;
    HashMap<std::string, std::shared_ptr<Tensor>> inMap = pipeline->get_inputs();
    TensorDesc sequenceDesc = (*(inMap.begin()->second)).get_desc();
    Vec<std::string> sequencePaths = load_sequences(sequenceDirectory, sequenceDesc, &sequences);

    double totalTime = 0;
    U32 sequenceIndex = 0;
    U32 invalidSequence = 0;
    std::cout << "[RESULT]:" << std::endl;
    for (auto sequence: sequences) {
        std::cout << sequencePaths[sequenceIndex] << std::endl;
        // stage3: set input
        Vec<Tensor> input;
        input = sequence;

        auto modelInputTensorNames = pipeline->get_model_input_tensor_names();
        HashMap<std::string, std::shared_ptr<U8>> modelInputTensors;
        for (int index = 0; index < (int)modelInputTensorNames.size(); index++) {
            std::shared_ptr<U8> tensorPointer = input[index].get_val();
            modelInputTensors.insert(std::pair(modelInputTensorNames[index], tensorPointer));
        }
        pipeline->set_input_tensors_value(modelInputTensors);

        // stage4: run
        double timeBegin = ut_time_ms();
        pipeline->run();
        double timeEnd = ut_time_ms();
        totalTime += (timeEnd - timeBegin);

        // stage5: process result
        HashMap<std::string, std::shared_ptr<Tensor>> outMap = pipeline->get_outputs();
        Tensor intentSoftmax = *(outMap["intent_softmax"]);
        U32 intentNum = tensorNumElements(intentSoftmax.get_desc());
        F16 *ptr = (F16*)(intentSoftmax.get_val().get());
        U32 intentMaxIndex = 0;
        for (U32 index = 1; index < intentNum; index++) {
            if (ptr[index] > ptr[intentMaxIndex])
                intentMaxIndex = index;
        }
        std::cout << "    intent: " << intentMaxIndex << " " << ptr[intentMaxIndex] << std::endl;
        Tensor slotSoftmax = *(outMap["slot_softmax"]);
        U32 slotNum = tensorNumElements(slotSoftmax.get_desc());
        ptr = (F16*)(slotSoftmax.get_val().get());
        std::cout << "    slot: ";
        for (U32 i = 0; i < 32; i++) {
            U32 slotMaxIndex = 0;
            ptr = (F16*)(slotSoftmax.get_val().get()) + i * slotNum / 32;
            for (U32 index = 1; index < slotNum / 32; index++) {
                if (ptr[index] > ptr[slotMaxIndex])
                    slotMaxIndex = index;
            }
            std::cout << slotMaxIndex << " ";
        }
        std::cout << std::endl;
        sequenceIndex++;
    }

    std::cout << "[SUMMARY]:" << std::endl;
    U32 validSequence = sequenceIndex - invalidSequence;
    std::cout << "time: " << 1.0 * totalTime / validSequence << " ms/sequence" << std::endl;

    return 0;
}
