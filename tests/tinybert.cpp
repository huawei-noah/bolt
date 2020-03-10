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
    char* cpuAffinityPolicyName = (char *)"";
    if (argc > 3) cpuAffinityPolicyName = argv[3];
    DeviceTypeIn device = d_CPU;

    auto pipeline = createPipeline(cpuAffinityPolicyName, modelPath, device);

    // load sequences
    HashMap<std::string, std::shared_ptr<Tensor>> inMap = pipeline->get_inputs();
    Vec<TensorDesc> sequenceDescs;
    TensorDesc wordInputDesc = (*(inMap["tinybert_words"])).get_desc();
    wordInputDesc.dt = DT_U32;
    sequenceDescs.push_back(wordInputDesc);
    TensorDesc positionInputDesc = (*(inMap["tinybert_positions"])).get_desc();
    positionInputDesc.dt = DT_U32;
    sequenceDescs.push_back(positionInputDesc);
    TensorDesc tokenTypeInputDesc = (*(inMap["tinybert_token_type"])).get_desc();
    tokenTypeInputDesc.dt = DT_U32;
    sequenceDescs.push_back(tokenTypeInputDesc);
    Vec<Vec<Tensor>> sequences, intents, slots;
    Vec<std::string> sequencePaths = load_data(sequenceDirectory+std::string("/input"), sequenceDescs, &sequences);

    // load result
    Vec<TensorDesc> intentDescs;
    TensorDesc intentDesc = tensor1d(DT_F32, 2);
    intentDescs.push_back(intentDesc);
    Vec<std::string> intentPaths = load_data(sequenceDirectory+std::string("/intent"), intentDescs, &intents);
    Vec<TensorDesc> slotDescs;
    slotDescs.push_back(wordInputDesc);
    Vec<std::string> slotPaths = load_data(sequenceDirectory+std::string("/slot"), slotDescs, &slots);

    double totalTime = 0;
    U32 sequenceIndex = 0;
    int falseIntent = 0;
    int falseSlot = 0;
    std::cout << "[RESULT]:" << std::endl;
    for (auto sequence: sequences) {
        std::cout << sequencePaths[sequenceIndex] << ":" << std::endl;

        Vec<TensorDesc> inputDescs;
        for (Tensor t: sequence) {
            inputDescs.push_back(t.get_desc());
        }
        pipeline->infer_output_tensors_size(inputDescs);

        auto modelInputTensorNames = pipeline->get_model_input_tensor_names();
        HashMap<std::string, std::shared_ptr<U8>> modelInputTensors;
        for (int index = 0; index < (int)modelInputTensorNames.size(); index++) {
            U8* tmp = sequence[index].get_val();
            pipeline->copy_to_named_input(modelInputTensorNames[index], tmp);
        }

        double timeBegin = ut_time_ms();
        pipeline->run();
        double timeEnd = ut_time_ms();
        totalTime += (timeEnd - timeBegin);

        Tensor intentSoftmax = pipeline->get_tensor_by_name("intent_softmax");
        U32 intentNum = tensorNumElements(intentSoftmax.get_desc());
        U32 intentMaxIndex = 0;
        for (U32 index = 1; index < intentNum; index++) {
            if (intentSoftmax.getElement(index) > intentSoftmax.getElement(intentMaxIndex))
                intentMaxIndex = index;
        }
        std::cout << "    intent: " << intentMaxIndex << " " << intentSoftmax.getElement(intentMaxIndex) << std::endl;
        if (intentPaths.size() > 0) {
            F32 *intentResult = (F32*)intents[sequenceIndex][0].get_val();
            if (intentMaxIndex != intentResult[0] || abs(intentSoftmax.getElement(intentMaxIndex) - intentResult[1]) > 0.1) {
                falseIntent++;
            }
        }
        Tensor slotSoftmax = pipeline->get_tensor_by_name("slot_softmax");
        U32 slotNum   = slotSoftmax.get_desc().dims[1];
        U32 slotRange = slotSoftmax.get_desc().dims[0];
        Vec<U32> slotSoftmaxResult;
        std::cout << "    slot: ";
        for (U32 i = 0; i < slotNum; i++) {
            U32 slotMaxIndex = 0;
            for (U32 index = 1; index < slotRange; index++) {
                if (slotSoftmax.getElement(i*slotRange + index) > 
                    slotSoftmax.getElement(i*slotRange +  slotMaxIndex))
                    slotMaxIndex = index;
            }
            slotSoftmaxResult.push_back(slotMaxIndex);
            std::cout << slotMaxIndex << " ";
        }
        std::cout << std::endl;
        if (slotPaths.size() > sequenceIndex) {
            U32 *slotResult = (U32*)slots[sequenceIndex][0].get_val();
            for (U32 i = 0; i < slotSoftmaxResult.size(); i++) {
                if (slotSoftmaxResult.size() != slots[sequenceIndex][0].get_desc().dims[0] || slotResult[i] != slotSoftmaxResult[i]) {
                    falseSlot++;
                    break;
                }
            }
        }

        sequenceIndex++;
    }

    UTIL_TIME_STATISTICS
    std::cout << "[SUMMARY]:" << std::endl;
    U32 validSequence = UNI_MAX(1, sequenceIndex);
    CI_info("intent correct rate: " << 100.0 * (validSequence - falseIntent) / validSequence  << " %");
    CI_info("slot   correct rate: " << 100.0 * (validSequence - falseSlot) / validSequence  << " %");
    CI_info("avg_time:" << 1.0 * totalTime / validSequence << "ms/sequence");

    return 0;
}
