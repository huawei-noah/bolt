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

std::map<std::string, Tensor> prepareStates(
    DataType dt, std::string sequenceDirectory, std::string shapeMapFileName)
{
    std::map<std::string, TensorDesc> shapeMap;
    std::string filePath = sequenceDirectory + "/" + shapeMapFileName;
    FILE *shapeMapFile = fopen(filePath.c_str(), "r");
    char buffer[NAME_LEN];
    while (fscanf(shapeMapFile, "%s", buffer) != EOF) {
        TensorDesc desc;
        fscanf(shapeMapFile, "%u", &(desc.nDims));
        for (U32 i = 0; i < desc.nDims; i++) {
            fscanf(shapeMapFile, "%u", &(desc.dims[desc.nDims - 1 - i]));
        }
        if (std::string(buffer) == std::string("label")) {
            desc.dt = DT_U32;
        } else {
            desc.dt = dt;
        }
        std::string inputName(buffer);
        if (inputName.find(std::string("layer1_mem")) != std::string::npos) {
            desc.df = DF_NCHWC8;
        } else {
            desc.df = DF_NCHW;
        }
        shapeMap[inputName] = desc;
    }
    fclose(shapeMapFile);

    std::map<std::string, Tensor> tensorMap;
    for (auto iter : shapeMap) {
        std::string filePath = sequenceDirectory + "/" + iter.first + ".txt";
        TensorDesc desc = iter.second;
        tensorMap[iter.first] = load_txt(filePath, std::vector<TensorDesc>{desc})[0];
    }
    return tensorMap;
}

void saveStates(std::shared_ptr<CNN> pipeline,
    std::string sequenceDirectory,
    std::string outputFileName,
    std::string outputStatesFileName)
{
    char buffer[NAME_LEN];
    std::string outputFilePath = sequenceDirectory + "/" + outputFileName;
    std::string outputStatesFilePath = sequenceDirectory + "/" + outputStatesFileName;
    FILE *outputFile = fopen(outputFilePath.c_str(), "r");
    FILE *outputStatesFile = fopen(outputStatesFilePath.c_str(), "w");
    while (!feof(outputFile)) {
        fscanf(outputFile, "%s", buffer);
        Tensor tensor = pipeline->get_tensor_by_name(buffer);
        TensorDesc desc = tensor.get_desc();

        // write states
        fprintf(outputStatesFile, "%s\n", buffer);
        fprintf(outputStatesFile, "%u\n", desc.nDims);
        for (U32 i = 0; i < desc.nDims; i++) {
            fprintf(outputStatesFile, "%u ", desc.dims[desc.nDims - 1 - i]);
        }

        // write data
        U32 num = tensorNumElements(desc);
        std::string outputDataPath = sequenceDirectory + "/" + std::string(buffer) + ".txt";
        FILE *outputDataFile = fopen(outputDataPath.c_str(), "w");
        for (U32 i = 0; i < num; i++) {
            fprintf(outputDataFile, "%f ", tensor.element(i));
            if (i % 10 == 9) {
                fprintf(outputDataFile, "\n");
            }
        }
        fclose(outputDataFile);
    }
    fclose(outputFile);
    fclose(outputStatesFile);
}

int verify(Tensor tensor, std::string subNetworkName, std::map<std::string, TensorDesc> inputDescMap)
{
    U32 num = tensor.length();
    F32 sum = 0;
    for (U32 i = 0; i < num; i++) {
        sum += tensor.element(i);
    }
    I32 result = 0;
    if (subNetworkName == std::string("encoder")) {
        if (inputDescMap["sounds"].dims[1] == 15) {
            if (abs(sum - 44.4) >= 1) {
                result = 1;
            }
        } else if (inputDescMap["sounds"].dims[1] == 8) {
            if (abs(sum - 102.3) >= 1) {
                result = 1;
            }
        } else {
            result = 1;
        }
    } else if (subNetworkName == std::string("prediction_net")) {
        if (abs(sum - 21.7) >= 1) {
            result = 1;
        }
    } else if (subNetworkName == std::string("joint_net")) {
        if (abs(sum - (-24.6)) >= 1) {
            result = 1;
        }
    }
    return result;
}

int main(int argc, char *argv[])
{
    UNI_TIME_INIT
    ParseRes parse_res;
    parseCommandLine(argc, argv, &parse_res, "examples");
    char *modelPath = (char *)"";
    char *sequenceDirectory = (char *)"";
    std::string subNetworkName = std::string("encoder");
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
    if (parse_res.subNetworkName.second) {
        subNetworkName = std::string(parse_res.subNetworkName.first);
    }

    std::string outputTensorName;
    if (subNetworkName == std::string("encoder")) {
        outputTensorName = "encoder_block3_transformer_ln";
    } else if (subNetworkName == std::string("prediction_net")) {
        outputTensorName = "prediction_net_ln";
    } else if (subNetworkName == std::string("joint_net")) {
        outputTensorName = "joint_output_fc";
    } else {
        UNI_ERROR_LOG("unrecognized sub network(encoder|prediction_net|joint_net) %s\n",
            subNetworkName.c_str());
    }

    DataType dt;
    std::string modelPathStr = std::string(modelPath);
    // "_f[16|32].bolt"
    std::string modelPathSuffix = modelPathStr.substr(modelPathStr.size() - 9);
    if (modelPathSuffix == std::string("_f16.bolt")) {
        dt = DT_F16;
    } else if (modelPathSuffix == std::string("_f32.bolt")) {
        dt = DT_F32;
    } else if (modelPathSuffix == std::string("t8_q.bolt")) {
        dt = DT_F16;
    } else {
        UNI_ERROR_LOG("unrecognized model file path suffix %s\n", modelPathSuffix.c_str());
    }
    auto pipeline = createPipeline(affinityPolicyName, modelPath);

    double totalTime = 0;
    int loops = 1;
    U32 falseResult = 0;
    for (int i = 0; i < loops; i++) {
        std::map<std::string, Tensor> input =
            prepareStates(dt, sequenceDirectory, "input_shape.txt");
        std::map<std::string, TensorDesc> inputDescMap;
        std::map<std::string, U8 *> inputMap;
        for (auto iter : input) {
            inputDescMap[iter.first] = iter.second.get_desc();
            inputMap[iter.first] = (U8 *)((CpuMemory *)(iter.second.get_memory()))->get_ptr();
        }
        pipeline->reready(inputDescMap);
        pipeline->set_input_by_copy(inputMap);

        double timeBegin = ut_time_ms();
        pipeline->run();
        double timeEnd = ut_time_ms();
        totalTime += (timeEnd - timeBegin);
        Tensor output = pipeline->get_tensor_by_name(outputTensorName);
        falseResult += verify(output, subNetworkName, inputDescMap);
        //saveStates(pipeline, sequenceDirectory, "output_name.txt", "output_shape.txt");
    }
    UNI_TIME_STATISTICS

    std::cout << "[SUMMARY]:" << std::endl;
    U32 validSequence = loops;
    UNI_CI_LOG(
        "speech recognition rate: %f %%\n", 100.0 * (validSequence - falseResult) / validSequence);
    UNI_CI_LOG("avg_time:%fms/sequence\n", 1.0 * totalTime / validSequence);
    if (falseResult > 0) {
        UNI_ERROR_LOG("verify failed\n");
    }

    return 0;
}
