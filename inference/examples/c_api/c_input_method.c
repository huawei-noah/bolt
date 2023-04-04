// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include <stdio.h>
#include <string.h>
#include "../../examples/c_api/c_test.h"

int main(int argc, char *argv[])
{
    ParseOptions(argc, argv);
    ModelHandle inferenceHandle;
    ResultHandle resultHandle;
    if (useFileStream) {
        char *modelFileStream = BuildFileStream(modelPath);
        CreateInference(useFileStream, modelFileStream, algorithmMapPath, affinity,
            &inferenceHandle, &resultHandle);
    } else {
        CreateInference(
            useFileStream, modelPath, algorithmMapPath, affinity, &inferenceHandle, &resultHandle);
    }

    int inputNum, *inputN, *inputC, *inputH, *inputW;
    DATA_TYPE *inputDT;
    DATA_FORMAT *inputDF;
    char **inputName;
    void **inputData;
    CreateInputTensorDesc(inferenceHandle, &inputNum, &inputName, &inputN, &inputC, &inputH,
        &inputW, &inputDT, &inputDF);
    inputData = (void **)malloc(sizeof(void *) * inputNum);

    int outputNum, *outputN, *outputC, *outputH, *outputW;
    DATA_TYPE *outputDT;
    DATA_FORMAT *outputDF;
    char **outputName;
    void **outputData, **lastOutputData;
    CreateOutputTensorDesc(resultHandle, &outputNum, &outputName, &outputN, &outputC, &outputH,
        &outputW, &outputDT, &outputDF);
    outputData = (void **)malloc(sizeof(void *) * outputNum);
    MallocTensor(outputNum, outputName, outputN, outputC, outputH, outputW, outputDT, outputDF,
        &lastOutputData);
    InitTensor(outputNum, outputName, outputN, outputC, outputH, outputW, outputDT, outputDF,
        lastOutputData, 0);

    //int data[1][4] = {{137, 667, 13439, 2835}};
    //int data[2][2] = {{137, 667}, {13439, 2835}};
    int data[4][1] = {{137}, {667}, {13439}, {2835}};
    for (unsigned int i = 0; i < sizeof(data) / sizeof(data[0]); i++) {
        printf("Inference step %u:\n", i);
        int num = sizeof(data[i]) / sizeof(int);
        for (int j = 0; j < inputNum; j++) {
            if (strcmp(inputName[j], "input") == 0) {
                inputData[j] = data[i];
                inputC[j] = num;
            } else {
                char *name;
                if (strcmp(inputName[j], "hidden1") == 0) {
                    name = "state1";
                } else if (strcmp(inputName[j], "hidden2") == 0) {
                    name = "state2";
                } else {
                    printf("[ERROR] can not build output->input map for %s.\n", inputName[j]);
                    return 1;
                }
                int stateId = -1;
                for (int k = 0; k < outputNum; k++) {
                    if (strcmp(outputName[k], name) == 0) {
                        stateId = k;
                        break;
                    }
                }
                if (stateId == -1) {
                    printf("[ERROR] can not find last output %s for next input %s\n", name,
                        inputName[j]);
                    return 1;
                }
                if (i == 0) {
                    inputData[j] = lastOutputData[stateId];
                } else {
                    inputData[j] = outputData[stateId];
                }
            }
        }
        PrintTensor(inputNum, inputName, inputN, inputC, inputH, inputW, inputDT, inputDF,
            inputData, "    input ", 8);
        ResizeModelInput(inferenceHandle, inputNum, (const char **)inputName, inputN, inputC,
            inputH, inputW, inputDT, inputDF);
        RunModel(inferenceHandle, resultHandle, inputNum, (const char **)inputName, inputData);
        GetOutputDataFromResultHandle(resultHandle, outputNum, outputData);
        PrintTensor(outputNum, outputName, outputN, outputC, outputH, outputW, outputDT, outputDF,
            outputData, "    output ", 8);
    }

    FreeTensorDesc(inputNum, inputName, inputN, inputC, inputH, inputW, inputDT, inputDF);
    free(inputData);
    FreeTensor(outputNum, outputName, outputN, outputC, outputH, outputW, outputDT, outputDF,
        lastOutputData);
    free(outputData);
    FreeResultHandle(resultHandle);
    DestroyModel(inferenceHandle);
    return 0;
}
