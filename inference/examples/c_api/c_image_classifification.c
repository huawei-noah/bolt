// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "../../examples/c_api/c_test.h"

int main(int argc, char *argv[])
{
    ParseOptions(argc, argv);
    ModelHandle inferenceHandle;
    ResultHandle resultHandle;
    int inNum;
    char **inName;
    if (useFileStream) {
        char *modelFileStream = BuildFileStream(modelPath);
        CreateInference(useFileStream, modelFileStream, algorithmMapPath, affinity,
            &inferenceHandle, &resultHandle, &inNum, &inName);
    } else {
        CreateInference(useFileStream, modelPath, algorithmMapPath, affinity, &inferenceHandle,
            &resultHandle, &inNum, &inName);
    }

    int inputNum, *inputN, *inputC, *inputH, *inputW;
    DATA_TYPE *inputDT;
    DATA_FORMAT *inputDF;
    char **inputName;
    void **inputData;
    CreateInputTensorDesc(inferenceHandle, &inputNum, &inputName, &inputN, &inputC, &inputH,
        &inputW, &inputDT, &inputDF);
    MallocTensor(inputNum, inputName, inputN, inputC, inputH, inputW, inputDT, inputDF, &inputData);
    InitTensor(inputNum, inputName, inputN, inputC, inputH, inputW, inputDT, inputDF, inputData, 1);

    int outputNum, *outputN, *outputC, *outputH, *outputW;
    DATA_TYPE *outputDT;
    DATA_FORMAT *outputDF;
    char **outputName;
    void **outputData;
    CreateOutputTensorDesc(resultHandle, &outputNum, &outputName, &outputN, &outputC, &outputH,
        &outputW, &outputDT, &outputDF);
    outputData = malloc(sizeof(void *) * outputNum);

    PrintTensor(inputNum, inputName, inputN, inputC, inputH, inputW, inputDT, inputDF, inputData,
        "input ", 8);
    RunModel(inferenceHandle, resultHandle, inNum, (const char **)inName, inputData);

    GetOutputDataFromResultHandle(resultHandle, outputNum, outputData);
    PrintTensor(outputNum, outputName, outputN, outputC, outputH, outputW, outputDT, outputDF,
        outputData, "output ", 8);

    FreeTensor(inputNum, inputName, inputN, inputC, inputH, inputW, inputDT, inputDF, inputData);
    FreeTensorDesc(outputNum, outputName, outputN, outputC, outputH, outputW, outputDT, outputDF);
    free(outputData);
    for (int i = 0; i < inNum; i++) {
        free(inName[i]);
    }
    free(inName);
    FreeResultHandle(resultHandle);
    DestroyModel(inferenceHandle);
    return 0;
}
