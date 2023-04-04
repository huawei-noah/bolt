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
#include <stdio.h>
#ifdef _USE_GPU
#include <CL/cl.h>
#endif

int main(int argc, char *argv[])
{
    int ret = ParseOptions(argc, argv);
    if (ret) {
        return ret;
    }

    SetNumThreads(threadsNum);

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

    int outputNum, *outputN, *outputC, *outputH, *outputW;
    DATA_TYPE *outputDT;
    DATA_FORMAT *outputDF;
    char **outputName;
    void **outputData;
    CreateOutputTensorDesc(resultHandle, &outputNum, &outputName, &outputN, &outputC, &outputH,
        &outputW, &outputDT, &outputDF);

    MallocTensor(inputNum, inputName, inputN, inputC, inputH, inputW, inputDT, inputDF, &inputData);
    MallocTensor(
        outputNum, outputName, outputN, outputC, outputH, outputW, outputDT, outputDF, &outputData);

#ifdef _USE_GPU
    cl_int err = CL_SUCCESS;
    cl_mem **inputMem = NULL, **outputMem = NULL;
    cl_context *ctx = GetDeviceContext();
    cl_command_queue *queue = GetDeviceCommandQueue();
    if (affinity == GPU) {
        inputMem = (cl_mem **)malloc(sizeof(cl_mem *) * inputNum);
        for (int i = 0; i < inputNum; i++) {
            inputMem[i] = (cl_mem *)malloc(sizeof(cl_mem));
            size_t size =
                inputN[i] * inputC[i] * inputH[i] * inputW[i] * GetDataTypeSize(inputDT[i]);
            *(inputMem[i]) = clCreateBuffer(*ctx, CL_MEM_READ_WRITE, size, NULL, &err);
            printf("[DEBUG] create mem:%p size:%ld.\n", *(inputMem[i]), size);
            if (err != CL_SUCCESS) {
                printf("[ERROR] can not create buffer:%d.\n", err);
            }
        }
        outputMem = (cl_mem **)malloc(sizeof(cl_mem *) * outputNum);
        for (int i = 0; i < outputNum; i++) {
            outputMem[i] = (cl_mem *)malloc(sizeof(cl_mem));
            size_t size =
                outputN[i] * outputC[i] * outputH[i] * outputW[i] * GetDataTypeSize(outputDT[i]);
            *(outputMem[i]) = clCreateBuffer(*ctx, CL_MEM_READ_WRITE, size, NULL, &err);
            printf("[DEBUG] create mem:%p size:%ld.\n", *(outputMem[i]), size);
            if (err != CL_SUCCESS) {
                printf("[ERROR] can not create buffer:%d.\n", err);
            }
        }
        SetInputOutput(inferenceHandle, inputNum, (const char**)inputName, (void **)inputMem);
        SetInputOutput(inferenceHandle, outputNum, (const char **)outputName, (void **)outputMem);
#else
    if (affinity == GPU) {
        printf("[ERROR] not support GPU.\n");
        return 1;
#endif
    } else {
        SetInputOutput(inferenceHandle, inputNum, (const char **)inputName, inputData);
        SetInputOutput(inferenceHandle, outputNum, (const char **)outputName, outputData);
    }
    for (int i = 0; i < loopTime; i++) {
        if (inputDataPath != NULL) {
            LoadTensor(inputNum, inputName, inputN, inputC, inputH, inputW, inputDT, inputDF,
                inputData, inputDataPath);
        } else {
            InitTensor(inputNum, inputName, inputN, inputC, inputH, inputW, inputDT, inputDF,
                inputData, 1);
        }
        PrintTensor(inputNum, inputName, inputN, inputC, inputH, inputW, inputDT, inputDF,
            inputData, "input ", 8);

#ifdef _USE_GPU
        if (affinity == GPU) {
            for (int i = 0; i < inputNum; i++) {
                size_t size =
                    inputN[i] * inputC[i] * inputH[i] * inputW[i] * GetDataTypeSize(inputDT[i]);
                err = clEnqueueWriteBuffer(
                    *queue, *(inputMem[i]), CL_TRUE, 0, size, inputData[i], 0, NULL, NULL);
                if (err != CL_SUCCESS) {
                    printf("[ERROR] can not trans data to device:%d.\n", err);
                }
            }
        }
#endif
        RunModel(inferenceHandle, NULL, 0, NULL, NULL);
#ifdef _USE_GPU
        if (affinity == GPU) {
            clFinish(*((cl_command_queue*)GetDeviceCommandQueue()));
            for (int i = 0; i < outputNum; i++) {
                size_t size = outputN[i] * outputC[i] * outputH[i] * outputW[i] *
                    GetDataTypeSize(outputDT[i]);
                err = clEnqueueReadBuffer(
                    *queue, *(outputMem[i]), CL_TRUE, 0, size, outputData[i], 0, NULL, NULL);
                if (err != CL_SUCCESS) {
                    printf("[ERROR] can not trans data to host.\n");
                }
            }
        }
#endif

        PrintTensor(outputNum, outputName, outputN, outputC, outputH, outputW, outputDT, outputDF,
            outputData, "output ", 8);
    }

    FreeTensor(inputNum, inputName, inputN, inputC, inputH, inputW, inputDT, inputDF, inputData);
    FreeTensor(
        outputNum, outputName, outputN, outputC, outputH, outputW, outputDT, outputDF, outputData);
#ifdef _USE_GPU
    if (affinity == GPU) {
        for (int i = 0; i < inputNum; i++) {
            clReleaseMemObject(*(inputMem[i]));
            free(inputMem[i]);
        }
        free(inputMem);
        for (int i = 0; i < outputNum; i++) {
            clReleaseMemObject(*(outputMem[i]));
            free(outputMem[i]);
        }
        free(outputMem);
    }
#endif
    FreeResultHandle(resultHandle);
    DestroyModel(inferenceHandle);
    return 0;
}
