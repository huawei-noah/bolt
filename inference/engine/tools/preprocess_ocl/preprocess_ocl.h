// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _H_PREPROCESS_OCL
#define _H_PREPROCESS_OCL

#include <vector>
#include <map>
#include <string>

#include "file.h"

#include "inference.hpp"
#include "../api/c/bolt.h"
#include "../api/c/bolt_simplify.h"

inline void update_kernels(
    std::map<std::string, std::vector<U8>> *kernels, std::string kernel_name, Program program)
{
    if (kernels == NULL) {
        return;
    }
    if (kernels->find(kernel_name) == kernels->end()) {
        U8 *binary = NULL;
        U32 binary_len = 0;
        CHECK_STATUS(get_program_binary(program, &binary, &binary_len));
#ifdef _DETAIL
        CHECK_STATUS(save_binary(std::string(kernel_name + ".bin").c_str(), binary, binary_len));
        char *source = NULL;
        U32 source_len;
        CHECK_STATUS(get_program_info(program, CL_PROGRAM_SOURCE, (void **)&source, &source_len));
        CHECK_STATUS(save_string(std::string(kernel_name + ".cl").c_str(), source));
        if (source != NULL) {
            free(source);
        }
#endif
        std::vector<U8> k(binary_len);
        for (U32 i = 0; i < binary_len; ++i) {
            k[i] = binary[i];
        }
        (*kernels)[kernel_name] = k;
        if (binary != NULL) {
            free(binary);
        }
    }
}

inline void run_model(
    const char *model_path, const char *algo_dir, std::map<std::string, std::vector<U8>> *kernels)
{
    if (!strstr(model_path, "f16.bolt")) {
        UNI_WARNING_LOG("Bolt gpu only support float16 inference, and model file is end with "
                        "_f16.bolt suffix.\n");
        return;
    }
    UNI_INFO_LOG("Build gpu kernels and algorithm map file for bolt model(%s)...\n", model_path);

    ModelHandle inferenceHandle;
    ResultHandle resultHandle;
    CreateInference(false, model_path, algo_dir, GPU, &inferenceHandle, &resultHandle);

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
    CreateOutputTensorDesc(resultHandle, &outputNum, &outputName, &outputN, &outputC, &outputH,
        &outputW, &outputDT, &outputDF);

    RunModel(inferenceHandle, resultHandle, inputNum, (const char **)inputName, inputData);

    GCLHandle_t handle = OCLContext::getInstance().handle.get();
    for (auto p : handle->kernelMap) {
        std::string device_name = handle->deviceName;
        std::string kernel_name = p.first;
        kernel_name.erase(0, device_name.size() + 1);
        Program program;
        CHECK_STATUS(get_program_info_from_kernel(p.second, &program));
        update_kernels(kernels, kernel_name, program);
    }
    for (auto p : handle->programMap) {
        update_kernels(kernels, p.first, p.second);
    }
    CHECK_STATUS(gcl_finish(handle));

    FreeTensor(inputNum, inputName, inputN, inputC, inputH, inputW, inputDT, inputDF, inputData);
    FreeTensorDesc(outputNum, outputName, outputN, outputC, outputH, outputW, outputDT, outputDF);
    FreeResultHandle(resultHandle);
    DestroyModel(inferenceHandle);
}

#endif
