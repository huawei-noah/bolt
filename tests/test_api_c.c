// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), 
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, 
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE 
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR 
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include "type.h"
#include "../exports/c/bolt.h"

void print_help(char* argv[]) {
    printf("usage: %s modelPath \n", argv[0]);
}

void classification(const char* modelPath, DEVICE_TYPE device) {
    const char* precision_BNN  = "f16.bolt";
    const char* precision_INT8 = "int8_q.bolt";
    const char* precision_FP16 = "f16.bolt";
    const char* precision_FP32 = "f32.bolt";
    DATA_TYPE precisionMode = FP_32;
    if (strstr(modelPath, precision_BNN)
        || strstr(modelPath, precision_INT8)
        || strstr(modelPath, precision_FP16))
    {
	    precisionMode = FP_16;
    } else if (strstr(modelPath, precision_FP32)) {
	    precisionMode = FP_32;
    } else {
        printf("[ERROR] inference precision currently support "
            "FP32(_f32.bolt)/FP16(_f16.bolt)/INT8(_int8_q.bolt)/BNN(_f16.bolt), "
            "unsupported %s\n", modelPath);
        exit(1);
    }

    IHandle model_address = model_create(modelPath, HIGH_PERFORMANCE, device);
    int num_input = 1;
    int n[1] = {1};
    int c[1] = {3};
    int h[1] = {224};
    int w[1] = {224};
    char* firstName = "data";
    char* name[1];
    name[0] = firstName;
    DATA_TYPE dt_input[1] = {precisionMode};
    DATA_FORMAT df_input[1] = {NCHW};
    model_ready(model_address, num_input, n, c, h, w, name, dt_input, df_input);
    IResult model_result = IResult_malloc_all(model_address);

    void* mem[1];
    int length = 3 * 224 * 224;
    switch (precisionMode) {
#ifdef _USE_FP32
        case FP_32: {
            F32 *ptr = (F32 *)malloc(sizeof(F32) * length);
            for (int i = 0; i < length; i++) {
                ptr[i] = 1;
            }
            mem[0] = (void*)ptr;
            break;
        }
#endif
#ifdef _USE_FP16
        case FP_16: {
            F16 *ptr = (F16 *)malloc(sizeof(F16) * length);
            for (int i = 0; i < length; i++) {
                ptr[i] = 1;
            }
            mem[0] = (void*)ptr;
            break;
        }
#endif
        default:
            printf("[ERROR] unsupported data precision in C API test\n");
            exit(1);
    }
    model_run(model_address, model_result, 1, name, mem);

    int model_result_num = IResult_num_outputs(model_result);

    char* outputNames[1];
    void* outputData[1];
    int output_n[1];
    int output_c[1];
    int output_h[1];
    int output_w[1];
    DATA_TYPE dt_output[1];
    DATA_FORMAT df_output[1];
    IResult_get(model_result, model_result_num, outputNames, outputData, output_n, output_c, output_h, output_w, dt_output, df_output);
    IResult_free(model_result);
    model_destroy(model_address);
    free(mem[0]);
}

int main() { 
    const char* mobilenet_v1_fp16_modelPath = "/data/local/tmp/CI/model_zoo/caffe_models/mobilenet_v1/mobilenet_v1_f16.bolt";
    classification(mobilenet_v1_fp16_modelPath, CPU);

    const char* mobilenet_v1_fp32_modelPath = "/data/local/tmp/CI/model_zoo/caffe_models/mobilenet_v1/mobilenet_v1_f32.bolt";
    classification(mobilenet_v1_fp32_modelPath, CPU);
     
    const char* mobilenet_v2_fp16_modelPath = "/data/local/tmp/CI/model_zoo/caffe_models/mobilenet_v2/mobilenet_v2_f16.bolt";
    classification(mobilenet_v2_fp16_modelPath, CPU);
    
    const char* mobilenet_v2_fp32_modelPath = "/data/local/tmp/CI/model_zoo/caffe_models/mobilenet_v2/mobilenet_v2_f32.bolt";
    classification(mobilenet_v2_fp32_modelPath, CPU);

    const char* resnet50_fp16_modelPath = "/data/local/tmp/CI/model_zoo/caffe_models/resnet50/resnet50_f16.bolt";
    classification(resnet50_fp16_modelPath, CPU);

    const char* resnet50_fp32_modelPath = "/data/local/tmp/CI/model_zoo/caffe_models/resnet50/resnet50_f32.bolt";
    classification(resnet50_fp32_modelPath, CPU);

    classification(mobilenet_v1_fp16_modelPath, GPU);
    classification(mobilenet_v2_fp16_modelPath, GPU);
    return 0;
} 
