// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include <float.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include "../api/c/bolt.h"
#include <sys/time.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <stdbool.h>
#include <unistd.h>
#ifdef _USE_FP16
#include <arm_neon.h>
typedef __fp16 F16;
#endif

double ut_time_ms()
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    double time = tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
    return time;
}

void print_help(char *argv[])
{
    printf("usage: %s modelPath \n", argv[0]);
}

void classification(const char *modelPath, AFFINITY_TYPE affinity, DATA_TYPE dt, const char *algoPath, bool useFileStream)
{
    DATA_TYPE precisionMode = dt;
    ModelHandle model_address;
    if(useFileStream) {
        model_address = CreateModelWithFileStream(modelPath, affinity, algoPath);
    } else {
        model_address = CreateModel(modelPath, affinity, algoPath);
    }

    int num_input = GetNumInputsFromModel(model_address);
    int *n = (int *)malloc(sizeof(int) * num_input);
    int *c = (int *)malloc(sizeof(int) * num_input);
    int *h = (int *)malloc(sizeof(int) * num_input);
    int *w = (int *)malloc(sizeof(int) * num_input);
    char **name = (char **)malloc(sizeof(char *) * num_input);
    for (int i = 0; i < num_input; i++) {
        name[i] = (char *)malloc(sizeof(char) * 1024);
    }
    DATA_TYPE *dt_input = (DATA_TYPE *)malloc(sizeof(DATA_TYPE) * num_input);
    DATA_FORMAT *df_input = (DATA_FORMAT *)malloc(sizeof(DATA_FORMAT) * num_input);

    GetInputDataInfoFromModel(model_address, num_input, name, n, c, h, w, dt_input, df_input);

    unsigned char **input_ptr = (unsigned char **)malloc(sizeof(unsigned char *) * num_input);
    for (int i = 0; i < num_input; i++) {
        printf("input name = %s in = %d ic = %d ih = %d iw = %d\n", name[i], n[i], c[i], h[i], w[i]);
        int length = n[i] * c[i] * h[i] * w[i];
        switch (precisionMode) {
#ifdef _USE_FP32
            case FP_32: {
                float *ptr = (float *)malloc(sizeof(float) * length);
                for (int i = 0; i < length; i++) {
                    ptr[i] = 1;
                }
                input_ptr[i] = (unsigned char *)ptr;
                break;
            }
#endif
#ifdef _USE_FP16
            case FP_16: {
                F16 *ptr = (F16 *)malloc(sizeof(F16) * length);
                for (int i = 0; i < length; i++) {
                    ptr[i] = 1;
                }
                input_ptr[i] = (unsigned char *)ptr;
                break;
            }
#endif
            default:
                printf("[ERROR] unsupported data precision in C API test\n");
                exit(1);
        }
    }

    PrepareModel(model_address, num_input, name, n, c, h, w, dt_input, df_input);

    ResultHandle model_result = AllocAllResultHandle(model_address);
    int model_result_num = GetNumOutputsFromResultHandle(model_result);

    int *output_n = (int *)malloc(sizeof(int) * model_result_num);
    int *output_c = (int *)malloc(sizeof(int) * model_result_num);
    int *output_h = (int *)malloc(sizeof(int) * model_result_num);
    int *output_w = (int *)malloc(sizeof(int) * model_result_num);
    char **outputNames = (char **)malloc(sizeof(char *) * model_result_num);
    for (int i = 0; i < model_result_num; i++) {
        outputNames[i] = (char *)malloc(sizeof(char) * 1024);
    }
    DATA_TYPE *dt_output = (DATA_TYPE *)malloc(sizeof(DATA_TYPE) * model_result_num);
    DATA_FORMAT *df_output = (DATA_FORMAT *)malloc(sizeof(DATA_FORMAT) * model_result_num);

    GetOutputDataInfoFromResultHandle(model_result, model_result_num, outputNames, output_n,
        output_c, output_h, output_w, dt_output, df_output);

    unsigned char **user_out_ptr =
        (unsigned char **)malloc(sizeof(unsigned char *) * model_result_num);
    for (int i = 0; i < model_result_num; i++) {
        printf("output name = %s on = %d oc = %d oh = %d ow = %d\n", outputNames[i], output_n[i],
            output_c[i], output_h[i], output_w[i]);
        int length = output_n[i] * output_c[i] * output_h[i] * output_w[i];
        switch (precisionMode) {
#ifdef _USE_FP32
            case FP_32: {
                float *ptr = (float *)malloc(sizeof(float) * length);
                user_out_ptr[i] = (unsigned char *)ptr;
                break;
            }
#endif
#ifdef _USE_FP16
            case FP_16: {
                F16 *ptr = (F16 *)malloc(sizeof(F16) * length);
                user_out_ptr[i] = (unsigned char *)ptr;
                break;
            }
#endif
            default:
                printf("[ERROR] unsupported data precision in C API test\n");
                exit(1);
        }
    }

    double totalTime = 0;
    double max_time = -DBL_MAX;
    double min_time = DBL_MAX;
    int loop = 1;

    /*warm up*/
    for (int i = 0; i < 1; i++) {
        RunModel(model_address, model_result, 1, name, (void **)input_ptr);
    }

    for (int i = 0; i < loop; i++) {
        double timeBegin = ut_time_ms();
        RunModel(model_address, model_result, 1, name, (void **)input_ptr);
        double timeEnd = ut_time_ms();
        double t = timeEnd - timeBegin;
        totalTime += t;
        if (t < min_time) {
            min_time = t;
        }
        if (t > max_time) {
            max_time = t;
        }
    }

    unsigned char **bolt_out_ptr =
        (unsigned char **)malloc(sizeof(unsigned char *) * model_result_num);
    GetPtrFromResultHandle(model_result, model_result_num, outputNames, (void **)bolt_out_ptr,
        output_n, output_c, output_h, output_w, dt_output, df_output);
    for (int i = 0; i < model_result_num; i++) {
        int length = output_n[i] * output_c[i] * output_h[i] * output_w[i];
        switch (precisionMode) {
#ifdef _USE_FP32
            case FP_32: {
                memcpy(user_out_ptr[i], bolt_out_ptr, sizeof(float) * length);
                break;
            }
#endif
#ifdef _USE_FP16
            case FP_16: {
                memcpy(user_out_ptr[i], bolt_out_ptr, sizeof(F16) * length);
                break;
            }
#endif
            default:
                printf("[ERROR] unsupported data precision in C API test\n");
                exit(1);
        }
    }
    FreeResultHandle(model_result);
    DestroyModel(model_address);
    free(n);
    free(c);
    free(h);
    free(w);
    free(dt_input);
    free(df_input);
    for (int i = 0; i < num_input; i++) {
        free(name[i]);
        free(input_ptr[i]);
    }
    free(name);
    free(input_ptr);
    free(output_n);
    free(output_c);
    free(output_h);
    free(output_w);
    free(dt_output);
    free(df_output);
    for (int i = 0; i < model_result_num; i++) {
        free(outputNames[i]);
        free(user_out_ptr[i]);
    }
    free(outputNames);
    free(user_out_ptr);
    free(bolt_out_ptr);

    const char* modelName = (useFileStream) ? "Use file stream" : modelPath;
    if (affinity == GPU) {
        printf("DeviceType = GPU, Model = %s\n", modelName);
    } else {
        printf("DeviceType = CPU, Model = %s\n", modelName);
    }
    printf("avg_time: %lf ms\n", 1.0 * totalTime / loop);
    printf("max_time: %lf ms\n", 1.0 * max_time);
    printf("min_time: %lf ms\n", 1.0 * min_time);
    fflush(stdout); 
}

char* buildFileStream(const char* fileName) {
    int fd;
    int length;
    struct stat ss;
    fd = open(fileName, O_RDONLY);
    if (-1 == fd) {
        printf("Open file %s failed\n", fileName);
        exit(-1);
    }
    if (-1 == fstat(fd, &ss)) {
        printf("Can not get size from file %s\n", fileName);
        exit(-1);
    }
    length = ss.st_size;
    char* bytes = (char*)mmap(NULL, length, PROT_READ, MAP_SHARED, fd, 0);
    if(MAP_FAILED == bytes) {
        printf("Map file %s failed\n", fileName);
        exit(-1);
    }
    char* res = malloc(length);
    memcpy(res, bytes, length);
    munmap(bytes, length);
    if (-1 != fd) {
        close(fd);
    }
    return res;
}

int main()
{
    const char *mobilenet_v1_fp16_modelPath = "/data/local/tmp/xyf/model/mobilenet_v1_f16.bolt";
    const char *algoPath = "./";
    bool useFileStream = false;
    classification(mobilenet_v1_fp16_modelPath, CPU_HIGH_PERFORMANCE, FP_16, algoPath, useFileStream);
    classification(mobilenet_v1_fp16_modelPath, GPU, FP_16, algoPath, useFileStream);

    /*Test use filestream to read algoFile*/
    useFileStream = true;
    const char* modelFileStream = buildFileStream(mobilenet_v1_fp16_modelPath);
    const char* algoFileStream = buildFileStream("./algorithmInfo_Mali_G52p_MOBILENET_2_4");
    classification(modelFileStream, GPU, FP_16, algoFileStream, useFileStream);
    free((void*)modelFileStream);
    free((void*)algoFileStream);
    return 0;
}
