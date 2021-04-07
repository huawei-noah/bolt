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
#include <stdbool.h>
#include <unistd.h>
#ifdef _USE_FP16
#include <arm_neon.h>
typedef __fp16 F16;
#endif
char *modelPath = (char *)"";
AFFINITY_TYPE affinity = CPU_HIGH_PERFORMANCE;
char *algorithmMapPath = (char *)"./";
int loopTime = 1;
bool useFileStream = false;
char *algorithmMapName = (char *)"";

double ut_time_ms()
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    double time = tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
    return time;
}

void print_help()
{
    printf("-m <boltModelPath>:      the file path for .bolt model\n");
    printf("-a <affinityPolicyName>: the arch used to run model, default to be "
           "CPU_HIGH_PERFORMANCE\n");
    printf("                         this value can be {CPU_HIGH_PERFORMANCE, CPU_LOW_POWER, "
           "GPU}\n");
    printf("-p <algorithmMapPath>:   the file path to store algoirthmFileName, default to be "
           "program run path \n");
    printf("-l <loopTime>:           the loopTime to run set_input + run + get_output, default to "
           "be 1\n");
    printf("-f <useFileStream>:      use file stream c api to read .bolt and algo file content, "
           "default to be 0\n");
    printf("                         this value can be {0, 1}\n");
    printf("-n <algorithmMapName>:   the algorithm map name\n");
    printf("                         you can get this file name from algorithmMapPath after run "
           "this program once\n");
}

void parse_options(int argc, char *argv[])
{
    if (argc < 2) {
        print_help();
        exit(-1);
    }
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--h") == 0 ||
            strcmp(argv[i], "-help") == 0 || strcmp(argv[i], "--help") == 0) {
            print_help();
            exit(-1);
        }
    }
    int option;
    const char *optionstring = "m:a:p:l:f:n:";
    while ((option = getopt(argc, argv, optionstring)) != -1) {
        switch (option) {
            case 'm':
                printf("option is -m <boltModelPath>, value is: %s\n", optarg);
                modelPath = optarg;
                break;
            case 'a':
                printf("option is -a [affinityPolicyName], value is: %s\n", optarg);
                if (strcmp(optarg, "CPU_HIGH_PERFORMANCE") == 0) {
                    affinity = CPU_HIGH_PERFORMANCE;
                } else if (strcmp(optarg, "CPU_LOW_POWER") == 0) {
                    affinity = CPU_LOW_POWER;
                } else if (strcmp(optarg, "GPU") == 0) {
                    affinity = GPU;
                } else {
                    print_help();
                    exit(-1);
                }
                break;
            case 'p':
                printf("option is -p [algorithmMapPath], value is: %s\n", optarg);
                algorithmMapPath = optarg;
                break;
            case 'l':
                printf("option is -l [loopTime], value is: %s\n", optarg);
                loopTime = atoi(optarg);
                break;
            case 'f':
                printf("option is -f [useFileStream], value is: %s\n", optarg);
                useFileStream = atoi(optarg);
                break;
            case 'n':
                printf("option is -n [algorithmMapName], value is: %s\n", optarg);
                algorithmMapName = optarg;
                break;
            default:
                print_help();
                exit(-1);
        }
    }
    fflush(stdout);
}

void classification(const char *modelPath,
    AFFINITY_TYPE affinity,
    DATA_TYPE dt,
    const char *algoPath,
    int loopTime,
    bool useFileStream)
{
#ifdef _USE_FP16
    if (affinity == GPU) {
        char deviceName[128];
        GetGpuDeviceName(deviceName);
        printf("Current GPU DeviceName %s\n", deviceName);
    }
#endif
    DATA_TYPE precisionMode = dt;
    ModelHandle model_address;
    if (useFileStream) {
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

    PrepareModel(model_address, num_input, (const char **)name, n, c, h, w, dt_input, df_input);

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

    /*warm up*/
    for (int i = 0; i < 1; i++) {
        RunModel(model_address, model_result, num_input, (const char **)name, (void **)input_ptr);
    }

    for (int i = 0; i < loopTime; i++) {
        double timeBegin = ut_time_ms();
        RunModel(model_address, model_result, num_input, (const char **)name, (void **)input_ptr);
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
    GetOutputDataFromResultHandle(model_result, model_result_num, (void **)bolt_out_ptr);
    for (int i = 0; i < model_result_num; i++) {
        int length = output_n[i] * output_c[i] * output_h[i] * output_w[i];
        printf("First 8 results of output %d:", i);
        int j = (length > 8) ? 8 : length;
        switch (precisionMode) {
#ifdef _USE_FP32
            case FP_32: {
                memcpy(user_out_ptr[i], bolt_out_ptr[i], sizeof(float) * length);
                float *val = (float *)user_out_ptr[i];
                for (int k = 0; k < j; k++) {
                    printf("%f ", val[k]);
                }
                printf("\n");
                break;
            }
#endif
#ifdef _USE_FP16
            case FP_16: {
                memcpy(user_out_ptr[i], bolt_out_ptr[i], sizeof(F16) * length);
                F16 *val = (F16 *)user_out_ptr[i];
                for (int k = 0; k < j; k++) {
                    printf("%f ", val[k]);
                }
                printf("\n");
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

    const char *modelName = (useFileStream) ? "Use file stream" : modelPath;
    if (affinity == GPU) {
        printf("DeviceType = GPU, Model = %s\n", modelName);
    } else {
        printf("DeviceType = CPU, Model = %s\n", modelName);
    }
    printf("avg_time: %lf ms\n", 1.0 * totalTime / loopTime);
    printf("max_time: %lf ms\n", 1.0 * max_time);
    printf("min_time: %lf ms\n", 1.0 * min_time);
    fflush(stdout);
}

char *buildFileStream(const char *fileName)
{
    FILE *file = fopen(fileName, "rb");
    if (file == NULL) {
        printf("Cannot open model file %s\n", fileName);
        return NULL;
    }

    fseek(file, 0, SEEK_END);
    size_t fileLength = ftell(file);
    rewind(file);

    char *bytes = (char *)malloc(sizeof(char) * fileLength);
    if (bytes == NULL) {
        printf("Memory allocate error.\n");
        return NULL;
    }

    size_t result = fread(bytes, 1, fileLength, file);
    if (result != fileLength) {
        printf("Read model file %s error.\n", fileName);
        return NULL;
    }
    fclose(file);
    return bytes;
}

int main(int argc, char *argv[])
{
    parse_options(argc, argv);
    if (!useFileStream) {
        classification(modelPath, affinity, FP_16, algorithmMapPath, loopTime, useFileStream);
    } else {
        const char *modelFileStream = buildFileStream(modelPath);
        char *algoInfo = malloc(strlen(algorithmMapPath) + strlen(algorithmMapName) + 1);
        const char *algoFileStream = NULL;
        if (strcmp(algorithmMapName, "") != 0) {
            strcpy(algoInfo, algorithmMapPath);
            strcpy(algoInfo, algorithmMapName);
            algoFileStream = buildFileStream(algoInfo);
        }
        classification(modelFileStream, affinity, FP_16, algoFileStream, loopTime, useFileStream);
        free((void *)algoInfo);
        free((void *)modelFileStream);
        free((void *)algoFileStream);
    }
    return 0;
}
