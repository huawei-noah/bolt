// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include <sys/time.h>
#include <unistd.h>
#include <stdio.h>
#include <string.h>
#include "../../examples/c_api/c_test.h"

char *modelPath = (char *)"";
AFFINITY_TYPE affinity = CPU_HIGH_PERFORMANCE;
char *algorithmMapPath = NULL;
int loopTime = 1;
int useFileStream = 0;
char *algorithmMapName = (char *)"";

double GetTimeMs()
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    double time = tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
    return time;
}

void PrintHelp()
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

void ParseOptions(int argc, char *argv[])
{
    if (argc < 2) {
        PrintHelp();
        exit(-1);
    }
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--h") == 0 ||
            strcmp(argv[i], "-help") == 0 || strcmp(argv[i], "--help") == 0) {
            PrintHelp();
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
                    PrintHelp();
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
                PrintHelp();
                exit(-1);
        }
    }
    fflush(stdout);
}

char *BuildFileStream(const char *fileName)
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

void InitTensor(int num,
    char **name,
    const int *n,
    const int *c,
    const int *h,
    const int *w,
    const DATA_TYPE *dt,
    const DATA_FORMAT *df,
    void **data,
    float value)
{
    for (int i = 0; i < num; i++) {
        int length = n[i] * c[i] * h[i] * w[i];
        switch (dt[i]) {
            case FP_32: {
                float *ptr = (float *)data[i];
                for (int i = 0; i < length; i++) {
                    ptr[i] = value;
                }
                break;
            }
#ifdef _USE_FP16
            case FP_16: {
                __fp16 *ptr = (__fp16 *)data[i];
                for (int i = 0; i < length; i++) {
                    ptr[i] = value;
                }
                break;
            }
#endif
            default:
                printf("[ERROR] unsupported data precision in %s\n", __func__);
                exit(1);
        }
    }
}

void PrintTensor(int num,
    char **name,
    const int *n,
    const int *c,
    const int *h,
    const int *w,
    const DATA_TYPE *dt,
    const DATA_FORMAT *df,
    void **data,
    const char *printPrefix,
    int printElementNum)
{
    for (int i = 0; i < num; i++) {
        printf("%sname:%s type:%s format:%s(%d,%d,%d,%d) data:", printPrefix, name[i],
            GetDataTypeString()[dt[i]], GetDataFormatString()[df[i]], n[i], c[i], h[i], w[i]);
        int length = n[i] * c[i] * h[i] * w[i];
        if (length > printElementNum) {
            length = printElementNum;
        }
        if (data[i] == NULL) {
            printf("\n");
            continue;
        }
        for (int j = 0; j < length; j++) {
            switch (dt[i]) {
                case FP_32: {
                    float *ptr = (float *)data[i];
                    printf("%f ", ptr[j]);
                    break;
                }
                case INT_32: {
                    int *ptr = (int *)data[i];
                    printf("%d ", ptr[j]);
                    break;
                }
                case UINT_32: {
                    unsigned *ptr = (unsigned *)data[i];
                    printf("%d ", ptr[j]);
                    break;
                }
#ifdef _USE_FP16
                case FP_16: {
                    __fp16 *ptr = (__fp16 *)data[i];
                    printf("%f ", ptr[j]);
                    break;
                }
#endif
                default:
                    printf("[ERROR] can not process data type in %s.\n", __func__);
                    exit(1);
            }
        }
        printf("\n");
    }
    fflush(stdout);
}
