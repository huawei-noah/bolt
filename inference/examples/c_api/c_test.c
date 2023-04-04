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

char *modelPath = NULL;
char *inputDataPath = NULL;
AFFINITY_TYPE affinity = CPU_HIGH_PERFORMANCE;
char *affinityPolicyName = (char *)"CPU";
char *algorithmMapPath = NULL;
int loopTime = 1;
int warmUp = 10;
int threadsNum = 1;
int useFileStream = 0;

double GetTimeMs()
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    double time = tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
    return time;
}

void PrintHelp()
{
    printf("usage: ./exe -m <boltModelPath> -i [inputDataPath] -a [affinityPolicyName] -p "
           "[algorithmMapPath] -l [loopTime] -w [warmTime] -t [threadsNum]\n"
           "\nParameter description: (<> must be filled with exact value, [] is optional)\n"
           "1. -m <boltModelPath>: Bolt model file path on disk.\n"
           "2. -i [inputDataPath]: Input data file path on disk.\n"
           "    If not set input data path, benchmark will use fake data.\n"
           "    If model only have one input, you can directly pass file path, currently support "
           ".txt and "
           ".bin format. File only contains data and split with space or newline.\n"
           "    If model have multiple inputs, you can pass directory path, benchmark will search "
           "input_name.txt in directory and read it.\n"
           "    If you want to change model input size, you can pass directory path with a "
           "shape.txt file "
           "in that directory. shape.txt need to be write in this format.\n"
           "        input_name0 1 3 224 224\n"
           "        input_name1 1 224 224\n"
           "3. -a [affinityPolicyName]: Affinity policy. you can choose one of "
           "{CPU_AFFINITY_HIGH_PERFORMANCE, CPU_AFFINITY_LOW_POWER, CPU, GPU}. default: %s.\n"
           "4. -p [algorithmMapPath]: Algorithm configration path.\n"
           "5. -l [loopTime]: Loop running times. default: %d.\n"
           "6. -w [warmTime]: Warm up times. default: %d.\n"
           "7. -t [threadsNum]: Parallel threads num. default: %d.\n"
           "8. -f [useFileStream]: Use model and algorithm map file stream. default: %d.\n"
           "Example:\n"
           "    ./exe -m /local/models/resnet50_f16.bolt\n"
           "    ./exe -m /local/models/resnet50_f16.bolt -i ./input.txt\n"
           "    ./exe -m /local/models/resnet50_f16.bolt -i ./data/\n"
           "Note:\n    If you want to profiling network and get execution time of each layer, please rebuild Bolt with --profile option.\n",
        affinityPolicyName, loopTime, warmUp, threadsNum, useFileStream);
}

int ParseOptions(int argc, char *argv[])
{
    if (argc < 2) {
        PrintHelp();
        return 1;
    }
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--h") == 0 ||
            strcmp(argv[i], "-help") == 0 || strcmp(argv[i], "--help") == 0) {
            PrintHelp();
            return 1;
        }
    }

    int option;
    const char *optionstring = "m:i:a:p:l:w:t:f:";
    while ((option = getopt(argc, argv, optionstring)) != -1) {
        switch (option) {
            case 'm':
                printf("option is -m <boltModelPath>, value is: %s\n", optarg);
                modelPath = optarg;
                break;
            case 'i':
                printf("option is -i [inputDataPath], value is: %s\n", optarg);
                inputDataPath = optarg;
                break;
            case 'a':
                printf("option is -a [affinityPolicyName], value is: %s\n", optarg);
                affinityPolicyName = optarg;
                break;
            case 'l':
                printf("option is -l [loopTime], value is: %s\n", optarg);
                loopTime = atoi(optarg);
                break;
            case 'w':
                printf("option is -w [warmTime], value is: %s\n", optarg);
                warmUp = atoi(optarg);
                break;
            case 't':
                printf("option is -t [threadsNum], value is: %s\n", optarg);
                threadsNum = atoi(optarg);
                break;
            case 'p':
                printf("option is -p [algorithmMapPath], value is: %s\n", optarg);
                algorithmMapPath = optarg;
                break;
            case 'f':
                printf("option is -f [useFileStream], value is: %s\n", optarg);
                useFileStream = atoi(optarg);
                break;
            default:
                PrintHelp();
                return 1;
        }
    }
    fflush(stdout);
    if (modelPath == NULL) {
        printf("Please give an valid bolt model path.\n");
        PrintHelp();
        return 1;
    }
    if (strcmp(affinityPolicyName, "CPU_AFFINITY_HIGH_PERFORMANCE") == 0) {
        affinity = CPU_HIGH_PERFORMANCE;
    } else if (strcmp(affinityPolicyName, "CPU_AFFINITY_LOW_POWER") == 0) {
        affinity = CPU_LOW_POWER;
    } else if (strcmp(affinityPolicyName, "CPU") == 0) {
        affinity = CPU;
    } else if (strcmp(affinityPolicyName, "GPU") == 0) {
        affinity = GPU;
    } else {
        PrintHelp();
        return 1;
    }
    return 0;
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
