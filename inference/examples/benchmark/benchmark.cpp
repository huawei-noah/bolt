// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include <getopt.h>
#include "inference.hpp"
#include "data_loader.hpp"
#include "profiling.h"

char *modelPath = NULL;
std::string inputDataPath = "";
std::string affinityPolicyName = "CPU_AFFINITY_HIGH_PERFORMANCE";
char *algorithmMapPath = NULL;
int loopTime = 1;
int warmUp = 10;
int threadsNum = OMP_MAX_NUM_THREADS;

void PrintHelp()
{
    printf("usage: ./benchmark -m <boltModelPath> -i [inputDataPath] -a [affinityPolicyName] -p "
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
           "Example:\n"
           "    ./benchmark -m /local/models/resnet50_f16.bolt\n"
           "    ./benchmark -m /local/models/resnet50_f16.bolt -i ./input.txt\n"
           "    ./benchmark -m /local/models/resnet50_f16.bolt -i ./data/\n"
           "Note:\n    If you want to profiling network and get execution time of each layer, please rebuild Bolt with --profile option.\n",
        affinityPolicyName.c_str(), loopTime, warmUp, threadsNum);
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
    const char *optionstring = "m:i:a:p:l:w:t:";
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
                affinityPolicyName = upper(optarg);
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
    return 0;
}

std::map<std::string, std::shared_ptr<U8>> create_tensors_from_path(
    std::string inputDataPath, std::shared_ptr<CNN> pipeline)
{
    std::map<std::string, TensorDesc> inputDescMap = pipeline->get_input_desc();
    std::vector<Tensor> input;
    if (inputDataPath != "" && is_directory(inputDataPath.c_str())) {
        std::map<std::string, TensorDesc> descs =
            load_shape(inputDataPath + "/shape.txt", inputDescMap);
        if (descs.size() > 0) {
            pipeline->reready(descs);
            inputDescMap = descs;
        }
        for (auto iter : inputDescMap) {
            std::string path = inputDataPath + "/" + iter.first + ".txt";
            input.push_back(load_txt(path, {iter.second})[0]);
        }
    } else {
        std::vector<DataType> sourceDataTypes;
        std::vector<TensorDesc> inputDescs;
        for (auto iter : inputDescMap) {
            TensorDesc curDesc = iter.second;
            sourceDataTypes.push_back(curDesc.dt);
            inputDescs.push_back(curDesc);
        }
        if (endswith(inputDataPath, ".txt")) {
            input = load_txt(inputDataPath, inputDescs);
        } else {
            input = load_bin(inputDataPath, sourceDataTypes, inputDescs);
        }
    }
    std::map<std::string, std::shared_ptr<U8>> model_tensors_input;
    int index = 0;
    printf("\nInput Information:\n");
    for (auto iter : inputDescMap) {
        printf("Input Tensor %s %s\n", iter.first.c_str(), input[index].string(8).c_str());
        model_tensors_input[iter.first] = ((CpuMemory *)input[index].get_memory())->get_shared_ptr();
        index++;
    }
    return model_tensors_input;
}

void print_result(std::map<std::string, std::shared_ptr<Tensor>> outMap)
{
    printf("\nBenchmark Result:\n");
    for (auto iter : outMap) {
        Tensor result = *(iter.second);
        printf("Output Tensor %s %s\n", iter.first.c_str(), result.string(8).c_str());
    }
}

std::map<std::string, std::shared_ptr<Tensor>> get_output(
    std::shared_ptr<CNN> pipeline, std::string affinity)
{
    std::map<std::string, std::shared_ptr<Tensor>> outMap = pipeline->get_output();
    if (affinity == "GPU") {
#ifdef _USE_GPU
        for (auto iter : outMap) {
            Tensor result = *(iter.second);
            auto mem = (OclMemory *)result.get_memory();
            UNI_PROFILE(mem->get_mapped_ptr(), "copy " + iter.first, std::string("output::copy"));
        }
#else
        UNI_WARNING_LOG("this binary not support GPU, please recompile project with GPU "
                        "compile options\n");
#endif
    }
    return outMap;
}

int benchmark(int argc, char *argv[])
{
    UNI_TIME_INIT
    int ret = ParseOptions(argc, argv);
    if (ret) {
        return 1;
    }

    set_cpu_num_threads(threadsNum);

    // 1: set up the pipeline
    double timeBegin = ut_time_ms();
    auto pipeline = createPipeline(affinityPolicyName.c_str(), modelPath, algorithmMapPath);
#ifdef _USE_GPU
    if (std::string(affinityPolicyName) == std::string("GPU")) {
        gcl_finish(OCLContext::getInstance().handle.get());
    }
#endif
    double timeEnd = ut_time_ms();
    double prepareTime = timeEnd - timeBegin;

    // 2: create input data and feed the pipeline with it
    auto model_tensors_input = create_tensors_from_path(inputDataPath, pipeline);

    std::map<std::string, std::shared_ptr<Tensor>> outMap;

    // 3: warm up and run
    UNI_TIME_STOP
    timeBegin = ut_time_ms();
    for (int i = 0; i < warmUp; i++) {
        pipeline->set_input_by_assign(model_tensors_input);
        pipeline->run();
        outMap = get_output(pipeline, affinityPolicyName);
    }
#ifdef _USE_GPU
    if (std::string(affinityPolicyName) == std::string("GPU")) {
        gcl_finish(OCLContext::getInstance().handle.get());
    }
#endif
    timeEnd = ut_time_ms();
    double warmUpTime = timeEnd - timeBegin;
    UNI_TIME_START

    double minTime = INT_MAX;
    double maxTime = 0;
    double totalTime = 0;
    for (int i = 0; i < loopTime; i++) {
        double timeBegin = ut_time_ms();
        pipeline->set_input_by_assign(model_tensors_input);
        pipeline->run();
        outMap = get_output(pipeline, affinityPolicyName);
        double timeEnd = ut_time_ms();
        double time = timeEnd - timeBegin;
        minTime = (minTime < time) ? minTime : time;
        maxTime = (maxTime > time) ? maxTime : time;
        totalTime += time;
    }
    if (minTime == INT_MAX) {
        minTime = 0;
    }

    // 4: process result
    print_result(outMap);

    UNI_TIME_STATISTICS
    UNI_CI_LOG("model prepare_time:%fms\n", 1.0 * prepareTime);
    UNI_CI_LOG("model warm_up_time:%fms\n", 1.0 * warmUpTime);
    UNI_CI_LOG("run total_time:%fms(loops=%d)\n", 1.0 * totalTime, loopTime);
    UNI_CI_LOG("run avg_time:%fms/data\n", 1.0 * totalTime / UNI_MAX(1, loopTime));
    UNI_CI_LOG("run min_time:%fms/data\n", 1.0 * minTime);
    UNI_CI_LOG("run max_time:%fms/data\n", 1.0 * maxTime);
    if (algorithmMapPath != NULL) {
        pipeline->saveAlgorithmMapToFile(algorithmMapPath);
    }
    return 0;
}

int main(int argc, char *argv[])
{
    int ret = benchmark(argc, argv);
    UNI_MEM_STATISTICS();
    return ret;
}
