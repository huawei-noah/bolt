// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include <iostream>
#include <getopt.h>
#include "inference.hpp"
#include "data_loader.hpp"
#include "profiling.h"

char *modelPath = (char *)"";
std::string inputData = "";
char *affinityPolicyName = (char *)"CPU_AFFINITY_HIGH_PERFORMANCE";
char *algorithmMapPath = (char *)"";
int loopTime = 1;
int warmUp = 10;

void print_benchmark_usage()
{
    std::cout << "benchmark usage: (<> must be filled in with exact value; [] is optional)\n"
                 "./benchmark -m <boltModelPath> -i [inputDataPath] -a [affinityPolicyName] -p "
                 "[algorithmMapPath] -l [loopTime]\n"
                 "\nParameter description:\n"
                 "1. -m <boltModelPath>: The path where .bolt is stored.\n"
                 "2. -i [inputDataPath]: The input data absolute path. If not input the option, "
                 "benchmark will run with fake data.\n"
                 "3. -a [affinityPolicyName]: The affinity policy. If not input the option, "
                 "affinityPolicyName is CPU_AFFINITY_HIGH_PERFORMANCE.Or you can only choose one "
                 "of {CPU_AFFINITY_HIGH_PERFORMANCE, CPU_AFFINITY_LOW_POWER, GPU}.\n"
                 "4. -p [algorithmMapPath]: The algorithm configration path.\n"
                 "5. -l [loopTime]: The running loopTimes.\n"
                 "6. -w [warmUp]: WarmUp times. The default value is 10.\n"
                 "Example: ./benchmark -m /local/models/resnet50_f16.bolt"
              << std::endl;
}

void parse_options(int argc, char *argv[])
{
    std::cout << "\nPlease enter this command './benchmark --help' to get more usage "
                 "information.\n";
    std::vector<std::string> lineArgs(argv, argv + argc);
    for (std::string arg : lineArgs) {
        if (arg == "--help" || arg == "-help" || arg == "--h" || arg == "-h") {
            print_benchmark_usage();
            exit(-1);
        }
    }

    int option;
    const char *optionstring = "m:i:a:p:l:w:";
    while ((option = getopt(argc, argv, optionstring)) != -1) {
        switch (option) {
            case 'm':
                std::cout << "option is -m <boltModelPath>, value is: " << optarg << std::endl;
                modelPath = optarg;
                break;
            case 'i':
                std::cout << "option is -i [inputDataPath], value is: " << optarg << std::endl;
                inputData = std::string(optarg);
                break;
            case 'a':
                std::cout << "option is -a [affinityPolicyName], value is: " << optarg << std::endl;
                affinityPolicyName = optarg;
                break;
            case 'p':
                std::cout << "option is -p [algorithmMapPath], value is: " << optarg << std::endl;
                algorithmMapPath = optarg;
                break;
            case 'l':
                std::cout << "option is -l [loopTime], value is: " << optarg << std::endl;
                loopTime = atoi(optarg);
                break;
            case 'w':
                std::cout << "option is -w [warmUp], value is: " << optarg << std::endl;
                warmUp = atoi(optarg);
                break;
            default:
                std::cout << "Input option gets error, please check the params meticulously.\n";
                print_benchmark_usage();
                exit(-1);
        }
    }
}

std::map<std::string, std::shared_ptr<U8>> create_tensors_from_path(
    std::string dataPath, std::shared_ptr<CNN> pipeline)
{
    std::map<std::string, TensorDesc> inputDescMap = pipeline->get_input_desc();
    std::vector<DataType> sourceDataTypes;
    std::vector<TensorDesc> inputDescs;
    for (auto iter : inputDescMap) {
        TensorDesc curDesc = iter.second;
        sourceDataTypes.push_back(curDesc.dt);
        inputDescs.push_back(curDesc);
    }
    std::vector<Tensor> input;
    if (string_end_with(inputData, ".txt")) {
        input = load_txt(inputData, inputDescs);
    } else {
        input = load_bin(inputData, sourceDataTypes, inputDescs);
    }
    std::map<std::string, std::shared_ptr<U8>> model_tensors_input;
    int index = 0;
    std::cout << "\nInput Information:" << std::endl;
    for (auto iter : inputDescMap) {
        std::cout << "Input Tensor " << iter.first << " " << input[index].string(8) << std::endl;
        model_tensors_input[iter.first] = ((CpuMemory *)input[index].get_memory())->get_shared_ptr();
        index++;
    }
    return model_tensors_input;
}

void print_result(std::map<std::string, std::shared_ptr<Tensor>> outMap)
{
    std::cout << "\nBenchmark Result:" << std::endl;
    for (auto iter : outMap) {
        Tensor result = *(iter.second);
        std::cout << "Output Tensor " << iter.first << " " << result.string(8) << std::endl;
    }
}

std::map<std::string, std::shared_ptr<Tensor>> get_output(
    std::shared_ptr<CNN> pipeline, std::string affinity)
{
    std::map<std::string, std::shared_ptr<Tensor>> outMap = pipeline->get_output();
    if (affinity == "GPU") {
#ifdef _USE_MALI
        for (auto iter : outMap) {
            Tensor result = *(iter.second);
            auto mem = (OclMemory *)result.get_memory();
            mem->get_mapped_ptr();
        }
#else
        UNI_WARNING_LOG("this binary not support GPU, please recompile project with GPU "
                        "compile options\n");
#endif
    }
    return outMap;
}

int main(int argc, char *argv[])
{
    UNI_TIME_INIT
    parse_options(argc, argv);

    // 1: set up the pipeline
    auto pipeline = createPipeline(affinityPolicyName, modelPath, algorithmMapPath);

    // 2: create input data and feed the pipeline with it
    auto model_tensors_input = create_tensors_from_path(inputData, pipeline);

    std::map<std::string, std::shared_ptr<Tensor>> outMap;

    // 3: warm up and run
    for (int i = 0; i < warmUp; i++) {
        pipeline->set_input_by_assign(model_tensors_input);
        pipeline->run();
        outMap = get_output(pipeline, affinityPolicyName);
    }
#ifdef _USE_MALI
    if (strcmp(affinityPolicyName, "GPU") == 0) {
        gcl_finish(OCLContext::getInstance().handle.get());
    }
#endif

    double timeBegin = ut_time_ms();
    for (int i = 0; i < loopTime; i++) {
        pipeline->set_input_by_assign(model_tensors_input);
        pipeline->run();
        outMap = get_output(pipeline, affinityPolicyName);
    }
    double timeEnd = ut_time_ms();
    double totalTime = (timeEnd - timeBegin);

    // 4: process result
    print_result(outMap);

    UNI_TIME_STATISTICS
    UNI_CI_LOG("total_time:%fms(loops=%d)\n", 1.0 * totalTime, loopTime);
    UNI_CI_LOG("avg_time:%fms/data\n", 1.0 * totalTime / loopTime);
    pipeline->saveAlgorithmMapToFile(algorithmMapPath);
    return 0;
}
