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

template <typename T>
void copyArr(T *targetPtr, F32 *sourcePtr, int dataSize)
{
    for (int i = 0; i < dataSize; i++) {
        targetPtr[i] = (T)sourcePtr[i];
    }
}

inline std::shared_ptr<CNN> createPipelineTinyGPT(
    const char *affinityPolicyName, const char *modelPath, const char *algorithmMapPath = "")
{
    ModelSpec ms;
    CHECK_STATUS(deserialize_model_from_file(modelPath, &ms));
    for (int i = 0; i < ms.num_inputs; i++) {
        std::string curInputName = std::string(ms.input_names[i]);
        if (curInputName == "input_ids" || curInputName == "position_ids") {
            ms.input_dims[i].dims[0] = 11;  // set1
                                            // ms.input_dims[i].dims[0] = 1;  // set2
        } else {
            ms.input_dims[i].dims[1] = 0;  // set1
                                           // ms.input_dims[i].dims[1] = 16;  // set2
        }
    }
    std::shared_ptr<CNN> pipeline = createPipelinefromMs(affinityPolicyName, &ms, algorithmMapPath);
    CHECK_STATUS(mt_destroy_model(&ms));
    return pipeline;
}

std::map<std::string, std::shared_ptr<U8>> create_tensors_for_tinyGPT(std::shared_ptr<CNN> pipeline)
{
    std::map<std::string, std::shared_ptr<Tensor>> inMap = pipeline->get_input();
    std::map<std::string, TensorDesc> inputDescsMap;
    std::vector<Tensor> input;

    int words_size = 11;

    //    float words_arr[11] = {50257, 75, 82, 84, 85, 82, 88, 64, 75, 68, 50260};
    //    float words_arr[11] = {15,26,45,94,37,60,5,4,5,6,788};
    //    float words_arr[11] = {38,93,46,990,32,67,44,98,36,47,66,13};
    float words_arr[11] = {67, 89, 59, 486, 785, 43, 53, 16, 5, 47, 233};
    //    float words_arr[11] = {965,789,578,943,632,147,145,235,774,346,9};
    //

    /*
     * initialize the fake data
    float words_arr[words_size];
    for (int i = 0; i < words_size; i++) {
        words_arr[i] = 6;
    }
    */

    TensorDesc inputIdsDesc = (*(inMap["input_ids"])).get_desc();
    inputIdsDesc.dims[0] = words_size;
    Tensor inputIds_tensor = Tensor::alloc_sized<CPUMem>(inputIdsDesc);
    U8 *input_ab_ptr = (U8 *)((CpuMemory *)(inputIds_tensor.get_memory()))->get_ptr();

    float pos_arr[words_size];
    for (int i = 0; i < words_size; i++) {
        pos_arr[i] = (float)i;
    }

    TensorDesc posIdsDesc = inputIdsDesc;
    Tensor posIds_tensor = Tensor::alloc_sized<CPUMem>(posIdsDesc);
    U8 *pos_ab_ptr = (U8 *)((CpuMemory *)(posIds_tensor.get_memory()))->get_ptr();

    if (inputIdsDesc.dt == DT_F32) {
        copyArr<F32>((F32 *)input_ab_ptr, words_arr, words_size);
        F32 *testPtr = (F32 *)input_ab_ptr;
        for (int i = 0; i < words_size; i++) {
            std::cout << testPtr[i] << "/";
        }
        copyArr<F32>((F32 *)pos_ab_ptr, pos_arr, words_size);
    }
#ifdef _USE_FP16
    else if (inputIdsDesc.dt == DT_F16) {
        copyArr<F16>((F16 *)input_ab_ptr, words_arr, words_size);
        F16 *testPtr = (F16 *)input_ab_ptr;
        for (int i = 0; i < words_size; i++) {
            std::cout << testPtr[i] << "/";
        }
        copyArr<F16>((F16 *)pos_ab_ptr, pos_arr, words_size);
    }
#endif

    TensorDesc share_desc;
    for (auto iter : inMap) {
        std::string tmpName = iter.first;
        if (tmpName != "input_ids" && tmpName != "position_ids") {
            share_desc = (*(inMap[tmpName])).get_desc();
            break;
        }
    }

    int share_len = 0;
    share_desc.dims[1] = share_len;
    Tensor share_tensor = Tensor::alloc_sized<CPUMem>(share_desc);
    /*
    U8* st_ab_ptr = (U8*)((CpuMemory*)(share_tensor.get_memory()))->get_ptr();
    F32* st_ad_ptr = (F32*)st_ab_ptr;
    for (int i = 0; i < (int)(tensorNumElements(share_desc)); i++) {
        st_ad_ptr[i] = 1.0;
    }
    */

    std::map<std::string, std::shared_ptr<U8>> model_tensors_input;
    for (auto iter : inMap) {
        std::string curName = iter.first;
        if (curName == "input_ids") {
            inputDescsMap[curName] = inputIds_tensor.get_desc();
            model_tensors_input[curName] =
                ((CpuMemory *)inputIds_tensor.get_memory())->get_shared_ptr();
        } else if (curName == "position_ids") {
            inputDescsMap[curName] = posIds_tensor.get_desc();
            model_tensors_input[curName] =
                ((CpuMemory *)posIds_tensor.get_memory())->get_shared_ptr();
        } else {
            inputDescsMap[curName] = share_tensor.get_desc();
            model_tensors_input[curName] =
                ((CpuMemory *)share_tensor.get_memory())->get_shared_ptr();
        }
    }
    return model_tensors_input;
}

void print_result(std::map<std::string, std::shared_ptr<Tensor>> outMap)
{
    std::cout << "\n\nBenchmark Result:\n";
    int outputIndex = 0;
    for (auto iter : outMap) {
        Tensor result = *(iter.second);
        std::cout << "Output Tensor" << outputIndex++ << " : " << iter.first << "\n"
                  << result.string(8) << "\n\n";
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
    auto pipeline = createPipelineTinyGPT(affinityPolicyName, modelPath, algorithmMapPath);

    // 2: create input data and feed the pipeline with it
    //  auto model_tensors_input = create_tensors_from_path(inputData, pipeline);
    auto model_tensors_input = create_tensors_for_tinyGPT(pipeline);
    std::map<std::string, std::shared_ptr<Tensor>> outMap;
    // 3: warm up and run
    for (int i = 0; i < warmUp; i++) {
        pipeline->set_input_by_assign(model_tensors_input);
        pipeline->run();
        outMap = get_output(pipeline, affinityPolicyName);
    }

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
