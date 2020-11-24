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
#include "online_conversion.h"
#include "model_tools.h"
#include "model_quantization.h"

void print_quantization_usage()
{
    std::cout << "post_training_quantization : " 
                 "./post_training_quantization -p <path2Model>\n"
                 "Parameter description:\n"
                 "1. -p <path2Model>: Path to the input model. The suffix should be _ptq_input.bolt.\n"
                 "2. -i [inferencePrecision]: The inference precision. Currently, you can only "
                 "choose one of "
                 "{FP32, FP16, INT8}. Default is INT8.\n"
                 "3. -b [BatchNormFusion]: Whether to fuse convolution or FC with BN. Default is true.\n"
                 "4. -q [quantStorage]: Store model in quantized form. You can choose one of"
                 "{FP16, INT8, MIX}. Default is MIX.\n"
                 "5. -c [clipValue]: To clip the input for gemm if clipValue > 0. The default "
                 "value is 0.\n"
                 "6. -s <scaleFileDirectory>: The directory of the scale file.\n"
                 "7. -V : Verbose mode.\n"
              << std::endl;
}

int main(int argc, char *argv[])
{
    std::cout << "\nEnter './post_training_quantization --help' to get more usage information." << std::endl;
    std::vector<std::string> lineArgs(argv, argv + argc);
    for (std::string arg : lineArgs) {
        if (arg == "--help" || arg == "-help" || arg == "--h" || arg == "-h") {
            print_quantization_usage();
            return -1;
        }
    }
    CHECK_REQUIREMENT(argc >= 2);
    char *modelPath = nullptr;
    char *inferPrecision = (char *)"INT8";
    bool fuseBN = true;
    char *quantStorage = (char *)"NOQUANT";
    F32 clipVal = 0.0;
    char *scaleFile = nullptr;
    bool verbose = false;

    int option;
    const char *optionstring = "p:i:b:q:c:s:V";
    while ((option = getopt(argc, argv, optionstring)) != -1) {
        switch (option) {
            case 'p':
                std::cout << "option is -p <path2Model>, value is: " << optarg << std::endl;
                modelPath = optarg;
                break;
            case 'i':
                std::cout << "option is -i [inferencePrecision], value is: " << optarg << std::endl;
                inferPrecision = optarg;
                break;
            case 'b':
                std::cout << "option is -b [BatchNormFusion], value is: " << optarg << std::endl;
                fuseBN = (std::string(optarg).compare("false") == 0) ? false : true;
                break;
            case 'q':
                std::cout << "option is -q [quantStorage], value is: " << optarg << std::endl;
                quantStorage = optarg;
                break;
            case 'c':
                std::cout << "option is -c [clipValue], value is: " << optarg << std::endl;
                clipVal = atof(optarg);
                break;
            case 's':
                std::cout << "option is -s [scaleFileDirectory], value is: " << optarg << std::endl;
                scaleFile = optarg;
                break;
            case 'V':
                verbose = true;
                break;
            default:
                std::cout << "Input option gets error. Please check the params meticulously."
                          << std::endl;
                print_quantization_usage();
                return -1;
        }
    }
    ModelSpec ms;
    std::string storePath = std::string(modelPath);
    CHECK_STATUS(deserialize_model_from_file(storePath.c_str(), &ms));
    if (ms.dt != DT_F32 || std::string::npos == storePath.find("ptq_input.bolt")) {
        CHECK_STATUS(mt_destroy_model(&ms));
        UNI_ERROR_LOG("Input model does not match. Please produce it with: ./X2bolt -i PTQ\n");
        return 0;
    }
    auto relationNum = ms.num_op_tensor_entries;
    auto relationPtr = ms.op_relationship_entries;
    ms.num_op_tensor_entries = 0;
    ms.op_relationship_entries = nullptr;
#ifdef _DEBUG
    print_ms(ms);
#endif

    DataConvertType converterMode = F32_to_F16;
    if (inferPrecision == std::string("INT8")) {
        converterMode = F32_to_F16;
    } else if (inferPrecision == std::string("HIDDEN")) {
        converterMode = F32_to_F16;
    } else if (inferPrecision == std::string("FP16")) {
        converterMode = F32_to_F16;
    } else if (inferPrecision == std::string("FP32")) {
        converterMode = F32_to_F32;
    } else {
        UNI_ERROR_LOG("Unknown converter data precision : %s", inferPrecision);
    }

    ModelSpecOptimizer msOptimizer;
    msOptimizer.suggest_for_ptq(inferPrecision, fuseBN, clipVal);
    msOptimizer.optimize(&ms);

    ModelSpec *targetMs = new ModelSpec();
    CHECK_STATUS(mt_create_model(targetMs));
    CHECK_STATUS(ms_datatype_converter(&ms, targetMs, converterMode, quantStorage));
    if ("INT8" == std::string(inferPrecision)) {
        targetMs->dt = DT_F16_8Q;
    }

    if (nullptr != scaleFile) {
        add_scale_from_file(targetMs, scaleFile);
    }

    auto suffixPos = storePath.find("ptq_input.bolt");
    storePath.erase(suffixPos, 14);
    if (0) {
#if _USE_INT8
    } else if (std::string(inferPrecision).compare(std::string("INT8")) == 0) {
        storePath += std::string("int8_q.bolt");
#endif
#if _USE_FP16
    } else if (std::string(inferPrecision).compare(std::string("FP16")) == 0) {
        storePath += std::string("f16_q.bolt");
#endif
#if _USE_FP32
    } else if (std::string(inferPrecision).compare(std::string("FP32")) == 0) {
        storePath += std::string("f32_q.bolt");
#endif
    } else {
        std::cerr << "NOT SUPPORT THIS PRECISION " << inferPrecision << std::endl;
        return -1;
    }
    
    CHECK_STATUS(serialize_model_to_file(targetMs, storePath.c_str()));
    CHECK_STATUS(mt_destroy_model(targetMs));
    delete targetMs;
    ms.num_op_tensor_entries = relationNum;
    ms.op_relationship_entries = relationPtr;
    CHECK_STATUS(mt_destroy_model(&ms));

    if (verbose) {
        ModelSpec resultMs;
        CHECK_STATUS(deserialize_model_from_file(storePath.c_str(), &resultMs));
        print_header(resultMs);
        print_operator_tensor_relationship(resultMs);
        print_weights(resultMs);
        CHECK_STATUS(mt_destroy_model(&resultMs));
    }
    std::cout << "Post Training Quantization Succeeded! " << std::endl;
    return 0;
}