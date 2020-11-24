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
#include "model_serialize_deserialize.hpp"

void print_X2bolt_usage()
{
    std::cout << "X2bolt(version:" << sg_boltVersion
              << ") "
                 "converter usage: (<> must be filled in with exact value; [] is "
                 "optional)\n"
                 "./X2bolt -d <modelDirectory> -m <modelName> -i <inferencePrecision> -v -s -h "
                 "-r [removeOperatorNum]\n"
                 "Parameter description:\n"
                 "1. -d <modelDirectory>: The directory where your model is stored.\n"
                 "2. -m <modelName>: The name of your model. "
                 "Tips: If your model trained from caffe, please ensure the names of prototxt and "
                 "caffemodel are the same, otherwise error occurs.\n"
                 "3. -i <inferencePrecision>: The inference precision. Currently, you can only "
                 "choose one of "
                 "{FP32, FP16, PTQ}. PTQ produces the input for post_training_quantization tool.\n"
                 "4. -r [removeOperatorNum]: The number of preprocession operator in onnx model."
                 "The default value is 0.\n"
                 "5. -v : X2bolt version information.\n"
                 "6. -s : Bolt Model detail information.\n"
                 "7. -h : X2bolt help information.\n"
                 "Example: ./X2bolt -d /local/models/ -m resnet50 -i FP16\n"
                 "If model conversion is successful, you can find the resnet50_f16.bolt file in "
                 "/local/models. Otherwise, you should check the usage Intro above.\n"
              << std::endl;
}

void print_version()
{
    std::cout << "Current mdoel converter version is : " << sg_boltVersion << std::endl;
}

int main(int argc, char *argv[])
{
    std::cout << "\nEnter './X2bolt --help' to get more usage information.\nEnter './X2bolt "
                 "--version' to get the version.\n\n";
    std::vector<std::string> lineArgs(argv, argv + argc);
    for (std::string arg : lineArgs) {
        if (arg == "--help" || arg == "-help" || arg == "--h" || arg == "-h") {
            print_X2bolt_usage();
            return -1;
        } else if (arg == "--version" || arg == "-version" || arg == "--v" || arg == "-v") {
            print_version();
            return -1;
        }
    }
    CHECK_REQUIREMENT(argc >= 4);
    char *storagePath = (char *)" ";
    char *modelName = (char *)" ";
    char *inferPrecision = (char *)" ";
    I32 removeProcessOpsNum = 0;
    bool show_model_info = false;

    int option;
    const char *optionstring = "d:m:i:r:s";
    while ((option = getopt(argc, argv, optionstring)) != -1) {
        switch (option) {
            case 'd':
                std::cout << "option is -d <modelDirectory>, value is: " << optarg << std::endl;
                storagePath = optarg;
                break;
            case 'm':
                std::cout << "option is -m <modelName>, value is: " << optarg << std::endl;
                modelName = optarg;
                break;
            case 'i':
                std::cout << "option is -i <inferencePrecision>, value is: " << optarg << std::endl;
                inferPrecision = optarg;
                break;
            case 'r':
                std::cout << "option is -r [removeOperatorNum], value is: " << optarg << std::endl;
                removeProcessOpsNum = atoi(optarg);
                break;
            case 's':
                show_model_info = true;
                break;
            default:
                std::cerr << "Input option gets error. Please check the params meticulously."
                          << std::endl;
                print_X2bolt_usage();
                return -1;
        }
    }

    void *onlineModel = OnlineModelConversion(
        storagePath, modelName, inferPrecision, removeProcessOpsNum);
    ModelSpec *ms = (ModelSpec *)onlineModel;

    std::string modelStorePath = std::string(storagePath) + "/" + std::string(modelName);
    if (0) {
#if _USE_FP32
    } else if (std::string(inferPrecision).compare(std::string("PTQ")) == 0) {
        modelStorePath += std::string("_ptq_input.bolt");
#endif
#if _USE_FP16
    } else if (std::string(inferPrecision).compare(std::string("FP16")) == 0) {
        modelStorePath += std::string("_f16.bolt");
#endif
#if _USE_FP32
    } else if (std::string(inferPrecision).compare(std::string("FP32")) == 0) {
        modelStorePath += std::string("_f32.bolt");
#endif
    } else {
        std::cerr << "NOT SUPPORT THIS PRECISION " << inferPrecision << std::endl;
        return -1;
    }
    CHECK_STATUS(serialize_model_to_file(ms, modelStorePath.c_str()));
    OnlineModelReclaim(onlineModel);
    if (show_model_info) {
        ModelSpec resultMs;
        CHECK_STATUS(deserialize_model_from_file(modelStorePath.c_str(), &resultMs));
        print_header(resultMs);
        print_operator_tensor_relationship(resultMs);
        print_weights(resultMs);
        CHECK_STATUS(mt_destroy_model(&resultMs));
    }
    std::cout << "Model Conversion Succeeded!" << std::endl;
    return 0;
}
