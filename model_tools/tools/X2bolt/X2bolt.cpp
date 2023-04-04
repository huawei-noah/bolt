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
#include "online_conversion.h"
#include "model_print.h"
#include "model_common.h"
#include <iostream>
#include <algorithm>
#include "file.h"

void print_X2bolt_usage()
{
    std::cout << "X2bolt(version:" << sg_boltVersion
              << ") usage: (<> must be filled in with exact value; [] is optional.)\n"
                 "Parameter description:\n"
                 "1. -d <modelDirectory>: The directory where your model is stored.\n"
                 "2. -m <modelFileName>: The name of your model file without file suffix. "
                 "Tips: If your model is trained from caffe, please ensure the model file name of "
                 "prototxt and caffemodel are the same, otherwise error occurs.\n"
                 "3. -i [inferencePrecision]: The inference precision. Currently, you can only "
                 "choose one of {FP32, FP16, PTQ, BNN}. PTQ produces the input for "
                 "int8 post_training_quantization tool. BNN supports 1-bit computation of convolution and "
                 "FP16 computation of other operators on ARMv8.2 machines. \n"
                 "4. -r [removeOperatorNum]: The number of preprocession operator in onnx model. default: 0.\n"
                 "5. -t : Generate training model for on-device finetuning.\n"
                 "6. -I : To modify input names of the model. You need to list all input names seperated with ','.\n"
                 "7. -O : To modify output names of the model. You need to list all output names seperated with ','.\n"
                 "8. -v : X2bolt version information.\n"
                 "9. -V : Bolt Model detail information.\n"
                 "10. -B : Bolt Model binary information.\n"
                 "11. -h : help information.\n"
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
    std::string storagePath = ".";
    std::string modelFileName;
    std::string inferPrecision = "FP32";
    I32 removeProcessOpsNum = 0;
    bool printModel = false;
    bool printBinaryModel = false;
    bool trainMode = false;
    std::string modifiedInputs = "";
    std::string modifiedOutputs = "";

    int option;
    const char *optionstring = "d:m:i:r:VBvtI:O:";
    while ((option = getopt(argc, argv, optionstring)) != -1) {
        switch (option) {
            case 'd':
                storagePath = optarg;
                std::cout << "option is -d <modelDirectory>, value is: " << storagePath << std::endl;
                break;
            case 'm':
                modelFileName = optarg;
                std::cout << "option is -m <modelFileName>, value is: " << modelFileName
                          << std::endl;
                break;
            case 'i':
                inferPrecision = optarg;
                std::cout << "option is -i <inferencePrecision>, value is: " << inferPrecision
                          << std::endl;
                break;
            case 'r':
                removeProcessOpsNum = atoi(optarg);
                std::cout << "option is -r [removeOperatorNum], value is: " << removeProcessOpsNum
                          << std::endl;
                break;
            case 'V':
                printModel = true;
                break;
            case 'b':
                printBinaryModel = true;
                break;
            case 't':
                trainMode = true;
                break;
            case 'I':
                modifiedInputs = optarg;
                break;
            case 'O':
                modifiedOutputs = optarg;
                break;
            default:
                std::cerr << "Input option gets error. Please check the params meticulously.\n"
                          << std::endl;
                print_X2bolt_usage();
                return 1;
        }
    }
    if (modelFileName == "") {
        std::cerr << "Please use -m <modelFileName> option to give an valid model file name "
                     "without file suffix."
                  << std::endl;
        return 1;
    }
    transform(inferPrecision.begin(), inferPrecision.end(), inferPrecision.begin(), toupper);

    void *onlineModel = OnlineModelConversion(storagePath.c_str(), modelFileName.c_str(),
        inferPrecision.c_str(), removeProcessOpsNum, trainMode);
    ModelSpec *ms = (ModelSpec *)onlineModel;

    std::string modelStorePath = storagePath + "/" + modelFileName;
    if (trainMode) {
        modelStorePath += std::string("_train.bolt");
    } else if (inferPrecision.compare(std::string("PTQ")) == 0) {
        modelStorePath += std::string("_ptq_input.bolt");
    } else if (inferPrecision.compare(std::string("BNN")) == 0) {
        modelStorePath += std::string("_f16_b.bolt");
    } else if (inferPrecision.compare(std::string("FP16")) == 0) {
        modelStorePath += std::string("_f16.bolt");
    } else if (inferPrecision.compare(std::string("FP32")) == 0) {
        modelStorePath += std::string("_f32.bolt");
    } else {
        std::cerr << "Unknown converter data precision: " << inferPrecision << std::endl;
        return 1;
    }

    // modified input names and output names
    modify_ms_inputs_and_outputs(ms, modifiedInputs, modifiedOutputs);

    UNI_INFO_LOG("Write bolt model to %s.\n", modelStorePath.c_str());
    CHECK_STATUS(serialize_model_to_file(ms, modelStorePath.c_str()));
    OnlineModelReclaim(onlineModel);
    if (printModel) {
        ModelSpec resultMs;
        CHECK_STATUS(deserialize_model_from_file(modelStorePath.c_str(), &resultMs, ms->dt));
        print_header(resultMs);
        print_operator_tensor_relationship(resultMs);
        print_weights(resultMs);
        CHECK_STATUS(mt_destroy_model(&resultMs));
    }
    if (printBinaryModel) {
        U8 *binary = NULL;
        size_t binary_len = 0;
        CHECK_STATUS(load_binary(modelStorePath.c_str(), (void **)&binary, &binary_len));
        std::string line = "const unsigned int model_len = " + std::to_string(binary_len) + ";\n";
        line += "const char model[] = " + hex_array(binary, binary_len) + ";\n";
        if (binary != NULL) {
            free(binary);
        }
        std::cout << line << std::endl;
    }

    std::cout << "Model Conversion Succeeded!" << std::endl;
    // UNI_MEM_STATISTICS();
    return 0;
}
