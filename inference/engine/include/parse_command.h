// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _H_PARSE_COMMAND
#define _H_PARSE_COMMAND
#include <getopt.h>
#include <iostream>
#include <utility>

#include "error.h"

#ifdef _USE_FP16

inline U32 getBinFileSize(CI8 *dataPath, CI8 *dataName)
{
    std::string filePath = dataPath;
    CI8 lastFlag = filePath[filePath.length() - 1];
    if (strcmp(&lastFlag, "/") != 0) {
        filePath += "/";
    }
    std::string fileName = dataName;
    fileName = filePath + fileName;
    FILE *file = fopen(fileName.c_str(), "rb");
    if (file == NULL) {
        UNI_WARNING_LOG("can not get %s file size.\n", fileName.c_str());
        return 0;
    }
    fseek(file, 0, SEEK_END);
    U32 size = (U32)ftell(file);
    fseek(file, 0, SEEK_SET);
    fclose(file);
    return size;
}

inline void writeF16ToF32Bin(F16 *data, U32 num, CI8 *dataPath, CI8 *dataName)
{
    std::string filePath = dataPath;
    CI8 lastFlag = filePath[filePath.length() - 1];
    if (strcmp(&lastFlag, "/") != 0) {
        filePath += "/";
    }
    std::string fileName = dataName;
    fileName = filePath + fileName;
    FILE *outfile = fopen(fileName.c_str(), "wb");
    if (outfile == NULL) {
        UNI_WARNING_LOG("can not write %s.\n", fileName.c_str());
        return;
    }
    F32 *dataTran = new F32[num];
    for (U32 i = 0; i < num; i++) {
        dataTran[i] = (F32)data[i];
    }
    fwrite(dataTran, sizeof(float), num, outfile);
    fclose(outfile);
    delete[] dataTran;
}

inline void readF32BinToF16(F16 *data, U32 num, CI8 *dataPath, CI8 *dataName)
{
    std::string filePath = dataPath;
    CI8 lastFlag = filePath[filePath.length() - 1];
    if (strcmp(&lastFlag, "/") != 0) {
        filePath += "/";
    }
    std::string fileName = dataName;
    fileName = filePath + fileName;
    FILE *infile = fopen(fileName.c_str(), "rb");
    if (infile == NULL) {
        UNI_WARNING_LOG("can not read %s.\n", fileName.c_str());
        return;
    }
    F32 *dataTran = new F32[num];
    fread(dataTran, sizeof(float), num, infile);
    for (U32 i = 0; i < num; i++) {
        data[i] = (F16)dataTran[i];
    }
    fclose(infile);
    delete[] dataTran;
}
#endif

const struct option long_options[]{
    {"model", 1, nullptr, 'm'},
    {"inputPath", 1, nullptr, 'i'},
    {"archInfo", 1, nullptr, 'a'},
    {"algoPath", 1, nullptr, 'p'},
    {"imageFormat", 1, nullptr, 'f'},
    {"scaleValue", 1, nullptr, 's'},
    {"topK", 1, nullptr, 't'},
    {"correctLable", 1, nullptr, 'c'},
    {"loopTime", 1, nullptr, 'l'},
    {"subNetworkName", 1, nullptr, 'S'},
    {"help", 1, nullptr, 'h'},
    {"readInputBinName", 1, nullptr, 1},
    {"writeOutputBinName", 1, nullptr, 2},
};

const char optstring[] = "m:i:a:p:f:s:t:c:l:S:h:";

typedef struct {
    std::pair<char *, bool> model;
    std::pair<char *, bool> inputPath;
    std::pair<char *, bool> archInfo;
    std::pair<char *, bool> algoPath;
    std::pair<ImageFormat, bool> imageFormat;
    std::pair<float, bool> scaleValue;
    std::pair<int, bool> topK;
    std::pair<int, bool> correctLable;
    std::pair<int, bool> loopTime;
    std::pair<char *, bool> subNetworkName;
    std::pair<char *, bool> readInputBinName;
    std::pair<char *, bool> writeOutputBinName;
} ParseRes;
typedef ParseRes *ParseRes_t;

inline void init_parse_res(ParseRes_t parse_res)
{
    parse_res->model.second = false;
    parse_res->inputPath.second = false;
    parse_res->archInfo.second = false;
    parse_res->algoPath.second = false;
    parse_res->imageFormat.second = false;
    parse_res->scaleValue.second = false;
    parse_res->topK.second = false;
    parse_res->correctLable.second = false;
    parse_res->loopTime.second = false;
    parse_res->subNetworkName.second = false;
    parse_res->readInputBinName.second = false;
    parse_res->writeOutputBinName.second = false;
}

inline void help_examples()
{
    std::cout << "<<<<<<<<<<<<<<<<<<<< Parameters specification for examples >>>>>>>>>>>>>>>>>>>>"
              << std::endl;
    std::cout << "--model         "
              << " or -m:  "
              << "--required-- "
              << "specific bolt model" << std::endl;
    std::cout << "--archInfo      "
              << " or -a:  "
              << "--optional-- "
              << "specific running arch: CPU_AFFINITY_HIGH_PERFORMANCE/CPU_AFFINITY_LOW_POWER/GPU,"
              << " the defaule value is CPU_AFFINITY_HIGH_PERFORMANCE" << std::endl;
    std::cout << "--inputPath     "
              << " or -i:  "
              << "--optional-- "
              << "specific file path to read input data" << std::endl;
    std::cout << "--algoPath      "
              << " or -p:  "
              << "--optional-- "
              << "specific file path to read or write algorithm auto tunning result" << std::endl;
    std::cout << "--imageFormat   "
              << " or -f:  "
              << "--optional-- "
              << "specific imageFormat if the input is an image: "
                 "BGR/RGB/RGB_SC/BGR_SC_RAW/BGR_SC_R,"
              << " the default value is RGB" << std::endl;
    std::cout << "--scaleValue    "
              << " or -s:  "
              << "--optional-- "
              << "specific scaleValue for image classification, the default value is 1" << std::endl;
    std::cout << "--topK          "
              << " or -t:  "
              << "--optional-- "
              << "specific topK value for image classification, the default value is 5" << std::endl;
    std::cout << "--correctLable  "
              << " or -c:  "
              << "--optional-- "
              << "specific correctLable for image classification, the deault value is -1"
              << std::endl;
    std::cout << "--loopTime      "
              << " or -l:  "
              << "--optional-- "
              << "specific loopTime for running set_input + run + get_output, the deault value is 1"
              << std::endl;
    std::cout << "--subNetworkName"
              << " or -S:   "
              << "--optional-- "
              << "specific subNetworkName for:" << std::endl;
    std::cout << "     asr convolution transformer: encoder/prediction_net_ln/joint_net, the "
                 "default value is encoder"
              << std::endl;
    std::cout << "     nmt_tsc                    : encoder/decoder" << std::endl;
    std::cout << "     tts                        : "
                 "encoder_decoder/postnet/melgan_vocoder/tinybert, the default value is "
                 "encoder_decoder"
              << std::endl;
    std::cout << "--readInputBinName              "
              << "--optional-- "
              << "specific read input as binary, the binary should be float value with nchw format"
              << std::endl;
    std::cout << "--writeOutputBinName            "
              << "--optional-- "
              << "specific save output as binary, the binary will be float value with nchw format"
              << std::endl;
}

inline void help(std::string name)
{
    if (name == "examples") {
        help_examples();
    }
}

inline void parseCommandLine(int argc, char *argv[], ParseRes_t parse_res, std::string name)
{
    int c = 0;
    int optionIndex;
    ImageFormat imageFormat;
    std::cout << "[PARAMETERS INFO]:" << std::endl;
    if (argc == 1) {
        help(name);
    }
    while ((c = getopt_long(argc, argv, optstring, long_options, &optionIndex)) != -1) {
        switch (c) {
            case 'm':
                parse_res->model.first = optarg;
                parse_res->model.second = true;
                std::cout << "<model>          - " << parse_res->model.first << std::endl;
                break;
            case 'i':
                parse_res->inputPath.first = optarg;
                parse_res->inputPath.second = true;
                std::cout << "<inputPath>      - " << parse_res->inputPath.first << std::endl;
                break;
            case 'a':
                parse_res->archInfo.first = optarg;
                parse_res->archInfo.second = true;
                std::cout << "<archInfo>       - " << parse_res->archInfo.first << std::endl;
                break;
            case 'p':
                parse_res->algoPath.first = optarg;
                parse_res->algoPath.second = true;
                std::cout << "<algoPath>       - " << parse_res->algoPath.first << std::endl;
                break;
            case 'f':
                if (std::string(optarg) == std::string("RGB")) {
                    imageFormat = RGB;
                } else if (std::string(optarg) == std::string("BGR")) {
                    imageFormat = BGR;
                } else if (std::string(optarg) == std::string("RGB_SC")) {
                    imageFormat = RGB_SC;
                } else if (std::string(optarg) == std::string("BGR_SC_RAW")) {
                    imageFormat = BGR_SC_RAW;
                } else if (std::string(optarg) == std::string("RGB_SC_RAW")) {
                    imageFormat = RGB_SC_RAW;
                } else {
                    imageFormat = RGB;
                    std::cout << "Unsupported image format, default to be RGB" << std::endl;
                }
                parse_res->imageFormat.first = imageFormat;
                parse_res->imageFormat.second = true;
                std::cout << "<imageFormat>    - " << optarg << std::endl;
                break;
            case 's':
                parse_res->scaleValue.first = atof(optarg);
                parse_res->scaleValue.second = true;
                std::cout << "<scaleValue>     - " << parse_res->scaleValue.first << std::endl;
                break;
            case 't':
                parse_res->topK.first = atoi(optarg);
                parse_res->topK.second = true;
                std::cout << "<topK>           - " << parse_res->topK.first << std::endl;
                break;
            case 'l':
                parse_res->loopTime.first = atoi(optarg);
                parse_res->loopTime.second = true;
                std::cout << "<loopTime>       - " << parse_res->loopTime.first << std::endl;
                break;
            case 'c':
                parse_res->correctLable.first = atoi(optarg);
                parse_res->correctLable.second = true;
                std::cout << "<correctLable>   - " << parse_res->correctLable.first << std::endl;
                break;
            case 'S':
                parse_res->subNetworkName.first = optarg;
                parse_res->subNetworkName.second = true;
                std::cout << "<subNetworkName> - " << parse_res->subNetworkName.first << std::endl;
                break;
            case 1:
                parse_res->readInputBinName.first = optarg;
                parse_res->readInputBinName.second = true;
                std::cout << "<loadInputBinaryName> - " << parse_res->readInputBinName.first
                          << std::endl;
                break;
            case 2:
                parse_res->writeOutputBinName.first = optarg;
                parse_res->writeOutputBinName.second = true;
                std::cout << "<writeOutputBinaryName> - " << parse_res->writeOutputBinName.first
                          << std::endl;
                break;
            case 'h':
                help(name);
                break;
            default:
                help(name);
                break;
        }
    }
}
#endif
