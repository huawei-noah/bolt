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
#include "inference.hpp"
#include "tensor.hpp"
#include "result_format.hpp"
#include <stdlib.h>
#include <fstream>
#include <vector>
#include <string>
#include <dirent.h>
#include <algorithm>
#include <fcntl.h>

#include "error.h"
#include "../api/c/bolt.h"

#ifdef _USE_FP16
inline std::vector<std::string> buildFileNamesArray(std::string path, std::string postfix)
{
    struct dirent *dirTp;
    DIR *handle = opendir(path.c_str());
    std::vector<std::string> names;
    if (handle != NULL) {
        while ((dirTp = readdir(handle)) != NULL) {
            std::string fileName = dirTp->d_name;
            U32 len = fileName.size();
            U32 postfix_len = postfix.size();
            if (postfix_len == 0) {
                if (fileName.find("algorithmInfo_") != std::string::npos) {
                    names.push_back(fileName);
                }
            } else if (len > postfix_len) {
                if (fileName.substr(len - postfix_len) == postfix) {
                    names.push_back(fileName);
                }
            }
        }
    } else {
        UNI_ERROR_LOG("opendir %s failed\n", path.c_str());
    }
    closedir(handle);
    return names;
}

inline void write_to_file(std::string str, std::string path, std::string name)
{
    std::string fileName = path + name;
    std::ofstream file(fileName.c_str());
    if (file.is_open()) {
        file << str.c_str();
        file.close();
    } else {
        UNI_ERROR_LOG("fail to write file %s\n", fileName.c_str());
    }
}

inline void runBoltModel(
    CI8 *modelPath, CI8 *algoPath, std::map<std::string, std::vector<U8>> *kernelInfos)
{
    UNI_INFO_LOG("Build gpu kernels and algorithm map file for bolt model(%s)...\n", modelPath);
    if (!strstr(modelPath, "f16.bolt")) {
        UNI_ERROR_LOG("Bolt gpu only support float16 inference, and model file is end with "
                      "_f16.bolt suffix.\n");
        exit(1);
    }

    ModelHandle model = CreateModel(modelPath, GPU, algoPath);
    int input_num = GetNumInputsFromModel(model);
    int *input_n = (int *)malloc(sizeof(int) * input_num);
    int *input_c = (int *)malloc(sizeof(int) * input_num);
    int *input_h = (int *)malloc(sizeof(int) * input_num);
    int *input_w = (int *)malloc(sizeof(int) * input_num);
    DATA_TYPE *input_dt = (DATA_TYPE *)malloc(sizeof(DATA_TYPE) * input_num);
    DATA_FORMAT *input_df = (DATA_FORMAT *)malloc(sizeof(DATA_FORMAT) * input_num);
    char **input_name = (char **)malloc(sizeof(char *) * input_num);
    for (int i = 0; i < input_num; i++) {
        input_name[i] = (char *)malloc(sizeof(char) * 1024);
    }
    GetInputDataInfoFromModel(
        model, input_num, input_name, input_n, input_c, input_h, input_w, input_dt, input_df);
    unsigned char **input_ptr = (unsigned char **)malloc(sizeof(unsigned char *) * input_num);
    for (int i = 0; i < input_num; i++) {
        int length = input_n[i] * input_c[i] * input_h[i] * input_w[i];
        input_ptr[i] = (unsigned char *)malloc(sizeof(F16) * length);
        UNI_INIT(length, DT_F16, 1, input_ptr[i]);
    }
    PrepareModel(model, input_num, (const char **)input_name, input_n, input_c, input_h, input_w,
        input_dt, input_df);
    ResultHandle result = AllocAllResultHandle(model);
    int output_num = GetNumOutputsFromResultHandle(result);
    int *output_n = (int *)malloc(sizeof(int) * output_num);
    int *output_c = (int *)malloc(sizeof(int) * output_num);
    int *output_h = (int *)malloc(sizeof(int) * output_num);
    int *output_w = (int *)malloc(sizeof(int) * output_num);
    DATA_TYPE *output_dt = (DATA_TYPE *)malloc(sizeof(DATA_TYPE) * output_num);
    DATA_FORMAT *output_df = (DATA_FORMAT *)malloc(sizeof(DATA_FORMAT) * output_num);
    char **output_name = (char **)malloc(sizeof(char *) * output_num);
    for (int i = 0; i < output_num; i++) {
        output_name[i] = (char *)malloc(sizeof(char) * 1024);
    }
    GetOutputDataInfoFromResultHandle(result, output_num, output_name, output_n, output_c, output_h,
        output_w, output_dt, output_df);
    RunModel(model, result, input_num, (const char **)input_name, (void **)input_ptr);

    GCLHandle_t handle = OCLContext::getInstance().handle.get();
    for (auto p : handle->kernelMap) {
        std::string device_name = handle->deviceName;
        std::string kernelName = p.first;
        kernelName.erase(0, device_name.size() + 1);
        if ((*kernelInfos).find(kernelName) == (*kernelInfos).end()) {
            Kernel kernel = p.second;
            Program program;
            get_program_info_from_kernel(kernel, &program);

            U8 *binary;
            U32 len;
            CHECK_STATUS(gcl_get_program_info(program, &binary, &len));
            std::vector<U8> binaryInfo;
            for (U32 i = 0; i < len; ++i) {
                binaryInfo.push_back(binary[i]);
            }
            (*kernelInfos)[kernelName] = binaryInfo;
        }
    }

    for (auto p : handle->programMap) {
        std::string kernelName = p.first;
        if ((*kernelInfos).find(kernelName) == (*kernelInfos).end()) {
            Program program = p.second;
            U8 *binary;
            U32 len;
            CHECK_STATUS(gcl_get_program_info(program, &binary, &len));
            std::vector<U8> binaryInfo;
            for (U32 i = 0; i < len; ++i) {
                binaryInfo.push_back(binary[i]);
            }
            (*kernelInfos)[kernelName] = binaryInfo;
        }
    }
    CHECK_STATUS(gcl_finish(handle));
    FreeResultHandle(result);
    DestroyModel(model);

    free(input_n);
    free(input_c);
    free(input_h);
    free(input_w);
    free(input_dt);
    free(input_df);
    for (int i = 0; i < input_num; i++) {
        free(input_name[i]);
        free(input_ptr[i]);
    }
    free(input_name);
    free(input_ptr);
    free(output_n);
    free(output_c);
    free(output_h);
    free(output_w);
    free(output_dt);
    free(output_df);
    for (int i = 0; i < output_num; i++) {
        free(output_name[i]);
    }
    free(output_name);
}

inline void buildFileStream(CI8 *fileName, U8 **bytesPtr, U32 *len)
{
    FILE *file = fopen(fileName, "rb");
    if (file == NULL) {
        UNI_ERROR_LOG("Cannot open file %s\n", fileName);
        return;
    }

    fseek(file, 0, SEEK_END);
    U32 fileLength = ftell(file);
    rewind(file);

    U8 *bytes = (U8 *)malloc(sizeof(U8) * fileLength);
    if (bytes == NULL) {
        UNI_ERROR_LOG("File memory allocate error.\n");
        return;
    }

    U32 result = fread(bytes, 1, fileLength, file);
    if (result != fileLength) {
        UNI_ERROR_LOG("Read file %s error.\n", fileName);
        return;
    }
    fclose(file);
    *len = fileLength;
    *bytesPtr = bytes;
}

inline void buildKernelBinFiles(std::map<std::string, std::vector<U8>> kernelInfos,
    std::string includePath,
    std::string cppPath,
    std::string algoPath,
    std::vector<std::string> algoBinNames)
{
    GCLHandle_t handle = OCLContext::getInstance().handle.get();
    std::string device_name = handle->deviceName;
    std::string device_name_up = device_name;
    std::transform(device_name_up.begin(), device_name_up.end(), device_name_up.begin(), ::toupper);

    std::string device_map_head;
    std::string device_map_head_name;
    device_map_head_name = device_name + "_map.h";
    device_map_head = "#ifndef " + device_name_up + "_MAP_H\n";
    device_map_head += "#define " + device_name_up + "_MAP_H\n";
    device_map_head += "extern \"C\" {\n";
    device_map_head += "    gcl_kernel_binmap* create_" + device_name + "_kernelbin_map();\n";
    device_map_head +=
        "    const char* get_" + device_name + "_algorithm_info(std::string algoName);\n";
    device_map_head += "}\n";
    device_map_head += "#endif";
    write_to_file(device_map_head, includePath, device_map_head_name);

    std::string device_map;
    std::string device_map_name;
    device_map_name = device_name + "_map.cpp";
    device_map += "#include \"gcl_kernel_binmap.h\"\n";
    device_map += "#include \"" + device_map_head_name + "\"\n";

    I8 buffer[16];
    for (auto p : kernelInfos) {
        std::string kernelName = p.first;
        std::vector<U8> binaryInfo = p.second;
        U32 len = binaryInfo.size();

        std::string func = device_name + "_" + kernelName;
        device_map += "const unsigned int " + func + "_len = " + std::to_string(len) + ";\n";
        device_map += "const unsigned char " + func + "[] = " + "{";
        for (U32 i = 0; i < len; i++) {
            if (i % 20 == 0) {
                device_map += "\n";
            }
            sprintf(buffer, "0x%02x", binaryInfo[i]);
            device_map += std::string(buffer);
            if (i != len - 1) {
                device_map += ", ";
            } else {
                device_map += "};\n";
            }
        }
    }

    for (auto algoName : algoBinNames) {
        U8 *bytes;
        U32 len;
        std::string algo = algoPath + algoName;
        buildFileStream(algo.c_str(), &bytes, &len);
        device_map += "const unsigned int " + algoName + "_len = " + std::to_string(len) + ";\n";
        device_map += "const unsigned char " + algoName + "[] = " + "{";
        for (U32 i = 0; i < len; i++) {
            if (i % 20 == 0) {
                device_map += "\n";
            }
            sprintf(buffer, "0x%02x", bytes[i]);
            device_map += std::string(buffer);
            if (i != len - 1) {
                device_map += ", ";
            } else {
                device_map += "};\n";
            }
        }
        free(bytes);
    }
    write_to_file(device_map, cppPath, device_map_name);

    device_map += "class " + device_name + " : public gcl_kernel_binmap {\n";
    device_map += "public:\n";
    device_map += "    " + device_name + "() {\n";
    device_map += "        loadKernelBin();\n";
    device_map += "    }\n";
    device_map += "    void loadKernelBin();\n";
    device_map += "};\n";
    device_map += "void " + device_name + "::loadKernelBin() {\n";
    for (auto p : kernelInfos) {
        std::string kernelName = p.first;
        std::string func = device_name + "_" + kernelName;
        device_map += "    put(\"" + func + "\", " + "{" + func + ", " + func + "_len});\n";
    }
    device_map += "}\n";
    device_map += "gcl_kernel_binmap* create_" + device_name + "_kernelbin_map(){\n";
    device_map += "    " + device_name + "* kernelbin = new " + device_name + "();\n";
    device_map += "    return (gcl_kernel_binmap*) kernelbin;\n";
    device_map += "}\n";
    write_to_file(device_map, cppPath, device_map_name);

    device_map += "const char* get_" + device_name + "_algorithm_info(std::string algoName){\n";
    for (auto algoName : algoBinNames) {
        device_map += "    if (algoName == \"" + algoName + "\") {\n";
        device_map += "        return (const char*)" + algoName + ";\n";
        device_map += "    }\n";
    }
    device_map += "    return nullptr;\n";
    device_map += "}\n";
    write_to_file(device_map, cppPath, device_map_name);
}
#endif

int main(int argc, char *argv[])
{
#ifdef _USE_FP16
    if (argc != 5) {
        UNI_INFO_LOG("Please set your models path, and put your bolt models into it\n");
        UNI_INFO_LOG("Please set your algosPath for save produced algo files, and ensure it is "
                     "clean\n");
        UNI_INFO_LOG("Please set your include Path for save ocl kernelBin headFile, and ensure it "
                     "is clean\n");
        UNI_INFO_LOG("Please set your cpp Path for save ocl kernelBin cpp, and ensure it is "
                     "clean\n");
        UNI_INFO_LOG("For example: ./preprocess_ocl ./boltModels/ ./algoFiles/ ./include/ "
                     "./cpp/\n");
        exit(1);
    }
    I8 lastFlag;
    std::string modelsPath = (CI8 *)argv[1] + std::string("/");
    std::string algoPath = (CI8 *)argv[2] + std::string("/");
    std::string includePath = (CI8 *)argv[3] + std::string("/");
    std::string cppPath = (CI8 *)argv[4] + std::string("/");

    std::vector<std::string> modelNamesArray;
    modelNamesArray = buildFileNamesArray(modelsPath, ".bolt");
    std::map<std::string, std::vector<U8>> kernelInfos;
    for (auto name : modelNamesArray) {
        name = modelsPath + name;
        runBoltModel(name.c_str(), algoPath.c_str(), &kernelInfos);
    }
    std::vector<std::string> algoBinNamesArray;
    algoBinNamesArray = buildFileNamesArray(algoPath, "");
    buildKernelBinFiles(kernelInfos, includePath, cppPath, algoPath, algoBinNamesArray);
#endif
    return 0;
}
