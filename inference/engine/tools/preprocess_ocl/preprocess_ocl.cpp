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
#include <string.h>
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
    if (!strstr(modelPath, "f16.bolt")) {
        UNI_ERROR_LOG("Bolt gpu only support F16(_f16.bolt) now\n");
        UNI_ERROR_LOG("Ensure your model is xxxx_f16.bolt\n");
        exit(1);
    }

    UNI_INFO_LOG("Building algofile and used kernelInfos for %s\n", modelPath);

    ModelHandle model_address = model_address = CreateModel(modelPath, GPU, algoPath);
    int num_input = GetNumInputsFromModel(model_address);
    int *n = (int *)malloc(sizeof(int) * num_input);
    int *c = (int *)malloc(sizeof(int) * num_input);
    int *h = (int *)malloc(sizeof(int) * num_input);
    int *w = (int *)malloc(sizeof(int) * num_input);
    char **name = (char **)malloc(sizeof(char *) * num_input);
    for (int i = 0; i < num_input; i++) {
        name[i] = (char *)malloc(sizeof(char) * 1024);
    }
    DATA_TYPE *dt_input = (DATA_TYPE *)malloc(sizeof(DATA_TYPE) * num_input);
    DATA_FORMAT *df_input = (DATA_FORMAT *)malloc(sizeof(DATA_FORMAT) * num_input);
    GetInputDataInfoFromModel(model_address, num_input, name, n, c, h, w, dt_input, df_input);
    unsigned char **input_ptr = (unsigned char **)malloc(sizeof(unsigned char *) * num_input);
    for (int i = 0; i < num_input; i++) {
        int length = n[i] * c[i] * h[i] * w[i];
        F16 *ptr = (F16 *)malloc(sizeof(F16) * length);
        for (int i = 0; i < length; i++) {
            ptr[i] = 1;
        }
        input_ptr[i] = (unsigned char *)ptr;
        break;
    }
    PrepareModel(model_address, num_input, (const char **)name, n, c, h, w, dt_input, df_input);
    ResultHandle model_result = AllocAllResultHandle(model_address);
    int model_result_num = GetNumOutputsFromResultHandle(model_result);
    int *output_n = (int *)malloc(sizeof(int) * model_result_num);
    int *output_c = (int *)malloc(sizeof(int) * model_result_num);
    int *output_h = (int *)malloc(sizeof(int) * model_result_num);
    int *output_w = (int *)malloc(sizeof(int) * model_result_num);
    char **outputNames = (char **)malloc(sizeof(char *) * model_result_num);
    for (int i = 0; i < model_result_num; i++) {
        outputNames[i] = (char *)malloc(sizeof(char) * 1024);
    }
    DATA_TYPE *dt_output = (DATA_TYPE *)malloc(sizeof(DATA_TYPE) * model_result_num);
    DATA_FORMAT *df_output = (DATA_FORMAT *)malloc(sizeof(DATA_FORMAT) * model_result_num);
    GetOutputDataInfoFromResultHandle(model_result, model_result_num, outputNames, output_n,
        output_c, output_h, output_w, dt_output, df_output);
    RunModel(model_address, model_result, num_input, (const char **)name, (void **)input_ptr);

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
    FreeResultHandle(model_result);
    DestroyModel(model_address);

    free(n);
    free(c);
    free(h);
    free(w);
    free(dt_input);
    free(df_input);
    for (int i = 0; i < num_input; i++) {
        free(name[i]);
        free(input_ptr[i]);
    }
    free(name);
    free(input_ptr);
    free(output_n);
    free(output_c);
    free(output_h);
    free(output_w);
    free(dt_output);
    free(df_output);
    for (int i = 0; i < model_result_num; i++) {
        free(outputNames[i]);
    }
    free(outputNames);
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

    for (auto p : kernelInfos) {
        std::string kernelName = p.first;
        std::vector<U8> binaryInfo = p.second;
        U32 len = binaryInfo.size();

        std::string func = device_name + "_" + kernelName;
        device_map += "const unsigned int " + func + "_len = " + std::to_string(len) + ";\n";
        device_map += "const unsigned char " + func + "[] = " + "{";
        for (U32 i = 0; i < len; i++) {
            I8 tempstr[4];
            if (i % 20 == 0) {
                device_map += "\n";
            }
            sprintf(tempstr, "0x%02x", binaryInfo[i]);
            device_map += std::string(tempstr);
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
            I8 tempstr[4];
            if (i % 20 == 0) {
                device_map += "\n";
            }
            sprintf(tempstr, "0x%02x", bytes[i]);
            device_map += std::string(tempstr);
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
    std::string modelsPath = (CI8 *)argv[1];
    lastFlag = modelsPath[modelsPath.length() - 1];
    if (strcmp(&lastFlag, "/") != 0) {
        modelsPath += "/";
    }

    std::string algoPath = (CI8 *)argv[2];
    lastFlag = algoPath[algoPath.length() - 1];
    if (strcmp(&lastFlag, "/") != 0) {
        algoPath += "/";
    }

    std::string includePath = (CI8 *)argv[3];
    lastFlag = includePath[includePath.length() - 1];
    if (strcmp(&lastFlag, "/") != 0) {
        includePath += "/";
    }

    std::string cppPath = (CI8 *)argv[4];
    lastFlag = cppPath[cppPath.length() - 1];
    if (strcmp(&lastFlag, "/") != 0) {
        cppPath += "/";
    }

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
