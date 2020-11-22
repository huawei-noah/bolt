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
#include "types.h"
#include "error.h"

#ifdef _USE_FP16
inline std::vector<std::string> buildModelsNameArray(std::string path, std::string postfix)
{
    struct dirent *dirTp;
    DIR *handle = opendir(path.c_str());
    std::vector<std::string> names;
    if (handle != NULL) {
        while ((dirTp = readdir(handle)) != NULL) {
            std::string modelName = dirTp->d_name;
            U32 len = modelName.size();
            U32 postfix_len = postfix.size();
            if (len > postfix_len) {
                if (modelName.substr(len - postfix_len) == postfix) {
                    modelName = path + modelName;
                    names.push_back(modelName);
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

inline void runBoltModel(CI8 *modelPath, CI8 *algoPath, std::vector<std::string> *kernelNames)
{
    if (!strstr(modelPath, "f16.bolt")) {
        UNI_ERROR_LOG("Bolt gpu only support F16(_f16.bolt) now\n");
        UNI_ERROR_LOG("Ensure your model is xxxx_f16.bolt\n");
        exit(1);
    }

    UNI_INFO_LOG("Building algofile and used kernelNames for %s\n", modelPath);
    auto cnn = createPipeline("GPU", modelPath, algoPath);
    std::vector<TensorDesc> inputDescs = cnn->get_model_input_tensor_descs();
    U8 **input_ptrs = new U8 *[inputDescs.size()];
    for (U32 i = 0; i < inputDescs.size(); i++) {
        U32 size = tensorNumBytes(inputDescs[i]);
        input_ptrs[i] = new U8[size];
    }

    std::vector<std::string> inputNames = cnn->get_model_input_tensor_names();
    for (U32 i = 0; i < inputNames.size(); i++) {
        cnn->copy_to_named_input(inputNames[i], input_ptrs[i]);
    }

    std::map<std::string, std::shared_ptr<Tensor>> outMap;
    cnn->run();
    outMap = cnn->get_outputs();
    cnn->saveAlgorithmMapToText(algoPath);
    GCLHandle_t handle = OCLContext::getInstance().handle.get();
    for (auto p : handle->kernelMap) {
        std::string device_name = handle->deviceName;
        std::string kernelName = p.first;
        kernelName.erase(0, device_name.size() + 1);
        if (find((*kernelNames).begin(), (*kernelNames).end(), kernelName) == (*kernelNames).end()) {
            (*kernelNames).push_back(kernelName);
        }
    }
    for (auto p : handle->programMap) {
        std::string kernelName = p.first;
        if (find((*kernelNames).begin(), (*kernelNames).end(), kernelName) == (*kernelNames).end()) {
            (*kernelNames).push_back(kernelName);
        }
    }

    for (U32 i = 0; i < inputDescs.size(); i++) {
        delete[] input_ptrs[i];
    }
    delete[] input_ptrs;
    CHECK_STATUS(gcl_finish(handle));
}

inline void buildKernelBinFiles(
    std::vector<std::string> kernelNames, std::string includePath, std::string cppPath)
{
    GCLHandle_t handle;
    CHECK_STATUS(gcl_create_handle(&handle));
    std::string device_name = handle->deviceName;
    std::string device_name_up = device_name;
    std::transform(device_name_up.begin(), device_name_up.end(), device_name_up.begin(), ::toupper);

    std::string inline_kernel_bin_head;
    std::string inline_kernel_bin_head_name;
    inline_kernel_bin_head_name = "inline_" + device_name + ".h";
    inline_kernel_bin_head = "#ifndef _INLINE_" + device_name_up + "_H\n";
    inline_kernel_bin_head += "#define _INLINE_" + device_name_up + "_H\n";

    std::string device_map_head;
    std::string device_map_head_name;
    device_map_head_name = device_name + "_map.h";
    device_map_head = "#ifndef " + device_name_up + "_MAP_H\n";
    device_map_head += "#define " + device_name_up + "_MAP_H\n";
    device_map_head += "extern \"C\" {\n";
    device_map_head += "    gcl_kernel_binmap* create_" + device_name + "_kernelbin_map();\n";
    device_map_head += "}\n";
    device_map_head += "#endif";
    write_to_file(device_map_head, includePath, device_map_head_name);

    std::string device_map;
    std::string device_map_name;
    device_map_name = device_name + "_map.cpp";
    device_map = "#include \"gcl_kernel_binmap.h\"\n";
    device_map += "#include\"" + device_map_head_name + "\"\n";
    device_map += "#include\"" + inline_kernel_bin_head_name + "\"\n";
    device_map += "class " + device_name + " : public gcl_kernel_binmap {\n";
    device_map += "public:\n";
    device_map += "    " + device_name + "() {\n";
    device_map += "        loadKernelBin();\n";
    device_map += "    }\n";
    device_map += "    void loadKernelBin();\n";
    device_map += "};\n";
    device_map += "void " + device_name + "::loadKernelBin() {\n";

    std::string device_kernel_bin;
    std::string device_kernel_bin_name;
    device_kernel_bin_name = device_name + "_kernel_bin.cpp";
    device_kernel_bin = "#include\"" + inline_kernel_bin_head_name + "\"\n";

    for (auto p : kernelNames) {
        Kernel kernel;
        U8 *binary;
        U32 len;
        CHECK_STATUS(gcl_create_kernel(handle, p.c_str(), &kernel));
        Program program = handle->programMap[p];
        CHECK_STATUS(gcl_get_program_info(program, &binary, &len));
        std::string func = device_name + "_" + p;
        inline_kernel_bin_head += "extern const unsigned int " + func + "_len;\n";
        inline_kernel_bin_head += "extern const unsigned char " + func + "[];\n";
        device_map += "    put(\"" + func + "\", " + "{" + func + ", " + func + "_len});\n";
        device_kernel_bin += "const unsigned int " + func + "_len = " + std::to_string(len) + ";\n";
        device_kernel_bin += "const unsigned char " + func + "[] = " + "{";
        for (U32 i = 0; i < len; i++) {
            char tempstr[4];
            if (i % 20 == 0) {
                device_kernel_bin += "\n";
            }
            sprintf(tempstr, "0x%02x", binary[i]);
            device_kernel_bin += std::string(tempstr);
            if (i != len - 1) {
                device_kernel_bin += ", ";
            } else {
                device_kernel_bin += "};\n";
            }
        }
        CHECK_STATUS(release_kernel(kernel));
    }
    inline_kernel_bin_head += "#endif";
    device_map += "}\n";
    device_map += "gcl_kernel_binmap* create_" + device_name + "_kernelbin_map(){\n";
    device_map += "    " + device_name + "* kernelbin = new " + device_name + "();\n";
    device_map += "    return (gcl_kernel_binmap*) kernelbin;\n";
    device_map += "}";
    write_to_file(inline_kernel_bin_head, cppPath, inline_kernel_bin_head_name);
    write_to_file(device_map, cppPath, device_map_name);
    write_to_file(device_kernel_bin, cppPath, device_kernel_bin_name);
    gcl_destroy_handle(handle);
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

    std::vector<std::string> modelsNameArray;
    modelsNameArray = buildModelsNameArray(modelsPath, ".bolt");
    std::vector<std::string> kernelNames;
    for (auto name : modelsNameArray) {
        runBoltModel(name.c_str(), algoPath.c_str(), &kernelNames);
    }

    buildKernelBinFiles(kernelNames, includePath, cppPath);
#endif
    return 0;
}
