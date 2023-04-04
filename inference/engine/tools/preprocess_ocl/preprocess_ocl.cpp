// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "preprocess_ocl.h"

void save_kernels(std::map<std::string, std::vector<U8>> kernels,
    std::string include_dir,
    std::string cpp_dir,
    std::string algo_dir,
    std::vector<std::string> algo_names)
{
    GCLHandle_t handle = OCLContext::getInstance().handle.get();
    std::string device_name = handle->deviceName;
    std::string device_name_up = upper(device_name);

    std::string device_map_head_name = device_name + "_map.h";
    std::string device_map_head;
    device_map_head = "#ifndef " + device_name_up + "_MAP_H\n";
    device_map_head += "#define " + device_name_up + "_MAP_H\n";
    device_map_head += "#include <string>\n";
    device_map_head += "#include \"gcl_kernel_binmap.h\"\n";
    device_map_head += "extern \"C\" {\n";
    device_map_head += "    gcl_kernel_binmap* create_" + device_name + "_kernelbin_map();\n";
    device_map_head +=
        "    const char* get_" + device_name + "_algorithm_info(std::string algo_name);\n";
    device_map_head += "}\n";
    device_map_head += "#endif";
    CHECK_STATUS(save_string((include_dir + device_map_head_name).c_str(), device_map_head.c_str()));

    std::string device_map_name = device_name + "_map.cpp";
    std::string device_map;
    device_map += "#include \"" + device_map_head_name + "\"\n";
    I8 buffer[16];
#ifndef _USE_SEPERATE_KERNELBIN
    for (auto p : kernels) {
        const std::string &kernel_name = p.first;
        std::string func = device_name + "_" + kernel_name;
        const std::vector<U8> &binary = kernels[kernel_name];
        size_t binary_len = binary.size();
        device_map += "const unsigned int " + func + "_len = " + std::to_string(binary_len) + ";\n";
        device_map += "const unsigned char " + func + "[] = " + "{";
        for (size_t i = 0; i < binary_len; i++) {
            if (i % 20 == 0) {
                device_map += "\n";
            }
            sprintf(buffer, "0x%02x", binary[i]);
            device_map += std::string(buffer);
            if (i != binary_len - 1) {
                device_map += ", ";
            } else {
                device_map += "};\n";
            }
        }
    }
#else
    gcl_kernel_binmap kernel_binmap;
    std::string spir_name = gcl_get_device_spir(handle);
    std::string hardware_name = device_name + "_" + spir_name;
    kernel_binmap.setHardwareName(hardware_name);
    for (auto p : kernels) {
        const std::string &kernel_name = p.first;
        std::string func = device_name + "_" + kernel_name;
        const std::vector<U8> &binary = kernels[kernel_name];
        kernel_binmap.insert(func, {binary.data(), (uint32_t)binary.size()});
    }
    kernel_binmap.saveKernelBin();
    UNI_INFO_LOG("save kernel.bin end.\n");
#endif
    for (auto algo_name : algo_names) {
        U8 *binary = NULL;
        size_t binary_len = 0;
        std::string algo_path = algo_dir + algo_name;
        CHECK_STATUS(load_binary(algo_path.c_str(), (void **)&binary, &binary_len));
        device_map +=
            "const unsigned int " + algo_name + "_len = " + std::to_string(binary_len) + ";\n";
        device_map += "const unsigned char " + algo_name + "[] = " + "{";
        for (size_t i = 0; i < binary_len; i++) {
            if (i % 20 == 0) {
                device_map += "\n";
            }
            sprintf(buffer, "0x%02x", binary[i]);
            device_map += std::string(buffer);
            if (i != binary_len - 1) {
                device_map += ", ";
            } else {
                device_map += "};\n";
            }
        }
        if (binary != NULL) {
            free(binary);
        }
    }

    device_map += "class " + device_name + " : public gcl_kernel_binmap {\n";
    device_map += "public:\n";
    device_map += "    " + device_name + "() {\n";
    device_map += "        loadKernelBin();\n";
    device_map += "    }\n";
#ifndef _USE_SEPERATE_KERNELBIN
    device_map += "    void loadKernelBin() {\n";
    for (auto p : kernels) {
        std::string kernel_name = p.first;
        std::string func = device_name + "_" + kernel_name;
        device_map += "        put(\"" + func + "\", " + "{" + func + ", " + func + "_len});\n";
    }
    device_map += "    }\n";
#endif
    device_map += "};\n";
    device_map += "gcl_kernel_binmap* create_" + device_name + "_kernelbin_map(){\n";
    device_map += "    " + device_name + "* kernelbin = new " + device_name + "();\n";
    device_map += "    return (gcl_kernel_binmap*) kernelbin;\n";
    device_map += "}\n";

    device_map += "const char* get_" + device_name + "_algorithm_info(std::string algo_name){\n";
    for (auto algo_name : algo_names) {
        device_map += "    if (algo_name == \"" + algo_name + "\") {\n";
        device_map += "        return (const char*)" + algo_name + ";\n";
        device_map += "    }\n";
    }
    device_map += "    return nullptr;\n";
    device_map += "}\n";
    CHECK_STATUS(save_string((cpp_dir + device_map_name).c_str(), device_map.c_str()));
}

int main(int argc, char *argv[])
{
    if (argc != 5) {
        UNI_INFO_LOG("Please set your models directory, and put your bolt models into it\n");
        UNI_INFO_LOG("Please set your algorithm directory for save produced algorithm files, and "
                     "ensure it is "
                     "clean\n");
        UNI_INFO_LOG("Please set your include directory for save OpenCL header files, and ensure "
                     "it "
                     "is clean\n");
        UNI_INFO_LOG("Please set your cpp directory for save OpenCL cpp files, and ensure it is "
                     "clean\n");
        UNI_INFO_LOG("For example: ./preprocess_ocl ./models ./algos ./include ./cpp\n");
        return 1;
    }
    std::string model_dir = argv[1] + std::string("/");
    std::string algo_dir = argv[2] + std::string("/");
    std::string include_dir = argv[3] + std::string("/");
    std::string cpp_dir = argv[4] + std::string("/");

    std::map<std::string, std::vector<U8>> kernels;
    std::vector<std::string> model_names = search_files(model_dir, ".bolt");
    for (auto model_name : model_names) {
        std::string model_path = model_dir + model_name;
        run_model(model_path.c_str(), algo_dir.c_str(), &kernels);
    }
    std::vector<std::string> algo_names = search_files(algo_dir, "", "algorithmInfo_");
    save_kernels(kernels, include_dir, cpp_dir, algo_dir, algo_names);
    return 0;
}
