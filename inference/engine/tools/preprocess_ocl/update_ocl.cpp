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

int main(int argc, char *argv[])
{
#ifdef _USE_SEPERATE_KERNELBIN
    GCLHandle_t handle = OCLContext::getInstance().handle.get();
    std::string device_name = handle->deviceName;
    std::string spir_name = gcl_get_device_spir(handle);
    std::string hardware_name = device_name + "_" + spir_name;
    UNI_INFO_LOG("update kernel.bin for %s...\n", hardware_name.c_str());

    std::map<std::string, std::vector<U8>> kernels;
    auto kernel_binmap = (gcl_kernel_binmap *)(handle->kernel_binmap);
    if (kernel_binmap == NULL) {
        if (argc != 2) {
            UNI_ERROR_LOG("Please set your models directory, and put your bolt models into it.\n"
                          "For example: ./update_ocl ./models\n");
            return 1;
        }
        std::string model_dir = argv[1] + std::string("/");
        std::vector<std::string> model_names = search_files(model_dir, ".bolt");
        for (auto model_name : model_names) {
            std::string model_path = model_dir + model_name;
            run_model(model_path.c_str(), NULL, &kernels);
        }
        gcl_kernel_binmap kernel_binmap;
        kernel_binmap.setHardwareName(hardware_name);
        for (auto p : kernels) {
            const std::string &kernel_name = p.first;
            std::string func = device_name + "_" + kernel_name;
            UNI_INFO_LOG("    update kernel:%s...\n", func.c_str());
            const std::vector<U8> &binary = kernels[kernel_name];
            kernel_binmap.insert(func, {binary.data(), (uint32_t)binary.size()});
        }
        kernel_binmap.saveKernelBin();
        UNI_INFO_LOG("save kernel.bin end.\n");
    } else {
        auto kernel_hardware_name = kernel_binmap->getHardwareName();
        if (hardware_name == kernel_hardware_name) {
            UNI_INFO_LOG("there is no need to update kernel.bin file.\n");
        } else {
            if (!handle->useBinMap) {
                UNI_ERROR_LOG("can not load prebuild library and kernel.bin file.\n");
                return 1;
            }
            auto bins = kernel_binmap->binMap();
            int offset = device_name.length() + 1;
            for (auto p : bins) {
                const auto &name = p.first;
                UNI_INFO_LOG("    update kernel:%s...\n", name.c_str());
                std::string kernelname = name.substr(offset);
                Kernel kernel;
                Program program;
                CHECK_STATUS(gcl_create_kernel_binary(handle, kernelname.c_str(), &kernel));
                CHECK_STATUS(get_program_info_from_kernel(kernel, &program));
                update_kernels(&kernels, name, program);
            }
            for (auto p : kernels) {
                const std::vector<U8> &binary = kernels[p.first];
                kernel_binmap->insert(p.first, {binary.data(), (uint32_t)binary.size()});
            }
            kernel_binmap->setHardwareName(hardware_name);
            kernel_binmap->saveKernelBin();
            UNI_INFO_LOG("update kernel.bin end.\n");
        }
    }
    return 0;
#else
    UNI_ERROR_LOG("not support to update kernel.bin.\n");
    return 1;
#endif
}
