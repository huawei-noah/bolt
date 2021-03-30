// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "gcl_func.h"
#include "ocl_context.h"

OCLContext::OCLContext()
{
    UNI_DEBUG_LOG("OCLContext %p constructor start\n", (char *)this);
    this->handle = std::shared_ptr<GCLHandle>(new GCLHandle());
    this->handle->platformId = 0;
    this->handle->deviceId = 0;
    this->handle->deviceType = CL_DEVICE_TYPE_GPU;
    this->handle->eventPtr = nullptr;
    this->handle->numWaitEvents = 0;
    this->handle->waitEvents = nullptr;
    this->handle->t_execute = 0;
    this->handle->t_total = 0;
    this->handle->curOpName = "unknow";
    this->handle->deviceName = "unknow";
    this->handle->kernel_source = nullptr;
    this->handle->kernel_binmap = nullptr;
    this->handle->kernel_binmap_handle = nullptr;
    this->handle->common_source_opt = "unknow";
    this->handle->common_source_ext = "unknow";
    this->handle->source_head_name[0] = "unknow";
    this->handle->useBinMap = false;
    this->handle->existProfilingQueue = false;
    CHECK_STATUS(get_platforms(&(this->handle->numPlatform), &(this->handle->platforms)));
    CHECK_STATUS(platform_get_devices(this->handle->platforms[this->handle->platformId],
        this->handle->deviceType, &this->handle->numDevice, &this->handle->devices));
    CHECK_STATUS(create_context(this->handle->platforms[this->handle->platformId],
        this->handle->numDevice, this->handle->devices, &this->handle->context));
    cl_queue_properties props[] = {CL_QUEUE_PROPERTIES, 0, 0};
#ifdef _DEBUG
    this->handle->eventPtr = &this->handle->eventObj;
    props[1] = props[1] | CL_QUEUE_PROFILING_ENABLE;
#endif
    CHECK_STATUS(create_command_queue_properties(this->handle->context,
        this->handle->devices[this->handle->deviceId], props, &this->handle->queue));
    this->setDeviceName();
    this->registerBinaryKernelMap();
    if (!this->handle->useBinMap) {
        this->registerSourceKernelMap();
    }
    CHECK_STATUS(gcl_get_device_max_ls_size(this->handle.get(), this->handle->device_max_ls_size));
    CHECK_STATUS(gcl_get_device_max_cu(this->handle.get(), &this->handle->device_max_cu));
    CHECK_STATUS(
        gcl_get_device_max_work_group(this->handle.get(), &this->handle->device_max_work_group));
    UNI_DEBUG_LOG("OCLContext %p constructor end\n", (char *)this);
}

OCLContext::~OCLContext()
{
    UNI_DEBUG_LOG("OCLContext %p deconstructor start\n", (char *)this);
    if (this->handle->platforms == nullptr) {
        return;
    }
    CHECK_STATUS(finish(this->handle->queue));
    for (auto k : this->handle->programMap) {
        CHECK_STATUS(release_program(k.second));
    }
    for (auto k : this->handle->kernelMap) {
        CHECK_STATUS(release_kernel(k.second));
    }
    if (this->handle->useBinMap) {
        delete (gcl_kernel_binmap *)this->handle->kernel_binmap;
        dlclose(this->handle->kernel_binmap_handle);
    } else {
        CHECK_STATUS(release_program(this->handle->source_head[0]));
        delete (gcl_kernel_source *)this->handle->kernel_source;
    }
    this->handle->kernelMap.clear();
    if (this->handle->existProfilingQueue) {
        CHECK_STATUS(finish(this->handle->queue_profiling));
        CHECK_STATUS(release_command_queue(this->handle->queue_profiling));
    }
    CHECK_STATUS(release_command_queue(this->handle->queue));
    CHECK_STATUS(release_context(this->handle->context));
    CHECK_STATUS(release_device(this->handle->devices[this->handle->deviceId]));
    free(this->handle->devices);
    free(this->handle->platforms);
    UNI_DEBUG_LOG("OCLContext %p deconstructor end\n", (char *)this);
}

void OCLContext::setDeviceName()
{
    cl_device_id device = this->handle->devices[this->handle->deviceId];
    U32 len;
    I8 *data;
    CHECK_STATUS(get_device_info(device, CL_DEVICE_NAME, (void **)&data, &len));
    std::string devName = std::string(data);
    for (U32 i = 0; i < len - 1; i++) {
        if (devName[i] == '-') {
            devName[i] = '_';
        }
        if (devName[i] == ' ') {
            devName[i] = '_';
        }
        if (devName[i] == '(') {
            devName[i] = '_';
        }
        if (devName[i] == ')') {
            devName[i] = '_';
        }
    }

    U32 version_len;
    free(data);
    CHECK_STATUS(get_device_info(device, CL_DEVICE_VERSION, (void **)&data, &version_len));
    std::string deviceV = std::string(data);
    if (devName.find("QUALCOMM") != std::string::npos) {
        U32 be = deviceV.find("Adreno");
        std::string subDevName = deviceV.substr(be);
        if (subDevName.find("(TM)") != std::string::npos) {
            subDevName.erase(subDevName.find("(TM)"), 4);
        }
        devName = "QUALCOMM_";
        devName += subDevName;
        for (U32 i = 0; i < devName.length(); i++) {
            if (devName[i] == '-') {
                devName[i] = '_';
            }
            if (devName[i] == ' ') {
                devName[i] = '_';
            }
            if (devName[i] == '(') {
                devName[i] = '_';
            }
            if (devName[i] == ')') {
                devName[i] = '_';
            }
        }
    }

    U32 be = deviceV.find("r");
    U32 end = deviceV.find("p", be + 1);
    std::string numV = deviceV.substr(be + 1, end - be - 1);
    U32 i = atoi(numV.c_str());
    if (i >= 14) {
        devName += "p";
    }
    free(data);
    this->handle->deviceName = devName;
}

void OCLContext::registerBinaryKernelMap()
{
    std::string libKernelBinName = "lib" + this->handle->deviceName + "_map.so";
    char *err;
    void *dvm_handle = dlopen(libKernelBinName.c_str(), RTLD_LAZY);
    if (dvm_handle) {
        std::string func = "create_" + this->handle->deviceName + "_kernelbin_map";
        gcl_kernel_binmap *(*create_kernelbin_map)();
        dlerror();
        create_kernelbin_map = (gcl_kernel_binmap * (*)()) dlsym(dvm_handle, func.c_str());
        if ((err = dlerror()) != NULL) {
            UNI_ERROR_LOG(
                "Get %s in %s failed, error %s\n", func.c_str(), libKernelBinName.c_str(), err);
            dlclose(dvm_handle);
        }
        gcl_kernel_binmap *kernel_binmap = create_kernelbin_map();
        this->handle->kernel_binmap = (void *)kernel_binmap;
        this->handle->useBinMap = true;
        this->handle->kernel_binmap_handle = dvm_handle;
    } else {
        UNI_DEBUG_LOG("try to dlopen %s failed, %s, create kernel from source code\n",
            libKernelBinName.c_str(), dlerror());
    }
}

void OCLContext::registerSourceKernelMap()
{
    gcl_kernel_source *kernel_source = new kernel_source_executor();
    this->handle->kernel_source = kernel_source;
    KernelOption *common_opt;
    if (!kernel_source->get_option("common", &common_opt)) {
        UNI_ERROR_LOG("the common doesn't exist in optionMap\n");
        CHECK_STATUS(NULL_POINTER);
    }
    this->handle->common_source_opt = common_opt->option;
    this->handle->common_source_ext = "#pragma OPENCL EXTENSION cl_khr_3d_image_writes : enable\n";
    this->handle->common_source_ext += "#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n";
    this->handle->source_head_name[0] = "kernel_def.h";
    KernelSource *head_source;
    if (!kernel_source->get_source("kernel_def", &head_source)) {
        UNI_ERROR_LOG("the kernel_def doesn't exist in sourceMap\n");
        CHECK_STATUS(NULL_POINTER);
    }
    CHECK_STATUS(create_program_from_source(this->handle->context, (U32 *)&head_source->len,
        head_source->data, this->handle->source_head));
}

OCLContext &OCLContext::getInstance()
{
    static OCLContext _instance;
    return _instance;
}
