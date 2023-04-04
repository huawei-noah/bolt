// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef H_GCL_FUNC
#define H_GCL_FUNC

#include <algorithm>
#include <map>
#include "gcl_common.h"
#include "platform.h"
#include "context.h"
#include "program.h"
#include "memory.h"
#include "kernel.h"
#include "event.h"
#include "gcl_kernel_binmap.h"
#include "gcl_kernel_source.h"
#include "libkernelsource.h"
#include "algorithm_map.h"
#include "dl_func.h"

inline std::string gcl_get_kernelbin_name(std::string device)
{
    std::string ret = "./lib" + device + "_map.";
#ifdef _WIN32
    ret += "dll";
#else
    ret += "so";
#endif
    return ret;
}

inline std::string gcl_get_type(DataType dt)
{
    static std::map<DataType, std::string> m = {
        {DT_F32, "float"},
        {DT_F16, "half"},
        {DT_U32, "uint"},
        {DT_I32, "int"},
        {DT_U8, "uchar"},
        {DT_I8, "char"},
    };
    std::string ret;
    if (m.find(dt) == m.end()) {
        UNI_ERROR_LOG("gcl currently not support %s.\n", DataTypeName()[dt]);
    } else {
        ret = m[dt];
    }
    return ret;
}

inline std::string gcl_get_device_spir(GCLHandle_t handle)
{
    std::string spirName = "";
#if CL_TARGET_OPENCL_VERSION > 200
    cl_device_id device = handle->devices[handle->deviceId];
    U32 len;
    I8 *data = NULL;
    CHECK_STATUS(get_device_info(device, CL_DEVICE_IL_VERSION, (void **)&data, &len));
    spirName = data;
    free(data);
#endif
    return spirName;
}

#ifdef __cplusplus
extern "C" {
#endif
inline EE gcl_regist_binMap(GCLHandle_t handle)
{
    std::string deviceName = handle->deviceName;
    std::string libKernelBinName = gcl_get_kernelbin_name(deviceName);
    void *dvm_handle = handle->kernel_binmap_handle;
    if (!dvm_handle) {
        dvm_handle = dlopen(libKernelBinName.c_str(), RTLD_LAZY);
    }
    if (dvm_handle) {
        std::string func = "create_" + deviceName + "_kernelbin_map";
        gcl_kernel_binmap *(*create_kernelbin_map)();
        create_kernelbin_map = (gcl_kernel_binmap * (*)()) dlsym(dvm_handle, func.c_str());
        const char *err = dlerror();
        if (err != NULL) {
            dlclose(dvm_handle);
            UNI_ERROR_LOG(
                "get %s from %s failed %s.\n", func.c_str(), libKernelBinName.c_str(), err);
            return NULL_POINTER;
        }
        gcl_kernel_binmap *kernel_binmap = create_kernelbin_map();
        handle->kernel_binmap = (void *)kernel_binmap;
        handle->useBinMap = true;
        handle->kernel_binmap_handle = dvm_handle;
    } else {
        UNI_DEBUG_LOG("try to dlopen %s failed %s, create kernel from source code.\n",
            libKernelBinName.c_str(), dlerror());
    }
    return SUCCESS;
}

inline const char *gcl_get_algorithm_info(GCLHandle_t handle, std::string algoName)
{
    std::string deviceName = handle->deviceName;
    std::string libKernelBinName = gcl_get_kernelbin_name(deviceName);
    void *dvm_handle = handle->kernel_binmap_handle;
    if (!dvm_handle) {
        dvm_handle = dlopen(libKernelBinName.c_str(), RTLD_LAZY);
    }
    const char *algoInfo = nullptr;
    if (dvm_handle) {
        std::string func = "get_" + deviceName + "_algorithm_info";
        const char *(*get_algorithm_info)(std::string);
        get_algorithm_info = (const char *(*)(std::string))dlsym(dvm_handle, func.c_str());
        const char *err = dlerror();
        if (err != NULL) {
            dlclose(dvm_handle);
            UNI_ERROR_LOG("Get %s in %s failed %s.\n", func.c_str(), libKernelBinName.c_str(), err);
        }
        algoInfo = (*get_algorithm_info)(algoName);
    } else {
        UNI_DEBUG_LOG("try to dlopen %s failed %s, create algorithm map by profiling.\n",
            libKernelBinName.c_str(), dlerror());
    }
    return algoInfo;
}

inline EE gcl_regist_sourceMap(GCLHandle_t handle)
{
    gcl_kernel_source *kernel_source = (gcl_kernel_source *)new kernel_source_executor();
    handle->kernel_source = kernel_source;
    KernelOption *common_opt;
    if (!kernel_source->get_option("common", &common_opt)) {
        UNI_ERROR_LOG("the common doesn't exist in optionMap\n");
        CHECK_STATUS(NULL_POINTER);
    }
    handle->common_source_opt = common_opt->option;
    handle->common_source_ext = "#pragma OPENCL EXTENSION cl_khr_3d_image_writes : enable\n";
    handle->common_source_ext += "#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n";
    handle->source_head_name[0] = "kernel_def.h";
    KernelSource *head_source;
    if (!kernel_source->get_source("kernel_def", &head_source)) {
        UNI_ERROR_LOG("the kernel_def doesn't exist in sourceMap\n");
        CHECK_STATUS(NULL_POINTER);
    }
    CHECK_STATUS(create_program_from_source(
        handle->context, (U32 *)&head_source->len, head_source->data, handle->source_head));
    return SUCCESS;
}

inline EE gcl_get_device_name(GCLHandle_t handle)
{
    cl_device_id device = handle->devices[handle->deviceId];
    U32 len;
    I8 *data = NULL;
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
    free(data);
    data = NULL;

    U32 version_len;
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
    free(data);
    data = NULL;

    U32 be = deviceV.find("r");
    U32 end = deviceV.find("p", be + 1);
    std::string numV = deviceV.substr(be + 1, end - be - 1);
    U32 i = atoi(numV.c_str());
    if (i >= 14) {
        devName += "p";
    }
    handle->deviceName = devName;
    return SUCCESS;
}

inline bool gcl_check_device_qualcomm(GCLHandle_t handle)
{
    std::string deviceName = handle->deviceName;
    bool qualCommDev = false;
    if (deviceName.find("QUALCOMM") != std::string::npos) {
        qualCommDev = true;
    }
    return qualCommDev;
}

inline bool gcl_check_meet_device_image3d_limits(
    GCLHandle_t handle, U32 width, U32 height, U32 depth)
{
    if (!handle->useQualcommDev) {
        return false;
    }
    if (width <= 0 || height <= 0 || depth <= 0) {
        return false;
    }
    if (width > handle->device_max_image3d_size[0] || height > handle->device_max_image3d_size[1] ||
        depth > handle->device_max_image3d_size[2]) {
        return false;
    }
    return true;
}

inline EE gcl_get_device_max_image3d_size(GCLHandle_t handle, U32 *size)
{
    CHECK_STATUS(get_device_max_image3d_sizes(handle->devices[handle->deviceId], size));
    return SUCCESS;
}

inline EE gcl_get_device_max_ls_size(GCLHandle_t handle, U32 *size)
{
    CHECK_STATUS(get_device_max_work_item_sizes(handle->devices[handle->deviceId], size));
    return SUCCESS;
}

inline EE gcl_get_device_max_cu(GCLHandle_t handle, U32 *size)
{
    CHECK_STATUS(get_device_max_compute_units(handle->devices[handle->deviceId], size));
    return SUCCESS;
}

inline EE gcl_get_device_max_work_group(GCLHandle_t handle, U32 *size)
{
    CHECK_STATUS(get_device_max_work_group_size(handle->devices[handle->deviceId], size));
    return SUCCESS;
}

inline EE gcl_create_handle(GCLHandle_t *handlePtr)
{
    if (handlePtr == NULL) {
        UNI_ERROR_LOG("the handlePtr set to gcl_create_handle is NULL\n");
    }
    GCLHandle_t handle = new GCLHandle();
    handle->platformId = 0;
    handle->deviceId = 0;
    handle->deviceType = CL_DEVICE_TYPE_GPU;
    handle->eventPtr = nullptr;
    handle->numWaitEvents = 0;
    handle->waitEvents = nullptr;
    handle->t_execute = 0;
    handle->t_total = 0;
    handle->curOpName = "unknow";
    handle->deviceName = "unknow";
    handle->kernel_source = nullptr;
    handle->kernel_binmap = nullptr;
    handle->kernel_binmap_handle = nullptr;
    handle->common_source_opt = "unknow";
    handle->common_source_ext = "unknow";
    handle->source_head_name[0] = "unknow";
    handle->useBinMap = false;
    handle->existProfilingQueue = false;
    U32 platformId = handle->platformId;
    U32 deviceId = handle->deviceId;
    CHECK_STATUS(get_platforms(&handle->numPlatform, &handle->platforms));
    CHECK_STATUS(get_devices(
        handle->platforms[platformId], handle->deviceType, &handle->numDevice, &handle->devices));
    CHECK_STATUS(create_context(
        handle->platforms[platformId], handle->numDevice, handle->devices, &handle->context));
    cl_queue_properties props[] = {CL_QUEUE_PROPERTIES, 0, 0};
#ifdef _DEBUG
    handle->eventPtr = &handle->eventObj;
    props[1] = props[1] | CL_QUEUE_PROFILING_ENABLE;
#endif
    CHECK_STATUS(create_command_queue_properties(
        handle->context, handle->devices[deviceId], props, &handle->queue));
    CHECK_STATUS(gcl_get_device_name(handle));
    CHECK_STATUS(gcl_regist_binMap(handle));
    if (!handle->useBinMap) {
        CHECK_STATUS(gcl_regist_sourceMap(handle));
    }
    CHECK_STATUS(gcl_get_device_max_ls_size(handle, handle->device_max_ls_size));
    CHECK_STATUS(gcl_get_device_max_cu(handle, &handle->device_max_cu));
    CHECK_STATUS(gcl_get_device_max_work_group(handle, &handle->device_max_work_group));
    *handlePtr = handle;
    return SUCCESS;
}

inline void gcl_destroy_handle(GCLHandle_t handle)
{
    U32 deviceId = handle->deviceId;
    CHECK_STATUS(finish(handle->queue));
    for (auto k : handle->programMap) {
        CHECK_STATUS(release_program(k.second));
    }
    for (auto k : handle->kernelMap) {
        CHECK_STATUS(release_kernel(k.second));
    }
    if (handle->useBinMap) {
        delete (gcl_kernel_binmap *)handle->kernel_binmap;
        dlclose(handle->kernel_binmap_handle);
    } else {
        CHECK_STATUS(release_program(handle->source_head[0]));
        delete (gcl_kernel_source *)handle->kernel_source;
    }
    handle->kernelMap.clear();
    if (handle->existProfilingQueue) {
        CHECK_STATUS(finish(handle->queue_profiling));
        CHECK_STATUS(release_command_queue(handle->queue_profiling));
    }
    CHECK_STATUS(release_command_queue(handle->queue));
    CHECK_STATUS(release_context(handle->context));
    CHECK_STATUS(release_device(handle->devices[deviceId]));
    free(handle->devices);
    free(handle->platforms);
    delete handle;
}

inline EE gcl_enable_queue_profiling(GCLHandle_t handle)
{
#ifndef _DEBUG
    handle->eventPtr = &handle->eventObj;
    bool enableProfiling;
    CHECK_STATUS(check_queue_profiling(handle->queue, &enableProfiling));
    if (enableProfiling) {
        return SUCCESS;
    }
    if (!handle->existProfilingQueue) {
        cl_queue_properties props[] = {CL_QUEUE_PROPERTIES, 0, 0};
        props[1] = props[1] | CL_QUEUE_PROFILING_ENABLE;
        CHECK_STATUS(create_command_queue_properties(
            handle->context, handle->devices[handle->deviceId], props, &handle->queue_profiling));
        handle->existProfilingQueue = true;
    }
    CommandQueue tmpQueue = handle->queue;
    handle->queue = handle->queue_profiling;
    handle->queue_profiling = tmpQueue;
#endif
    return SUCCESS;
}

inline EE gcl_off_queue_profiling(GCLHandle_t handle)
{
#ifndef _DEBUG
    handle->eventPtr = NULL;
    bool enableProfiling;
    CHECK_STATUS(check_queue_profiling(handle->queue, &enableProfiling));
    if (!enableProfiling) {
        return SUCCESS;
    }
    CHECK_STATUS(check_queue_profiling(handle->queue_profiling, &enableProfiling));
    if (!enableProfiling) {
        CHECK_STATUS(finish(handle->queue));
        CommandQueue tmpQueue = handle->queue;
        handle->queue = handle->queue_profiling;
        handle->queue_profiling = tmpQueue;
    } else {
        return NOT_SUPPORTED;
    }
#endif
    return SUCCESS;
}

inline GCLMemDesc gcl_mem_desc(U32 stride[], U32 offset[], DataType dt, DataFormat memFormat)
{
    GCLMemDesc desc;
    U32 s0, s1, s2;
    s0 = stride[0];
    s1 = stride[1];
    s2 = stride[2];
    desc.stride[0] = s0;
    desc.stride[1] = s1;
    desc.stride[2] = s2;
    desc.offset[0] = offset[0];
    desc.offset[1] = offset[1];
    desc.offset[2] = offset[2];
    desc.memFormat = memFormat;
    desc.memType = GCL_MEM_BUF;
    desc.num = s0 * s1 * s2;
    desc.byteSize = s0 * s1 * s2 * bytesOf(dt);
    desc.flags = CL_MEM_READ_WRITE;
    desc.host_ptr = NULL;
    desc.imgFormat.image_channel_order = CL_RGBA;
    desc.imgFormat.image_channel_data_type = CL_HALF_FLOAT;
    desc.need_pad = false;
    return desc;
}

inline GCLMem_t gcl_create_gclmem()
{
    GCLMem_t ret = new GCLMem;
    ret->mem = NULL;
    U32 str[3] = {0, 0, 0};
    U32 off[3] = {0, 0, 0};
    ret->desc = gcl_mem_desc(str, off, DT_U8, DF_NCHWC4);
    return ret;
}

inline EE gcl_release_subMem(GCLMem_t gclMem)
{
    if (gclMem == nullptr) {
        CHECK_STATUS(NULL_POINTER);
    }
    if (gclMem->subMem.size()) {
        for (int i = gclMem->subMem.size() - 1; i >= 0; i--) {
            auto p = gclMem->subMem[i];
            CHECK_STATUS(release_memory(p));
        }
        gclMem->subMem.clear();
    }
    return SUCCESS;
}

inline EE gcl_release_memory(GCLMem_t gclMem)
{
    if (gclMem == nullptr) {
        CHECK_STATUS(NULL_POINTER);
    }
    if (gclMem->mem) {
        CHECK_STATUS(release_memory(gclMem->mem));
        gclMem->mem = NULL;
    }
    return SUCCESS;
}

inline void gcl_destroy_gclmem(GCLMem_t mem)
{
    CHECK_STATUS(gcl_release_subMem(mem));
    CHECK_STATUS(gcl_release_memory(mem));
    delete mem;
}

inline EE gcl_finish(GCLHandle_t handle)
{
    CHECK_STATUS(finish(handle->queue));
    return SUCCESS;
}

inline EE gcl_unmap_memory(GCLHandle_t handle, GCLMem_t gclMem)
{
    for (auto p : gclMem->mapPtrArray) {
        CHECK_STATUS(enqueue_unmap_memory(handle->queue, gclMem->mem, (void *)p,
            handle->numWaitEvents, handle->waitEvents, handle->eventPtr));
#ifdef _DEBUG
        double time = 0;
        CHECK_STATUS(event_counting_time(handle->eventPtr, NULL, NULL, NULL, NULL, &time));
        CHECK_STATUS(release_event(handle->eventObj));
        UNI_DETAIL_LOG("DATA_UNMAP>>> enqueue_unmap_memory time:%.2fus\n", time);
#endif
        CHECK_STATUS(gcl_finish(handle));
    }
    if (gclMem->mapPtrArray.size()) {
        gclMem->mapPtrArray.clear();
    }
    return SUCCESS;
}

inline EE gcl_produce_program_kernel_with_source(GCLHandle_t handle,
    U32 *len,
    const char *src,
    const char *option,
    Program *program,
    U32 numKernel,
    Kernel *kernels)
{
    U32 deviceId = handle->deviceId;
    CHECK_STATUS(create_build_program_from_source(
        handle->context, len, src, handle->devices[deviceId], option, program));
    CHECK_STATUS(create_kernels_in_program(*program, numKernel, kernels));
    return SUCCESS;
}

inline EE gcl_get_program_info(Program program, U8 **binary, U32 *len)
{
    CHECK_STATUS(get_program_binary(program, binary, len));
    return SUCCESS;
}

inline EE gcl_kernelmap_put(GCLHandle_t handle, std::string kernelName, Kernel kernel)
{
    handle->kernelMap.insert(std::pair<std::string, Kernel>(kernelName, kernel));
    return SUCCESS;
}

inline Kernel gcl_kernelmap_get(GCLHandle_t handle, std::string kernelName)
{
    auto it = handle->kernelMap.find(std::string(kernelName));
    if (it == handle->kernelMap.end()) {
        CHECK_STATUS(NOT_MATCH);
    }
    return it->second;
}

inline EE gcl_create_kernel_binary(GCLHandle_t handle, const char *kernelName, Kernel *kernel)
{
    std::string binmapname = handle->deviceName;
    std::string binmap_kernelname = binmapname + "_" + std::string(kernelName);
    gcl_kernel_binmap *kernel_binmap = (gcl_kernel_binmap *)handle->kernel_binmap;
    KernelBin *binmap;
    if (!kernel_binmap->get(binmap_kernelname, &binmap)) {
        UNI_ERROR_LOG("get kernel %s from %s kernel_binmap failed\n", kernelName,
            gcl_get_kernelbin_name(binmapname).c_str());
        return NULL_POINTER;
    }
    U32 length = binmap->len;
    const U8 *data = binmap->data;
    I32 binsta;
    Program program;
    const char *options = "";
    Device device = handle->devices[handle->deviceId];
    CHECK_STATUS(
        create_program_from_binary(handle->context, device, &length, &data, &binsta, &program));
    CHECK_STATUS(build_program(program, device, options));
    CHECK_STATUS(create_kernel(program, kernelName, kernel));
    CHECK_STATUS(release_program(program));
    return SUCCESS;
}

inline EE gcl_create_kernel_with_source_map(
    GCLHandle_t handle, const char *kernelName, Kernel *kernel, KernelOpt *opt = NULL)
{
    UNI_DETAIL_LOG("create kernel %s.\n", kernelName);
    Program program;
    auto it = handle->programMap.find(kernelName);
    if (it == handle->programMap.end()) {
        gcl_kernel_source *kernel_source = (gcl_kernel_source *)handle->kernel_source;
        KernelOption *option_ptr;
        KernelSource *source_ptr;
        const char *sourceName;
        std::string option = "";
        std::string optionName = kernelName;
        bool use_common_opt = false;
        if (!kernel_source->get_option(optionName, &option_ptr)) {
            if (opt) {
                sourceName = opt->sourceName;
                option = opt->option;
                char buffer[256] = {0};
                std::string type = gcl_get_type(opt->kernelDataType);
                const char *p = type.c_str();
                UNI_SNPRINTF(buffer, 256,
                    "-cl-single-precision-constant -cl-std=CL2.0 -DT=%s -DT2=%s2 -DT3=%s3 -DT4=%s4 -DT8=%s8 -DT16=%s16", p, p, p,
                    p, p, p);
                option += std::string(buffer);
                if (opt->kernelDataType == DT_F16) {
                    option += std::string(" -DUSE_HALF");
                }
                option += std::string(" ");
            } else {
                sourceName = kernelName;
                use_common_opt = true;
            }
        } else {
            use_common_opt = option_ptr->use_common_opt;
            sourceName = option_ptr->sourceName;
            option = option_ptr->option;
        }
        if (use_common_opt) {
            option = handle->common_source_opt + " " + option;
        }
        if (!kernel_source->get_source(sourceName, &source_ptr)) {
            UNI_ERROR_LOG(
                "the %s doesn't exist in sourceMap to find kernel %s.\n", sourceName, kernelName);
            CHECK_STATUS(NULL_POINTER);
        }

        U32 len = source_ptr->len + handle->common_source_ext.size();
        std::string source = source_ptr->data;
        source = handle->common_source_ext + source;
        bool use_kernel_def_head = source_ptr->use_kernel_def_head;
        create_program_from_source(handle->context, &len, source.c_str(), &program);
        Device device = handle->devices[handle->deviceId];
        if (use_kernel_def_head) {
            CHECK_STATUS(compile_program(
                program, device, option.c_str(), 1, handle->source_head, handle->source_head_name));
            CHECK_STATUS(link_program(handle->context, device, NULL, 1, &program, &program));
        } else {
            CHECK_STATUS(build_program(program, device, option.c_str()));
        }
        handle->programMap.insert(std::pair<std::string, Program>(kernelName, program));
    } else {
        program = it->second;
    }
    CHECK_STATUS(create_kernel(program, kernelName, kernel));
    return SUCCESS;
}

inline EE gcl_create_kernel(
    GCLHandle_t handle, const char *kernelName, Kernel *kernel, KernelOpt *opt = NULL)
{
    if (handle->useBinMap) {
        CHECK_STATUS(gcl_create_kernel_binary(handle, kernelName, kernel));
    } else {
        CHECK_STATUS(gcl_create_kernel_with_source_map(handle, kernelName, kernel, opt));
    }
    return SUCCESS;
}

inline EE gcl_get_kernel_from_map(
    GCLHandle_t handle, const char *kernelName, Kernel *kernel, KernelOpt *opt = NULL)
{
    std::string binmapname = handle->deviceName;
    std::string binmap_kernelname = binmapname + "_" + std::string(kernelName);
    if (handle->kernelMap.find(binmap_kernelname) == handle->kernelMap.end()) {
        CHECK_STATUS(gcl_create_kernel(handle, kernelName, kernel, opt));
        CHECK_STATUS(gcl_kernelmap_put(handle, binmap_kernelname, *kernel));
    } else {
        *kernel = gcl_kernelmap_get(handle, binmap_kernelname);
    }
    return SUCCESS;
}

inline EE gcl_infer_best_ls_with_device_info(GCLHandle_t handle, U32 dim, U32 *gs, U32 *ls)
{
    U32 *max_ls_size = handle->device_max_ls_size;
    U32 max_cu = handle->device_max_cu;
    U32 max_work_group = handle->device_max_work_group;
    U32 current_total_size = 1;
    for (U32 i = 0; i < dim; i++) {
        U32 gs_size = gs[i];
        U32 ls_size = gs_size / max_cu;
        U32 ls_rest = gs_size % max_cu;
        U32 ls_up_size = max_work_group / current_total_size;
        if (ls_up_size > max_ls_size[i]) {
            ls_up_size = max_ls_size[i];
        }
        if (ls_size > ls_up_size) {
            ls_size = ls_up_size;
        }
        if (ls_rest != 0 || i == 2) {
            while (ls_size) {
                ls_size = ls_size - 1;
                ls_rest = gs_size % ls_size;
                if (ls_rest == 0 || (ls_size <= max_work_group && i != 2)) {
                    break;
                }
            }
        }
        if (ls_size < 1) {
            ls_size = 1;
        }
        current_total_size *= ls_size;
        gs[i] = (gs_size + ls_size - 1) / ls_size * ls_size;
        ls[i] = ls_size;
    }
    return SUCCESS;
}
inline EE gcl_set_kernelVec(GCLHandle_t handle,
    Kernel kernel,
    U32 work_dim,
    U32 global_work_size[],
    U32 local_work_size[],
    const char *kernelName = NULL)
{
    if (handle == NULL || handle->kernelVec == NULL) {
        return NULL_POINTER;
    }
    GCLKernelInfo kernelInfo;
    kernelInfo.kernel = kernel;
    kernelInfo.dim = work_dim;
    kernelInfo.name = handle->curOpName + "_" + std::string(kernelName);
    switch (work_dim) {
        case 1: {
            kernelInfo.gs[0] = global_work_size[0];
            kernelInfo.gs[1] = 1;
            kernelInfo.gs[2] = 1;
            kernelInfo.ls[0] = local_work_size[0];
            kernelInfo.ls[1] = 0;
            kernelInfo.ls[2] = 0;
            break;
        }
        case 2: {
            kernelInfo.gs[0] = global_work_size[0];
            kernelInfo.gs[1] = global_work_size[1];
            kernelInfo.gs[2] = 1;
            kernelInfo.ls[0] = local_work_size[0];
            kernelInfo.ls[1] = local_work_size[1];
            kernelInfo.ls[2] = 0;
            break;
        }
        case 3: {
            kernelInfo.gs[0] = global_work_size[0];
            kernelInfo.gs[1] = global_work_size[1];
            kernelInfo.gs[2] = global_work_size[2];
            kernelInfo.ls[0] = local_work_size[0];
            kernelInfo.ls[1] = local_work_size[1];
            kernelInfo.ls[2] = local_work_size[2];
            break;
        }
        default:
            return NOT_SUPPORTED;
    }
    handle->kernelVec->push_back(kernelInfo);
    return SUCCESS;
}

inline EE gcl_run_kernelVec(GCLHandle_t handle, U32 *index = NULL)
{
    if (handle == NULL || handle->kernelVec == NULL) {
        return NULL_POINTER;
    }
    CommandQueue queue = handle->queue;
    U32 numWaitEvents = handle->numWaitEvents;
    Event *waitEvents = handle->waitEvents;
    Event *eventPtr = handle->eventPtr;
    U32 runBe;
    U32 runEnd;
    if (index) {
        runBe = index[0];
        runEnd = index[1];
    } else {
        runBe = 0;
        runEnd = handle->kernelVec->size();
    }
    for (U32 i = runBe; i < runEnd; ++i) {
        auto kernelInfo = (*handle->kernelVec)[i];
        for (U32 j = 0; j < kernelInfo.dim; j++) {
            if (kernelInfo.ls[j] != 0) {
                kernelInfo.gs[j] =
                    (kernelInfo.gs[j] + kernelInfo.ls[j] - 1) / kernelInfo.ls[j] * kernelInfo.ls[j];
            }
        }
        CHECK_STATUS(enqueue_ndrange_kernel(queue, kernelInfo.kernel, kernelInfo.dim, NULL,
            kernelInfo.gs, kernelInfo.ls, numWaitEvents, waitEvents, eventPtr));
#ifdef _DEBUG
        double time = 0;
        CHECK_STATUS(event_counting_time(eventPtr, NULL, NULL, NULL, NULL, &time));
        CHECK_STATUS(release_event(*eventPtr));
        handle->t_execute = time;
        UNI_DETAIL_LOG("KERNEL>>> %s ls<%d %d %d> time:%.2fus\n", kernelInfo.name.c_str(),
            kernelInfo.ls[0], kernelInfo.ls[1], kernelInfo.ls[2], time);
        CHECK_STATUS(gcl_finish(handle));
#endif
    }
    return SUCCESS;
}

inline EE gcl_run_kernelVec_timing(
    GCLHandle_t handle, U32 be, U32 end, std::vector<double> *kernelArrayTime = NULL)
{
    if (handle == NULL || handle->kernelVec == NULL) {
        return NULL_POINTER;
    }
    EE ret = NOT_SUPPORTED;
#ifndef _DEBUG
    CHECK_STATUS(gcl_enable_queue_profiling(handle));
#endif
    bool enableProfiling;
    CHECK_STATUS(check_queue_profiling(handle->queue, &enableProfiling));
    if (enableProfiling) {
        double time = 0;
        double total = 0;
        CommandQueue queue = handle->queue;
        U32 numWaitEvents = handle->numWaitEvents;
        Event *waitEvents = handle->waitEvents;
        Event *eventPtr = handle->eventPtr;
        for (U32 i = be; i < end; ++i) {
            double minTime = DBL_MAX;
            auto kernelInfo = (*handle->kernelVec)[i];
            U32 gs[3] = {1, 1, 1};
            U32 ls[3] = {0, 0, 0};
            U32 dim = kernelInfo.dim;
            bool noNeedInferLS = true;
            for (U32 j = 0; j < dim; j++) {
                gs[j] = kernelInfo.gs[j];
                ls[j] = kernelInfo.ls[j];
                bool k = false;
                if (ls[j] > 0) {
                    k = true;
                }
                noNeedInferLS = noNeedInferLS & k;
            }
            if (noNeedInferLS) {
                for (U32 j = 0; j < dim; j++) {
                    kernelInfo.gs[j] = (gs[j] + ls[j] - 1) / ls[j] * ls[j];
                }
                CHECK_STATUS(enqueue_ndrange_kernel(queue, kernelInfo.kernel, kernelInfo.dim, NULL,
                    kernelInfo.gs, kernelInfo.ls, numWaitEvents, waitEvents, eventPtr));
                CHECK_STATUS(event_counting_time(handle->eventPtr, NULL, NULL, NULL, NULL, &time));
                CHECK_STATUS(release_event(handle->eventObj));
                CHECK_STATUS(gcl_finish(handle));
            } else {
                for (U32 j = 0; j < 2; j++) {
                    CHECK_STATUS(enqueue_ndrange_kernel(queue, kernelInfo.kernel, kernelInfo.dim,
                        NULL, kernelInfo.gs, kernelInfo.ls, 0, NULL, NULL));
                }
                gcl_finish(handle);
                std::vector<U32> test_gs;
                std::vector<U32> test_ls;
                for (U32 j = 0; j < 3; j++) {
                    test_gs.push_back(gs[j]);
                    test_ls.push_back(ls[j]);
                }
                for (U32 j = 2; j <= 256; j *= 2) {
                    test_gs.push_back((gs[0] + j - 1) / j * j);
                    test_gs.push_back(gs[1]);
                    test_gs.push_back(gs[2]);
                    test_ls.push_back(j);
                    test_ls.push_back(1);
                    test_ls.push_back(1);
                }
                int timeUpCount = 0;
                for (U32 j = 0; j < test_gs.size(); j += 3) {
                    for (U32 k = 0; k < 3; k++) {
                        gs[k] = test_gs[j + k];
                        ls[k] = test_ls[j + k];
                    }
                    CHECK_STATUS(enqueue_ndrange_kernel(queue, kernelInfo.kernel, dim, NULL, gs, ls,
                        numWaitEvents, waitEvents, eventPtr));
                    CHECK_STATUS(
                        event_counting_time(handle->eventPtr, NULL, NULL, NULL, NULL, &time));
                    CHECK_STATUS(release_event(handle->eventObj));
                    gcl_finish(handle);
                    if (minTime > time) {
                        minTime = time;
                        for (U32 k = 0; k < 3; k++) {
                            kernelInfo.ls[k] = ls[k];
                        }
                        timeUpCount = 0;
                    } else {
                        timeUpCount++;
                    }
                    if (timeUpCount >= 3) {
                        break;
                    }
                }
                time = minTime;
            }

            UNI_DETAIL_LOG("KERNEL>>> %s ls<%d %d %d> time:%.2fus\n", kernelInfo.name.c_str(),
                kernelInfo.ls[0], kernelInfo.ls[1], kernelInfo.ls[2], time);
            total += time;
            if (kernelArrayTime) {
                (*kernelArrayTime).push_back(time);
            }
        }
        handle->t_execute = total;
        ret = SUCCESS;
    }
#ifndef _DEBUG
    CHECK_STATUS(gcl_off_queue_profiling(handle));
#endif
    return ret;
}

inline EE gcl_clean_kernels(std::vector<GCLKernelInfo> *kernels)
{
    if (kernels != NULL && kernels->size() > 0) {
        auto p = kernels->data();
        for (U32 i = 0; i < kernels->size(); i++) {
            CHECK_STATUS(release_kernel(p[i].kernel));
        }
        kernels->clear();
    }
    return SUCCESS;
}

inline EE gcl_clean_kernelVec(GCLHandle_t handle)
{
    EE ret = SUCCESS;
    if (handle != NULL && handle->kernelVec != NULL) {
        ret = gcl_clean_kernels(handle->kernelVec);
    }
    return ret;
}

inline EE gcl_clean_programMap(GCLHandle_t handle)
{
    if (handle != NULL) {
        for (auto k : handle->programMap) {
            CHECK_STATUS(release_program(k.second));
        }
        handle->programMap.clear();
    }
    return SUCCESS;
}

inline EE gcl_run_kernel(
    GCLHandle_t handle, Kernel kernel, U32 work_dim, U32 *gs, U32 *ls, const char *kernelName = NULL)
{
    for (U32 i = 0; i < work_dim; i++) {
        if (ls[i] != 0) {
            gs[i] = (gs[i] + ls[i] - 1) / ls[i] * ls[i];
        }
    }
    CHECK_STATUS(enqueue_ndrange_kernel(handle->queue, kernel, work_dim, NULL, gs, ls,
        handle->numWaitEvents, handle->waitEvents, handle->eventPtr));
#ifdef _DEBUG
    std::string name = "unknown kernel";
    if (kernelName) {
        name = handle->curOpName + "_" + std::string(kernelName);
    }
    double time = 0;
    CHECK_STATUS(event_counting_time(handle->eventPtr, NULL, NULL, NULL, NULL, &time));
    CHECK_STATUS(release_event(handle->eventObj));
    handle->t_execute = time;
    UNI_DETAIL_LOG("KERNEL>>> %s time:%.2fus\n", name.c_str(), time);
    CHECK_STATUS(gcl_finish(handle));
#else
    UNUSED(kernelName);
#endif
    return SUCCESS;
}

inline EE gcl_get_kernel_name(Kernel kernel, I8 *kernelName)
{
    char name[256];
    U32 len;
    CHECK_STATUS(get_kernel_name(kernel, name, &len));
    if (len > 256) {
        UNI_ERROR_LOG("KernelName length %d > 256, please reset name array length\n", len);
        CHECK_STATUS(NOT_MATCH);
    } else {
        UNI_STRCPY(kernelName, name);
    }
    return SUCCESS;
}

inline void gcl_set_kernel_ls_to_cache(
    GCLHandle_t handle, const char *kernelName, U32 gs[3], U32 ls[3])
{
    std::string name = kernelName;
    name += "_" + std::to_string(gs[0]);
    name += "_" + std::to_string(gs[1]);
    name += "_" + std::to_string(gs[2]);
    std::vector<U32> lsVec = {ls[0], ls[1], ls[2]};
    if (handle->kernelLSCache.find(name) == handle->kernelLSCache.end()) {
        handle->kernelLSCache[name] = lsVec;
    }
}

inline bool gcl_get_kernel_ls_from_cache(
    GCLHandle_t handle, const char *kernelName, U32 gs[3], U32 ls[3])
{
    std::string name = kernelName;
    name += "_" + std::to_string(gs[0]);
    name += "_" + std::to_string(gs[1]);
    name += "_" + std::to_string(gs[2]);
    if (handle->kernelLSCache.find(name) != handle->kernelLSCache.end()) {
        for (U32 i = 0; i < 3; i++) {
            ls[i] = handle->kernelLSCache[name][i];
        }
        UNI_DETAIL_LOG("get kernel %s args from cache success: gs<%d %d %d> ls<%d %d %d>\n",
            kernelName, gs[0], gs[1], gs[2], ls[0], ls[1], ls[2]);
        return true;
    } else {
        UNI_DETAIL_LOG("get kernel %s args from cache failed, try to find best ls. gs<%d %d %d>\n",
            kernelName, gs[0], gs[1], gs[2]);
        return false;
    }
}

inline U32 get_next_ls_size(U32 ls_size)
{
    return (ls_size << 1);
}

inline EE gcl_run_kernel_select_ls(GCLHandle_t handle, GCLKernelInfo *kernelInfo)
{
    UNI_DETAIL_LOG("try to find %s best args.\n", kernelInfo->name.c_str());
    auto kernel = kernelInfo->kernel;
    auto work_dim = kernelInfo->dim;
    auto gs = kernelInfo->gs;
    double minTime = DBL_MAX;
    double time;
    U32 test_ls[3] = {0};
    U32 best_ls[3] = {0};
    U32 test_gs[3] = {0};
    U32 lsMaxSize[3];
    CHECK_STATUS(gcl_get_device_max_ls_size(handle, lsMaxSize));
    U32 maxSize = 384;
    U32 gs_x = 256;
    U32 gs_y = (work_dim > 1) ? 256 : 1;
    U32 gs_z = (work_dim > 2) ? gs[2] : 1;
    for (U32 z = 1; z <= gs_z; z = get_next_ls_size(z)) {
        if (0 != gs_z % z || z > maxSize || z > lsMaxSize[2]) {
            continue;
        }
        for (U32 y = 1; y <= gs_y; y = get_next_ls_size(y)) {
            if (0 != gs_y % y || y > maxSize || y > lsMaxSize[1]) {
                continue;
            }
            for (U32 x = 1; x <= gs_x; x = get_next_ls_size(x)) {
                if (0 != gs_x % x || x > maxSize || x > lsMaxSize[0]) {
                    continue;
                }
                U32 total = x * y * z;
                if (total <= maxSize) {
                    test_gs[0] = (gs[0] + x - 1) / x * x;
                    test_gs[1] = (gs[1] + y - 1) / y * y;
                    test_gs[2] = (gs[2] + z - 1) / z * z;
                    test_ls[0] = x;
                    test_ls[1] = y;
                    test_ls[2] = z;
                    CHECK_STATUS(
                        enqueue_ndrange_kernel(handle->queue, kernel, work_dim, NULL, test_gs,
                            test_ls, handle->numWaitEvents, handle->waitEvents, handle->eventPtr));
                    CHECK_STATUS(
                        event_counting_time(handle->eventPtr, NULL, NULL, NULL, NULL, &time));
                    if (minTime > time) {
                        minTime = time;
                        best_ls[0] = test_ls[0];
                        best_ls[1] = test_ls[1];
                        best_ls[2] = test_ls[2];
                    }
                    CHECK_STATUS(release_event(handle->eventObj));
                }
            }
        }
    }
    test_ls[0] = 0;
    test_ls[1] = 0;
    test_ls[2] = 0;
    CHECK_STATUS(enqueue_ndrange_kernel(handle->queue, kernel, work_dim, NULL, gs, test_ls,
        handle->numWaitEvents, handle->waitEvents, handle->eventPtr));
    CHECK_STATUS(event_counting_time(handle->eventPtr, NULL, NULL, NULL, NULL, &time));
    if (minTime > time) {
        minTime = time;
        best_ls[0] = test_ls[0];
        best_ls[1] = test_ls[1];
        best_ls[2] = test_ls[2];
    }
    CHECK_STATUS(release_event(handle->eventObj));
    if (best_ls[0] != 0 && best_ls[1] != 0 && best_ls[2] != 0) {
        kernelInfo->gs[0] = (gs[0] + best_ls[0] - 1) / best_ls[0] * best_ls[0];
        kernelInfo->gs[1] = (gs[1] + best_ls[1] - 1) / best_ls[1] * best_ls[1];
        kernelInfo->gs[2] = (gs[2] + best_ls[2] - 1) / best_ls[2] * best_ls[2];
    }
    kernelInfo->ls[0] = best_ls[0];
    kernelInfo->ls[1] = best_ls[1];
    kernelInfo->ls[2] = best_ls[2];
    handle->t_execute = minTime;
    UNI_DETAIL_LOG("KERNEL_SELECT_LS>>> %s ls<%u %u %u> time:%.2fus\n", kernelInfo->name.c_str(),
        best_ls[0], best_ls[1], best_ls[2], minTime);
    return SUCCESS;
}

inline EE gcl_run_kernelVec_select_ls(GCLHandle_t handle, std::vector<U32> kernelIndex)
{
    if (kernelIndex.size() == 0) {
        return SUCCESS;
    }
    CHECK_STATUS(gcl_enable_queue_profiling(handle));
    for (auto index : kernelIndex) {
        auto kernelInfo = (*handle->kernelVec)[index];
        bool needSelectLs = false;
        U32 gs[3] = {0, 0, 0};
        for (U32 i = 0; i < kernelInfo.dim; i++) {
            if (kernelInfo.ls[i] == 0) {
                needSelectLs = true;
            }
            gs[i] = kernelInfo.gs[i];
        }
        if (!needSelectLs) {
            continue;
        }
        CHECK_STATUS(gcl_run_kernel_select_ls(handle, &kernelInfo));
        char kernelName[256];
        gcl_get_kernel_name(kernelInfo.kernel, kernelName);
        gcl_set_kernel_ls_to_cache(handle, kernelName, gs, kernelInfo.ls);
        (*handle->kernelVec)[index].gs[0] = kernelInfo.gs[0];
        (*handle->kernelVec)[index].gs[1] = kernelInfo.gs[1];
        (*handle->kernelVec)[index].gs[2] = kernelInfo.gs[2];
        (*handle->kernelVec)[index].ls[0] = kernelInfo.ls[0];
        (*handle->kernelVec)[index].ls[1] = kernelInfo.ls[1];
        (*handle->kernelVec)[index].ls[2] = kernelInfo.ls[2];
    }
    CHECK_STATUS(gcl_off_queue_profiling(handle));
    return SUCCESS;
}

inline EE gcl_infer_best_kernelVec_ls_with_map(
    GCLHandle_t handle, std::shared_ptr<AlgorithmMap> algoMap)
{
    std::vector<U32> kernelIndex;
    U32 len = handle->kernelVec->size();
    bool needSaveKernelThreadInfoToMap = false;
    for (U32 i = 0; i < len; i++) {
        auto kernelInfo = (*handle->kernelVec)[i];
        U32 gs[3] = {0};
        U32 ls[3] = {0};
        bool findKernelThreadInfo = false;
        findKernelThreadInfo = algoMap->getKernelThreadInfoFromMap(kernelInfo.name, gs, ls);
        U32 dim = (*handle->kernelVec)[i].dim;
        if (findKernelThreadInfo) {
            U32 cur_gs[3];
            for (U32 j = 0; j < dim; j++) {
                cur_gs[j] = kernelInfo.gs[j];
                if (ls[j] != 0) {
                    cur_gs[j] = (cur_gs[j] + ls[j] - 1) / ls[j] * ls[j];
                }
                (*handle->kernelVec)[i].gs[j] = cur_gs[j];
                (*handle->kernelVec)[i].ls[j] = ls[j];
            }
        } else {
            bool noNeedInferLS = true;
            needSaveKernelThreadInfoToMap = true;
            for (U32 j = 0; j < dim; j++) {
                gs[j] = kernelInfo.gs[j];
                ls[j] = kernelInfo.ls[j];
                if (ls[j] == 0) {
                    noNeedInferLS = false;
                }
            }
            if (!noNeedInferLS) {
                char kernelName[256];
                gcl_get_kernel_name(kernelInfo.kernel, kernelName);
                if (gcl_get_kernel_ls_from_cache(handle, kernelName, gs, ls)) {
                    for (U32 j = 0; j < dim; j++) {
                        (*handle->kernelVec)[i].ls[j] = ls[j];
                    }
                    noNeedInferLS = true;
                }
            }
            if (noNeedInferLS) {
                for (U32 j = 0; j < dim; j++) {
                    if (ls[j] > 0) {
                        (*handle->kernelVec)[i].gs[j] = (gs[j] + ls[j] - 1) / ls[j] * ls[j];
                    }
                }
            }
            if (!noNeedInferLS) {
                kernelIndex.push_back(i);
            }
        }
    }
    CHECK_STATUS(gcl_run_kernelVec_select_ls(handle, kernelIndex));
    if (needSaveKernelThreadInfoToMap) {
        for (U32 i = 0; i < len; i++) {
            auto kernelInfo = (*handle->kernelVec)[i];
            algoMap->setKernelThreadInfoToMap(kernelInfo.name, kernelInfo.gs, kernelInfo.ls);
        }
    }
    return SUCCESS;
}

#ifdef _DEBUG
inline EE gcl_run_kernel_profiling(
    GCLHandle_t handle, Kernel kernel, U32 work_dim, U32 *gs, U32 *ls, const char *kernelName = NULL)
{
    std::string name = "unknown kernel";
    if (kernelName) {
        name = kernelName;
    }
    UNI_DETAIL_LOG("KERNEL>>> %s\n", name.c_str());
    double total = 0;
    double time = 0;
    U32 loop = 10;
    for (U32 i = 0; i < loop; i++) {
        CHECK_STATUS(enqueue_ndrange_kernel(handle->queue, kernel, work_dim, NULL, gs, ls,
            handle->numWaitEvents, handle->waitEvents, handle->eventPtr));
        CHECK_STATUS(event_counting_time(handle->eventPtr, NULL, NULL, NULL, NULL, &time));
        CHECK_STATUS(release_event(handle->eventObj));
        UNI_DETAIL_LOG("    loop:%u time:%.2fus\n", i, time);
        total += time;
    }
    double average = total / loop;
    UNI_DETAIL_LOG("    average time:%.2fus\n", average);
    CHECK_STATUS(gcl_finish(handle));
    return SUCCESS;
}
#endif

inline EE gcl_create_memory(GCLHandle_t handle, GCLMem_t gclMem)
{
    GCLMemDesc_t desc = &gclMem->desc;
    switch (desc->memType) {
        case GCL_MEM_BUF: {
            CHECK_STATUS(create_buffer(
                handle->context, desc->flags, desc->byteSize, desc->host_ptr, &gclMem->mem));
            break;
        }
        case GCL_MEM_IMG_1D: {
            CHECK_STATUS(create_image1D(handle->context, desc->flags, &desc->imgFormat,
                desc->stride[0], 0, desc->host_ptr, &gclMem->mem));
            break;
        }
        case GCL_MEM_IMG_2D: {
            CHECK_STATUS(create_image2D(handle->context, desc->flags, &desc->imgFormat,
                desc->stride[0], desc->stride[1], 0, desc->host_ptr, &gclMem->mem));
            break;
        }
        case GCL_MEM_IMG_3D: {
            CHECK_STATUS(
                create_image3D(handle->context, desc->flags, &desc->imgFormat, desc->stride[0],
                    desc->stride[1], desc->stride[2], 0, 0, desc->host_ptr, &gclMem->mem));
            break;
        }
        default:
            CHECK_STATUS(NOT_SUPPORTED);
            break;
    }
    return SUCCESS;
}

inline EE gcl_trans_memory(GCLHandle_t handle,
    void *src,
    void *dst,
    U32 *size,
    GCLMemTransType type,
    cl_bool blocking,
    U32 *offset = NULL)
{
#ifdef _DEBUG
    std::string line = "DATA_TRANS1D>>> ";
#endif
    switch (type) {
        case HOST_TO_DEVICE_BUF: {
            U8 *hostPtr = (U8 *)src;
            GCLMem_t gclMem = (GCLMem_t)dst;
            U32 dstOff = (offset) ? offset[0] : 0;
            CHECK_STATUS(enqueue_write_buffer(handle->queue, gclMem->mem, blocking, dstOff, *size,
                hostPtr, handle->numWaitEvents, handle->waitEvents, handle->eventPtr));
#ifdef _DEBUG
            line += "enqueue_write_buffer";
#endif
            break;
        }
        case HOST_TO_DEVICE_IMG: {
            U8 *hostPtr = (U8 *)src;
            GCLMem_t gclMem = (GCLMem_t)dst;
            U32 origin[3] = {0, 0, 0};
            if (offset) {
                origin[0] = offset[0];
                origin[1] = offset[1];
                origin[2] = offset[2];
            }
            CHECK_STATUS(enqueue_write_image(handle->queue, gclMem->mem, blocking, origin, size, 0,
                0, hostPtr, handle->numWaitEvents, handle->waitEvents, handle->eventPtr));
#ifdef _DEBUG
            line += "enqueue_write_image";
#endif
            break;
        }
        case DEVICE_BUF_TO_HOST: {
            U8 *hostPtr = (U8 *)dst;
            GCLMem_t gclMem = (GCLMem_t)src;
            U32 srcOff = (offset) ? offset[0] : 0;
            CHECK_STATUS(enqueue_read_buffer(handle->queue, gclMem->mem, blocking, srcOff, *size,
                hostPtr, handle->numWaitEvents, handle->waitEvents, handle->eventPtr));
#ifdef _DEBUG
            line += "enqueue_read_buffer";
#endif
            break;
        }
        case DEVICE_IMG_TO_HOST: {
            U8 *hostPtr = (U8 *)dst;
            GCLMem_t gclMem = (GCLMem_t)src;
            U32 origin[3] = {0, 0, 0};
            if (offset) {
                origin[0] = offset[0];
                origin[1] = offset[1];
                origin[2] = offset[2];
            }
            CHECK_STATUS(enqueue_read_image(handle->queue, gclMem->mem, blocking, origin, size, 0,
                0, hostPtr, handle->numWaitEvents, handle->waitEvents, handle->eventPtr));
#ifdef _DEBUG
            line += "enqueue_read_image";
#endif
            break;
        }
        case DEVICE_BUF_TO_BUF: {
            GCLMem_t srcBuf = (GCLMem_t)src;
            GCLMem_t dstBuf = (GCLMem_t)dst;
            U32 srcOff = 0;
            U32 dstOff = 0;
            if (offset) {
                srcOff = offset[0];
                dstOff = offset[1];
            }
            CHECK_STATUS(enqueue_copy_buffer(handle->queue, srcBuf->mem, dstBuf->mem, srcOff,
                dstOff, *size, handle->numWaitEvents, handle->waitEvents, handle->eventPtr));
#ifdef _DEBUG
            line += "enqueue_copy_buffer";
#endif
            break;
        }
        case DEVICE_BUF_TO_IMG: {
            GCLMem_t srcBuf = (GCLMem_t)src;
            GCLMem_t dstImg = (GCLMem_t)dst;
            U32 origin[3] = {0, 0, 0};
            U32 srcOff = 0;
            if (offset) {
                srcOff = offset[0];
                origin[0] = offset[1];
                origin[1] = offset[2];
                origin[2] = offset[3];
            }
            CHECK_STATUS(enqueue_copy_buffer_to_image(handle->queue, srcBuf->mem, dstImg->mem,
                srcOff, origin, size, handle->numWaitEvents, handle->waitEvents, handle->eventPtr))
#ifdef _DEBUG
            line += "enqueue_copy_buffer_to_image";
#endif
            break;
        }
        case DEVICE_IMG_TO_BUF: {
            GCLMem_t srcImg = (GCLMem_t)src;
            GCLMem_t dstBuf = (GCLMem_t)dst;
            U32 origin[3] = {0, 0, 0};
            U32 dstOff = 0;
            if (offset) {
                origin[0] = offset[0];
                origin[1] = offset[1];
                origin[2] = offset[2];
                dstOff = offset[3];
            }
            CHECK_STATUS(enqueue_copy_image_to_buffer(handle->queue, srcImg->mem, dstBuf->mem,
                origin, size, dstOff, handle->numWaitEvents, handle->waitEvents, handle->eventPtr))
#ifdef _DEBUG
            line += "enqueue_copy_image_to_buffer";
#endif
            break;
        }
        case DEVICE_IMG_TO_IMG: {
            return NOT_SUPPORTED;
            break;
        }
        default:
            return NOT_SUPPORTED;
    }
#ifdef _DEBUG
    double time = 0;
    CHECK_STATUS(event_counting_time(handle->eventPtr, NULL, NULL, NULL, NULL, &time));
    CHECK_STATUS(release_event(handle->eventObj));
    UNI_DETAIL_LOG("%s time:%.2fus\n", line.c_str(), time);
    CHECK_STATUS(gcl_finish(handle));
#endif
    return SUCCESS;
}

inline EE gcl_trans_buffer_rect(GCLHandle_t handle,
    void *src,
    void *dst,
    U32 *host_org,
    U32 *buf_org,
    U32 *region,
    U32 host_row_pitch,
    U32 host_slice_pitch,
    U32 buf_row_pitch,
    U32 buf_slice_pitch,
    GCLMemTransType type,
    cl_bool blocking)
{
#ifdef _DEBUG
    std::string line = "DATA_TRANS2D>>> ";
#endif
    switch (type) {
        case HOST_TO_DEVICE_BUF: {
            GCLMem_t dstBuf = (GCLMem_t)dst;
            CHECK_STATUS(enqueue_write_buffer_rect(handle->queue, dstBuf->mem, blocking, buf_org,
                host_org, region, buf_row_pitch, buf_slice_pitch, host_row_pitch, host_slice_pitch,
                src, handle->numWaitEvents, handle->waitEvents, handle->eventPtr));
#ifdef _DEBUG
            line += "enqueue_write_buffer_rect";
#endif
            break;
        }
        case DEVICE_BUF_TO_HOST: {
            CHECK_STATUS(NOT_SUPPORTED);
            break;
        }
        default:
            CHECK_STATUS(NOT_SUPPORTED);
            break;
    }
#ifdef _DEBUG
    double time = 0;
    CHECK_STATUS(event_counting_time(handle->eventPtr, NULL, NULL, NULL, NULL, &time));
    CHECK_STATUS(release_event(handle->eventObj));
    UNI_DETAIL_LOG("%s time:%.2fus\n", line.c_str(), time);
    CHECK_STATUS(gcl_finish(handle));
#endif
    return SUCCESS;
}

inline EE gcl_map_memory(
    GCLHandle_t handle, GCLMem_t gclMem, U32 *offset, U32 *size, cl_map_flags flags, cl_bool blocking)
{
#ifdef _DEBUG
    std::string line = "DATAT_MAP>>> ";
#endif
    if (gclMem->desc.memType == GCL_MEM_BUF) {
        U8 *map_ptr;
        CHECK_STATUS(enqueue_map_buffer(handle->queue, gclMem->mem, blocking, flags, *offset, *size,
            handle->numWaitEvents, handle->waitEvents, handle->eventPtr, (void **)&map_ptr));
        gclMem->mapPtrArray.push_back(map_ptr);
#ifdef _DEBUG
        line += "enqueue_map_buffer";
#endif
    } else {
        return NOT_SUPPORTED;
    }
#ifdef _DEBUG
    double time = 0;
    CHECK_STATUS(event_counting_time(handle->eventPtr, NULL, NULL, NULL, NULL, &time));
    CHECK_STATUS(release_event(handle->eventObj));
    UNI_DETAIL_LOG("%s time:%.2fus\n", line.c_str(), time);
    CHECK_STATUS(gcl_finish(handle));
#endif
    return SUCCESS;
}

inline EE gcl_fill_memory_zero(GCLHandle_t handle, GCLMem_t gclMem)
{
#ifdef _DEBUG
    std::string line = "DATA_SET>>> ";
#endif
    if (gclMem->desc.memType == GCL_MEM_BUF) {
#ifdef _DEBUG
        line += "enqueue_fill_buffer";
#endif
        U8 pat_val = 0;
        CHECK_STATUS(enqueue_fill_buffer(handle->queue, gclMem->mem, &pat_val, sizeof(pat_val), 0,
            gclMem->desc.byteSize, handle->numWaitEvents, handle->waitEvents, handle->eventPtr));
    } else {
#ifdef _DEBUG
        line += "enqueue_fill_image";
#endif
        F32 color[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        U32 origin[3] = {0, 0, 0};
        U32 region[3];
        region[0] = gclMem->desc.stride[0];
        region[1] = gclMem->desc.stride[1];
        region[2] = gclMem->desc.stride[2];
        CHECK_STATUS(enqueue_fill_image(handle->queue, gclMem->mem, color, origin, region,
            handle->numWaitEvents, handle->waitEvents, handle->eventPtr));
    }
#ifdef _DEBUG
    double time = 0;
    CHECK_STATUS(event_counting_time(handle->eventPtr, NULL, NULL, NULL, NULL, &time));
    CHECK_STATUS(release_event(handle->eventObj));
    UNI_DETAIL_LOG("%s time:%.2fus\n", line.c_str(), time);
    CHECK_STATUS(gcl_finish(handle));
#endif
    return SUCCESS;
}

inline EE gcl_get_mem_size(GCLMem_t gclMem, U32 *size)
{
    CHECK_STATUS(get_memory_size(gclMem->mem, size));
    return SUCCESS;
}

inline EE gcl_create_sub_buffer(U32 size, U32 *offset, GCLMem_t src, Mem *subbuf)
{
    CHECK_STATUS(create_sub_buffer(src->mem, CL_MEM_READ_WRITE, *offset, size, subbuf));
    src->subMem.push_back(*subbuf);
    *offset += (size + 127) / 128 * 128;
    return SUCCESS;
}
#ifdef __cplusplus
}
#endif

template <typename Tuple, U32 N>
struct DummpyWrapper {
    static void set_kernel_arg_wrapper(Kernel kernel, const Tuple &t)
    {
        DummpyWrapper<Tuple, N - 1>::set_kernel_arg_wrapper(kernel, t);
        auto arg = std::get<N - 1>(t);
        set_kernel_arg(kernel, N - 1, sizeof(arg), (void *)&arg);
    }
};

template <typename Tuple>
struct DummpyWrapper<Tuple, 0> {
    static void set_kernel_arg_wrapper(Kernel kernel, const Tuple &t)
    {
        UNUSED(kernel);
        UNUSED(t);
    }
};

template <typename... Args>
inline EE gcl_set_kernelArgs(Kernel kernel, Args... args)
{
    std::tuple<Args...> t = std::make_tuple(args...);
    DummpyWrapper<decltype(t), sizeof...(Args)>::set_kernel_arg_wrapper(kernel, t);
    return SUCCESS;
}

inline EE gcl_get_image_size(GCLMem_t gclMem, U32 *width, U32 *height, U32 *depth)
{
    CHECK_STATUS(get_image_size(gclMem->mem, width, height, depth));
    return SUCCESS;
}

inline void gcl_set_runInfo_to_cache(
    GCLHandle_t handle, std::vector<I32> flag, ForwardRunInfoMali runInfo)
{
    if (handle->runInfoCache.find(flag) == handle->runInfoCache.end()) {
        handle->runInfoCache[flag] = runInfo;
    }
}

inline bool gcl_get_runInfo_from_cache(
    GCLHandle_t handle, std::vector<I32> flag, ForwardRunInfoMali_t runInfo)
{
    if (handle->runInfoCache.find(flag) != handle->runInfoCache.end()) {
        *runInfo = handle->runInfoCache[flag];
        UNI_DETAIL_LOG("get forward run info from cache success\n");
        return true;
    } else {
        UNI_DETAIL_LOG("get forward run info from cache fail, try to find best forward run info\n");
        return false;
    }
}

#ifdef _DEBUG
template <typename T>
inline EE gcl_print_memory(GCLHandle_t handle, GCLMem_t gclMem, const char *gclMemName = NULL)
{
    UNUSED(handle);
    UNUSED(gclMem);
    UNUSED(gclMemName);
    return SUCCESS;
}

template <typename T>
inline EE gcl_print_buffer(GCLHandle_t handle, Mem buf, U32 num, const char *bufferName = NULL)
{
    UNUSED(handle);
    UNUSED(buf);
    UNUSED(num);
    return SUCCESS;
}

template <typename T>
inline EE gcl_check_mem(GCLHandle_t handle,
    Mem mem,
    U32 *elementsNum,
    GCLMemType type,
    bool write2bin,
    const char *dataName = NULL)
{
    U32 num;
    T *p;
    F32 *pt;
    if (type == GCL_MEM_BUF) {
        num = *elementsNum;
        U32 size = num * sizeof(T);
        p = new U8[size];
        pt = new F32[num];
        CHECK_STATUS(enqueue_read_buffer(handle->queue, mem, CL_TRUE, 0, size, p,
            handle->numWaitEvents, handle->waitEvents, handle->eventPtr));
    } else {
        num = elementsNum[0] * elementsNum[1] * elementsNum[2] * 4;
        U32 size = num * sizeof(T);
        p = new U8[size];
        pt = new F32[num];
        U32 origin[3] = {0, 0, 0};
        CHECK_STATUS(enqueue_read_image(handle->queue, mem, CL_TRUE, origin, elementsNum, 0, 0, p,
            handle->numWaitEvents, handle->waitEvents, handle->eventPtr));
    }
    for (U32 i = 0; i < num; i++) {
        pt[i] = p[i];
    }

    if (write2bin) {
        if (!dataName) {
            dataName = "unknow";
        }
        std::string fileName = dataName;
        replace(fileName.begin(), fileName.end(), '/', '_');
        replace(fileName.begin(), fileName.end(), '.', '_');
        replace(fileName.begin(), fileName.end(), ' ', '_');
        fileName += "_gpu";
        fileName += ".out";
        FILE *outfile = fopen(fileName.c_str(), "wb");
        if (outfile == NULL) {
            UNI_ERROR_LOG("can not open file %s.\n", fileName.c_str());
        }
        fwrite(pt, sizeof(float), num, outfile);
        fclose(outfile);
    } else {
        U32 len = num;
        std::string line = "";
        for (U32 i = 0; i < len; i++) {
            line = line + std::to_string(pt[i]) + " ";
        }
        UNI_DETAIL_LOG("%s\n", line.c_str());
    }
    delete[] p;
    delete[] pt;
    return SUCCESS;
}

/*ptrType:
* GPU: 0
* CPU: 1
*/
inline std::string gcl_string(GCLHandle_t handle,
    GCLMemDesc memDesc,
    void *ptr,
    U32 len,
    U32 ptrType,
    bool write2bin = false,
    const char *dataName = NULL)
{
    std::string line = "";
    DataType tdt = memDesc.dt;
    DataFormat tdf = memDesc.df;
    U32 dims = memDesc.nDims;
    U32 tn = 1, tc = 1, th = 1, tw = 1, tt = 1;
    tw = memDesc.dims[0];
    if (dims == 5) {
        th = memDesc.dims[1];
        tt = memDesc.dims[2];
        tc = memDesc.dims[3];
        tn = memDesc.dims[4];
    } else {
        if (dims > 1) {
            th = memDesc.dims[1];
        }
        if (dims > 2) {
            tc = memDesc.dims[2];
        }
        if (dims > 3) {
            tn = memDesc.dims[3];
        }
        if (dims > 4) {
            UNI_ERROR_LOG("not support %u-dim data.\n", dims);
            return line;
        }
    }
    size_t num = tn * tc * th * tw * tt;
    if (num == 0) {
        return line;
    }
    std::string name = handle->curOpName;
    if (dataName) {
        name = dataName;
    }
    float *pt = new float[num];
    if (ptrType == 0) {
        GCLMem_t mem = (GCLMem_t)ptr;
        GCLMemDesc desc = memDesc;
        GCLMemType type = desc.memType;
        DataFormat df = desc.memFormat;
        U32 s0 = desc.stride[0];
        U32 s1 = desc.stride[1];
        U32 off0 = desc.offset[0];
        U32 off1 = desc.offset[1];
        U32 byteSize = desc.byteSize;
        U8 *p = new U8[byteSize];

        GCLMemTransType tranType = DEVICE_BUF_TO_HOST;
        U32 size[3] = {byteSize, 1, 1};
        if (type != GCL_MEM_BUF) {
            tranType = DEVICE_IMG_TO_HOST;
            size[0] = s0;
            size[1] = s1;
            size[2] = desc.stride[2];
        }
        gcl_trans_memory(handle, (void *)mem, p, size, tranType, CL_TRUE);

        if (df == DF_NCHWC4) {
            if (dims >= 4) {
                for (U32 i = 0; i < num; i++) {
                    U32 iw = i % tw;
                    U32 ih = (i / tw) % th;
                    U32 ic = i / (tw * th);
                    size_t j = ((ic / 4) * s1 + ih + off1) * s0 * 4 + (iw + off0) * 4 + (ic & 3);
                    transformToFloat(tdt, p + j * bytesOf(tdt), pt + i, 1, 1);
                }
            } else {
                UNI_ERROR_LOG("not support %u-dim data.\n", dims);
            }
        } else if (df == DF_NCHW) {
            for (U32 i = 0; i < num; i++) {
                U32 iw = i % tw;
                U32 ih = (i / tw) % th;
                U32 ic = i / (tw * th);
                size_t j = (ic * s1 + ih + off1) * s0 + (iw + off0);
                transformToFloat(tdt, p + j * bytesOf(tdt), pt + i, 1, 1);
            }
        } else if (df == DF_NHWC) {
            for (U32 i = 0; i < num; i++) {
                U32 iw = i % tw;
                U32 ih = (i / tw) % th;
                U32 ic = i / (tw * th);
                size_t j = ((ih + off1) * s1 + (iw + off0)) * s0 + ic;
                transformToFloat(tdt, p + j * bytesOf(tdt), pt + i, 1, 1);
            }
        } else if (df == DF_NORMAL) {
            for (U32 i = 0; i < num; i++) {
                transformToFloat(tdt, p + i * bytesOf(tdt), pt + i, 1, 1);
            }
        } else {
            UNI_ERROR_LOG("not support %s format data %s.\n", DataFormatName()[tdf], name.c_str());
        }
        delete[] p;
    } else if (ptrType == 1) {
        U8 *p = (U8 *)ptr;
        if (tdf == DF_NCHWC8) {
            for (U32 i = 0; i < num; i++) {
                U32 iw = i % tw;
                U32 ih = (i / tw) % th;
                U32 ic = i / (tw * th);
                size_t j = ((ic / 8) * th + ih) * tw * 8 + iw * 8 + (ic & 7);
                transformToFloat(tdt, p + j * bytesOf(tdt), pt + i, 1, 1);
            }
        } else if (tdf == DF_NORMAL || tdf == DF_NCHW) {
            for (U32 i = 0; i < num; i++) {
                transformToFloat(tdt, p + i * bytesOf(tdt), pt + i, 1, 1);
            }
        } else if (tdf == DF_MTK) {
            //write as MKT, for compare with gpu
            for (U32 i = 0; i < num; i++) {
                U32 it = i % th;
                U32 ik = i / th;
                size_t j = it * tw + ik;
                transformToFloat(tdt, p + j * bytesOf(tdt), pt + i, 1, 1);
            }
        } else {
            UNI_ERROR_LOG("not support %s format data %s.\n", DataFormatName()[tdf], name.c_str());
        }
    } else {
        UNI_ERROR_LOG("currently only support GPU(0) and CPU(1) type pointer.\n");
    }
    if (write2bin) {
        std::string fileName = name;
        replace(fileName.begin(), fileName.end(), '/', '_');
        replace(fileName.begin(), fileName.end(), '.', '_');
        replace(fileName.begin(), fileName.end(), ' ', '_');
        replace(fileName.begin(), fileName.end(), ':', '_');
        if (ptrType == 0) {
            fileName += "_gpu";
        }
        if (ptrType == 1) {
            fileName += "_cpu";
        }
        fileName += ".out";
        UNI_DETAIL_LOG("write data to file %s.\n", fileName.c_str());

        FILE *outfile = fopen(fileName.c_str(), "wb");
        if (outfile == NULL) {
            UNI_ERROR_LOG("can not open file %s.\n", fileName.c_str());
        }
        fwrite(pt, bytesOf(tdt), num, outfile);
        fclose(outfile);
    }
    if (len > num) {
        len = num;
    }
    line += "NCHW ";
    float sum = 0;
    for (U32 i = 0; i < num; i++) {
        if (i < len) {
            line += std::to_string(pt[i]) + " ";
        }
        sum += pt[i];
    }
    line += "sum:" + std::to_string(sum);
    delete[] pt;
    return line;
}
#endif
#endif
