// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef H_GCL_COMMON
#define H_GCL_COMMON

#include "uni.h"
#include "tensor_desc.h"
#include "gcl_kernel_type.h"
#include "CL/cl.h"
#include <map>
#include <unordered_map>
#include <vector>
#include <string>
#include <iostream>
/**
 * @file
 */
#define ERROR_CASE(x) \
    case x:           \
        return (#x)

#ifdef __cplusplus
extern "C" {
#endif

typedef cl_platform_id Platform;
typedef cl_device_id Device;
typedef cl_context Context;
typedef cl_command_queue CommandQueue;
typedef cl_program Program;
typedef cl_mem Mem;
typedef cl_sampler Sampler;
typedef cl_kernel Kernel;
typedef cl_event Event;
typedef cl_mem_flags MemFlags;
typedef cl_image_format ImgFormat;

inline CI8 *map_cl_error_2_string(cl_int err)
{
    switch (err) {
        ERROR_CASE(CL_SUCCESS);
        ERROR_CASE(CL_DEVICE_NOT_FOUND);
        ERROR_CASE(CL_DEVICE_NOT_AVAILABLE);
        ERROR_CASE(CL_COMPILER_NOT_AVAILABLE);
        ERROR_CASE(CL_MEM_OBJECT_ALLOCATION_FAILURE);
        ERROR_CASE(CL_OUT_OF_RESOURCES);
        ERROR_CASE(CL_OUT_OF_HOST_MEMORY);
        ERROR_CASE(CL_PROFILING_INFO_NOT_AVAILABLE);
        ERROR_CASE(CL_MEM_COPY_OVERLAP);
        ERROR_CASE(CL_IMAGE_FORMAT_MISMATCH);
        ERROR_CASE(CL_IMAGE_FORMAT_NOT_SUPPORTED);
        ERROR_CASE(CL_BUILD_PROGRAM_FAILURE);
        ERROR_CASE(CL_MAP_FAILURE);
#ifdef CL_VERSION_1_1
        ERROR_CASE(CL_MISALIGNED_SUB_BUFFER_OFFSET);
        ERROR_CASE(CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST);
#endif
#ifdef CL_VERSION_1_2
        ERROR_CASE(CL_COMPILE_PROGRAM_FAILURE);
        ERROR_CASE(CL_LINKER_NOT_AVAILABLE);
        ERROR_CASE(CL_LINK_PROGRAM_FAILURE);
        ERROR_CASE(CL_DEVICE_PARTITION_FAILED);
        ERROR_CASE(CL_KERNEL_ARG_INFO_NOT_AVAILABLE);
#endif
        ERROR_CASE(CL_INVALID_VALUE);
        ERROR_CASE(CL_INVALID_DEVICE_TYPE);
        ERROR_CASE(CL_INVALID_PLATFORM);
        ERROR_CASE(CL_INVALID_DEVICE);
        ERROR_CASE(CL_INVALID_CONTEXT);
        ERROR_CASE(CL_INVALID_QUEUE_PROPERTIES);
        ERROR_CASE(CL_INVALID_COMMAND_QUEUE);
        ERROR_CASE(CL_INVALID_HOST_PTR);
        ERROR_CASE(CL_INVALID_MEM_OBJECT);
        ERROR_CASE(CL_INVALID_IMAGE_FORMAT_DESCRIPTOR);
        ERROR_CASE(CL_INVALID_IMAGE_SIZE);
        ERROR_CASE(CL_INVALID_SAMPLER);
        ERROR_CASE(CL_INVALID_BINARY);
        ERROR_CASE(CL_INVALID_BUILD_OPTIONS);
        ERROR_CASE(CL_INVALID_PROGRAM);
        ERROR_CASE(CL_INVALID_PROGRAM_EXECUTABLE);
        ERROR_CASE(CL_INVALID_KERNEL_NAME);
        ERROR_CASE(CL_INVALID_KERNEL_DEFINITION);
        ERROR_CASE(CL_INVALID_KERNEL);
        ERROR_CASE(CL_INVALID_ARG_INDEX);
        ERROR_CASE(CL_INVALID_ARG_VALUE);
        ERROR_CASE(CL_INVALID_ARG_SIZE);
        ERROR_CASE(CL_INVALID_KERNEL_ARGS);
        ERROR_CASE(CL_INVALID_WORK_DIMENSION);
        ERROR_CASE(CL_INVALID_WORK_GROUP_SIZE);
        ERROR_CASE(CL_INVALID_WORK_ITEM_SIZE);
        ERROR_CASE(CL_INVALID_GLOBAL_OFFSET);
        ERROR_CASE(CL_INVALID_EVENT_WAIT_LIST);
        ERROR_CASE(CL_INVALID_EVENT);
        ERROR_CASE(CL_INVALID_OPERATION);
        ERROR_CASE(CL_INVALID_GL_OBJECT);
        ERROR_CASE(CL_INVALID_BUFFER_SIZE);
        ERROR_CASE(CL_INVALID_MIP_LEVEL);
        ERROR_CASE(CL_INVALID_GLOBAL_WORK_SIZE);
#ifdef CL_VERSION_1_1
        ERROR_CASE(CL_INVALID_PROPERTY);
#endif
#ifdef CL_VERSION_1_2
        ERROR_CASE(CL_INVALID_IMAGE_DESCRIPTOR);
        ERROR_CASE(CL_INVALID_COMPILER_OPTIONS);
        ERROR_CASE(CL_INVALID_LINKER_OPTIONS);
        ERROR_CASE(CL_INVALID_DEVICE_PARTITION_COUNT);
#endif
#ifdef CL_VERSION_2_0
        ERROR_CASE(CL_INVALID_PIPE_SIZE);
        ERROR_CASE(CL_INVALID_DEVICE_QUEUE);
#endif
#ifdef CL_VERSION_2_2
        ERROR_CASE(CL_INVALID_SPEC_ID);
        ERROR_CASE(CL_MAX_SIZE_RESTRICTION_EXCEEDED);
#endif

        default:
            return "CL_UNKNOW_ERROR";
    }
}

#define map_cl_error_2_ee(err)                                                \
    {                                                                         \
        if (err == 0) {                                                       \
            return SUCCESS;                                                   \
        } else {                                                              \
            UNI_ERROR_LOG("GCLAPI error: %s.\n", map_cl_error_2_string(err)); \
            return GCL_ERROR;                                                 \
        }                                                                     \
    }

inline EE has_dedicated_local(Device device, I32 *b)
{
    void *value;
    I32 ret = clGetDeviceInfo(device, CL_DEVICE_LOCAL_MEM_TYPE, sizeof(void *), &value, nullptr);
    if (CL_SUCCESS == ret) {
        *b = (*((cl_device_local_mem_type *)value) == CL_LOCAL);
    }
    free(value);
    map_cl_error_2_ee(ret);
}

typedef enum {
    HOST_TO_DEVICE_BUF = 0,
    HOST_TO_DEVICE_IMG = 1,
    DEVICE_BUF_TO_HOST = 2,
    DEVICE_IMG_TO_HOST = 3,
    DEVICE_BUF_TO_BUF = 4,
    DEVICE_BUF_TO_IMG = 5,
    DEVICE_IMG_TO_BUF = 6,
    DEVICE_IMG_TO_IMG = 7
} GCLMemTransType;
/**
 *@ struct define
 **/
struct GCLKernelInfo {
    Kernel kernel = NULL;
    U32 dim = 0;
    U32 gs[3] = {0};
    U32 ls[3] = {0};
    std::string name;
};

typedef struct {
    I32 algorithm;
    U32 best_h[6];
    U32 best_c[6];
    U32 best_k[6];
} ForwardRunInfoMali;
typedef ForwardRunInfoMali *ForwardRunInfoMali_t;

struct GCLHandle {
    Platform *platforms;
    U32 numPlatform;
    U32 platformId;

    Device *devices;
    U32 numDevice;
    U32 deviceId;
    cl_device_type deviceType;
    U32 device_max_ls_size[3];
    U32 device_max_cu;
    U32 device_max_work_group;
    U32 device_max_image3d_size[3];
    bool useQualcommDev;

    Context context;
    CommandQueue queue;
    CommandQueue queue_profiling;
    bool existProfilingQueue;

    Event eventObj;
    Event *eventPtr;
    U32 numWaitEvents;
    Event *waitEvents;
    double t_execute;
    double t_total;

    std::string deviceName;
    std::map<std::string, Kernel> kernelMap;
    std::map<std::string, Program> programMap;
    std::map<std::vector<I32>, ForwardRunInfoMali> runInfoCache;
    std::map<std::string, std::vector<U32>> kernelLSCache;
    std::vector<GCLKernelInfo> *kernelVec;
    std::string curOpName;
    void *kernel_source;
    void *kernel_binmap_handle;
    void *kernel_binmap;
    bool useBinMap;
    std::string common_source_opt;
    std::string common_source_ext;
    Program source_head[1];
    CI8 *source_head_name[1];
};

typedef struct GCLHandle *GCLHandle_t;

struct GCLHandleConfig {
    CI8 *deviceBinmapName;
};

typedef GCLHandleConfig *GCLHandleConfig_t;

typedef struct {
    GCLHandle_t handle;
    GCLMemDesc_t gclmemFilterDesc;
    ForwardRunInfoMali_t forwardRunInfo;
} MaliPara;
typedef MaliPara *MaliPara_t;

#ifdef __cplusplus
}
#endif
#endif
