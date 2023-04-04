// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _H_BOLT_C_COMMON
#define _H_BOLT_C_COMMON

#include "../api/c/bolt.h"
#include "inference.hpp"

struct ModelHandleInner {
    void *ms;
    void *cnn;
    HARDWARE_TYPE device;
    void *algoPath;
    bool useFileStream;
};

typedef struct DataDesc {
    DataType dt;
    DataFormat df;
    U32 nDims = 0;
    U32 dims[DIM_LEN] = {0};
    char name[NAME_LEN] = {0};
    void *data;
} DataDesc;

typedef struct {
    U32 num_data;
    DataDesc *data;
    HARDWARE_TYPE device;
} ResultHandleInner;

inline DataType DATA_TYPE2DataType(DATA_TYPE dt_user)
{
    static std::map<DATA_TYPE, DataType> m = {
        {FP_32, DT_F32},
        {FP_16, DT_F16},
        {INT_32, DT_I32},
        {UINT_32, DT_U32},
        {INT_8, DT_I8},
        {UINT_8, DT_U8},
    };
    DataType ret = DT_F32;
    if (m.find(dt_user) != m.end()) {
        ret = m[dt_user];
    } else {
        UNI_ERROR_LOG("C API can not recognize enum DATA_TYPE %d.\n", dt_user);
    }
    return ret;
}

inline DATA_TYPE DataType2DATA_TYPE(DataType dt_bolt)
{
    static std::map<DataType, DATA_TYPE> m = {
        {DT_F32, FP_32},
        {DT_F16, FP_16},
        {DT_I32, INT_32},
        {DT_U32, UINT_32},
        {DT_I8, INT_8},
        {DT_U8, UINT_8},
    };
    DATA_TYPE ret = FP_32;
    if (m.find(dt_bolt) != m.end()) {
        ret = m[dt_bolt];
    } else {
        UNI_ERROR_LOG("C API can not process inner DataType %s.\n", DataTypeName()[dt_bolt]);
    }
    return ret;
}

inline DataFormat DATA_FORMAT2DataFormat(DATA_FORMAT df_user)
{
    static std::map<DATA_FORMAT, DataFormat> m = {
        {NCHW, DF_NCHW},
        {NHWC, DF_NHWC},
        {NCHWC8, DF_NCHWC8},
        {NCHWC4, DF_NCHWC4},
        {MTK, DF_MTK},
        {NORMAL, DF_NORMAL},
        {SCALAR, DF_SCALAR},
    };
    DataFormat ret = DF_NCHW;
    if (m.find(df_user) != m.end()) {
        ret = m[df_user];
    } else {
        UNI_ERROR_LOG("C API can not recognize enum DATA_FORMAT %d.\n", df_user);
    }
    return ret;
}

inline DATA_FORMAT DataFormat2DATA_FORMAT(DataFormat df_bolt)
{
    static std::map<DataFormat, DATA_FORMAT> m = {
        {DF_NCHW, NCHW},
        {DF_NHWC, NHWC},
        {DF_NCHWC8, NCHWC8},
        {DF_NCHWC4, NCHWC4},
        {DF_MTK, MTK},
        {DF_NORMAL, NORMAL},
        {DF_SCALAR, SCALAR},
    };
    DATA_FORMAT ret = NCHW;
    if (m.find(df_bolt) != m.end()) {
        ret = m[df_bolt];
    } else {
        UNI_ERROR_LOG("C API can not process inner DataFormat %s.\n", DataFormatName()[df_bolt]);
    }
    return ret;
}

inline AffinityPolicy AFFINITY_TYPE2AffinityPolicy(AFFINITY_TYPE affinity)
{
    static std::map<AFFINITY_TYPE, AffinityPolicy> m = {
        {CPU, AFFINITY_CPU},
        {CPU_HIGH_PERFORMANCE, AFFINITY_CPU_HIGH_PERFORMANCE},
        {CPU_LOW_POWER, AFFINITY_CPU_LOW_POWER},
        {GPU, AFFINITY_GPU},
    };
    AffinityPolicy ret = AFFINITY_CPU_HIGH_PERFORMANCE;
    if (m.find(affinity) != m.end()) {
        ret = m[affinity];
    } else {
        UNI_ERROR_LOG("C API can not recognize enum AFFINITY_TYPE %d.\n", affinity);
    }
    return ret;
}

inline Arch HARDWARE_TYPE2Arch(HARDWARE_TYPE device)
{
    static std::map<HARDWARE_TYPE, Arch> m = {
        {CPU_ARM_V7, ARM_V7},
        {CPU_ARM_V8, ARM_V8},
        {CPU_ARM_A55, ARM_A55},
        {CPU_ARM_A76, ARM_A76},
        {GPU_MALI, MALI},
        {GPU_QUALCOMM, QUALCOMM},
        {CPU_X86_AVX2, X86_AVX2},
        {CPU_X86_AVX512, X86_AVX512},
        {CPU_SERIAL, CPU_GENERAL},
    };
    Arch ret = ARM_V8;
    if (m.find(device) != m.end()) {
        ret = m[device];
    } else {
        UNI_ERROR_LOG("C API can not recognize enum HARDWARE_TYPE %d.\n", device);
    }
    return ret;
}

inline HARDWARE_TYPE Arch2HARDWARE_TYPE(Arch arch)
{
    static std::map<Arch, HARDWARE_TYPE> m = {
        {ARM_V7, CPU_ARM_V7},
        {ARM_V8, CPU_ARM_V8},
        {ARM_A55, CPU_ARM_A55},
        {ARM_A76, CPU_ARM_A76},
        {MALI, GPU_MALI},
        {QUALCOMM, GPU_QUALCOMM},
        {X86_AVX2, CPU_X86_AVX2},
        {X86_AVX512, CPU_X86_AVX512},
        {CPU_GENERAL, CPU_SERIAL},
    };
    HARDWARE_TYPE ret = CPU_ARM_V8;
    if (m.find(arch) != m.end()) {
        ret = m[arch];
    } else {
        UNI_ERROR_LOG("C API can not process inner Arch %s.\n", ArchName()[arch]);
    }
    return ret;
}

inline void TensorDesc2DataDesc(TensorDesc srcDesc, DataDesc *dstDesc)
{
    dstDesc->dt = srcDesc.dt;
    dstDesc->df = srcDesc.df;
    dstDesc->nDims = srcDesc.nDims;
    if (srcDesc.nDims > DIM_LEN) {
        UNI_ERROR_LOG(
            "C API DataDesc only support %d dimensions, not %d.\n", DIM_LEN, srcDesc.nDims);
    }
    for (U32 i = 0; i < srcDesc.nDims; i++) {
        dstDesc->dims[i] = srcDesc.dims[srcDesc.nDims - 1 - i];
    }
    for (int i = srcDesc.nDims; i < DIM_LEN; i++) {
        dstDesc->dims[i] = 1;
    }
}

inline TensorDesc DataDesc2TensorDesc(DataDesc a)
{
    TensorDesc desc;
    desc.dt = a.dt;
    desc.df = a.df;
    desc.nDims = a.nDims;
    for (U32 i = 0; i < a.nDims; i++) {
        desc.dims[i] = a.dims[a.nDims - 1 - i];
    }
    return desc;
}

inline void assert_not_nullptr(const char *funcName, const char *ptrName, const void *ptr)
{
#ifndef _USE_LITE
    if (ptr == NULL) {
        UNI_WARNING_LOG("C API %s received null ptr %s.\n", funcName, ptrName);
    }
#endif
}

inline void print_model_handle(ModelHandleInner *handle)
{
#ifndef _USE_LITE
    if (handle == nullptr) {
        UNI_DEBUG_LOG("ModelHandle %p\n", handle);
    } else {
        UNI_DEBUG_LOG("ModelHandle %p(modelspec:%p engine:%p device:%d algorithm:%s file "
                      "stream:%d)\n",
            handle, handle->ms, handle->cnn, handle->device, (const char *)handle->algoPath,
            handle->useFileStream);
    }
#endif
}

inline void print_result_handle(ResultHandleInner *handle)
{
#ifndef _USE_LITE
    if (handle == nullptr) {
        UNI_DEBUG_LOG("ResultHandle %p\n", handle);
    } else {
        UNI_DEBUG_LOG("ResultHandle %p(num:%u data:%p device:%d)\n", handle, handle->num_data,
            handle->data, handle->device);
    }
#endif
}
#endif
