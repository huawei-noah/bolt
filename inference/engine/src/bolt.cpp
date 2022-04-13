// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "../api/c/bolt.h"
#include "inference.hpp"

#define NAME_VALUE_PAIR(x) #x, x
const int DataDescMaxDims = 8;

struct ModelHandleInner {
    void *ms;
    void *cnn;
    DEVICE_TYPE deviceType;
    void *algoPath;
    bool useFileStream;
};

typedef struct DataDesc {
    U32 dims[DataDescMaxDims] = {0};
    char name[NAME_LEN] = {0};
    DataType dt;
    DataFormat df;
    void *dataPtr;
} DataDesc;

typedef struct {
    U32 num_outputs;
    DataDesc *outputArr;
    DEVICE_TYPE deviceType;
} ResultHandleInner;

inline static DataType DATA_TYPE2DataType(DATA_TYPE dt_user)
{
    DataType ret = DT_F32;
    switch (dt_user) {
        case FP_32:
            ret = DT_F32;
            break;
#ifdef _USE_FP16
        case FP_16:
            ret = DT_F16;
            break;
#endif
        case INT_32:
            ret = DT_I32;
            break;
        case UINT_32:
            ret = DT_U32;
            break;
        default:
            UNI_ERROR_LOG("C API can not recognize enum DATA_TYPE %d.\n", dt_user);
            break;
    }
    return ret;
}

inline static DATA_TYPE DataType2DATA_TYPE(DataType dt_bolt)
{
    DATA_TYPE ret = FP_32;
    switch (dt_bolt) {
        case DT_F32:
            ret = FP_32;
            break;
#ifdef _USE_FP16
        case DT_F16:
            ret = FP_16;
            break;
#endif
        case DT_I32:
            ret = INT_32;
            break;
        case DT_U32:
            ret = UINT_32;
            break;
        default:
            UNI_ERROR_LOG("C API can not process inner DataType %s.\n", DataTypeName()[dt_bolt]);
            break;
    }
    return ret;
}

inline static DataFormat DATA_FORMAT2DataFormat(DATA_FORMAT df_user)
{
    DataFormat ret = DF_NCHW;
    switch (df_user) {
        case NCHW:
            ret = DF_NCHW;
            break;
        case NHWC:
            ret = DF_NHWC;
            break;
        case NCHWC8:
            ret = DF_NCHWC8;
            break;
        case MTK:
            ret = DF_MTK;
            break;
        case NORMAL:
            ret = DF_NORMAL;
            break;
        default: {
            UNI_ERROR_LOG("C API can not recognize enum DATA_FORMAT %d.\n", df_user);
            break;
        }
    }
    return ret;
}

inline static DATA_FORMAT DataFormat2DATA_FORMAT(DataFormat df_bolt)
{
    DATA_FORMAT ret = NCHW;
    switch (df_bolt) {
        case DF_NCHW:
            ret = NCHW;
            break;
        case DF_NHWC:
            ret = NHWC;
            break;
        case DF_NCHWC8:
            ret = NCHWC8;
            break;
        case DF_NCHWC4:
            ret = NCHWC4;
            break;
        case DF_MTK:
            ret = MTK;
            break;
        case DF_NORMAL:
            ret = NORMAL;
            break;
        default: {
            UNI_ERROR_LOG("C API can not process inner DataFormat %s.\n", DataFormatName()[df_bolt]);
            break;
        }
    }
    return ret;
}

inline static AffinityPolicy AFFINITY_TYPE2AffinityPolicy(AFFINITY_TYPE affinity)
{
    AffinityPolicy ret = AFFINITY_CPU_HIGH_PERFORMANCE;
    switch (affinity) {
        case CPU:
            ret = AFFINITY_CPU;
            break;
        case CPU_HIGH_PERFORMANCE:
            ret = AFFINITY_CPU_HIGH_PERFORMANCE;
            break;
        case CPU_LOW_POWER:
            ret = AFFINITY_CPU_LOW_POWER;
            break;
        case GPU:
            ret = AFFINITY_GPU;
            break;
        default: {
            UNI_ERROR_LOG("C API can not recognize enum AFFINITY_TYPE %d.\n", affinity);
            break;
        }
    }
    return ret;
}

inline static Arch DEVICE_TYPE2Arch(DEVICE_TYPE device)
{
    Arch ret = ARM_V8;
    switch (device) {
        case CPU_ARM_V7:
            ret = ARM_V7;
            break;
        case CPU_ARM_V8:
            ret = ARM_V8;
            break;
        case CPU_ARM_A55:
            ret = ARM_A55;
            break;
        case CPU_ARM_A76:
            ret = ARM_A76;
            break;
        case GPU_MALI:
            ret = MALI;
            break;
        case GPU_QUALCOMM:
            ret = QUALCOMM;
            break;
        case CPU_X86_AVX2:
            ret = X86_AVX2;
            break;
        case CPU_X86_AVX512:
            ret = X86_AVX512;
            break;
        case CPU_SERIAL:
            ret = CPU_GENERAL;
            break;
        default: {
            UNI_ERROR_LOG("C API can not recognize enum DEVICE_TYPE %d.\n", device);
            break;
        }
    }
    return ret;
}

inline static DEVICE_TYPE Arch2DEVICE_TYPE(Arch arch)
{
    DEVICE_TYPE ret = CPU_ARM_V8;
    switch (arch) {
        case ARM_V7:
            ret = CPU_ARM_V7;
            break;
        case ARM_V8:
            ret = CPU_ARM_V8;
            break;
        case ARM_A55:
            ret = CPU_ARM_A55;
            break;
        case ARM_A76:
            ret = CPU_ARM_A76;
            break;
        case MALI:
            ret = GPU_MALI;
            break;
        case QUALCOMM:
            ret = GPU_QUALCOMM;
            break;
        case X86_AVX2:
            ret = CPU_X86_AVX2;
            break;
        case CPU_GENERAL:
            ret = CPU_SERIAL;
            break;
        case X86_AVX512:
            ret = CPU_X86_AVX512;
            break;
        default: {
            UNI_ERROR_LOG("C API can not process inner Arch %s.\n", ArchName()[arch]);
            break;
        }
    }
    return ret;
}

inline static void TensorDesc2DataDesc(TensorDesc srcDesc, DataDesc *dstDesc)
{
    dstDesc->dt = srcDesc.dt;
    dstDesc->df = srcDesc.df;
    if (srcDesc.nDims > DataDescMaxDims) {
        UNI_ERROR_LOG(
            "C API DataDesc only support %d dimensions, not %d.\n", DataDescMaxDims, srcDesc.nDims);
    }
    for (U32 i = 0; i < srcDesc.nDims; i++) {
        dstDesc->dims[i] = srcDesc.dims[srcDesc.nDims - 1 - i];
    }
    for (int i = srcDesc.nDims; i < DataDescMaxDims; i++) {
        dstDesc->dims[i] = 1;
    }
}

inline static void assert_not_nullptr(const char *funcName, const char *ptrName, const void *ptr)
{
    if (ptr == NULL) {
        UNI_WARNING_LOG("C API %s received null ptr %s.\n", funcName, ptrName);
    }
}

static void print_model_handle(ModelHandleInner *handle)
{
    if (handle == nullptr) {
        UNI_DEBUG_LOG("ModelHandle %p\n", handle);
    } else {
        UNI_DEBUG_LOG("ModelHandle %p(modelspec:%p engine:%p device:%d algorithm:%s file "
                      "stream:%d)\n",
            handle, handle->ms, handle->cnn, handle->deviceType, (const char *)handle->algoPath,
            handle->useFileStream);
    }
}

static void print_result_handle(ResultHandleInner *handle)
{
    if (handle == nullptr) {
        UNI_DEBUG_LOG("ResultHandle %p\n", handle);
    } else {
        UNI_DEBUG_LOG("ResultHandle %p(num:%u data:%p device:%d)\n", handle, handle->num_outputs,
            handle->outputArr, handle->deviceType);
    }
}

const char *const *GetDataTypeString()
{
    static const char *const names[] = {"FP_32", "FP_16", "INT_32", "UINT_32"};
    return names;
}

const char *const *GetDataFormatString()
{
    static const char *const names[] = {"NCHW", "NHWC", "NCHWC8", "MTK", "NORMAL"};
    return names;
}

void GetGpuDeviceName(char *gpuDeviceName)
{
    UNI_DEBUG_LOG("C API %s(%p)...\n", __FUNCTION__, gpuDeviceName);
    std::string deviceName = "unKnown";
#ifdef _USE_GPU
    deviceName = OCLContext::getInstance().handle->deviceName;
#endif
    UNI_STRCPY(gpuDeviceName, deviceName.c_str());
    UNI_DEBUG_LOG("C API %s(%s) end.\n", __FUNCTION__, gpuDeviceName);
}

ModelHandle CreateModel(const char *modelPath, AFFINITY_TYPE affinity, const char *algorithmMapPath)
{
    UNI_DEBUG_LOG("C API %s(%p, %d, %p)...\n", __FUNCTION__, modelPath, affinity, algorithmMapPath);
    assert_not_nullptr(__FUNCTION__, NAME_VALUE_PAIR(modelPath));
    ModelHandleInner *handle = nullptr;
    if (modelPath != nullptr) {
        ModelSpec *ms = new ModelSpec();
        if (SUCCESS != deserialize_model_from_file(modelPath, ms)) {
            UNI_WARNING_LOG("C API %s failed to load model %s.\n", __FUNCTION__, modelPath);
            delete ms;
        } else {
            CNN *cnn = new CNN(AFFINITY_TYPE2AffinityPolicy(affinity), ms->dt, ms->model_name);
            cnn->sort_operators_sequential(ms);
            cnn->initialize_ops(ms);

            handle = new ModelHandleInner();
            handle->cnn = (void *)cnn;
            handle->ms = (void *)ms;
            handle->deviceType = Arch2DEVICE_TYPE(cnn->get_runtime_device());
            handle->algoPath = (void *)algorithmMapPath;
            handle->useFileStream = false;
        }
    }
    UNI_DEBUG_LOG("C API %s(%p) end.\n", __FUNCTION__, handle);
    print_model_handle(handle);
    return (ModelHandle)handle;
}

ModelHandle CloneModel(ModelHandle ih)
{
    UNI_DEBUG_LOG("C API %s(%p)...\n", __FUNCTION__, ih);
    ModelHandleInner *handle = (ModelHandleInner *)ih;
    print_model_handle(handle);
    assert_not_nullptr(__FUNCTION__, "ModelHandle", handle);
    ModelHandleInner *cloneHandle = nullptr;
    if (handle != nullptr) {
        CNN *cnn = (CNN *)handle->cnn;
        assert_not_nullptr(__FUNCTION__, "ModelHandle.cnn", cnn);
        if (cnn != nullptr) {
            cloneHandle = new ModelHandleInner();
            *cloneHandle = *handle;
            CNN *cloneCnn = new CNN();
            *cloneCnn = cnn->clone();
            cloneHandle->cnn = cloneCnn;
        }
    }
    UNI_DEBUG_LOG("C API %s(%p) end.\n", __FUNCTION__, cloneHandle);
    print_model_handle(cloneHandle);
    return (ModelHandle)cloneHandle;
}

ModelHandle CreateModelWithFileStream(
    const char *modelFileStream, AFFINITY_TYPE affinity, const char *algorithmMapFileStream)
{
    UNI_DEBUG_LOG("C API %s(%p, %d, %p)...\n", __FUNCTION__, modelFileStream, affinity,
        algorithmMapFileStream);
    assert_not_nullptr(__FUNCTION__, NAME_VALUE_PAIR(modelFileStream));
    ModelHandleInner *handle = nullptr;
    if (modelFileStream != nullptr) {
        ModelSpec *ms = new ModelSpec();
        if (SUCCESS != deserialize_model_from_file(modelFileStream, ms, true)) {
            UNI_WARNING_LOG("C API %s failed to parse model.\n", __FUNCTION__);
            delete ms;
        } else {
            CNN *cnn = new CNN(AFFINITY_TYPE2AffinityPolicy(affinity), ms->dt, ms->model_name);
            cnn->sort_operators_sequential(ms);
            cnn->initialize_ops(ms);

            handle = new ModelHandleInner();
            handle->cnn = (void *)cnn;
            handle->ms = (void *)ms;
            handle->deviceType = Arch2DEVICE_TYPE(cnn->get_runtime_device());
            handle->algoPath = (void *)algorithmMapFileStream;
            handle->useFileStream = true;
        }
    }
    UNI_DEBUG_LOG("C API %s(%p) end.\n", __FUNCTION__, handle);
    print_model_handle(handle);
    return (ModelHandle)handle;
}

int GetNumInputsFromModel(ModelHandle ih)
{
    UNI_DEBUG_LOG("C API %s(%p)...\n", __FUNCTION__, ih);
    ModelHandleInner *ihInfo = (ModelHandleInner *)ih;
    assert_not_nullptr(__FUNCTION__, "ModelHandle", ihInfo);
    print_model_handle(ihInfo);
    int ret = 0;
    if (ihInfo != nullptr) {
        CNN *cnn = (CNN *)ihInfo->cnn;
        assert_not_nullptr(__FUNCTION__, "ModelHandle.cnn", cnn);
        if (cnn != nullptr) {
            ret = (cnn->get_input_desc()).size();
        }
    }
    UNI_DEBUG_LOG("C API %s(%d) end.\n", __FUNCTION__, ret);
    return ret;
}

void GetInputDataInfoFromModel(ModelHandle ih,
    int num_inputs,
    char **name,
    int *n,
    int *c,
    int *h,
    int *w,
    DATA_TYPE *dt,
    DATA_FORMAT *df)
{
    UNI_DEBUG_LOG("C API %s...\n", __FUNCTION__);
    GetInputDataInfoFromModel5D(ih, num_inputs, name, n, c, NULL, h, w, dt, df);
    UNI_DEBUG_LOG("C API %s end.\n", __FUNCTION__);
}

void GetInputDataInfoFromModel5D(ModelHandle handle,
    int num_inputs,
    char **name,
    int *n,
    int *c,
    int *t,
    int *h,
    int *w,
    DATA_TYPE *dt,
    DATA_FORMAT *df)
{
    UNI_DEBUG_LOG("C API %s(%p, %d, %p, %p, %p, %p, %p, %p, %p, %p)...\n", __FUNCTION__, handle,
        num_inputs, name, n, c, t, h, w, dt, df);
    ModelHandleInner *ihInfo = (ModelHandleInner *)handle;
    assert_not_nullptr(__FUNCTION__, "ModelHandle", ihInfo);
    print_model_handle(ihInfo);

    CNN *cnn = (CNN *)ihInfo->cnn;
    assert_not_nullptr(__FUNCTION__, "ModelHandle.cnn", cnn);

    std::map<std::string, TensorDesc> inputTensorDescs = cnn->get_input_desc();
    if (num_inputs != (int)inputTensorDescs.size()) {
        UNI_ERROR_LOG("C API %s: number of inputs is not match, please use GetNumInputsFromModel "
                      "to get the right value.\n",
            __FUNCTION__);
    }
    assert_not_nullptr(__FUNCTION__, NAME_VALUE_PAIR(name));
    for (int i = 0; i < num_inputs; i++) {
        assert_not_nullptr(__FUNCTION__, NAME_VALUE_PAIR(name[i]));
    }
    assert_not_nullptr(__FUNCTION__, NAME_VALUE_PAIR(n));
    assert_not_nullptr(__FUNCTION__, NAME_VALUE_PAIR(c));
    assert_not_nullptr(__FUNCTION__, NAME_VALUE_PAIR(h));
    assert_not_nullptr(__FUNCTION__, NAME_VALUE_PAIR(w));
    assert_not_nullptr(__FUNCTION__, NAME_VALUE_PAIR(dt));
    assert_not_nullptr(__FUNCTION__, NAME_VALUE_PAIR(df));

    DataType idt;
    DataFormat idf;
    U32 in, ic, it, ih, iw;
    int i = 0;
    for (auto iter : inputTensorDescs) {
        UNI_STRCPY(name[i], iter.first.c_str());
        TensorDesc desc = iter.second;
        in = ic = it = ih = iw = 1;
        if (tensorIs1d(desc)) {
            CHECK_STATUS(tensor1dGet(desc, &idt, &idf, &in));
        } else if (tensorIs2d(desc)) {
            CHECK_STATUS(tensor2dGet(desc, &idt, &idf, &in, &ic));
        } else if (tensorIs3d(desc)) {
            CHECK_STATUS(tensor3dGet(desc, &idt, &idf, &in, &ic, &ih));
        } else if (tensorIs4d(desc)) {
            CHECK_STATUS(tensor4dGet(desc, &idt, &idf, &in, &ic, &ih, &iw));
        } else if (tensorIs5d(desc)) {
            CHECK_STATUS(tensor5dGet(desc, &idt, &idf, &in, &ic, &it, &ih, &iw));
            assert_not_nullptr(__FUNCTION__, NAME_VALUE_PAIR(t));
            t[i] = it;
        } else {
            UNI_ERROR_LOG("C API only support 1d,2d,3d,4d,5d query.\n");
        }
        dt[i] = DataType2DATA_TYPE(idt);
        df[i] = DataFormat2DATA_FORMAT(idf);
        n[i] = in;
        c[i] = ic;
        h[i] = ih;
        w[i] = iw;
        ++i;
    }
    UNI_DEBUG_LOG("C API %s end.\n", __FUNCTION__);
}

static std::map<std::string, TensorDesc> getInputDataFormatFromUser(ModelHandle ih,
    int num_inputs,
    const char **name,
    const int *n,
    const int *c,
    const int *t,
    const int *h,
    const int *w,
    const DATA_TYPE *dt,
    const DATA_FORMAT *df)
{
    UNI_DEBUG_LOG("C API %s...\n", __FUNCTION__);
    ModelHandleInner *ihInfo = (ModelHandleInner *)ih;
    assert_not_nullptr(__FUNCTION__, "ModelHandle", ihInfo);

    CNN *cnn = (CNN *)ihInfo->cnn;
    assert_not_nullptr(__FUNCTION__, "ModelHandle.cnn", cnn);

    std::map<std::string, TensorDesc> inputTensorDescs = cnn->get_input_desc();
    int num = inputTensorDescs.size();
    if (num != num_inputs) {
        UNI_ERROR_LOG("C API failed. model has %d inputs, not %d.\n", num, num_inputs);
    }
    assert_not_nullptr(__FUNCTION__, NAME_VALUE_PAIR(name));
    for (int i = 0; i < num_inputs; i++) {
        assert_not_nullptr(__FUNCTION__, NAME_VALUE_PAIR(name[i]));
    }
    assert_not_nullptr(__FUNCTION__, NAME_VALUE_PAIR(n));
    assert_not_nullptr(__FUNCTION__, NAME_VALUE_PAIR(c));
    assert_not_nullptr(__FUNCTION__, NAME_VALUE_PAIR(h));
    assert_not_nullptr(__FUNCTION__, NAME_VALUE_PAIR(w));
    assert_not_nullptr(__FUNCTION__, NAME_VALUE_PAIR(dt));
    assert_not_nullptr(__FUNCTION__, NAME_VALUE_PAIR(df));

    bool setInputDataAs5d = (t == NULL) ? false : true;
    std::map<std::string, TensorDesc> modelInputDims;
    for (int i = 0; i < num; ++i) {
        std::string inputName = name[i];
        if (inputTensorDescs.find(inputName) == inputTensorDescs.end()) {
            UNI_ERROR_LOG(
                "C API inner function %s received %s is not model input.\n", __FUNCTION__, name[i]);
            continue;
        }
        DataType idt = DATA_TYPE2DataType(dt[i]);
        DataFormat idf = DATA_FORMAT2DataFormat(df[i]);
        switch (idf) {
            case DF_NORMAL:
                modelInputDims[inputName] = tensor2df(idt, idf, n[i], c[i]);
                break;
            case DF_MTK:
                modelInputDims[inputName] = tensor3df(idt, idf, n[i], c[i], h[i]);
                break;
            case DF_NCHW:
                if (setInputDataAs5d) {
                    modelInputDims[inputName] = tensor5df(idt, idf, n[i], c[i], t[i], h[i], w[i]);
                } else {
                    modelInputDims[inputName] = tensor4df(idt, idf, n[i], c[i], h[i], w[i]);
                }
                break;
            default:
                UNI_ERROR_LOG("C API inner function %s can not process DataFormat %s.\n",
                    __FUNCTION__, DataFormatName()[idf]);
                break;
        }
    }
    UNI_DEBUG_LOG("C API %s end.\n", __FUNCTION__);
    return modelInputDims;
}

void PrepareModel(ModelHandle ih,
    int num_inputs,
    const char **name,
    const int *n,
    const int *c,
    const int *h,
    const int *w,
    const DATA_TYPE *dt,
    const DATA_FORMAT *df)
{
    UNI_DEBUG_LOG("C API %s...\n", __FUNCTION__);
    PrepareModel5D(ih, num_inputs, name, n, c, NULL, h, w, dt, df);
    UNI_DEBUG_LOG("C API %s end.\n", __FUNCTION__);
}

void PrepareModel5D(ModelHandle ih,
    int num_inputs,
    const char **name,
    const int *n,
    const int *c,
    const int *t,
    const int *h,
    const int *w,
    const DATA_TYPE *dt,
    const DATA_FORMAT *df)
{
    UNI_DEBUG_LOG("C API %s(%p, %d, %p, %p, %p, %p, %p, %p, %p, %p)...\n", __FUNCTION__, ih,
        num_inputs, name, n, c, t, h, w, dt, df);
    ModelHandleInner *ihInfo = (ModelHandleInner *)ih;
    assert_not_nullptr(__FUNCTION__, "ModelHandle", ihInfo);
    print_model_handle(ihInfo);

    CNN *cnn = (CNN *)ihInfo->cnn;
    assert_not_nullptr(__FUNCTION__, "ModelHandle.cnn", cnn);

    std::map<std::string, TensorDesc> modelInputDims =
        getInputDataFormatFromUser(ih, num_inputs, name, n, c, t, h, w, dt, df);
    //cnn->loadAlgorithmMap((const char *)ihInfo->algoPath, ihInfo->useFileStream);
    cnn->ready(modelInputDims);
    cnn->mark_input_output();

    ModelSpec *ms = (ModelSpec *)ihInfo->ms;
    CHECK_STATUS(mt_destroy_model(ms));
    delete ms;
    ihInfo->ms = nullptr;
    UNI_DEBUG_LOG("C API %s end.\n", __FUNCTION__);
}

void ResizeModelInput(ModelHandle ih,
    int num_inputs,
    const char **name,
    const int *n,
    const int *c,
    const int *h,
    const int *w,
    const DATA_TYPE *dt,
    const DATA_FORMAT *df)
{
    UNI_DEBUG_LOG("C API %s(%p, %d, %p, %p, %p, %p, %p, %p, %p)...\n", __FUNCTION__, ih, num_inputs,
        name, n, c, h, w, dt, df);
    ModelHandleInner *ihInfo = (ModelHandleInner *)ih;
    assert_not_nullptr(__FUNCTION__, "ModelHandle", ihInfo);
    print_model_handle(ihInfo);

    CNN *cnn = (CNN *)ihInfo->cnn;
    assert_not_nullptr(__FUNCTION__, "ModelHandle.cnn", cnn);

    std::map<std::string, TensorDesc> modelInputDims =
        getInputDataFormatFromUser(ih, num_inputs, name, n, c, NULL, h, w, dt, df);
    cnn->reready(modelInputDims);
    UNI_DEBUG_LOG("C API %s end.\n", __FUNCTION__);
}

ResultHandle AllocAllResultHandle(ModelHandle ih)
{
    UNI_DEBUG_LOG("C API %s(%p)...\n", __FUNCTION__, ih);
    ModelHandleInner *ihInfo = (ModelHandleInner *)ih;
    assert_not_nullptr(__FUNCTION__, "ModelHandle", ihInfo);
    print_model_handle(ihInfo);
    ResultHandleInner *model_result_ptr = nullptr;
    if (ihInfo != nullptr) {
        CNN *cnn = (CNN *)ihInfo->cnn;
        assert_not_nullptr(__FUNCTION__, "ModelHandle.cnn", cnn);
        if (cnn != nullptr) {
            model_result_ptr = (ResultHandleInner *)UNI_MALLOC(sizeof(ResultHandleInner));
            std::map<std::string, TensorDesc> outputTensorDescs = cnn->get_output_desc();
            int num_outputs = outputTensorDescs.size();
            DataDesc *outputArrPtr = (DataDesc *)UNI_MALLOC(sizeof(DataDesc) * num_outputs);
            int i = 0;
            for (auto iter : outputTensorDescs) {
                std::string name = iter.first;
                U32 length = name.size();
                length = (length > NAME_LEN) ? NAME_LEN : length;
                UNI_MEMCPY(outputArrPtr[i].name, name.c_str(), length);
                if (length < NAME_LEN) {
                    outputArrPtr[i].name[length] = '\0';
                }
                TensorDesc2DataDesc(iter.second, &outputArrPtr[i]);
                i++;
            }
            model_result_ptr->num_outputs = num_outputs;
            model_result_ptr->outputArr = outputArrPtr;
            model_result_ptr->deviceType = ihInfo->deviceType;
        }
    }
    UNI_DEBUG_LOG("C API %s(%p) end.\n", __FUNCTION__, model_result_ptr);
    print_result_handle(model_result_ptr);
    return (ResultHandle)model_result_ptr;
}

ResultHandle AllocSpecificResultHandle(ModelHandle ih, int num_outputs, const char **name)
{
    UNI_DEBUG_LOG("C API %s(%p, %d, %p)...\n", __FUNCTION__, ih, num_outputs, name);
    ModelHandleInner *ihInfo = (ModelHandleInner *)ih;
    assert_not_nullptr(__FUNCTION__, "ModelHandle", ihInfo);
    print_model_handle(ihInfo);
    ResultHandleInner *model_result_ptr = nullptr;
    if (ihInfo != nullptr) {
        CNN *cnn = (CNN *)ihInfo->cnn;
        assert_not_nullptr(__FUNCTION__, "ModelHandle.cnn", cnn);
        if (cnn != nullptr) {
            model_result_ptr = (ResultHandleInner *)UNI_MALLOC(sizeof(ResultHandleInner));
            int model_num_outputs = num_outputs;
            DataDesc *outputArrPtr = (DataDesc *)UNI_MALLOC(sizeof(DataDesc) * model_num_outputs);
            for (int i = 0; i < num_outputs; i++) {
                U32 length = UNI_MIN(strlen(name[i]), NAME_LEN - 1);
                UNI_MEMCPY(outputArrPtr[i].name, name[i], length);
                if (length < NAME_LEN) {
                    outputArrPtr[i].name[length] = '\0';
                }
                TensorDesc srcDesc = cnn->get_tensor_desc_by_name(name[i]);
                TensorDesc2DataDesc(srcDesc, &outputArrPtr[i]);
            }
            model_result_ptr->num_outputs = model_num_outputs;
            model_result_ptr->outputArr = outputArrPtr;
            model_result_ptr->deviceType = ihInfo->deviceType;
        }
    }
    UNI_DEBUG_LOG("C API %s(%p) end.\n", __FUNCTION__, model_result_ptr);
    print_result_handle(model_result_ptr);
    return (void *)model_result_ptr;
}

void SetRuntimeDevice(ModelHandle ih, int cpu_id, DEVICE_TYPE device)
{
    UNI_DEBUG_LOG("C API %s(%p, %d, %d)...\n", __FUNCTION__, ih, cpu_id, device);
    ModelHandleInner *ihInfo = (ModelHandleInner *)ih;
    assert_not_nullptr(__FUNCTION__, "ModelHandle", ihInfo);
    print_model_handle(ihInfo);
    CNN *cnn = (CNN *)ihInfo->cnn;
    assert_not_nullptr(__FUNCTION__, "ModelHandle.cnn", cnn);
    cnn->set_runtime_device(cpu_id, DEVICE_TYPE2Arch(device));
    ihInfo->deviceType = device;
    UNI_DEBUG_LOG("C API %s end.\n", __FUNCTION__);
}

void SetRuntimeDeviceDynamic(ModelHandle ih)
{
    UNI_DEBUG_LOG("C API %s(%p)...\n", __FUNCTION__, ih);
    ModelHandleInner *ihInfo = (ModelHandleInner *)ih;
    assert_not_nullptr(__FUNCTION__, "ModelHandle", ihInfo);
    print_model_handle(ihInfo);
    CNN *cnn = (CNN *)ihInfo->cnn;
    assert_not_nullptr(__FUNCTION__, "ModelHandle.cnn", cnn);
    cnn->set_runtime_device_dynamic();
    ihInfo->deviceType = Arch2DEVICE_TYPE(cnn->get_runtime_device());
    UNI_DEBUG_LOG("C API %s end.\n", __FUNCTION__);
}

void SetNumThreads(int threadNum)
{
    UNI_DEBUG_LOG("C API %s(%d)...\n", __FUNCTION__, threadNum);
    set_cpu_num_threads(threadNum);
    UNI_DEBUG_LOG("C API %s end.\n", __FUNCTION__);
}

void RunModel(ModelHandle ih, ResultHandle ir, int num_inputs, const char **name, void **data)
{
    UNI_DEBUG_LOG("C API %s(%p, %p, %d, %p, %p)...\n", __FUNCTION__, ih, ir, num_inputs, name, data);
    ModelHandleInner *ihInfo = (ModelHandleInner *)ih;
    assert_not_nullptr(__FUNCTION__, "ModelHandle", ihInfo);
    print_model_handle(ihInfo);
    CNN *cnn = (CNN *)ihInfo->cnn;
    assert_not_nullptr(__FUNCTION__, "ModelHandle.cnn", cnn);
    ResultHandleInner *ir_inner = (ResultHandleInner *)ir;
    assert_not_nullptr(__FUNCTION__, "ResultHandle", ir_inner);
    print_result_handle(ir_inner);
    if (num_inputs > 0) {
        assert_not_nullptr(__FUNCTION__, NAME_VALUE_PAIR(name));
        assert_not_nullptr(__FUNCTION__, NAME_VALUE_PAIR(data));
    }

#if 0
    std::map<std::string, U8 *> input;
    for (int index = 0; index < num_inputs; index++) {
        assert_not_nullptr(__FUNCTION__, NAME_VALUE_PAIR(data[index]));
        input[name[index]] = (U8 *)data[index];
    }
    cnn->set_input_by_copy(input);
#else
    std::map<std::string, std::shared_ptr<U8>> input;
    for (int index = 0; index < num_inputs; index++) {
        assert_not_nullptr(__FUNCTION__, NAME_VALUE_PAIR(data[index]));
        std::shared_ptr<U8> shared_data((U8 *)data[index], [](U8 *ptr) {});
        input[name[index]] = shared_data;
    }
    cnn->set_input_by_assign(input);
#endif
    cnn->run();

    DataDesc *outputArrPtr = ir_inner->outputArr;
    assert_not_nullptr(__FUNCTION__, "ResultHandle.outputArr", outputArrPtr);
    DEVICE_TYPE device = ihInfo->deviceType;
    for (U32 curIndex = 0; curIndex < ir_inner->num_outputs; curIndex++) {
        Tensor output_tensor = cnn->get_tensor_by_name(outputArrPtr[curIndex].name);
        if (device == GPU_MALI || device == GPU_QUALCOMM) {
#ifdef _USE_GPU
            auto mem = (OclMemory *)output_tensor.get_memory();
            outputArrPtr[curIndex].dataPtr = mem->get_mapped_ptr();
#else
            UNI_WARNING_LOG("this binary not support GPU, please recompile project with GPU "
                            "compile options.\n");
#endif
        } else {
            outputArrPtr[curIndex].dataPtr = ((CpuMemory *)(output_tensor.get_memory()))->get_ptr();
        }
        TensorDesc2DataDesc(output_tensor.get_desc(), &(outputArrPtr[curIndex]));
    }
    UNI_DEBUG_LOG("C API %s end.\n", __FUNCTION__);
}

int GetNumOutputsFromResultHandle(ResultHandle ir)
{
    UNI_DEBUG_LOG("C API %s(%p)...\n", __FUNCTION__, ir);
    ResultHandleInner *ir_inner = (ResultHandleInner *)ir;
    assert_not_nullptr(__FUNCTION__, "ResultHandle", ir_inner);
    print_result_handle(ir_inner);
    int ret = 0;
    if (ir_inner != nullptr) {
        ret = ir_inner->num_outputs;
    }
    UNI_DEBUG_LOG("C API %s(%d) end.\n", __FUNCTION__, ret);
    return ret;
}

void GetOutputDataInfoFromResultHandle(ResultHandle ir,
    int num_outputs,
    char **name,
    int *n,
    int *c,
    int *h,
    int *w,
    DATA_TYPE *dt,
    DATA_FORMAT *df)
{
    UNI_DEBUG_LOG("C API %s(%p, %d, %p, %p, %p, %p, %p, %p, %p)...\n", __FUNCTION__, ir,
        num_outputs, name, n, c, h, w, dt, df);
    if (num_outputs <= 0) {
        UNI_WARNING_LOG("C API %s received num_outputs = 0.\n", __FUNCTION__);
        return;
    }
    ResultHandleInner *ir_inner = (ResultHandleInner *)ir;
    assert_not_nullptr(__FUNCTION__, "ResultHandle", ir_inner);
    print_result_handle(ir_inner);
    if (num_outputs != (int)ir_inner->num_outputs) {
        UNI_ERROR_LOG("C API %s received num_outputs %d != num_outputs %d in ResultHandle.\n",
            __FUNCTION__, num_outputs, ir_inner->num_outputs);
        return;
    }
    DataDesc *outputArrPtr = ir_inner->outputArr;
    assert_not_nullptr(__FUNCTION__, "ResultHandle.outputArr", outputArrPtr);
    assert_not_nullptr(__FUNCTION__, NAME_VALUE_PAIR(name));
    assert_not_nullptr(__FUNCTION__, NAME_VALUE_PAIR(n));
    assert_not_nullptr(__FUNCTION__, NAME_VALUE_PAIR(c));
    assert_not_nullptr(__FUNCTION__, NAME_VALUE_PAIR(h));
    assert_not_nullptr(__FUNCTION__, NAME_VALUE_PAIR(w));
    assert_not_nullptr(__FUNCTION__, NAME_VALUE_PAIR(dt));
    assert_not_nullptr(__FUNCTION__, NAME_VALUE_PAIR(df));
    for (int i = 0; i < num_outputs; i++) {
        UNI_STRCPY(name[i], outputArrPtr[i].name);
        dt[i] = DataType2DATA_TYPE(outputArrPtr[i].dt);
        df[i] = DataFormat2DATA_FORMAT(outputArrPtr[i].df);
        n[i] = outputArrPtr[i].dims[0];
        c[i] = outputArrPtr[i].dims[1];
        h[i] = outputArrPtr[i].dims[2];
        w[i] = outputArrPtr[i].dims[3];
    }
    UNI_DEBUG_LOG("C API %s end.\n", __FUNCTION__);
}

void GetOutputDataFromResultHandle(ResultHandle ir, int num_outputs, void **data)
{
    UNI_DEBUG_LOG("C API %s(%p, %d, %p)...\n", __FUNCTION__, ir, num_outputs, data);
    if (num_outputs <= 0) {
        UNI_WARNING_LOG("C API %s received num_outputs = 0.\n", __FUNCTION__);
        return;
    }
    ResultHandleInner *ir_inner = (ResultHandleInner *)ir;
    assert_not_nullptr(__FUNCTION__, "ResultHandle", ir_inner);
    print_result_handle(ir_inner);
    if (num_outputs != (int)ir_inner->num_outputs) {
        UNI_ERROR_LOG("C API %s received num_outputs %d != num_outputs %d in ResultHandle.\n",
            __FUNCTION__, num_outputs, ir_inner->num_outputs);
        return;
    }
    DataDesc *outputArrPtr = ir_inner->outputArr;
    assert_not_nullptr(__FUNCTION__, "ResultHandle.outputArr", outputArrPtr);
    assert_not_nullptr(__FUNCTION__, NAME_VALUE_PAIR(data));
    for (int i = 0; i < num_outputs; i++) {
        data[i] = outputArrPtr[i].dataPtr;
    }
    UNI_DEBUG_LOG("C API %s end.\n", __FUNCTION__);
}

ResultHandle CloneResultHandle(ResultHandle ir)
{
    UNI_DEBUG_LOG("C API %s(%p)...\n", __FUNCTION__, ir);
    ResultHandleInner *ir_inner = (ResultHandleInner *)ir;
    assert_not_nullptr(__FUNCTION__, "ResultHandle", ir_inner);
    print_result_handle(ir_inner);
    ResultHandleInner *clone_ir_inner = nullptr;
    if (ir_inner != nullptr) {
        clone_ir_inner = (ResultHandleInner *)UNI_MALLOC(sizeof(ResultHandleInner));
        *clone_ir_inner = *ir_inner;
        U32 size = sizeof(DataDesc) * clone_ir_inner->num_outputs;
        if (size > 0) {
            clone_ir_inner->outputArr = (DataDesc *)UNI_MALLOC(size);
            DataDesc *outputArrPtr = ir_inner->outputArr;
            assert_not_nullptr(__FUNCTION__, "ResultHandle.outputArr", outputArrPtr);
            UNI_MEMCPY(clone_ir_inner->outputArr, outputArrPtr, size);
        } else {
            clone_ir_inner->outputArr = nullptr;
        }
    }
    UNI_DEBUG_LOG("C API %s(%p) end.\n", __FUNCTION__, clone_ir_inner);
    print_result_handle(clone_ir_inner);
    return (ResultHandle)clone_ir_inner;
}

void FreeResultHandle(ResultHandle ir)
{
    UNI_DEBUG_LOG("C API %s(%p)...\n", __FUNCTION__, ir);
    ResultHandleInner *ir_inner = (ResultHandleInner *)ir;
    assert_not_nullptr(__FUNCTION__, "ResultHandle", ir_inner);
    print_result_handle(ir_inner);
    if (ir_inner != nullptr) {
        DataDesc *outputArrPtr = ir_inner->outputArr;
        if (ir_inner->num_outputs > 0) {
            assert_not_nullptr(__FUNCTION__, "ResultHandle.outputArr", outputArrPtr);
            ir_inner->num_outputs = 0;
        }
        if (outputArrPtr != nullptr) {
            UNI_FREE(outputArrPtr);
            ir_inner->outputArr = nullptr;
        }
        UNI_FREE(ir_inner);
    }
    UNI_DEBUG_LOG("C API %s end.\n", __FUNCTION__);
}

void DestroyModel(ModelHandle ih)
{
    UNI_DEBUG_LOG("C API %s(%p)...\n", __FUNCTION__, ih);
    ModelHandleInner *ihInfo = (ModelHandleInner *)ih;
    assert_not_nullptr(__FUNCTION__, "ModelHandle", ihInfo);
    print_model_handle(ihInfo);
    if (ihInfo != nullptr) {
        CNN *cnn = (CNN *)ihInfo->cnn;
        assert_not_nullptr(__FUNCTION__, "ModelHandle.cnn", cnn);
        if (cnn != nullptr) {
            //if (ihInfo->algoPath != nullptr && !ihInfo->useFileStream) {
            //    const char *algoPath = (const char *)ihInfo->algoPath;
            //    UNI_THREAD_SAFE(cnn->saveAlgorithmMapToFile(algoPath));
            //}
            delete cnn;
            ihInfo->cnn = nullptr;
        }
        delete ihInfo;
    }
    UNI_DEBUG_LOG("C API %s end.\n", __FUNCTION__);
}

void MemoryCheck()
{
    UNI_DEBUG_LOG("C API %s()...\n", __FUNCTION__);
#ifndef _USE_MEM_CHECK
    UNI_WARNING_LOG("please set USE_MEM_CHECK to ON at common/cmakes/bolt.cmake, and rebuild "
                    "library.\n");
#endif
    UNI_MEM_STATISTICS();
    UNI_DEBUG_LOG("C API %s end.\n", __FUNCTION__);
}
#undef NAME_VALUE_PAIR
