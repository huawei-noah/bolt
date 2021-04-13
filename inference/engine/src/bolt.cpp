// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "inference.hpp"
#include "../api/c/bolt.h"

#define NAME_VALUE_PAIR(x) #x, x
const int DataDescMaxDims = 6;

struct ModelHandleInner {
    void *ms;
    void *cnn;
    DEVICE_TYPE deviceType;
    void *algoPath;
    bool useFileStream;
};

typedef struct {
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

inline DataType DATA_TYPE2DataType(DATA_TYPE dt_user)
{
    DataType ret = DT_F32;
    switch (dt_user) {
        case FP_32:
            ret = DT_F32;
            break;
#ifdef __aarch64__
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
            UNI_ERROR_LOG("C API can not recognize DATA_TYPE %d\n", dt_user);
            break;
    }
    return ret;
}

inline DATA_TYPE DataType2DATA_TYPE(DataType dt_bolt)
{
    DATA_TYPE ret = FP_32;
    switch (dt_bolt) {
        case DT_F32:
            ret = FP_32;
            break;
#ifdef __aarch64__
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
            UNI_ERROR_LOG("C API can not process inner DataType %d\n", dt_bolt);
            break;
    }
    return ret;
}

inline DataFormat DATA_FORMAT2DataFormat(DATA_FORMAT df_user)
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
            UNI_ERROR_LOG("C API can not recognize DATA_FORMAT %d\n", df_user);
            break;
        }
    }
    return ret;
}

inline DATA_FORMAT DataFormat2DATA_FORMAT(DataFormat df_bolt)
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
        case DF_MTK:
            ret = MTK;
            break;
        case DF_NORMAL:
            ret = NORMAL;
            break;
        default: {
            UNI_ERROR_LOG("C API can not process inner DataFormat %d\n", df_bolt);
            break;
        }
    }
    return ret;
}

inline AffinityPolicy AFFINITY_TYPE2AffinityPolicy(AFFINITY_TYPE affinity)
{
    AffinityPolicy ret = AFFINITY_CPU_HIGH_PERFORMANCE;
    switch (affinity) {
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
            UNI_ERROR_LOG("C API can not recognize AFFINITY_TYPE %d\n", affinity);
            break;
        }
    }
    return ret;
}

inline Arch DEVICE_TYPE2Arch(DEVICE_TYPE device)
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
        case CPU_X86_AVX2:
            ret = X86_AVX2;
            break;
        case CPU_SERIAL:
            ret = CPU_GENERAL;
            break;
        default: {
            UNI_ERROR_LOG("C API can not recognize DEVICE_TYPE %d\n", device);
            break;
        }
    }
    return ret;
}

inline DEVICE_TYPE Arch2DEVICE_TYPE(Arch arch)
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
        case X86_AVX2:
            ret = CPU_X86_AVX2;
            break;
        case CPU_GENERAL:
            ret = CPU_SERIAL;
            break;
        default: {
            UNI_ERROR_LOG("C API can not process inner Arch %d\n", arch);
            break;
        }
    }
    return ret;
}

void TensorDesc2DataDesc(TensorDesc srcDesc, DataDesc *dstDesc)
{
    dstDesc->dt = srcDesc.dt;
    dstDesc->df = srcDesc.df;
    if (srcDesc.nDims > DataDescMaxDims) {
        UNI_ERROR_LOG(
            "C API DataDesc only support %d dimensions, not %d\n", DataDescMaxDims, srcDesc.nDims);
    }
    for (U32 i = 0; i < srcDesc.nDims; i++) {
        dstDesc->dims[i] = srcDesc.dims[srcDesc.nDims - 1 - i];
    }
    for (int i = srcDesc.nDims; i < DataDescMaxDims; i++) {
        dstDesc->dims[i] = 1;
    }
}

void assert_not_nullptr(const char *funcName, const char *ptrName, const void *ptr)
{
    if (ptr == NULL) {
        UNI_ERROR_LOG("C API %s received null ptr %s\n", funcName, ptrName);
    }
}

void GetGpuDeviceName(char *gpuDeviceName)
{
    std::string deviceName = "unKnown";
#ifdef _USE_MALI
    deviceName = OCLContext::getInstance().handle->deviceName;
#endif
    strcpy(gpuDeviceName, deviceName.c_str());
}

ModelHandle CreateModel(const char *modelPath, AFFINITY_TYPE affinity, const char *algorithmMapPath)
{
    assert_not_nullptr(__FUNCTION__, NAME_VALUE_PAIR(modelPath));
    ModelHandleInner *handle = new ModelHandleInner();
    ModelSpec *ms = new ModelSpec();
    if (SUCCESS != deserialize_model_from_file(modelPath, ms)) {
        UNI_ERROR_LOG("C API %s failed\n", __FUNCTION__);
        delete ms;
        handle->cnn = nullptr;
        return (ModelHandle)handle;
    }
    CNN *cnn = new CNN(AFFINITY_TYPE2AffinityPolicy(affinity), ms->dt, ms->model_name);
    cnn->sort_operators_sequential(ms);
    cnn->initialize_ops(ms);

    handle->cnn = (void *)cnn;
    handle->ms = (void *)ms;
    handle->deviceType = Arch2DEVICE_TYPE(cnn->get_runtime_device());
    handle->algoPath = (void *)algorithmMapPath;
    handle->useFileStream = false;
    return (ModelHandle)handle;
}

ModelHandle CloneModel(ModelHandle ih)
{
    ModelHandleInner *handle = (ModelHandleInner *)ih;
    assert_not_nullptr(__FUNCTION__, "ModelHandle", handle);
    CNN *cnn = (CNN *)handle->cnn;
    assert_not_nullptr(__FUNCTION__, "ModelHandle.cnn", cnn);
    ModelHandleInner *cloneHandle = new ModelHandleInner();
    *cloneHandle = *handle;
    CNN *cloneCnn = new CNN();
    *cloneCnn = cnn->clone();
    cloneHandle->cnn = cloneCnn;
    return (ModelHandle)cloneHandle;
}

ModelHandle CreateModelWithFileStream(
    const char *modelFileStream, AFFINITY_TYPE affinity, const char *algorithmMapFileStream)
{
    assert_not_nullptr(__FUNCTION__, NAME_VALUE_PAIR(modelFileStream));
    ModelHandleInner *handle = new ModelHandleInner();
    ModelSpec *ms = new ModelSpec();
    if (SUCCESS != deserialize_model_from_file(modelFileStream, ms, true)) {
        UNI_ERROR_LOG("C API %s failed\n", __FUNCTION__);
        delete ms;
        handle->cnn = nullptr;
        return (ModelHandle)handle;
    }
    CNN *cnn = new CNN(AFFINITY_TYPE2AffinityPolicy(affinity), ms->dt, ms->model_name);
    cnn->sort_operators_sequential(ms);
    cnn->initialize_ops(ms);

    handle->cnn = (void *)cnn;
    handle->ms = (void *)ms;
    handle->deviceType = Arch2DEVICE_TYPE(cnn->get_runtime_device());
    handle->algoPath = (void *)algorithmMapFileStream;
    handle->useFileStream = true;
    return (ModelHandle)handle;
}

int GetNumInputsFromModel(ModelHandle ih)
{
    ModelHandleInner *ihInfo = (ModelHandleInner *)ih;
    assert_not_nullptr(__FUNCTION__, "ModelHandle", ihInfo);

    CNN *cnn = (CNN *)ihInfo->cnn;
    assert_not_nullptr(__FUNCTION__, "ModelHandle.cnn", cnn);

    return (cnn->get_input_desc()).size();
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
    GetInputDataInfoFromModel5D(ih, num_inputs, name, n, c, NULL, h, w, dt, df);
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
    ModelHandleInner *ihInfo = (ModelHandleInner *)handle;
    assert_not_nullptr(__FUNCTION__, "ModelHandle", ihInfo);

    CNN *cnn = (CNN *)ihInfo->cnn;
    assert_not_nullptr(__FUNCTION__, "ModelHandle.cnn", cnn);

    std::map<std::string, TensorDesc> inputTensorDescs = cnn->get_input_desc();
    if (num_inputs != (int)inputTensorDescs.size()) {
        UNI_ERROR_LOG("GetInputDataInfoFromModel: number of inputs is not match, "
                      "please use GetNumInputsFromModel to get the right value\n");
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
        strcpy(name[i], iter.first.c_str());
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
            UNI_ERROR_LOG("C API only support 1d,2d,3d,4d,5d query\n");
        }
        dt[i] = DataType2DATA_TYPE(idt);
        df[i] = DataFormat2DATA_FORMAT(idf);
        n[i] = in;
        c[i] = ic;
        h[i] = ih;
        w[i] = iw;
        ++i;
    }
}

std::map<std::string, TensorDesc> getInputDataFormatFromUser(ModelHandle ih,
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
    ModelHandleInner *ihInfo = (ModelHandleInner *)ih;
    assert_not_nullptr(__FUNCTION__, "ModelHandle", ihInfo);

    CNN *cnn = (CNN *)ihInfo->cnn;
    assert_not_nullptr(__FUNCTION__, "ModelHandle.cnn", cnn);

    std::map<std::string, TensorDesc> inputTensorDescs = cnn->get_input_desc();
    int num = inputTensorDescs.size();
    if (num != num_inputs) {
        UNI_ERROR_LOG("C API failed. model has %d inputs, not %d\n", num, num_inputs);
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
            exit(1);
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
                UNI_ERROR_LOG(
                    "C API inner function %s can not process DataFormat %d\n", __FUNCTION__, idf);
                break;
        }
    }
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
    PrepareModel5D(ih, num_inputs, name, n, c, NULL, h, w, dt, df);
}

void PrepareModel5D(ModelHandle ih,
    int num_input,
    const char **name,
    const int *n,
    const int *c,
    const int *t,
    const int *h,
    const int *w,
    const DATA_TYPE *dt,
    const DATA_FORMAT *df)
{
    ModelHandleInner *ihInfo = (ModelHandleInner *)ih;
    assert_not_nullptr(__FUNCTION__, "ModelHandle", ihInfo);

    CNN *cnn = (CNN *)ihInfo->cnn;
    assert_not_nullptr(__FUNCTION__, "ModelHandle.cnn", cnn);

    std::map<std::string, TensorDesc> modelInputDims =
        getInputDataFormatFromUser(ih, num_input, name, n, c, t, h, w, dt, df);
    cnn->loadAlgorithmMap((const char *)ihInfo->algoPath, ihInfo->useFileStream);
    cnn->ready(modelInputDims);
    cnn->mark_input_output();

    ModelSpec *ms = (ModelSpec *)ihInfo->ms;
    CHECK_STATUS(mt_destroy_model(ms));
    delete ms;
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
    ModelHandleInner *ihInfo = (ModelHandleInner *)ih;
    assert_not_nullptr(__FUNCTION__, "ModelHandle", ihInfo);
    CNN *cnn = (CNN *)ihInfo->cnn;
    assert_not_nullptr(__FUNCTION__, "ModelHandle.cnn", cnn);

    std::map<std::string, TensorDesc> modelInputDims =
        getInputDataFormatFromUser(ih, num_inputs, name, n, c, NULL, h, w, dt, df);
    cnn->reready(modelInputDims);
}

ResultHandle AllocAllResultHandle(ModelHandle ih)
{
    ModelHandleInner *ihInfo = (ModelHandleInner *)ih;
    assert_not_nullptr(__FUNCTION__, "ModelHandle", ihInfo);
    CNN *cnn = (CNN *)ihInfo->cnn;
    assert_not_nullptr(__FUNCTION__, "ModelHandle.cnn", cnn);

    ResultHandleInner *model_result_ptr = (ResultHandleInner *)malloc(sizeof(ResultHandleInner));
    std::map<std::string, TensorDesc> outputTensorDescs = cnn->get_output_desc();
    int num_outputs = outputTensorDescs.size();
    DataDesc *outputArrPtr = (DataDesc *)malloc(sizeof(DataDesc) * num_outputs);
    int i = 0;
    for (auto iter : outputTensorDescs) {
        std::string name = iter.first;
        U32 length = name.size();
        length = (length > NAME_LEN) ? NAME_LEN : length;
        memcpy(outputArrPtr[i].name, name.c_str(), length);
        if (length < NAME_LEN) {
            outputArrPtr[i].name[length] = '\0';
        }
        TensorDesc2DataDesc(iter.second, &outputArrPtr[i]);
        i++;
    }
    model_result_ptr->num_outputs = num_outputs;
    model_result_ptr->outputArr = outputArrPtr;
    model_result_ptr->deviceType = ihInfo->deviceType;
    return (void *)model_result_ptr;
}

ResultHandle AllocSpecificResultHandle(ModelHandle ih, int num_outputs, const char **name)
{
    ModelHandleInner *ihInfo = (ModelHandleInner *)ih;
    assert_not_nullptr(__FUNCTION__, "ModelHandle", ihInfo);
    CNN *cnn = (CNN *)ihInfo->cnn;
    assert_not_nullptr(__FUNCTION__, "ModelHandle.cnn", cnn);

    ResultHandleInner *model_result_ptr = (ResultHandleInner *)malloc(sizeof(ResultHandleInner));
    int model_num_outputs = num_outputs;
    DataDesc *outputArrPtr = (DataDesc *)malloc(sizeof(DataDesc) * model_num_outputs);
    for (int i = 0; i < num_outputs; i++) {
        U32 length = UNI_MIN(strlen(name[i]), NAME_LEN - 1);
        memcpy(outputArrPtr[i].name, name[i], length);
        if (length < NAME_LEN) {
            outputArrPtr[i].name[length] = '\0';
        }
        TensorDesc srcDesc = cnn->get_tensor_desc_by_name(name[i]);
        TensorDesc2DataDesc(srcDesc, &outputArrPtr[i]);
    }
    model_result_ptr->num_outputs = model_num_outputs;
    model_result_ptr->outputArr = outputArrPtr;
    model_result_ptr->deviceType = ihInfo->deviceType;
    return (void *)model_result_ptr;
}

void SetRuntimeDevice(ModelHandle ih, int cpu_id, DEVICE_TYPE device)
{
    ModelHandleInner *ihInfo = (ModelHandleInner *)ih;
    assert_not_nullptr(__FUNCTION__, "ModelHandle", ihInfo);
    CNN *cnn = (CNN *)ihInfo->cnn;
    assert_not_nullptr(__FUNCTION__, "ModelHandle.cnn", cnn);
    cnn->set_runtime_device(cpu_id, DEVICE_TYPE2Arch(device));
    ihInfo->deviceType = device;
}

void SetRuntimeDeviceDynamic(ModelHandle ih)
{
    ModelHandleInner *ihInfo = (ModelHandleInner *)ih;
    assert_not_nullptr(__FUNCTION__, "ModelHandle", ihInfo);
    CNN *cnn = (CNN *)ihInfo->cnn;
    assert_not_nullptr(__FUNCTION__, "ModelHandle.cnn", cnn);
    cnn->set_runtime_device_dynamic();
    ihInfo->deviceType = Arch2DEVICE_TYPE(cnn->get_runtime_device());
}

void RunModel(ModelHandle ih, ResultHandle ir, int num_inputs, const char **name, void **data)
{
    ModelHandleInner *ihInfo = (ModelHandleInner *)ih;
    assert_not_nullptr(__FUNCTION__, "ModelHandle", ihInfo);
    CNN *cnn = (CNN *)ihInfo->cnn;
    assert_not_nullptr(__FUNCTION__, "ModelHandle.cnn", cnn);
    ResultHandleInner *ir_inner = (ResultHandleInner *)ir;
    assert_not_nullptr(__FUNCTION__, "ResultHandle", ir_inner);
    if (num_inputs > 0) {
        assert_not_nullptr(__FUNCTION__, NAME_VALUE_PAIR(name));
        assert_not_nullptr(__FUNCTION__, NAME_VALUE_PAIR(data));
    }

    std::map<std::string, U8 *> input;
    for (int index = 0; index < num_inputs; index++) {
        assert_not_nullptr(__FUNCTION__, NAME_VALUE_PAIR(data[index]));
        input[name[index]] = (U8 *)data[index];
    }
    cnn->set_input_by_copy(input);
    cnn->run();

    DataDesc *outputArrPtr = ir_inner->outputArr;
    assert_not_nullptr(__FUNCTION__, "ResultHandle.outputArr", outputArrPtr);
    DEVICE_TYPE device = ihInfo->deviceType;
    for (U32 curIndex = 0; curIndex < ir_inner->num_outputs; curIndex++) {
        Tensor output_tensor = cnn->get_tensor_by_name(outputArrPtr[curIndex].name);
        TensorDesc2DataDesc(output_tensor.get_desc(), &(outputArrPtr[curIndex]));
        if (device == GPU_MALI) {
#ifdef _USE_MALI
            auto mem = (OclMemory *)output_tensor.get_memory();
            outputArrPtr[curIndex].dataPtr = mem->get_mapped_ptr();
#else
            UNI_WARNING_LOG("this binary not support GPU, please recompile project with GPU "
                            "compile options\n");
#endif
        } else {
            outputArrPtr[curIndex].dataPtr = ((CpuMemory *)(output_tensor.get_memory()))->get_ptr();
        }
    }
}

int GetNumOutputsFromResultHandle(ResultHandle ir)
{
    ResultHandleInner *ir_inner = (ResultHandleInner *)ir;
    assert_not_nullptr(__FUNCTION__, "ResultHandle", ir_inner);
    return ir_inner->num_outputs;
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
    if (num_outputs <= 0) {
        UNI_WARNING_LOG("C API %s received num_outputs = 0.\n", __FUNCTION__);
        return;
    }
    ResultHandleInner *ir_inner = (ResultHandleInner *)ir;
    assert_not_nullptr(__FUNCTION__, "ResultHandle", ir_inner);
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
        strcpy(name[i], outputArrPtr[i].name);
        dt[i] = DataType2DATA_TYPE(outputArrPtr[i].dt);
        df[i] = DataFormat2DATA_FORMAT(outputArrPtr[i].df);
        n[i] = outputArrPtr[i].dims[0];
        c[i] = outputArrPtr[i].dims[1];
        h[i] = outputArrPtr[i].dims[2];
        w[i] = outputArrPtr[i].dims[3];
    }
}

void GetOutputDataFromResultHandle(ResultHandle ir, int num_outputs, void **data)
{
    if (num_outputs <= 0) {
        UNI_WARNING_LOG("C API %s received num_outputs = 0.\n", __FUNCTION__);
        return;
    }
    ResultHandleInner *ir_inner = (ResultHandleInner *)ir;
    assert_not_nullptr(__FUNCTION__, "ResultHandle", ir_inner);
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
}

ResultHandle CloneResultHandle(ResultHandle ir)
{
    ResultHandleInner *ir_inner = (ResultHandleInner *)ir;
    assert_not_nullptr(__FUNCTION__, "ResultHandle", ir_inner);
    ResultHandleInner *clone_ir_inner = new ResultHandleInner();
    *clone_ir_inner = *ir_inner;
    U32 size = sizeof(DataDesc) * clone_ir_inner->num_outputs;
    if (size > 0) {
        clone_ir_inner->outputArr = (DataDesc *)malloc(size);
        DataDesc *outputArrPtr = ir_inner->outputArr;
        assert_not_nullptr(__FUNCTION__, "ResultHandle.outputArr", outputArrPtr);
        memcpy(clone_ir_inner->outputArr, outputArrPtr, size);
    } else {
        clone_ir_inner->outputArr = nullptr;
    }
    return (ResultHandle)clone_ir_inner;
}

void FreeResultHandle(ResultHandle ir)
{
    ResultHandleInner *ir_inner = (ResultHandleInner *)ir;
    assert_not_nullptr(__FUNCTION__, "ResultHandle", ir_inner);
    DataDesc *outputArrPtr = ir_inner->outputArr;
    if (ir_inner->num_outputs > 0) {
        assert_not_nullptr(__FUNCTION__, "ResultHandle.outputArr", outputArrPtr);
        free(outputArrPtr);
        ir_inner->num_outputs = 0;
    }
    (*ir_inner).outputArr = nullptr;
    free(ir_inner);
}

void DestroyModel(ModelHandle ih)
{
    ModelHandleInner *ihInfo = (ModelHandleInner *)ih;
    assert_not_nullptr(__FUNCTION__, "ModelHandle", ihInfo);

    CNN *cnn = (CNN *)ihInfo->cnn;
    assert_not_nullptr(__FUNCTION__, "ModelHandle.cnn", cnn);

    if (nullptr != ihInfo->algoPath) {
        const char *algoPath = (const char *)ihInfo->algoPath;
        UNI_THREAD_SAFE(cnn->saveAlgorithmMapToFile(algoPath));
    }

    delete cnn;
    ihInfo->cnn = nullptr;
    delete ihInfo;
}

#undef NAME_VALUE_PAIR
