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

struct ModelHandleInfo {
    void *cnn;
    DEVICE_TYPE deviceType;
    void *algoPath;
    bool useFileStream;
};

typedef struct {
    U32 dims[4] = {0};
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

DataType dt_mapping_user2bolt(DATA_TYPE dt_user)
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
            UNI_ERROR_LOG("unsupported user data type in API\n");
    }
    return ret;
}

DATA_TYPE dt_mapping_bolt2user(DataType dt_bolt)
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
            UNI_ERROR_LOG("unsupported bolt data type in API\n");
    }
    return ret;
}

DataFormat df_mapping_user2bolt(DATA_FORMAT df_user)
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
            UNI_ERROR_LOG("unsupported user data format in API\n");
        }
    }
    return ret;
}

DATA_FORMAT df_mapping_bolt2user(DataFormat df_bolt)
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
            UNI_ERROR_LOG("unsupported bolt data format in API\n");
        }
    }
    return ret;
}

inline AffinityPolicy affinity_mapping_user2bolt(AFFINITY_TYPE affinity)
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
            UNI_ERROR_LOG("unsupported user affinity type in API\n");
        }
    }
    return ret;
}

inline Arch device_mapping_user2bolt(DEVICE_TYPE device)
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
            UNI_ERROR_LOG("unsupported user device type %d in API\n", device);
            break;
        }
    }
    return ret;
}

inline DEVICE_TYPE device_mapping_bolt2user(Arch arch)
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
            UNI_ERROR_LOG("unsupported bolt device type %d in API\n", arch);
            break;
        }
    }
    return ret;
}

void copyTensorDescToDataDesc(TensorDesc srcDesc, DataDesc *dstDesc)
{
    dstDesc->dt = srcDesc.dt;
    dstDesc->df = srcDesc.df;
    if (srcDesc.nDims > 4) {
        UNI_ERROR_LOG("user interface only support 4 dimensions, not %d\n", srcDesc.nDims);
    }
    for (U32 i = 0; i < srcDesc.nDims; i++) {
        dstDesc->dims[i] = srcDesc.dims[srcDesc.nDims - 1 - i];
    }
    for (int i = srcDesc.nDims; i < 4; i++) {
        dstDesc->dims[i] = 1;
    }
}

ModelHandle CreateModel(const char *modelPath, AFFINITY_TYPE affinity, const char *algoPath)
{
    ModelHandleInfo *handle = new ModelHandleInfo();
    ModelSpec *ms = new ModelSpec();
    if (SUCCESS != deserialize_model_from_file(modelPath, ms)) {
        UNI_ERROR_LOG("CreateModel failed\n");
        delete ms;
        handle->cnn = nullptr;
        return (ModelHandle)handle;
    }
    CNN *cnn = new CNN(affinity_mapping_user2bolt(affinity), ms->dt, ms->model_name);
    cnn->sort_operators_sequential(ms);
    cnn->initialize_ops(ms);

    handle->cnn = (void *)cnn;
    handle->deviceType = device_mapping_bolt2user(cnn->get_runtime_device());
    handle->algoPath = (void *)algoPath;
    handle->useFileStream = false;
    CHECK_STATUS(mt_destroy_model(ms));
    delete ms;
    return (ModelHandle)handle;
}

ModelHandle CloneModel(ModelHandle ih)
{
    ModelHandleInfo *handle = (ModelHandleInfo *)ih;
    ModelHandleInfo *cloneHandle = new ModelHandleInfo();
    *cloneHandle = *handle;
    CNN *cloneCnn = new CNN();
    *cloneCnn = ((CNN *)handle->cnn)->clone();
    cloneHandle->cnn = cloneCnn;
    return (ModelHandle)cloneHandle;
}

ModelHandle CreateModelWithFileStream(
    const char *modelFileStream, AFFINITY_TYPE affinity, const char *algoFileStream)
{
    ModelHandleInfo *handle = new ModelHandleInfo();
    ModelSpec *ms = new ModelSpec();
    if (SUCCESS != deserialize_model_from_file(modelFileStream, ms, true)) {
        UNI_ERROR_LOG("CreateModelWithFileStream failed\n");
        delete ms;
        handle->cnn = nullptr;
        return (ModelHandle)handle;
    }
    CNN *cnn = new CNN(affinity_mapping_user2bolt(affinity), ms->dt, ms->model_name);
    cnn->sort_operators_sequential(ms);
    cnn->initialize_ops(ms);

    handle->cnn = (void *)cnn;
    handle->deviceType = device_mapping_bolt2user(cnn->get_runtime_device());
    handle->algoPath = (void *)algoFileStream;
    handle->useFileStream = true;
    CHECK_STATUS(mt_destroy_model(ms));
    delete ms;
    return (ModelHandle)handle;
}

int GetNumInputsFromModel(ModelHandle ih)
{
    ModelHandleInfo *ihInfo = (ModelHandleInfo *)ih;
    CNN *cnn = (CNN *)ihInfo->cnn;
    if (ihInfo == NULL) {
        UNI_ERROR_LOG("GetNumInputsFromModel: inference handle is nullptr\n");
    }
    return (cnn->get_model_input_tensor_names()).size();
}

void GetInputDataInfoFromModel(ModelHandle handle,
    const int number_inputs,
    char **inputNames,
    int *n,
    int *c,
    int *h,
    int *w,
    DATA_TYPE *dt_input,
    DATA_FORMAT *df_input)
{
    ModelHandleInfo *ihInfo = (ModelHandleInfo *)handle;
    CNN *cnn = (CNN *)ihInfo->cnn;
    if (ihInfo == NULL) {
        UNI_ERROR_LOG("GetInputDataInfoFromModel: inference handle is nullptr\n");
    }
    std::vector<TensorDesc> inputTensorDescs = cnn->get_model_input_tensor_descs();
    std::vector<std::string> inputTensorNames = cnn->get_model_input_tensor_names();
    if (number_inputs != (int)inputTensorDescs.size() ||
        number_inputs != (int)inputTensorNames.size()) {
        UNI_ERROR_LOG("GetInputDataInfoFromModel: number of inputs is not match, "
                      "please use GetNumInputsFromModel to get the right value\n");
    }
    DataType dt;
    DataFormat df;
    U32 in, ic, ih, iw;
    for (int i = 0; i < number_inputs; i++) {
        strcpy(inputNames[i], inputTensorNames[i].c_str());
        in = ic = ih = iw = 0;
        if (tensorIs1d(inputTensorDescs[i])) {
            CHECK_STATUS(tensor1dGet(inputTensorDescs[i], &dt, &df, &in));
        } else if (tensorIs2d(inputTensorDescs[i])) {
            CHECK_STATUS(tensor2dGet(inputTensorDescs[i], &dt, &df, &in, &ic));
        } else if (tensorIs3d(inputTensorDescs[i])) {
            CHECK_STATUS(tensor3dGet(inputTensorDescs[i], &dt, &df, &in, &ic, &ih));
        } else if (tensorIs4d(inputTensorDescs[i])) {
            CHECK_STATUS(tensor4dGet(inputTensorDescs[i], &dt, &df, &in, &ic, &ih, &iw));
        } else {
            UNI_ERROR_LOG("C API currently only support 1d,2d,3d,4d query\n");
        }
        n[i] = in;
        c[i] = ic;
        h[i] = ih;
        w[i] = iw;
        dt_input[i] = dt_mapping_bolt2user(dt);
        df_input[i] = df_mapping_bolt2user(df);
    }
}

std::map<std::string, TensorDesc> getInputDataFormatFromUser(ModelHandle ih,
    const int num_input,
    char **name,
    const int *n,
    const int *c,
    const int *h,
    const int *w,
    const DATA_TYPE *dt_input,
    const DATA_FORMAT *df_input)
{
    ModelHandleInfo *ihInfo = (ModelHandleInfo *)ih;
    CNN *cnn = (CNN *)ihInfo->cnn;
    std::vector<std::string> inputTensorNames = cnn->get_model_input_tensor_names();
    U32 num = inputTensorNames.size();
    if (num != (U32)num_input) {
        UNI_ERROR_LOG("getInputDataFormatFromUser: model has %d inputs, not %d\n", num, num_input);
    }
    if (ihInfo == NULL) {
        UNI_ERROR_LOG("getInputDataFormatFromUser: ih is nullptr\n");
    }
    if (n == NULL) {
        UNI_ERROR_LOG("getInputDataFormatFromUser: n is nullptr\n");
    }
    if (c == NULL) {
        UNI_ERROR_LOG("getInputDataFormatFromUser: c is nullptr\n");
    }
    if (h == NULL) {
        UNI_ERROR_LOG("getInputDataFormatFromUser: h is nullptr\n");
    }
    if (w == NULL) {
        UNI_ERROR_LOG("getInputDataFormatFromUser: w is nullptr\n");
    }
    if (name == NULL) {
        UNI_ERROR_LOG("getInputDataFormatFromUser: name is nullptr\n");
    }
    for (U32 i = 0; i < num; ++i) {
        if (name[i] == NULL) {
            UNI_ERROR_LOG("getInputDataFormatFromUser: input name %d is nullptr\n", i);
        }
    }

    std::map<std::string, TensorDesc> modelInputDims;
    for (U32 i = 0; i < num; ++i) {
        std::string inputName = name[i];
        bool findTensorName = false;
        for (U32 j = 0; j < num; ++j) {
            std::string modelName = inputTensorNames[j];
            if (modelName == inputName) {
                DataType dt = (dt_input == NULL) ? DT_F32 : dt_mapping_user2bolt(dt_input[i]);
                DataFormat df = (df_input == NULL) ? DF_NCHW : df_mapping_user2bolt(df_input[i]);
                switch (df) {
                    case DF_NORMAL:
                        modelInputDims[inputName] = tensor2df(dt, df, n[i], c[i]);
                        break;
                    case DF_MTK:
                        modelInputDims[inputName] = tensor3df(dt, df, n[i], c[i], h[i]);
                        break;
                    case DF_NCHW:
                        modelInputDims[inputName] = tensor4df(dt, df, n[i], c[i], h[i], w[i]);
                        break;
                    default:
                        UNI_ERROR_LOG("unsupported data format in %s\n", __func__);
                }
                findTensorName = true;
                break;
            }
        }

        if (!findTensorName) {
            std::string errorLog = "(";
            for (U32 j = 0; j < num; ++j) {
                errorLog.append(inputTensorNames[j]);
                if (j != num - 1) {
                    errorLog.append(", ");
                }
            }
            errorLog.append(")");
            UNI_ERROR_LOG("input data %s is not a valid model input %s\n", inputName.c_str(),
                errorLog.c_str());
        }
    }
    return modelInputDims;
}

void PrepareModel(ModelHandle ih,
    const int num_input,
    char **name,
    const int *n,
    const int *c,
    const int *h,
    const int *w,
    const DATA_TYPE *dt_input = NULL,
    const DATA_FORMAT *df_input = NULL)
{
    ModelHandleInfo *ihInfo = (ModelHandleInfo *)ih;
    CNN *cnn = (CNN *)ihInfo->cnn;

    std::map<std::string, TensorDesc> modelInputDims =
        getInputDataFormatFromUser(ih, num_input, name, n, c, h, w, dt_input, df_input);
    if (ihInfo->algoPath) {
        const char *algoPath = (const char *)ihInfo->algoPath;
        if (ihInfo->useFileStream) {
            cnn->loadAlgorithmMapFromFileStream(algoPath);
        } else {
            cnn->loadAlgorithmMapFromText(algoPath);
        }
    }
    cnn->ready(modelInputDims);
    cnn->mark_input_output();
    return;
}

void ResizeModelInput(ModelHandle ih,
    const int num_input,
    char **name,
    const int *n,
    const int *c,
    const int *h,
    const int *w,
    const DATA_TYPE *dt_input = NULL,
    const DATA_FORMAT *df_input = NULL)
{
    ModelHandleInfo *ihInfo = (ModelHandleInfo *)ih;
    CNN *cnn = (CNN *)ihInfo->cnn;

    std::map<std::string, TensorDesc> modelInputDims =
        getInputDataFormatFromUser(ih, num_input, name, n, c, h, w, dt_input, df_input);
    cnn->reready(modelInputDims);
}

ResultHandle AllocAllResultHandle(ModelHandle ih)
{
    ModelHandleInfo *ihInfo = (ModelHandleInfo *)ih;
    CNN *cnn = (CNN *)ihInfo->cnn;
    DEVICE_TYPE device = ihInfo->deviceType;

    ResultHandleInner *model_result_ptr = (ResultHandleInner *)malloc(sizeof(ResultHandleInner));
    std::vector<std::string> modelOutputTensorNames = cnn->get_model_output_tensor_names();
    int model_num_outputs = modelOutputTensorNames.size();
    DataDesc *outputArrPtr = (DataDesc *)malloc(sizeof(DataDesc) * model_num_outputs);
    for (int i = 0; i < model_num_outputs; ++i) {
        std::string name = modelOutputTensorNames[i];
        U32 length = name.size();
        length = (length > NAME_LEN) ? NAME_LEN : length;
        memcpy(outputArrPtr[i].name, name.c_str(), length);
        if (length < NAME_LEN) {
            outputArrPtr[i].name[length] = '\0';
        }
        TensorDesc srcDesc = cnn->get_tensor_desc_by_name(name);
        copyTensorDescToDataDesc(srcDesc, &outputArrPtr[i]);
    }
    model_result_ptr->num_outputs = model_num_outputs;
    model_result_ptr->outputArr = outputArrPtr;
    model_result_ptr->deviceType = device;
    return (void *)model_result_ptr;
}

ResultHandle AllocSpecificResultHandle(ModelHandle ih, const int num_outputs, char **outputNames)
{
    ModelHandleInfo *ihInfo = (ModelHandleInfo *)ih;
    CNN *cnn = (CNN *)ihInfo->cnn;
    DEVICE_TYPE device = ihInfo->deviceType;

    ResultHandleInner *model_result_ptr = (ResultHandleInner *)malloc(sizeof(ResultHandleInner));
    int model_num_outputs = num_outputs;
    DataDesc *outputArrPtr = (DataDesc *)malloc(sizeof(DataDesc) * model_num_outputs);
    for (int i = 0; i < num_outputs; i++) {
        U32 length = UNI_MIN(strlen(outputNames[i]), NAME_LEN - 1);
        memcpy(outputArrPtr[i].name, outputNames[i], length);
        if (length < NAME_LEN) {
            outputArrPtr[i].name[length] = '\0';
        }
        std::string name = outputNames[i];
        TensorDesc srcDesc = cnn->get_tensor_desc_by_name(name);
        copyTensorDescToDataDesc(srcDesc, &outputArrPtr[i]);
    }
    model_result_ptr->num_outputs = model_num_outputs;
    model_result_ptr->outputArr = outputArrPtr;
    model_result_ptr->deviceType = device;
    return (void *)model_result_ptr;
}

void SetRuntimeDevice(ModelHandle ih, int cpu_id, DEVICE_TYPE device)
{
    ModelHandleInfo *ihInfo = (ModelHandleInfo *)ih;
    CNN *cnn = (CNN *)ihInfo->cnn;
    cnn->set_runtime_device(cpu_id, device_mapping_user2bolt(device));
    ihInfo->deviceType = device;
}

void SetRuntimeDeviceDynamic(ModelHandle ih)
{
    ModelHandleInfo *ihInfo = (ModelHandleInfo *)ih;
    CNN *cnn = (CNN *)ihInfo->cnn;
    cnn->set_runtime_device_dynamic();
    ihInfo->deviceType = device_mapping_bolt2user(cnn->get_runtime_device());
}

void RunModel(ModelHandle ih, ResultHandle ir, const int num_input, char **inputNames, void **mem)
{
    ModelHandleInfo *ihInfo = (ModelHandleInfo *)ih;
    CNN *cnn = (CNN *)ihInfo->cnn;
    DEVICE_TYPE device = ihInfo->deviceType;
    ResultHandleInner *ir_inner = (ResultHandleInner *)ir;

    for (int index = 0; index < num_input; index++) {
        std::string input_name(inputNames[index]);
        cnn->copy_to_named_input(input_name, (U8 *)(mem[index]));
    }
    cnn->run();

    DataDesc *outputArrPtr = ir_inner->outputArr;
    for (U32 curIndex = 0; curIndex < ir_inner->num_outputs; curIndex++) {
        Tensor output_tensor = cnn->get_tensor_by_name(outputArrPtr[curIndex].name);
        copyTensorDescToDataDesc(output_tensor.get_desc(), &(outputArrPtr[curIndex]));
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
    return (*ir_inner).num_outputs;
}

void GetOutputDataInfoFromResultHandle(ResultHandle ir,
    int num_outputs,
    char **outputNames,
    int *n,
    int *c,
    int *h,
    int *w,
    DATA_TYPE *dt_output,
    DATA_FORMAT *df_output)
{
    ResultHandleInner *ir_inner = (ResultHandleInner *)ir;
    DataDesc *outputArrPtr = (*ir_inner).outputArr;
    for (int i = 0; i < num_outputs; i++) {
        n[i] = outputArrPtr[i].dims[0];
        c[i] = outputArrPtr[i].dims[1];
        h[i] = outputArrPtr[i].dims[2];
        w[i] = outputArrPtr[i].dims[3];
        strcpy(outputNames[i], outputArrPtr[i].name);
        DataType dt = outputArrPtr[i].dt;
        dt_output[i] = dt_mapping_bolt2user(dt);
        df_output[i] = df_mapping_bolt2user(outputArrPtr[i].df);
    }
}

void GetPtrFromResultHandle(ResultHandle ir,
    int num_outputs,
    char **outputNames,
    void **data,
    int *n,
    int *c,
    int *h,
    int *w,
    DATA_TYPE *dt_output,
    DATA_FORMAT *df_output)
{
    ResultHandleInner *ir_inner = (ResultHandleInner *)ir;
    DataDesc *outputArrPtr = (*ir_inner).outputArr;
    for (int i = 0; i < num_outputs; i++) {
        n[i] = outputArrPtr[i].dims[0];
        c[i] = outputArrPtr[i].dims[1];
        h[i] = outputArrPtr[i].dims[2];
        w[i] = outputArrPtr[i].dims[3];
        strcpy(outputNames[i], outputArrPtr[i].name);
        DataType dt = outputArrPtr[i].dt;
        dt_output[i] = dt_mapping_bolt2user(dt);
        df_output[i] = df_mapping_bolt2user(outputArrPtr[i].df);
        data[i] = outputArrPtr[i].dataPtr;
    }
}

void CopyOutputsFromResultHandle(ResultHandle ir, int num_outputs, const int *size, void **data)
{
    ResultHandleInner *ir_inner = (ResultHandleInner *)ir;
    DataDesc *outputArrPtr = (*ir_inner).outputArr;
    for (int i = 0; i < num_outputs; i++) {
        U32 dataSize = size[i];
        memcpy((void *)data[i], (void *)outputArrPtr[i].dataPtr, dataSize);
    }
}

ResultHandle CloneResultHandle(ResultHandle ir)
{
    ResultHandleInner *irInner = (ResultHandleInner *)ir;
    ResultHandleInner *cloneIrInner = new ResultHandleInner();
    *cloneIrInner = *irInner;
    U32 size = sizeof(DataDesc) * cloneIrInner->num_outputs;
    cloneIrInner->outputArr = (DataDesc *)malloc(size);
    memcpy(cloneIrInner->outputArr, irInner->outputArr, size);
    return (ResultHandle)cloneIrInner;
}

void FreeResultHandle(ResultHandle ir)
{
    ResultHandleInner *ir_inner = (ResultHandleInner *)ir;
    DataDesc *outputArrPtr = (*ir_inner).outputArr;
    free(outputArrPtr);
    (*ir_inner).outputArr = nullptr;
    free(ir_inner);
}

void DestroyModel(ModelHandle ih)
{
    ModelHandleInfo *ihInfo = (ModelHandleInfo *)ih;
    if (nullptr == ihInfo) {
        UNI_WARNING_LOG("DestroyModel received null handle.\n");
        return;
    }
    CNN *cnn = (CNN *)ihInfo->cnn;
    if (nullptr != ihInfo->algoPath) {
        const char *algoPath = (const char *)ihInfo->algoPath;
        UNI_THREAD_SAFE(cnn->saveAlgorithmMapToText(algoPath));
    }
    if (nullptr == cnn) {
        UNI_WARNING_LOG("nullptr in DestroyModel. Resource cleared.\n");
    } else {
        delete cnn;
        ihInfo->cnn = nullptr;
    }
    delete ihInfo;
}
