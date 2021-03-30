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
#include "../api/dllite/Bolt.h"
#include "inference.hpp"
#include "tensor.hpp"

struct ModelHandleInfo {
    void *ms;
    void *cnn;
    DEVICE_TYPE deviceType;
    void *algoPath;
    bool useFileStream;
};

struct DLLiteInfo {
    ModelHandle modelHandle;
    bool isReady;
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

inline AFFINITY_TYPE AffinityMapDLLite2c(bolt::AffinityType affinity)
{
    AFFINITY_TYPE ret = CPU_HIGH_PERFORMANCE;
    switch (affinity) {
        case bolt::AffinityType::CPU_HIGH_PERFORMANCE:
            ret = CPU_HIGH_PERFORMANCE;
            break;
        case bolt::AffinityType::CPU_LOW_POWER:
            ret = CPU_LOW_POWER;
            break;
        case bolt::AffinityType::GPU:
            ret = GPU;
            break;
        default: {
            UNI_ERROR_LOG("Unsupported affinity type in dllite API\n");
        }
    }
    return ret;
}

bolt::TensorType TypeMapBolt2DLLite(DataType dt)
{
    bolt::TensorType ret = bolt::TensorType::FP32;
    switch (dt) {
        case DT_F32:
            ret = bolt::TensorType::FP32;
            break;
#ifdef _USE_FP16
        case DT_F16:
            ret = bolt::TensorType::FP16;
            break;
#endif
        case DT_I32:
            ret = bolt::TensorType::INT32;
            break;
        case DT_U32:
            ret = bolt::TensorType::UINT32;
            break;
        default:
            UNI_ERROR_LOG("unsupported bolt data type in DLLite API\n");
    }
    return ret;
}

DataType TypeMapDLLite2bolt(bolt::TensorType dt)
{
    DataType ret = DT_F32;
    switch (dt) {
        case bolt::TensorType::FP32:
            ret = DT_F32;
            break;
#ifdef _USE_FP16
        case bolt::TensorType::FP16:
            ret = DT_F16;
            break;
#endif
        case bolt::TensorType::INT32:
            ret = DT_I32;
            break;
        case bolt::TensorType::UINT32:
            ret = DT_U32;
            break;
        default:
            UNI_ERROR_LOG("unsupported data type in DLLite API\n");
    }
    return ret;
}

bolt::TensorLayout LayoutMapBolt2DLLite(DataFormat df)
{
    bolt::TensorLayout ret = bolt::TensorLayout::NCHW;
    switch (df) {
        case DF_NCHW:
            ret = bolt::TensorLayout::NCHW;
            break;
        case DF_NHWC:
            ret = bolt::TensorLayout::NHWC;
            break;
        case DF_NCHWC8:
            ret = bolt::TensorLayout::NCHWC8;
            break;
        case DF_MTK:
            ret = bolt::TensorLayout::RNN_MTK;
            break;
        case DF_NORMAL:
            ret = bolt::TensorLayout::ROW_MAJOR;
            break;
        default: {
            UNI_ERROR_LOG("unsupported bolt data layout in DLLite API\n");
        }
    }
    return ret;
}

DataFormat LayoutMapDLLite2bolt(bolt::TensorLayout df)
{
    DataFormat ret = DF_NCHW;
    switch (df) {
        case bolt::TensorLayout::NCHW:
            ret = DF_NCHW;
            break;
        case bolt::TensorLayout::NHWC:
            ret = DF_NHWC;
            break;
        case bolt::TensorLayout::NCHWC8:
            ret = DF_NCHWC8;
            break;
        case bolt::TensorLayout::RNN_MTK:
            ret = DF_MTK;
            break;
        case bolt::TensorLayout::ROW_MAJOR:
            ret = DF_NORMAL;
            break;
        default: {
            UNI_ERROR_LOG("unsupported data layout in DLLite API\n");
        }
    }
    return ret;
}

std::map<std::string, TensorDesc> GetInputInfoFromDLLite(
    bolt::ModelHandle ih, const std::vector<bolt::IOTensor> &inputs)
{
    DLLiteInfo *handle = (DLLiteInfo *)ih;
    ModelHandleInfo *ihInfo = (ModelHandleInfo *)handle->modelHandle;
    CNN *cnn = (CNN *)ihInfo->cnn;
    std::map<std::string, TensorDesc> inputTensorDescs = cnn->get_input_desc();
    int num = inputTensorDescs.size();
    if (num != (int)inputs.size()) {
        UNI_ERROR_LOG(
            "GetInputInfoFromDLLite: model has %d inputs, not %d\n", num, (int)inputs.size());
    }

    std::map<std::string, TensorDesc> modelInputDims;
    for (int i = 0; i < num; ++i) {
        std::string inputName = inputs[i].name;
        if (inputTensorDescs.find(inputName) == inputTensorDescs.end()) {
            UNI_ERROR_LOG(
                "Bolt DLLite API received input %s is not model input.\n", inputName.c_str());
        }
        DataType dt = TypeMapDLLite2bolt(inputs[i].type);
        DataFormat df = LayoutMapDLLite2bolt(inputs[i].layout);
        switch (df) {
            case DF_NORMAL:
                modelInputDims[inputName] =
                    tensor2df(dt, df, inputs[i].shape[0], inputs[i].shape[1]);
                break;
            case DF_NCHW:
                modelInputDims[inputName] = tensor4df(dt, df, inputs[i].shape[0],
                    inputs[i].shape[1], inputs[i].shape[2], inputs[i].shape[3]);
                break;
            case DF_MTK:
                modelInputDims[inputName] =
                    tensor3df(dt, df, inputs[i].shape[0], inputs[i].shape[1], inputs[i].shape[2]);
                break;
            default:
                UNI_ERROR_LOG("unsupported data format in %s\n", __FUNCTION__);
                break;
        }
    }
    return modelInputDims;
}

void UpdateDataDesc(TensorDesc srcDesc, DataDesc *dstDesc)
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

bolt::ModelHandle bolt::CreateModel(const bolt::ModelConfig &modelConfig)
{
    DLLiteInfo *handle = new DLLiteInfo();
    if (nullptr != modelConfig.modelStream.first && modelConfig.modelStream.second > 0) {
        handle->modelHandle = CreateModelWithFileStream((char *)modelConfig.modelStream.first,
            AffinityMapDLLite2c(modelConfig.affinity),
            modelConfig.algoStream.second > 0 ? (char *)modelConfig.algoStream.first : nullptr);
    } else if ("" != modelConfig.modelPath) {
        handle->modelHandle = CreateModel(modelConfig.modelPath.c_str(),
            AffinityMapDLLite2c(modelConfig.affinity), modelConfig.algoPath.c_str());
    } else {
        handle->modelHandle = nullptr;
    }
    handle->isReady = false;
    return (bolt::ModelHandle)handle;
}

bolt::ReturnStatus bolt::GetIOFormats(bolt::ModelHandle modelHandle,
    std::vector<bolt::IOTensor> &inputs,
    std::vector<bolt::IOTensor> &outputs)
{
    DLLiteInfo *handle = (DLLiteInfo *)modelHandle;
    ModelHandleInfo *ihInfo = (ModelHandleInfo *)handle->modelHandle;
    if (nullptr == ihInfo) {
        return bolt::ReturnStatus::NULLPTR;
    }
    CNN *cnn = (CNN *)ihInfo->cnn;
    std::map<std::string, TensorDesc> inputDescMap = cnn->get_input_desc();

    if (ihInfo->algoPath) {
        const char *algoPath = (const char *)ihInfo->algoPath;
        cnn->loadAlgorithmMap(algoPath);
    }
    cnn->ready(inputDescMap);
    cnn->mark_input_output();
    if (ihInfo->algoPath) {
        const char *algoPath = (const char *)ihInfo->algoPath;
        cnn->saveAlgorithmMapToFile(algoPath);
    }
    handle->isReady = true;

    std::map<std::string, std::shared_ptr<Tensor>> inMap = cnn->get_input();
    inputs.clear();

    for (auto iter : inMap) {
        bolt::IOTensor in;
        in.name = iter.first;
        TensorDesc inDesc = iter.second->get_desc();
        in.type = TypeMapBolt2DLLite(inDesc.dt);
        in.shape.clear();
        for (U32 j = 0; j < inDesc.nDims; j++) {
            in.shape.push_back(inDesc.dims[inDesc.nDims - 1 - j]);
        }
        in.layout = LayoutMapBolt2DLLite(inDesc.df);
        inputs.push_back(in);
    }

    std::map<std::string, std::shared_ptr<Tensor>> outMap = cnn->get_output();
    outputs.clear();
    for (auto iter : outMap) {
        IOTensor out;
        out.name = iter.first;
        TensorDesc outDesc = iter.second->get_desc();
        out.type = TypeMapBolt2DLLite(outDesc.dt);
        out.shape.clear();
        for (U32 j = 0; j < outDesc.nDims; j++) {
            out.shape.push_back(outDesc.dims[outDesc.nDims - 1 - j]);
        }
        out.layout = LayoutMapBolt2DLLite(outDesc.df);
        outputs.push_back(out);
    }

    return bolt::ReturnStatus::SUCCESS;
}

bolt::ReturnStatus bolt::PrepareModel(
    bolt::ModelHandle modelHandle, const std::vector<bolt::IOTensor> &inputs)
{
    DLLiteInfo *handle = (DLLiteInfo *)modelHandle;
    if (handle->isReady) {
        return bolt::ReturnStatus::SUCCESS;
    }
    ModelHandleInfo *ihInfo = (ModelHandleInfo *)handle->modelHandle;
    if (nullptr == ihInfo) {
        return bolt::ReturnStatus::NULLPTR;
    }
    CNN *cnn = (CNN *)ihInfo->cnn;

    std::map<std::string, TensorDesc> modelInputDims = GetInputInfoFromDLLite(modelHandle, inputs);
    if (ihInfo->algoPath) {
        const char *algoPath = (const char *)ihInfo->algoPath;
        cnn->loadAlgorithmMap(algoPath);
    }
    cnn->ready(modelInputDims);
    cnn->mark_input_output();
    if (ihInfo->algoPath) {
        const char *algoPath = (const char *)ihInfo->algoPath;
        cnn->saveAlgorithmMapToFile(algoPath);
    }

    ModelSpec *ms = (ModelSpec *)ihInfo->ms;
    CHECK_STATUS(mt_destroy_model(ms));
    delete ms;

    return bolt::ReturnStatus::SUCCESS;
}

bolt::ReturnStatus bolt::GetInputTensors(
    bolt::ModelHandle modelHandle, std::vector<bolt::IOTensor> &inputs)
{
    DLLiteInfo *handle = (DLLiteInfo *)modelHandle;
    ModelHandleInfo *ihInfo = (ModelHandleInfo *)handle->modelHandle;
    if (nullptr == ihInfo) {
        return bolt::ReturnStatus::NULLPTR;
    }
    CNN *cnn = (CNN *)ihInfo->cnn;

    std::map<std::string, std::shared_ptr<Tensor>> inMap = cnn->get_input();

    for (U32 i = 0; i < inputs.size(); i++) {
        auto tensorPtr = inMap[inputs[i].name];
        if (nullptr == tensorPtr) {
            return bolt::ReturnStatus::FAIL;
        }
        inputs[i].buffer.first = ((CpuMemory *)(tensorPtr->get_memory()))->get_ptr();
        inputs[i].buffer.second = tensorPtr->bytes();
    }
    return bolt::ReturnStatus::SUCCESS;
}

bolt::ReturnStatus bolt::ResizeInput(
    bolt::ModelHandle modelHandle, const std::vector<bolt::IOTensor> &inputs)
{
    DLLiteInfo *handle = (DLLiteInfo *)modelHandle;
    ModelHandleInfo *ihInfo = (ModelHandleInfo *)handle->modelHandle;
    if (nullptr == ihInfo) {
        return bolt::ReturnStatus::NULLPTR;
    }
    CNN *cnn = (CNN *)ihInfo->cnn;

    std::map<std::string, TensorDesc> modelInputDims = GetInputInfoFromDLLite(modelHandle, inputs);
    cnn->reready(modelInputDims);
    return bolt::ReturnStatus::SUCCESS;
}

bolt::ResultHandle bolt::AllocResult(
    bolt::ModelHandle modelHandle, const std::vector<bolt::IOTensor> &outputs)
{
    DLLiteInfo *handle = (DLLiteInfo *)modelHandle;
    char **outputNames = (char **)malloc(outputs.size() * sizeof(char *));
    for (size_t i = 0; i < outputs.size(); i++) {
        U32 length = outputs[i].name.length();
        outputNames[i] = (char *)malloc(length + 1);
        memcpy(outputNames[i], outputs[i].name.c_str(), length);
        outputNames[i][length] = '\0';
    }
    bolt::ResultHandle rh = (bolt::ResultHandle)AllocSpecificResultHandle(
        handle->modelHandle, outputs.size(), (const char **)outputNames);
    for (size_t i = 0; i < outputs.size(); i++) {
        free(outputNames[i]);
    }
    free(outputNames);
    return rh;
}

bolt::ReturnStatus bolt::RunModel(bolt::ModelHandle modelHandle,
    bolt::ResultHandle resultHandle,
    const std::vector<bolt::IOTensor> &inputs)
{
    DLLiteInfo *handle = (DLLiteInfo *)modelHandle;
    ModelHandleInfo *ihInfo = (ModelHandleInfo *)handle->modelHandle;
    if (nullptr == ihInfo) {
        return bolt::ReturnStatus::NULLPTR;
    }
    CNN *cnn = (CNN *)ihInfo->cnn;
    DEVICE_TYPE device = ihInfo->deviceType;
    ResultHandleInner *ir_inner = (ResultHandleInner *)resultHandle;

    std::map<std::string, U8 *> input;
    for (size_t index = 0; index < inputs.size(); index++) {
        input[inputs[index].name] = (U8 *)(inputs[index].buffer.first);
    }
    cnn->set_input_by_copy(input);
    cnn->run();

    DataDesc *outputArrPtr = ir_inner->outputArr;
    for (U32 curIndex = 0; curIndex < ir_inner->num_outputs; curIndex++) {
        Tensor output_tensor = cnn->get_tensor_by_name(outputArrPtr[curIndex].name);
        UpdateDataDesc(output_tensor.get_desc(), &(outputArrPtr[curIndex]));
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
    return bolt::ReturnStatus::SUCCESS;
}

bolt::ReturnStatus bolt::GetOutputTensors(
    bolt::ResultHandle resultHandle, std::vector<bolt::IOTensor> &outputs)
{
    ResultHandleInner *ir_inner = (ResultHandleInner *)resultHandle;
    if (nullptr == ir_inner) {
        return bolt::ReturnStatus::NULLPTR;
    }
    DataDesc *outputArrPtr = (*ir_inner).outputArr;

    for (size_t i = 0; i < outputs.size(); i++) {
        U32 n = outputArrPtr[i].dims[0];
        U32 c = outputArrPtr[i].dims[1];
        U32 h = outputArrPtr[i].dims[2];
        U32 w = outputArrPtr[i].dims[3];
        DataType dt = outputArrPtr[i].dt;
        U32 size = n * c * h * w * bytesOf(dt);
        outputs[i].buffer = std::make_pair((void *)outputArrPtr[i].dataPtr, size);
    }
    return bolt::ReturnStatus::SUCCESS;
}

bolt::ReturnStatus bolt::FreeResult(bolt::ResultHandle resultHandle)
{
    if (nullptr == resultHandle) {
        return bolt::ReturnStatus::NULLPTR;
    }
    FreeResultHandle((ResultHandle)resultHandle);
    return bolt::ReturnStatus::SUCCESS;
}

bolt::ReturnStatus bolt::DestroyModel(bolt::ModelHandle modelHandle)
{
    DLLiteInfo *handle = (DLLiteInfo *)modelHandle;
    ModelHandleInfo *ihInfo = (ModelHandleInfo *)handle->modelHandle;

    if (nullptr == ihInfo) {
        UNI_ERROR_LOG("DestroyModel received null handle.\n");
        return bolt::ReturnStatus::NULLPTR;
    }
    CNN *cnn = (CNN *)ihInfo->cnn;
    if (nullptr == cnn) {
        UNI_WARNING_LOG("nullptr in DestroyModel. Resource cleared.\n");
        delete ihInfo;
        return bolt::ReturnStatus::SUCCESS;
    }
    delete cnn;
    delete ihInfo;
    return bolt::ReturnStatus::SUCCESS;
}
