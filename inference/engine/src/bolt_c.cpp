// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "bolt_c_common.h"
#include "../../tensor/src/cpu/tensor_computing_cpu.h"

#define NAME_VALUE_PAIR(x) #x, x
#ifdef _USE_GPU
pthread_mutex_t gpuLock = PTHREAD_MUTEX_INITIALIZER;
#endif

ModelHandle CreateInference(const char *modelPath,
    AFFINITY_TYPE affinity,
    const char *algorithmMapPath,
    bool useFileStream = false)
{
    assert_not_nullptr(__FUNCTION__, NAME_VALUE_PAIR(modelPath));
    ModelHandleInner *handle = nullptr;
    if (modelPath != nullptr) {
        ModelSpec *ms = new ModelSpec();
        AffinityPolicy affinityPolicy = AFFINITY_TYPE2AffinityPolicy(affinity);
        DataType targetDt = getTargetDtFromAffinity(affinityPolicy);
        if (SUCCESS != deserialize_model_from_file(modelPath, ms, targetDt, useFileStream)) {
            UNI_WARNING_LOG("C API %s failed to load model %s.\n", __FUNCTION__, modelPath);
            delete ms;
        } else {
#ifdef _USE_GPU
            if (affinity == GPU) {
                pthread_mutex_lock(&gpuLock);
            }
#endif
            CNN *cnn = new CNN(affinityPolicy, ms->dt, ms->model_name);
            cnn->sort_operators_sequential(ms);
            cnn->initialize_ops(ms);

            handle = new ModelHandleInner();
            handle->cnn = (void *)cnn;
            handle->ms = (void *)ms;
            handle->device = Arch2HARDWARE_TYPE(cnn->get_runtime_device());
            handle->algoPath = (void *)algorithmMapPath;
            handle->useFileStream = useFileStream;
#ifdef _USE_GPU
            if (affinity == GPU) {
                pthread_mutex_unlock(&gpuLock);
            }
#endif
        }
    }
    print_model_handle(handle);
    return (ModelHandle)handle;
}

ModelHandle CreateModel(
    const char *modelFilePath, AFFINITY_TYPE affinity, const char *algorithmMapFilePath)
{
    UNI_DEBUG_LOG(
        "C API %s(%s, %d, %s)...\n", __FUNCTION__, modelFilePath, affinity, algorithmMapFilePath);
    auto handle = CreateInference(modelFilePath, affinity, algorithmMapFilePath, false);
    UNI_DEBUG_LOG("C API %s(%p) end.\n", __FUNCTION__, handle);
    return handle;
}

ModelHandle CreateModelWithFileStream(
    const char *modelFileStream, AFFINITY_TYPE affinity, const char *algorithmMapFileStream)
{
    UNI_DEBUG_LOG("C API %s(%p, %d, %p)...\n", __FUNCTION__, modelFileStream, affinity,
        algorithmMapFileStream);
    auto handle = CreateInference(modelFileStream, affinity, algorithmMapFileStream, true);
    UNI_DEBUG_LOG("C API %s(%p) end.\n", __FUNCTION__, handle);
    return handle;
}

ModelHandle CloneModel(ModelHandle ih)
{
    UNI_DEBUG_LOG("C API %s(%p)...\n", __FUNCTION__, ih);
    ModelHandleInner *cloneHandle = nullptr;
#ifndef _USE_LITE
    ModelHandleInner *ihInfo = (ModelHandleInner *)ih;
    print_model_handle(ihInfo);
    assert_not_nullptr(__FUNCTION__, "ModelHandle", ihInfo);
    if (ihInfo != nullptr) {
        CNN *cnn = (CNN *)ihInfo->cnn;
        assert_not_nullptr(__FUNCTION__, "ModelHandle.cnn", cnn);
        if (cnn != nullptr) {
#ifdef _USE_GPU
            if (ihInfo->device == GPU_MALI || ihInfo->device == GPU_QUALCOMM) {
                pthread_mutex_lock(&gpuLock);
            }
#endif
            cloneHandle = new ModelHandleInner();
            *cloneHandle = *ihInfo;
            CNN *cloneCnn = new CNN();
            *cloneCnn = cnn->clone();
            cloneHandle->cnn = cloneCnn;
#ifdef _USE_GPU
            if (ihInfo->device == GPU_MALI || ihInfo->device == GPU_QUALCOMM) {
                pthread_mutex_unlock(&gpuLock);
            }
#endif
        }
    }
    print_model_handle(cloneHandle);
#endif
    UNI_DEBUG_LOG("C API %s(%p) end.\n", __FUNCTION__, cloneHandle);
    return (ModelHandle)cloneHandle;
}

int GetNumInputsFromModel(ModelHandle ih)
{
    int ret = 0;
    UNI_DEBUG_LOG("C API %s(%p)...\n", __FUNCTION__, ih);
    ModelHandleInner *ihInfo = (ModelHandleInner *)ih;
    assert_not_nullptr(__FUNCTION__, "ModelHandle", ihInfo);
    print_model_handle(ihInfo);
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
    UNI_DEBUG_LOG("C API %s(%p, %d, %p, %p, %p, %p, %p, %p, %p)...\n", __FUNCTION__, ih, num_inputs,
        name, n, c, h, w, dt, df);
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

    std::map<std::string, TensorDesc> inputDescs = cnn->get_input_desc();
    if (num_inputs != (int)inputDescs.size()) {
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
    for (auto iter : inputDescs) {
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
    std::map<std::string, TensorDesc> descs;
    ModelHandleInner *ihInfo = (ModelHandleInner *)ih;
    assert_not_nullptr(__FUNCTION__, "ModelHandle", ihInfo);

    CNN *cnn = (CNN *)ihInfo->cnn;
    assert_not_nullptr(__FUNCTION__, "ModelHandle.cnn", cnn);

    std::map<std::string, TensorDesc> inputDescs = cnn->get_input_desc();
    int num = inputDescs.size();
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

    for (int i = 0; i < num; ++i) {
        std::string inputName = name[i];
        if (inputDescs.find(inputName) == inputDescs.end()) {
            UNI_ERROR_LOG(
                "C API inner function %s received %s is not model input.\n", __FUNCTION__, name[i]);
            continue;
        }
        descs[inputName] = inputDescs[inputName];
        descs[inputName].dt = DATA_TYPE2DataType(dt[i]);
        descs[inputName].df = DATA_FORMAT2DataFormat(df[i]);
        switch (descs[inputName].nDims) {
            case 0:
                break;
            case 1:
                descs[inputName].dims[0] = n[i];
                break;
            case 2:
                descs[inputName].dims[0] = c[i];
                descs[inputName].dims[1] = n[i];
                break;
            case 3:
                descs[inputName].dims[0] = h[i];
                descs[inputName].dims[1] = c[i];
                descs[inputName].dims[2] = n[i];
                break;
            case 4:
                descs[inputName].dims[0] = w[i];
                descs[inputName].dims[1] = h[i];
                descs[inputName].dims[2] = c[i];
                descs[inputName].dims[3] = n[i];
                break;
#if DIM_LEN >= 5
            case 5:
                descs[inputName].dims[0] = w[i];
                descs[inputName].dims[1] = h[i];
                descs[inputName].dims[2] = t[i];
                descs[inputName].dims[3] = c[i];
                descs[inputName].dims[4] = n[i];
                break;
#endif
            default:
                UNI_ERROR_LOG("C API inner function %s can not process input(name:%s) from user.\n",
                    __FUNCTION__, inputName.c_str());
                break;
        }
    }
    return descs;
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
    UNI_DEBUG_LOG("C API %s(%p, %d, %p, %p, %p, %p, %p, %p, %p)...\n", __FUNCTION__, ih, num_inputs,
        name, n, c, h, w, dt, df);
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

#ifdef _USE_GPU
    if (ihInfo->device == GPU_MALI || ihInfo->device == GPU_QUALCOMM) {
        pthread_mutex_lock(&gpuLock);
    }
#endif
    std::map<std::string, TensorDesc> modelInputDims =
        getInputDataFormatFromUser(ih, num_inputs, name, n, c, t, h, w, dt, df);
    if (ihInfo->device == GPU_MALI || ihInfo->device == GPU_QUALCOMM) {
        cnn->loadAlgorithmMap((const char *)ihInfo->algoPath, ihInfo->useFileStream);
    }
    cnn->ready(modelInputDims);
    cnn->mark_input_output();

    ModelSpec *ms = (ModelSpec *)ihInfo->ms;
    CHECK_STATUS(mt_destroy_model(ms));
    delete ms;
    ihInfo->ms = nullptr;
#ifdef _USE_GPU
    if (ihInfo->device == GPU_MALI || ihInfo->device == GPU_QUALCOMM) {
        gcl_finish(OCLContext::getInstance().handle.get());
        pthread_mutex_unlock(&gpuLock);
    }
#endif
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
#ifndef _USE_LITE
    UNI_DEBUG_LOG("C API %s(%p, %d, %p, %p, %p, %p, %p, %p, %p)...\n", __FUNCTION__, ih, num_inputs,
        name, n, c, h, w, dt, df);
    ModelHandleInner *ihInfo = (ModelHandleInner *)ih;
    assert_not_nullptr(__FUNCTION__, "ModelHandle", ihInfo);
    print_model_handle(ihInfo);

    CNN *cnn = (CNN *)ihInfo->cnn;
    assert_not_nullptr(__FUNCTION__, "ModelHandle.cnn", cnn);

#ifdef _USE_GPU
    if (ihInfo->device == GPU_MALI || ihInfo->device == GPU_QUALCOMM) {
        pthread_mutex_lock(&gpuLock);
    }
#endif
    std::map<std::string, TensorDesc> modelInputDims =
        getInputDataFormatFromUser(ih, num_inputs, name, n, c, NULL, h, w, dt, df);
    cnn->reready(modelInputDims);
#ifdef _USE_GPU
    if (ihInfo->device == GPU_MALI || ihInfo->device == GPU_QUALCOMM) {
        pthread_mutex_unlock(&gpuLock);
    }
#endif
    UNI_DEBUG_LOG("C API %s end.\n", __FUNCTION__);
#endif
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
            std::map<std::string, TensorDesc> outputDescs = cnn->get_output_desc();
            int num_outputs = outputDescs.size();
            DataDesc *data = (DataDesc *)UNI_MALLOC(sizeof(DataDesc) * num_outputs);
            int i = 0;
            for (auto iter : outputDescs) {
                std::string name = iter.first;
                U32 length = name.size();
                length = (length > NAME_LEN) ? NAME_LEN : length;
                UNI_MEMCPY(data[i].name, name.c_str(), length);
                if (length < NAME_LEN) {
                    data[i].name[length] = '\0';
                }
                TensorDesc2DataDesc(iter.second, &data[i]);
                i++;
            }
            model_result_ptr->num_data = num_outputs;
            model_result_ptr->data = data;
            model_result_ptr->device = ihInfo->device;
        }
    }
    UNI_DEBUG_LOG("C API %s(%p) end.\n", __FUNCTION__, model_result_ptr);
    print_result_handle(model_result_ptr);
    return (ResultHandle)model_result_ptr;
}

ResultHandle AllocSpecificResultHandle(ModelHandle ih, int num_outputs, const char **name)
{
    ResultHandleInner *model_result_ptr = nullptr;
#ifndef _USE_LITE
    UNI_DEBUG_LOG("C API %s(%p, %d, %p)...\n", __FUNCTION__, ih, num_outputs, name);
    ModelHandleInner *ihInfo = (ModelHandleInner *)ih;
    assert_not_nullptr(__FUNCTION__, "ModelHandle", ihInfo);
    print_model_handle(ihInfo);
    if (ihInfo != nullptr) {
        CNN *cnn = (CNN *)ihInfo->cnn;
        assert_not_nullptr(__FUNCTION__, "ModelHandle.cnn", cnn);
        if (cnn != nullptr) {
            model_result_ptr = (ResultHandleInner *)UNI_MALLOC(sizeof(ResultHandleInner));
            int model_num_outputs = num_outputs;
            DataDesc *data = (DataDesc *)UNI_MALLOC(sizeof(DataDesc) * model_num_outputs);
            for (int i = 0; i < num_outputs; i++) {
                U32 length = UNI_MIN(strlen(name[i]), NAME_LEN - 1);
                UNI_MEMCPY(data[i].name, name[i], length);
                if (length < NAME_LEN) {
                    data[i].name[length] = '\0';
                }
                TensorDesc srcDesc = cnn->get_tensor_desc_by_name(name[i]);
                TensorDesc2DataDesc(srcDesc, &data[i]);
            }
            model_result_ptr->num_data = model_num_outputs;
            model_result_ptr->data = data;
            model_result_ptr->device = ihInfo->device;
        }
    }
    UNI_DEBUG_LOG("C API %s(%p) end.\n", __FUNCTION__, model_result_ptr);
    print_result_handle(model_result_ptr);
#endif
    return (void *)model_result_ptr;
}

void SetRuntimeDevice(ModelHandle ih, int cpu_id, HARDWARE_TYPE device)
{
#ifndef _USE_LITE
    UNI_DEBUG_LOG("C API %s(%p, %d, %d)...\n", __FUNCTION__, ih, cpu_id, device);
    ModelHandleInner *ihInfo = (ModelHandleInner *)ih;
    assert_not_nullptr(__FUNCTION__, "ModelHandle", ihInfo);
    print_model_handle(ihInfo);
    CNN *cnn = (CNN *)ihInfo->cnn;
    assert_not_nullptr(__FUNCTION__, "ModelHandle.cnn", cnn);
    cnn->set_runtime_device(cpu_id, HARDWARE_TYPE2Arch(device));
    ihInfo->device = device;
    UNI_DEBUG_LOG("C API %s end.\n", __FUNCTION__);
#endif
}

void SetRuntimeDeviceDynamic(ModelHandle ih)
{
#ifndef _USE_LITE
    UNI_DEBUG_LOG("C API %s(%p)...\n", __FUNCTION__, ih);
    ModelHandleInner *ihInfo = (ModelHandleInner *)ih;
    assert_not_nullptr(__FUNCTION__, "ModelHandle", ihInfo);
    print_model_handle(ihInfo);
    CNN *cnn = (CNN *)ihInfo->cnn;
    assert_not_nullptr(__FUNCTION__, "ModelHandle.cnn", cnn);
    cnn->set_runtime_device_dynamic();
    ihInfo->device = Arch2HARDWARE_TYPE(cnn->get_runtime_device());
    UNI_DEBUG_LOG("C API %s end.\n", __FUNCTION__);
#endif
}

void SetNumThreads(int threadNum)
{
#ifndef _USE_LITE
    UNI_DEBUG_LOG("C API %s(%d)...\n", __FUNCTION__, threadNum);
    set_cpu_num_threads(threadNum);
    UNI_DEBUG_LOG("C API %s end.\n", __FUNCTION__);
#endif
}

int TransformDataType(ModelHandle ih,
    DATA_TYPE inputType,
    const void *inputData,
    const float *inputScale,
    DATA_TYPE outputType,
    void *outputData,
    float *outputScale,
    unsigned int length)
{
#ifndef _USE_LITE
    UNI_DEBUG_LOG("C API %s(%p, %s, %p, %p %s, %p, %p, %u)...\n", __FUNCTION__, ih,
        GetDataTypeString()[inputType], inputData, inputScale, GetDataTypeString()[outputType],
        outputData, outputScale, length);
    int ret = 0;
    DataType srcType = DATA_TYPE2DataType(inputType);
    DataType dstType = DATA_TYPE2DataType(outputType);
    if (length == 0) {
    } else if (outputScale != NULL && (outputType == UINT_8 || outputType == INT_8)) {
        ModelHandleInner *ihInfo = (ModelHandleInner *)ih;
        assert_not_nullptr(__FUNCTION__, "ModelHandle", ihInfo);
        if (outputType == UINT_8) {
            dstType = DT_U8_Q;
        }
        TensorDesc inputDesc = tensor1d(srcType, length);
        TensorDesc outputDesc = tensor1d(dstType, length);
        CHECK_STATUS(quantize_cpu(inputDesc, inputData, &outputDesc, outputData, outputScale,
            HARDWARE_TYPE2Arch(ihInfo->device), 1));
    } else if (inputType == FP_32) {
        transformFromFloat(dstType, (const float *)inputData, outputData, length);
    } else if (outputType == FP_32) {
        transformToFloat(srcType, inputData, (float *)outputData, length);
    } else {
        UNI_ERROR_LOG("C API %s not support transform data from %s to %s.\n", __FUNCTION__,
            GetDataTypeString()[inputType], GetDataTypeString()[outputType]);
        ret = 1;
    }
    UNI_DEBUG_LOG("C API %s(%d) end.\n", __FUNCTION__, ret);
#else
    int ret = 1;
#endif
    return ret;
}

void RunModel(ModelHandle ih, ResultHandle ir, int num_inputs, const char **name, void **data)
{
    UNI_DEBUG_LOG("C API %s(%p, %p, %d, %p, %p)...\n", __FUNCTION__, ih, ir, num_inputs, name, data);
    RunModelWithType(ih, ir, num_inputs, name, NULL, data, nullptr);
    UNI_DEBUG_LOG("C API %s end.\n", __FUNCTION__);
}

int GetDeviceName(char *gpuDeviceName)
{
    int ret = 0;
    UNI_DEBUG_LOG("C API %s(%p)...\n", __FUNCTION__, gpuDeviceName);
    if (gpuDeviceName == NULL) {
        ret = 1;
    } else {
#ifdef _USE_GPU
        std::string deviceName = OCLContext::getInstance().handle->deviceName;
#else
        std::string deviceName = "unKnown";
#endif
        UNI_STRCPY(gpuDeviceName, deviceName.c_str());
    }
    UNI_DEBUG_LOG("C API %s(%s,%d) end.\n", __FUNCTION__, gpuDeviceName, ret);
    return ret;
}

void *GetDeviceContext()
{
    UNI_DEBUG_LOG("C API %s()...\n", __FUNCTION__);
    void *ret = NULL;
#ifdef _USE_GPU
    ret = &(OCLContext::getInstance().handle->context);
#endif
    UNI_DEBUG_LOG("C API %s(%p)...\n", __FUNCTION__, ret);
    return ret;
}

void *GetDeviceCommandQueue()
{
    UNI_DEBUG_LOG("C API %s()...\n", __FUNCTION__);
    void *ret = NULL;
#ifdef _USE_GPU
    ret = &(OCLContext::getInstance().handle->queue);
#endif
    UNI_DEBUG_LOG("C API %s(%p)...\n", __FUNCTION__, ret);
    return ret;
}

int SetInputOutput(ModelHandle ih, int num, const char **name, void **data)
{
    int ret = 0;
    UNI_DEBUG_LOG("C API %s(%p, %d, %p, %p)...\n", __FUNCTION__, ih, num, name, data);
    ModelHandleInner *ihInfo = (ModelHandleInner *)ih;
    assert_not_nullptr(__FUNCTION__, "ModelHandle", ihInfo);
    print_model_handle(ihInfo);
    CNN *cnn = (CNN *)ihInfo->cnn;
    assert_not_nullptr(__FUNCTION__, "ModelHandle.cnn", cnn);
    cnn->set_input_output(num, name, data);
    UNI_DEBUG_LOG("C API %s(%d) end.\n", __FUNCTION__, ret);
    return ret;
}

void RunModelWithType(ModelHandle ih,
    ResultHandle ir,
    int num_inputs,
    const char **name,
    DATA_TYPE *dt,
    void **data,
    float **scale)
{
    UNI_DEBUG_LOG("C API %s(%p, %p, %d, %p, %p, %p, %p)...\n", __FUNCTION__, ih, ir, num_inputs,
        name, dt, data, scale);
    ModelHandleInner *ihInfo = (ModelHandleInner *)ih;
    assert_not_nullptr(__FUNCTION__, "ModelHandle", ihInfo);
    print_model_handle(ihInfo);
    CNN *cnn = (CNN *)ihInfo->cnn;
    assert_not_nullptr(__FUNCTION__, "ModelHandle.cnn", cnn);
    ResultHandleInner *ir_inner = (ResultHandleInner *)ir;

#ifdef _USE_GPU
    if (ihInfo->device == GPU_MALI || ihInfo->device == GPU_QUALCOMM) {
        pthread_mutex_lock(&gpuLock);
    }
#endif
    if (num_inputs > 0) {
        assert_not_nullptr(__FUNCTION__, NAME_VALUE_PAIR(name));
        assert_not_nullptr(__FUNCTION__, NAME_VALUE_PAIR(data));
        std::vector<U8 *> vec(num_inputs);
        auto ptr = cnn->get_input();
        for (int i = 0; i < num_inputs; i++) {
            if (ptr.count(name[i])) {
                auto &t = ptr[name[i]];
                TensorDesc desc = t->get_desc();
                DATA_TYPE dstType = DataType2DATA_TYPE(desc.dt);
                if (dt != NULL && dt[i] != dstType) {
                    auto p = ((CpuMemory *)(t->get_memory()))->get_ptr();
                    CHECK_REQUIREMENT(TransformDataType(ih, dt[i], data[i], NULL, dstType, p, NULL,
                                          t->length()) == 0);
                    vec[i] = (U8 *)p;
                } else {
                    vec[i] = (U8 *)data[i];
                }
            } else {
                vec[i] = (U8 *)data[i];
            }
        }
#if 0
	    std::map<std::string, U8 *> input;
	    for (int index = 0; index < num_inputs; index++) {
	        assert_not_nullptr(__FUNCTION__, NAME_VALUE_PAIR(data[index]));
	        input[name[index]] = vec[index];
	    }
	    cnn->set_input_by_copy(input);
#else
        std::map<std::string, std::shared_ptr<U8>> input;
        std::map<std::string, F32 *> scale_input;
        for (int index = 0; index < num_inputs; index++) {
            assert_not_nullptr(__FUNCTION__, NAME_VALUE_PAIR(data[index]));
            std::shared_ptr<U8> shared_data(vec[index], [](U8 *ptr) {});
            input[name[index]] = shared_data;
        }
        if (scale != nullptr) {
            for (int index = 0; index < num_inputs; index++) {
                if (scale[index] != nullptr) {
                    scale_input[name[index]] = (F32 *)scale[index];
                }
            }
        }
        cnn->set_input_by_assign(input, scale_input);
#endif
    }
    cnn->run();

    if (ir_inner) {
        print_result_handle(ir_inner);
        DataDesc *p = ir_inner->data;
        assert_not_nullptr(__FUNCTION__, "ResultHandle.data", p);
        for (U32 curIndex = 0; curIndex < ir_inner->num_data; curIndex++) {
            Tensor output_tensor = cnn->get_tensor_by_name(p[curIndex].name);
            if (ihInfo->device == GPU_MALI || ihInfo->device == GPU_QUALCOMM) {
#ifdef _USE_GPU
                auto mem = (OclMemory *)output_tensor.get_memory();
                p[curIndex].data = mem->get_mapped_ptr();
#else
                UNI_WARNING_LOG("this binary not support GPU, please recompile project with GPU "
                                "compile options.\n");
#endif
            } else {
                p[curIndex].data = ((CpuMemory *)(output_tensor.get_memory()))->get_ptr();
            }
            TensorDesc2DataDesc(output_tensor.get_desc(), &(p[curIndex]));
        }
    }
#ifdef _USE_GPU
    if (ihInfo->device == GPU_MALI || ihInfo->device == GPU_QUALCOMM) {
        pthread_mutex_unlock(&gpuLock);
    }
#endif
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
        ret = ir_inner->num_data;
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
    if (num_outputs != (int)ir_inner->num_data) {
        UNI_ERROR_LOG("C API %s received num_outputs %d != num_data %d in ResultHandle.\n",
            __FUNCTION__, num_outputs, ir_inner->num_data);
        return;
    }
    DataDesc *data = ir_inner->data;
    assert_not_nullptr(__FUNCTION__, "ResultHandle.data", data);
    assert_not_nullptr(__FUNCTION__, NAME_VALUE_PAIR(name));
    assert_not_nullptr(__FUNCTION__, NAME_VALUE_PAIR(n));
    assert_not_nullptr(__FUNCTION__, NAME_VALUE_PAIR(c));
    assert_not_nullptr(__FUNCTION__, NAME_VALUE_PAIR(h));
    assert_not_nullptr(__FUNCTION__, NAME_VALUE_PAIR(w));
    assert_not_nullptr(__FUNCTION__, NAME_VALUE_PAIR(dt));
    assert_not_nullptr(__FUNCTION__, NAME_VALUE_PAIR(df));
    for (int i = 0; i < num_outputs; i++) {
        UNI_STRCPY(name[i], data[i].name);
        dt[i] = DataType2DATA_TYPE(data[i].dt);
        df[i] = DataFormat2DATA_FORMAT(data[i].df);
#ifdef _USE_GPU
        if (ir_inner->device == GPU_MALI || ir_inner->device == GPU_QUALCOMM) {
            if (df[i] == NCHWC4) {
                df[i] = NCHW;
            }
        }
#endif
        n[i] = data[i].dims[0];
        c[i] = data[i].dims[1];
        h[i] = data[i].dims[2];
        w[i] = data[i].dims[3];
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
    if (num_outputs != (int)ir_inner->num_data) {
        UNI_ERROR_LOG("C API %s received num_outputs %d != num_data %d in ResultHandle.\n",
            __FUNCTION__, num_outputs, ir_inner->num_data);
        return;
    }
    DataDesc *p = ir_inner->data;
    assert_not_nullptr(__FUNCTION__, "ResultHandle.data", p);
    assert_not_nullptr(__FUNCTION__, NAME_VALUE_PAIR(data));
    for (int i = 0; i < num_outputs; i++) {
        data[i] = p[i].data;
    }
    UNI_DEBUG_LOG("C API %s end.\n", __FUNCTION__);
}

int GetOutputDataFromResultHandleWithTransformFormat(
    ResultHandle ir, int num_outputs, unsigned int *lengths, void **data)
{
    UNI_DEBUG_LOG("C API %s(%p, %d, %p, %p)...\n", __FUNCTION__, ir, num_outputs, lengths, data);
    int ret = 0;
    if (num_outputs > 0) {
        ResultHandleInner *ir_inner = (ResultHandleInner *)ir;
        assert_not_nullptr(__FUNCTION__, "ResultHandle", ir_inner);
        print_result_handle(ir_inner);
        DataDesc *p = ir_inner->data;
        for (int i = 0; i < num_outputs; i++) {
            assert_not_nullptr(__FUNCTION__, NAME_VALUE_PAIR(data[i]));
            TensorDesc sDesc = DataDesc2TensorDesc(p[i]);
            TensorDesc dDesc = sDesc;
            if (!(ir_inner->device == GPU_MALI || ir_inner->device == GPU_QUALCOMM) &&
                (dDesc.df == DF_NCHWC8 || dDesc.df == DF_NCHWC4)) {
                dDesc.df = DF_NCHW;
                unsigned int length = 0;
                if (lengths != NULL) {
                    length = lengths[i];
                    int axis = sDesc.nDims - 2;
                    unsigned int c = sDesc.dims[axis];
                    unsigned int other = tensorNumElements(sDesc) / c;
                    if (length != c * other) {
                        dDesc.dims[axis] = length / other;
                    }
                }
                UNI_DEBUG_LOG("transform data from %s to %s, with lengths[%d]=%u\n.",
                    tensorDesc2Str(sDesc).c_str(), tensorDesc2Str(sDesc).c_str(), i, length);
                if (SUCCESS != transformToNCHW(sDesc, p[i].data, dDesc, data[i])) {
                    ret = 1;
                    break;
                }
            } else {
                if (lengths != NULL && lengths[i] != tensorNumElements(sDesc)) {
                    UNI_ERROR_LOG("lengths[%d]=%u is invalid, desc is %s.\n", i, lengths[i],
                        tensorDesc2Str(sDesc).c_str());
                    ret = 1;
                    break;
                }
                UNI_MEMCPY(data[i], p[i].data, tensorNumBytes(sDesc));
            }
        }
    }
    UNI_DEBUG_LOG("C API %s(%d) end.\n", __FUNCTION__, ret);
    return ret;
}

ResultHandle CloneResultHandle(ResultHandle ir)
{
    ResultHandleInner *clone_ir_inner = nullptr;
#ifndef _USE_LITE
    UNI_DEBUG_LOG("C API %s(%p)...\n", __FUNCTION__, ir);
    ResultHandleInner *ir_inner = (ResultHandleInner *)ir;
    assert_not_nullptr(__FUNCTION__, "ResultHandle", ir_inner);
    print_result_handle(ir_inner);
    if (ir_inner != nullptr) {
        clone_ir_inner = (ResultHandleInner *)UNI_MALLOC(sizeof(ResultHandleInner));
        *clone_ir_inner = *ir_inner;
        U32 size = sizeof(DataDesc) * clone_ir_inner->num_data;
        if (size > 0) {
            clone_ir_inner->data = (DataDesc *)UNI_MALLOC(size);
            DataDesc *data = ir_inner->data;
            assert_not_nullptr(__FUNCTION__, "ResultHandle.data", data);
            UNI_MEMCPY(clone_ir_inner->data, data, size);
        } else {
            clone_ir_inner->data = nullptr;
        }
    }
    UNI_DEBUG_LOG("C API %s(%p) end.\n", __FUNCTION__, clone_ir_inner);
    print_result_handle(clone_ir_inner);
#endif
    return (ResultHandle)clone_ir_inner;
}

void FreeResultHandle(ResultHandle ir)
{
    UNI_DEBUG_LOG("C API %s(%p)...\n", __FUNCTION__, ir);
    ResultHandleInner *ir_inner = (ResultHandleInner *)ir;
    assert_not_nullptr(__FUNCTION__, "ResultHandle", ir_inner);
    print_result_handle(ir_inner);
    if (ir_inner != nullptr) {
        DataDesc *data = ir_inner->data;
        if (ir_inner->num_data > 0) {
            assert_not_nullptr(__FUNCTION__, "ResultHandle.data", data);
            ir_inner->num_data = 0;
        }
        if (data != nullptr) {
            UNI_FREE(data);
            ir_inner->data = nullptr;
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
#ifdef _USE_GPU
    if (ihInfo->device == GPU_MALI || ihInfo->device == GPU_QUALCOMM) {
        pthread_mutex_lock(&gpuLock);
    }
#endif
    if (ihInfo != nullptr) {
        CNN *cnn = (CNN *)ihInfo->cnn;
        assert_not_nullptr(__FUNCTION__, "ModelHandle.cnn", cnn);
        if (cnn != nullptr) {
            ModelSpec *ms = (ModelSpec *)ihInfo->ms;
            if (ms != nullptr) {
                CHECK_STATUS(mt_destroy_model(ms));
                delete ms;
                ihInfo->ms = nullptr;
            }
            if (ihInfo->device == GPU_MALI || ihInfo->device == GPU_QUALCOMM) {
                if (ihInfo->algoPath != nullptr && !ihInfo->useFileStream) {
                    const char *algoPath = (const char *)ihInfo->algoPath;
                    UNI_THREAD_SAFE(cnn->saveAlgorithmMapToFile(algoPath));
                }
            }
            delete cnn;
            ihInfo->cnn = nullptr;
        }
        delete ihInfo;
    }
#ifdef _USE_GPU
    if (ihInfo->device == GPU_MALI || ihInfo->device == GPU_QUALCOMM) {
        pthread_mutex_unlock(&gpuLock);
    }
#endif
    UNI_DEBUG_LOG("C API %s end.\n", __FUNCTION__);
}

void MemoryCheck()
{
#ifndef _USE_LITE
    UNI_DEBUG_LOG("C API %s()...\n", __FUNCTION__);
#ifndef _USE_MEM_CHECK
    UNI_WARNING_LOG("please set USE_MEM_CHECK to ON at common/cmakes/bolt.cmake, and rebuild "
                    "library.\n");
#endif
    UNI_MEM_STATISTICS();
    UNI_DEBUG_LOG("C API %s end.\n", __FUNCTION__);
#endif
}
#undef NAME_VALUE_PAIR
