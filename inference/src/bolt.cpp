// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.


// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), 
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, 
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE 
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR 
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


#include <iostream>
#include <limits.h>
#include "inference.hpp"
#include "tensor.hpp"
#include "data_loader.hpp"
#include "result_format.hpp"
#include "utils.hpp"
#include "tensor_desc.h"
#include "../exports/c/bolt.h"

struct IHandleInfo {
    void*       cnn;
    void*       ms;
    DEVICE_TYPE deviceType;
};

typedef struct {
    U32 dims[4] = {0};
    char name[NAME_LEN] = {0};
    DataType dt;
    DataFormat df;
    void* dataPtr;
} DataDesc;

typedef struct {
    U32 num_outputs;
    DataDesc* outputArr;
    DEVICE_TYPE deviceType;
} IResultInner;

DataType dt_mapping_user2bolt(DATA_TYPE dt_user) {
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
	        std::cerr << "[ERROR] unsupported user data type in C API" << std::endl;
            exit(1);
    }
    return ret;
}

DATA_TYPE dt_mapping_bolt2user(DataType dt_bolt) {
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
            std::cerr << "[ERROR] unsupported bolt data type in C API" << std::endl;
            exit(1);
    }
    return ret;
}

DataFormat df_mapping_user2bolt(DATA_FORMAT df_user) {
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
	case NORMAL:
	    ret = DF_NORMAL;
	    break;
        default: {
            std::cerr << "[ERROR] unsupported user data format in C API" << std::endl;
            exit(1);
        }
    }
    return ret;
}

DATA_FORMAT df_mapping_bolt2user(DataFormat df_bolt) {
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
	case DF_NORMAL:
	    ret = NORMAL;
	    break;
        default: {
            std::cerr << "[ERROR] unsupported bolt data format in C API" << std::endl;
            exit(1);
        }
    }
    return ret;
}

CpuAffinityPolicy affinity_mapping(AFFINITY_TYPE affinity_user) {
    CpuAffinityPolicy ret = CPU_AFFINITY_HIGH_PERFORMANCE;
    if (affinity_user == HIGH_PERFORMANCE) {
        ret = CPU_AFFINITY_HIGH_PERFORMANCE;
    } else if (affinity_user == LOW_POWER) {
        ret = CPU_AFFINITY_LOW_POWER;
    } else {
        std::cerr << "[ERROR] unsupported user CPU affinity setting in C API" << std::endl;
        exit(1);
    }
    return ret;
}

inline Arch arch_acquire(AFFINITY_TYPE affinity, DEVICE_TYPE device)
{
    Arch ret = ARM_V8;
    switch(device) {
        case GPU:
            ret = MALI;
            break;
        case CPU: {
            Arch* archs;
            int* cpuids;
            int cpuNum;
            thread_affinity_init(&cpuNum, &archs, &cpuids);
            CpuAffinityPolicy affinityPolicy = affinity_mapping(affinity);
            ret = thread_affinity_set_by_policy(cpuNum, archs, cpuids, affinityPolicy, 0);
            thread_affinity_destroy(&cpuNum, &archs, &cpuids);
            break; 
        }
        default: {
            std::cerr << "[ERROR] unsupported device type in C API" << std::endl;
            exit(1);
        }
    }
    return ret;
}

IHandle model_create(const char* modelPath, AFFINITY_TYPE affinity, DEVICE_TYPE device) {
    ModelSpec* ms = new ModelSpec();
    Arch       arch;

    arch = arch_acquire(affinity, device);
    deserialize_model_from_file(modelPath, ms);
    CNN* cnn = new CNN(arch, ms->dt, ms->model_name);
    cnn->sort_operators_sequential(ms);
    cnn->initialize_ops(ms);
    
    IHandleInfo* handle = new IHandleInfo();
    handle->cnn         = (void*)cnn;
    handle->ms          = (void*)ms;
    handle->deviceType  = device;
    return (IHandle)handle;
}

Vec<TensorDesc> getInputDataFormatFromUser(IHandle ih,
    const int num_input,
    const int* n, const int* c, const int* h, const int* w,
    char** name,
    const DATA_TYPE* dt_input,
    const DATA_FORMAT* df_input)
{
    IHandleInfo* ihInfo = (IHandleInfo*)ih;
    ModelSpec*ms        = (ModelSpec*)ihInfo->ms;
    U32 num             = ms->num_inputs;
    if(num != (U32)num_input) {
        std::cerr << "[ERROR] model has " << num << " input, not " << num_input << std::endl;
        exit(1);
    }

    Vec<TensorDesc> modelInputDims(num);
    for(U32 i = 0; i < num; ++i) {
        std::string inputName = name[i];
        bool findTensorName = false;
        for (U32 j = 0; j < num; ++j) {
            std::string modelName = ms->input_names[j];
            if (modelName == inputName) {
                DataType dt = (dt_input == NULL) ? DT_F32 : dt_mapping_user2bolt(dt_input[i]);
                DataFormat df = (df_input == NULL) ? DF_NCHW : df_mapping_user2bolt(df_input[i]);
                switch (df) {
                    case DF_NORMAL:
                        modelInputDims[j] = tensor2df(dt, df, n[i], c[i]);
                        break;
                    case DF_NCHW:
                        modelInputDims[j] = tensor4df(dt, df, n[i], c[i], h[i], w[i]);
                        break;
                    default:
                        std::cerr << "[ERROR] unsupported data format in " << __func__ << std::endl;
                        exit(1);
                }
                findTensorName = true;
                break;
            }
        }

        if(!findTensorName) {
            std::cerr << "[ERROR] input data " << inputName << " is not a valid model input(";
            for (U32 j = 0; j < num; ++j) {
                std::cerr << ms->input_names[j];
                if (j != num - 1)
                    std::cerr << ", " << std::endl;
            }
            std::cerr << ")" << std::endl;
            exit(1);
        }
    }
    return modelInputDims;
}

void model_ready(IHandle ih,
    const int num_input,
    const int* n, const int* c, const int* h, const int* w,
    char** name,
    const DATA_TYPE* dt_input = NULL,
    const DATA_FORMAT* df_input = NULL)
{
    IHandleInfo* ihInfo = (IHandleInfo*)ih;
    CNN* cnn            = (CNN*)ihInfo->cnn;
    ModelSpec*ms        = (ModelSpec*)ihInfo->ms;

    Vec<TensorDesc> modelInputDims = getInputDataFormatFromUser(ih,
        num_input, n, c, h, w, name, dt_input, df_input);
    cnn->ready(modelInputDims);
    cnn->mark_input_output(ms);
#ifdef _USE_MALI    
    if (ihInfo->deviceType == GPU)
        cnn->mali_prepare();
#endif    
    return;
}

void model_resize_input(IHandle ih,
    const int num_input,
    const int* n, const int* c, const int* h, const int* w,
    char** name,
    const DATA_TYPE* dt_input = NULL,
    const DATA_FORMAT* df_input = NULL)
{
    IHandleInfo* ihInfo = (IHandleInfo*)ih;
    CNN* cnn            = (CNN*)ihInfo->cnn;

    Vec<TensorDesc> modelInputDims = getInputDataFormatFromUser(ih,
        num_input, n, c, h, w, name, dt_input, df_input);
    cnn->infer_output_tensors_size(modelInputDims);
}

IResult IResult_malloc_all(IHandle ih) {
    IHandleInfo* ihInfo = (IHandleInfo*)ih;
    CNN* cnn            = (CNN*)ihInfo->cnn;
    DEVICE_TYPE device  = ihInfo->deviceType;

    IResultInner* model_result_ptr = (IResultInner*)malloc(sizeof(IResultInner));
    HashMap<std::string, std::shared_ptr<Tensor>> outMap = cnn->get_outputs();
    int model_num_outputs = outMap.size();
    DataDesc* outputArrPtr = (DataDesc*)malloc(sizeof(DataDesc) * model_num_outputs);
    int curIndex = 0;
    for (auto iter: outMap) {
        U32 length = iter.first.size();
        memcpy(outputArrPtr[curIndex].name, iter.first.c_str(), length);
        if (length < NAME_LEN)
            outputArrPtr[curIndex].name[length] = '\0';
        curIndex++;
    }
    model_result_ptr->num_outputs = model_num_outputs;
    model_result_ptr->outputArr = outputArrPtr;
    model_result_ptr->deviceType = device;
    return (void*)model_result_ptr;
}

IResult IResult_malloc_part(IHandle ih, const int num_outputs,
    char** outputNames)
{
    IHandleInfo* ihInfo = (IHandleInfo*)ih;
    DEVICE_TYPE device  = ihInfo->deviceType;

    IResultInner* model_result_ptr = (IResultInner*)malloc(sizeof(IResultInner));
    int model_num_outputs = num_outputs;
    DataDesc* outputArrPtr = (DataDesc*)malloc(sizeof(DataDesc) * model_num_outputs);
    for (int i = 0; i < num_outputs; i++) {
        U32 length = strlen(outputNames[i]);
        memcpy(outputArrPtr[i].name, outputNames[i], strlen(outputNames[i]));
        if (length < NAME_LEN)
            outputArrPtr[i].name[length] = '\0';
    }
    model_result_ptr->num_outputs = model_num_outputs;
    model_result_ptr->outputArr = outputArrPtr;
    model_result_ptr->deviceType = device;
    return (void*)model_result_ptr;
}

void copyTensorDescToDataDesc(TensorDesc srcDesc, DataDesc *dstDesc) {
    dstDesc->dt = srcDesc.dt;
    dstDesc->df = srcDesc.df;
    if (srcDesc.nDims > 4) {
        std::cerr << "[ERROR] user interface only support 4 dimensions, not " << srcDesc.nDims << std::endl;
        exit(1);
    }
    for (U32 i = 0; i < srcDesc.nDims; i++)
        dstDesc->dims[i] = srcDesc.dims[srcDesc.nDims-1-i];
    for (int i = srcDesc.nDims; i < 4; i++)
        dstDesc->dims[i] = 1;
}

void model_run(IHandle ih, IResult ir, const int num_input, char** inputNames, void** mem) {
    IHandleInfo* ihInfo = (IHandleInfo*)ih;
    CNN* cnn            = (CNN*)ihInfo->cnn;
    DEVICE_TYPE device  = ihInfo->deviceType;
    IResultInner* ir_inner = (IResultInner*)ir;
   
    if(device == CPU){
        for (int index = 0; index < num_input; index++) {
            std::string input_name(inputNames[index]);
            cnn->copy_to_named_input(input_name, (U8*)(mem[index]));
        }
    } else if(device == GPU){
        HashMap<std::string, std::shared_ptr<U8>> modelInputTensors;
        for (int index = 0; index < num_input; ++index) {
            U8* tmp = (U8*)(mem[index]);
            std::shared_ptr<U8> tensorPointer(tmp);
            modelInputTensors.insert(std::pair(inputNames[index], tensorPointer));
        }
        cnn->set_input_tensors_value(modelInputTensors);//GPU will copy data to GPU memory
    } else {
        std::cerr << "[ERROR] unsupported device type in " << __func__ << std::endl;
        exit(1);
    }
    cnn->run();
    
    DataDesc* outputArrPtr = ir_inner->outputArr;
    if(device == CPU) {
        for (U32 curIndex = 0; curIndex < ir_inner->num_outputs; curIndex++) {
            Tensor output_tensor = cnn->get_tensor_by_name(outputArrPtr[curIndex].name);
            copyTensorDescToDataDesc(output_tensor.get_desc(), &(outputArrPtr[curIndex]));
            outputArrPtr[curIndex].dataPtr = output_tensor.get_val();
        }
    } 
#ifdef _USE_MALI
    else if(device == GPU) {
        HashMap<std::string, std::shared_ptr<Tensor>> outMap = cnn->get_outputs();
//        HashMap<std::string, std::tuple<TensorDesc, U8*>> outMap = cnn->get_outputs_mali_map();
        if (ir_inner->num_outputs != outMap.size()) {
            std::cerr << "[ERROR] GPU currently not support IResult_malloc_part" << std::endl;
            exit(1);
        }
        
        int curIndex = 0;
        for(const auto &p : outMap) {
            std::string   output_name = p.first;
            copyTensorDescToDataDesc(p.second->get_desc(), &(outputArrPtr[curIndex]));
            outputArrPtr[curIndex].dataPtr = p.second ->get_val();
//            copyTensorDescToDataDesc(std::get<0>(p.second), &(outputArrPtr[curIndex]));
//            outputArrPtr[curIndex].dataPtr = std::get<1>(p.second);
            curIndex++;
        }
    }
#endif
}

int IResult_num_outputs(IResult ir) {
    IResultInner* ir_inner = (IResultInner*)ir;
    return (*ir_inner).num_outputs;
}

void IResult_get(IResult ir,
    int num_outputs,
    char** outputNames,
    void** data,
    int* n, int* c, int* h, int* w,
    DATA_TYPE* dt_output, DATA_FORMAT* df_output)
{
    IResultInner* ir_inner = (IResultInner*)ir;
    DataDesc* outputArrPtr = (*ir_inner).outputArr;
    for (int i = 0; i < num_outputs; i++) {
        n[i] = outputArrPtr[i].dims[0];
        c[i] = outputArrPtr[i].dims[1];
        h[i] = outputArrPtr[i].dims[2];
        w[i] = outputArrPtr[i].dims[3];
        outputNames[i] = const_cast<char*>(outputArrPtr[i].name);
        DataType dt = outputArrPtr[i].dt;
        dt_output[i] = dt_mapping_bolt2user(dt);
        df_output[i] = df_mapping_bolt2user(outputArrPtr[i].df);
        data[i] = outputArrPtr[i].dataPtr;
    }
}

void IResult_get_nocopy(IResult ir,
    int num_outputs,
    char** outputNames,
    void** data,
    int* n, int* c, int* h, int* w,
    DATA_TYPE* dt_output, DATA_FORMAT* df_output)
{
    IResultInner* ir_inner = (IResultInner*)ir;
    DataDesc* outputArrPtr = (*ir_inner).outputArr;
    for (int i = 0; i < num_outputs; i++) {
        n[i] = outputArrPtr[i].dims[0];
        c[i] = outputArrPtr[i].dims[1];
        h[i] = outputArrPtr[i].dims[2];
        w[i] = outputArrPtr[i].dims[3];
        outputNames[i] = const_cast<char*>(outputArrPtr[i].name);
        DataType dt = outputArrPtr[i].dt;
        dt_output[i] = dt_mapping_bolt2user(dt);
        df_output[i] = df_mapping_bolt2user(outputArrPtr[i].df);
        data[i] = outputArrPtr[i].dataPtr;
    }
}

void IResult_free(IResult ir) {
    IResultInner* ir_inner = (IResultInner*)ir;
    DataDesc* outputArrPtr = (*ir_inner).outputArr;
    free(outputArrPtr);
    free(ir_inner);
}

void model_destroy(IHandle ih) {
    IHandleInfo* ihInfo = (IHandleInfo*)ih;
    CNN* cnn            = (CNN*)ihInfo->cnn;
    ModelSpec* ms       = (ModelSpec*)ihInfo->ms;
    CHECK_STATUS(mt_destroy_model(ms));
    delete ms;
    delete cnn;
}

