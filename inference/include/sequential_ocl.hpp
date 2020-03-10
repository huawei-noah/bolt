// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), 
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, 
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE 
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR 
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifdef _USE_MALI
#ifndef _SEQUENTIAL_OCL_HPP
#define _SEQUENTIAL_OCL_HPP

#include <cstring>
#include "sys.h"
#include "error.h"
#include "type.h"
#include <string>
#include "tensor.hpp"
#include "operator.hpp"
#include "model.hpp"
#include "op_type.h"
#include "tensor_desc.h"
#include "memory.hpp"
#include "weight_operator.hpp"
#include "pooling.hpp"
#include "convolution.hpp"
#include "bilateral_slice_apply.hpp"
#include "ocl/pooling_ocl.hpp"
#include "ocl/memory_ocl.hpp"
#include "ocl/convolution_ocl.hpp"
#include "ocl/bilateral_slice_apply_ocl.hpp"
#include "ocl/fully_connected_eltwise_ocl.hpp"
#include "ocl/scale_ocl.hpp"
#include "libkernelbin.h"


class SequentialOcl : public Model {
public:
    SequentialOcl(Arch A, DataType dt, std::string name) : Model(A, dt, name) {
        gclTempMem = NULL;
        input_output_same = false;
    }
    virtual ~SequentialOcl(){
        if(gclTempMem) {
            CHECK_STATUS(gcl_release_memory(this->gclTempMem));
            gcl_destroy_gclmem(gclTempMem);
        }
    }

    EE ready(Vec<TensorDesc> dims, std::shared_ptr<U8> modelPtr, U32 numOutput)
    {
        CHECK_STATUS(this->ops[0]->set_mali_handle(this->handle));
        this->ops[0]->set_op_schedule(this->schedule);
        input_output_same = this->ops[0]->can_input_output_the_same();
        CHECK_STATUS(this->infer_output_tensors_size(dims, numOutput));
        Vec<Tensor> inTensors;
        Vec<Tensor> outTensors;
        for(U32 i = 0; i < inputTensors.size(); i++)  inTensors.push_back(*inputTensors[i].get());
        for(U32 i = 0; i < outputTensors.size(); i++) outTensors.push_back(*outputTensors[i].get());
        this->ops[0]->set_input_output_tensors(inTensors, outTensors);

        if(this->ops[0]->is_weight()){
            U8* curPtr = modelPtr.get();
            if(this->ops[0]->get_op_type() == OT_Conv){
                auto convOpPtr = dynamic_cast<Convolution*>(this->ops[0].get());
                auto weightOp = (WeightOperator*) convOpPtr;
                weightOp->set_hasBias(true);
                CHECK_STATUS(convOpPtr->init_weight_bias_from_model(&curPtr));
                CHECK_STATUS(convOpPtr->infer_forward_algorithm(this->algorithmMap));
                CHECK_STATUS(convOpPtr->transform_filter());
            }
            if(this->ops[0]->get_op_type() == OT_FC){
                auto fcOpPtr = dynamic_cast<FullyConnectedEltwise*>(this->ops[0].get());
                auto weightOp = (WeightOperator*) fcOpPtr;
                weightOp->set_hasBias(true);
                CHECK_STATUS(fcOpPtr->init_weight_bias_from_model(&curPtr));
                CHECK_STATUS(fcOpPtr->transform_filter());
            }
            if(this->ops[0]->get_op_type() == OT_Scale){
                auto scaleOpPtr = dynamic_cast<Scale*>(this->ops[0].get());
                auto weightOp = (WeightOperator*) scaleOpPtr;
                weightOp->set_hasBias(true);
                CHECK_STATUS(scaleOpPtr->init_weight_bias_from_model(&curPtr));
            }
        }
        this->infer_tmp_memory_size();
        this->assign_tmp_tensor();
        this->alloc_output_host_tensors(numOutput);
        return SUCCESS;
    }

    virtual EE infer_output_tensors_size(Vec<TensorDesc>) override{return NOT_SUPPORTED;}
    virtual void assign_output_tensor() override{}
    EE infer_output_tensors_size(Vec<TensorDesc> dims, U32 outputTensorNum)
    {
        /*infer GCLMemDesc & alloc gpu mem for tensors*/
        UNUSED(dims);
        Vec<TensorDesc> inTensorDescs;
        Vec<TensorDesc> outTensorDescs;
        Vec<GCLMemDesc> inGCLMemDescs;
        Vec<GCLMemDesc> outGCLMemDescs;
        for (U32 i = 0; i < dims.size(); ++i){
            inTensorDescs.push_back(dims[i]);
            U32 stride[3] = {0, 0, 0};
            U32 offset[3] = {0, 0, 0};
            GCLMemDesc tmpDesc = gcl_mem_desc(stride, offset, DT_U8, DF_NCWHC4);
            inGCLMemDescs.push_back(tmpDesc);
        }

        for (U32 i = 0; i < outputTensorNum; ++i){
            TensorDesc tmpDesc;
            outTensorDescs.push_back(tmpDesc);
            U32 stride[3] = {0, 0, 0};
            U32 offset[3] = {0, 0, 0};
            GCLMemDesc gclTmpDesc = gcl_mem_desc(stride, offset, DT_U8, DF_NCWHC4);
            outGCLMemDescs.push_back(gclTmpDesc);
        }

        CHECK_STATUS(this->ops[0]->infer_output_tensors_size(inTensorDescs, &outTensorDescs, &inGCLMemDescs, &outGCLMemDescs));

        if(!input_output_same) {
            for(U32 i = 0; i < dims.size(); ++i){
                std::shared_ptr<Tensor> tmpTensor(new Tensor(this->handle));
                tmpTensor->set_desc(inTensorDescs[i]);
                Memory_* tmpmem = tmpTensor->get_memory();
                OclMemory* mem = (OclMemory*)tmpmem;
                mem->set_mem_desc(inGCLMemDescs[i]);
                tmpTensor->alloc();
                inputTensors.push_back(tmpTensor);
            }
        }

        for(U32 i = 0; i < outputTensorNum; ++i){
            std::shared_ptr<Tensor> tmpTensor(new Tensor(this->handle));
            tmpTensor->set_desc(outTensorDescs[i]);
            Memory_* tmpmem = tmpTensor->get_memory();
            OclMemory* mem = (OclMemory*)tmpmem;
            mem->set_mem_desc(outGCLMemDescs[i]);
            tmpTensor->alloc();
            outputTensors.push_back(tmpTensor);
            if(input_output_same) inputTensors.push_back(tmpTensor);
        }
        return SUCCESS;
    }


    void alloc_output_host_tensors(U32 outputTensorNum){
        for(U32 i = 0; i < outputTensorNum; i++){
            std::shared_ptr<Tensor> tmpTensor(new Tensor());
            tmpTensor->set_desc(outputTensors[i]->get_desc());
            std::shared_ptr<GCLMem> val = outputTensors[i]->get_shared_ptr();
            auto host_desc   = outputTensors[i]->get_desc();
            auto device_desc = val->desc;
            auto host_df     = host_desc.df;
            auto device_df   = device_desc.memFormat;
            if(host_df == device_df){
                CHECK_STATUS(gcl_release_memory(val.get()));
                val->desc.use_map = true;
                val->desc.flags   = CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR;
                outputTensors[i]->set_shared_ptr(val);
                outputTensors[i]->alloc();
            }
            tmpTensor->alloc();
            outputTensorsOCLHost.push_back(tmpTensor);
        }
    }

    void infer_tmp_memory_size() override
    {
        maxTmpElements = 0;
        for (auto op: this->ops) {
            auto len = op->infer_tmp_memory_size();
            if(len > maxTmpElements) maxTmpElements = len;
        }
    }

    void assign_tmp_tensor() override
    {
        // design for serial , if parallel running should redesign
        if(this->schedule == MALI){
            gclTempMem = gcl_create_gclmem();
            gclTempMem->desc.memType  = GCL_MEM_BUF;
            gclTempMem->desc.byteSize = maxTmpElements;
            gclTempMem->desc.flags    = CL_MEM_READ_WRITE;
            gclTempMem->desc.host_ptr = nullptr;
            if(maxTmpElements) CHECK_STATUS(gcl_create_memory(this->handle.get(), gclTempMem));
            for(auto op: this->ops) {
                op->set_tmp_gclmem(this->maxTmpElements, gclTempMem);
            }
        } else {
    //        auto sPtr = std::shared_ptr<U8>((U8*)operator new(this->maxTmpElements));
    //	    for (auto op: this->ops) {
    //		    op->set_tmp_memory(this->maxTmpElements, sPtr);
    //	    }
        }
    }


    void add(std::shared_ptr<Operator> op)
    {
        this->ops.push_back(op);
    }

    void mark_input_output()
    {
        if(this->schedule == MALI){
           U32 tmpBufSize = 0;
           for(U32 i = 0; i < inputTensors.size(); i++) {
               Tensor* inputTensor = inputTensors[i].get();
               TensorDesc desc = inputTensor->desc;
               GCLMem_t   mem  = inputTensor->get_val();
               U32        size = 0;
               tensor_computing_set_input_infer_tmpBuf_size(mem, desc, &size, MALI); 
               tmpBufSize = (tmpBufSize < size) ? size : tmpBufSize;
           }

           for(U32 i = 0; i < outputTensors.size(); i++) {
               Tensor* outputTensor = outputTensors[i].get();
               TensorDesc desc = outputTensor->desc;
               GCLMem_t   mem  = outputTensor->get_val();
               U32        size = 0;
               tensor_computing_get_output_infer_tmpBuf_size(mem, desc, &size, MALI); 
               tmpBufSize = (tmpBufSize < size) ? size : tmpBufSize;
           }

           if(tmpBufSize > maxTmpElements){
               maxTmpElements = tmpBufSize;
               gcl_release_memory(gclTempMem);
               gclTempMem->desc.memType  = GCL_MEM_BUF;
               gclTempMem->desc.byteSize = this->maxTmpElements;
               gclTempMem->desc.flags    = CL_MEM_READ_WRITE;
               gclTempMem->desc.host_ptr = nullptr;
               CHECK_STATUS(gcl_create_memory(this->handle.get(), this->gclTempMem));
               for(auto op: this->ops) {
                   op->set_tmp_gclmem(this->maxTmpElements, gclTempMem);
               }
           }
           DataType dt = inputTensors[0]->desc.dt;
           gclTempMem->desc.stride[0] = this->maxTmpElements / bytesOf(dt);
           gclTempMem->desc.stride[1] = 1;
           gclTempMem->desc.stride[2] = 1;
           gclTempMem->desc.memFormat = DF_NCHW;
        }
    }

    void mali_prepare(){
        Model::run_mali_prepare();
        CHECK_STATUS(gcl_finish(this->handle.get())); 
    }

    void set_input_tensors(Vec<Tensor> modelInputTensors)
    {
        for(U32 i = 0; i < modelInputTensors.size(); i++){
            U8*        tmpTensorPtr  = modelInputTensors[i].get_val();
            TensorDesc tmpTensorDesc = modelInputTensors[i].get_desc();
            Memory_*   mem = inputTensors[i]->get_memory();
            OclMemory* oclMem = (OclMemory*) mem;
            oclMem->set_tmpBuf(gclTempMem);
            oclMem->set_val_from_hostptr(tmpTensorDesc, tmpTensorPtr, CL_FALSE);
        }
        gcl_finish(this->handle.get());
    }

    Vec<std::shared_ptr<Tensor>> get_output_tensors()
    {
        for(U32 i = 0; i < outputTensorsOCLHost.size(); i++) {
            auto outputTensor    = outputTensorsOCLHost[i];
            auto outputTensorOcl = outputTensors[i];
            auto mem = (OclMemory*)outputTensorOcl->get_memory();
            std::shared_ptr<GCLMem> val = outputTensorOcl->get_shared_ptr();
            auto host_desc   = outputTensor->get_desc();
            mem->set_tmpBuf(gclTempMem);
            U8* tmpVal  = outputTensor->get_val();
            val->desc.use_map = false;//this API need to copy memory
            mem->get_val_to_hostptr(host_desc, &tmpVal, CL_FALSE);
        }
        gcl_finish(this->handle.get());
        return this->outputTensorsOCLHost;
    }

    Vec<U8*> get_output_tensors_map()
    {
        for(U32 i = 0; i < outputTensorsOCLHost.size(); i++) {
            auto outputTensorOcl = outputTensors[i];
            auto mem = (OclMemory*)outputTensorOcl->get_memory();
            std::shared_ptr<GCLMem> val = outputTensorOcl->get_shared_ptr();
            auto host_desc   = outputTensorOcl->get_desc();
            auto device_desc = val->desc;
            auto host_df   = host_desc.df;
            auto device_df = device_desc.memFormat;
            mem->set_tmpBuf(gclTempMem);
            if(host_df == device_df){
                U8* tmpVal = NULL;
                mem->get_val_to_hostptr(host_desc, &tmpVal, CL_FALSE);
                outputTensorsOCLHostMap.push_back(tmpVal);
            } else {
                CHECK_STATUS(NOT_SUPPORTED);
            }
        }
        gcl_finish(this->handle.get());
        return this->outputTensorsOCLHostMap;
    }
#ifdef _USE_MALI
#else

    //TODO 0823
    EE ConvBiasAssignmentAndWeightTransform() {
        return SUCCESS;
    }

    //TODO 0823
    EE FCBiasAssignmentAndWeight() {
        return SUCCESS;
    }




#endif
private:
    using Model::ready;
    U32 maxTmpElements;
    Vec<std::shared_ptr<Tensor>> inputTensors;
    Vec<std::shared_ptr<Tensor>> outputTensors;
    Vec<std::shared_ptr<Tensor>> outputTensorsOCLHost;
    Vec<U8*> outputTensorsOCLHostMap;
    GCLMem_t    gclTempMem;
    bool input_output_same;
};
#endif
#endif

