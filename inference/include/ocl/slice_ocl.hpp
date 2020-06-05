// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), 
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, 
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE 
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR 
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


#ifndef _SLICE_OCL_H
#define _SLICE_OCL_H

#include "operator.hpp"
#include "tensor_computing.h"
#include "slice.hpp"

class SliceOCL: public Slice
{
public:
    SliceOCL(DataType dt, I32 axis, I32* slicePointsPtr, I32 sliceSize) : Slice(dt, axis, slicePointsPtr, sliceSize) {}


    virtual void run() override
    {
        UTIL_TIME_TIC(__CLASS_FUNCTION__)
        this->handle->curOpName = this->get_op_name();
        Tensor inputTensor = this->inputTensors[0];
        TensorDesc inputDesc = inputTensor.get_desc();

        Vec<Tensor> outputTensors = this->get_output_tensors();
        Vec<TensorDesc> outputTensorDescs;
        Vec<void*> outputPtrs;
        for (U32 i = 0; i < outputTensors.size(); i++) {
            outputTensors[i].set_scale(inputTensor.get_scale());
            outputTensorDescs.push_back(outputTensors[i].get_desc());
            outputPtrs.push_back(outputTensors[i].get_val());
        }
        CHECK_STATUS(slice(inputDesc, inputTensor.get_val(), this->axis, outputTensorDescs, &outputPtrs, this->schedule, &this->oclExtInfo));
        UTIL_TIME_TOC(__CLASS_FUNCTION__)
    }

    virtual EE infer_output_tensors_size(Vec<TensorDesc>inDims, Vec<TensorDesc>* outDims) override
    {
        this->oclExtInfo.maliInfo.gclmemInputDesc =  NULL;
        this->oclExtInfo.maliInfo.gclmemOutputDesc = NULL;
        CHECK_STATUS(slice_infer_output_size(inDims[0], outDims, this->axis, this->slicePoints.data(), this->schedule, &this->oclExtInfo));
        return SUCCESS;
    }

    virtual EE infer_gclmem_desc(Vec<GCLMemDesc>* gclmemInputDesc, Vec<GCLMemDesc>* gclmemOutputDesc) override
    {
        TensorDesc inputDesc  = this->inputTensors[0].get_desc();
        Vec<TensorDesc> outputDesc;
        U32 sliceNum = (*gclmemOutputDesc).size();
        for(U32 i = 0; i < sliceNum; ++i) outputDesc.push_back(this->outputTensors[i].get_desc());
        GCLMemDesc_t memOutDesc = (GCLMemDesc_t) operator new(sizeof(struct GCLMemDesc) * sliceNum) ;
        for(U32 i = 0; i < sliceNum; i++) memOutDesc[i] = (*gclmemOutputDesc)[i];
        this->oclExtInfo.maliInfo.gclmemInputDesc =  &((*gclmemInputDesc)[0]);
        this->oclExtInfo.maliInfo.gclmemOutputDesc = memOutDesc;
        CHECK_STATUS(slice_infer_output_size(inputDesc, &outputDesc, this->axis, this->slicePoints.data(), this->schedule, &this->oclExtInfo));
        for(U32 i = 0; i < sliceNum; i++) (*gclmemOutputDesc)[i] = memOutDesc[i];
        delete memOutDesc;
        return SUCCESS;
    }
};

#endif //_SLICE_OCL_H
