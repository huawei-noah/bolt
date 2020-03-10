// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), 
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, 
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE 
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR 
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


#ifndef _BILATERAL_SLICE_APPLY_OCL_H
#define _BILATERAL_SLICE_APPLY_OCL_H
#include <math.h>
#include "operator.hpp"
#include "tensor_computing.h"
#include "tensor_desc.h"
#include "op_type.h"
#include "bilateral_slice_apply.hpp"

class BilateralSliceApplyOCL: public BilateralSliceApply {
public:

/**
 * @param coefficient_len
 * @param has_offset
 */
    BilateralSliceApplyOCL(U32 coefficient_len, bool has_offset, BilateralSliceApplyMode mode) : BilateralSliceApply(coefficient_len, has_offset, mode){} 
    virtual ~BilateralSliceApplyOCL(){}
    virtual void run() override
    {
        UTIL_TIME_TIC(__CLASS_FUNCTION__)
        Tensor inputTensor = this->inputTensors[0];
        Tensor gridTensor  = this->inputTensors[1];
        TensorDesc inputDesc = inputTensor.get_desc();
        TensorDesc gridDesc  = gridTensor.get_desc();
        BilateralSliceApplyDesc bilateralSliceDesc = BilateralSliceApply::create_BilateralSliceApplyDesc(this->coefficient_len, this->has_offset, this->mode);
        Tensor outputTensor = this->outputTensors[0];
        TensorDesc outputDesc = outputTensor.get_desc();

        U8* guidePtr = NULL;
        DataFormat df;
        U32 in, ih, iw;
        tensorSelectGet(inputDesc, NULL, &df, &in, NULL, &ih, &iw);
        TensorDesc guideDesc = tensor4df(DT_F16, df, in, 1, ih, iw);
        if(mode == BSliceApply_NULL) guidePtr = this->inputTensors[2].get_val();

        CHECK_STATUS(bilateral_slice_apply(inputDesc, inputTensor.get_val(),
                                           guideDesc, guidePtr,
                                           gridDesc,  gridTensor.get_val(),
                                           bilateralSliceDesc, this->lenOfTemp, this->gclTempMem,
                                           outputDesc, outputTensor.get_val(), this->schedule, &this->oclExtInfo));
        UTIL_TIME_TOC(__CLASS_FUNCTION__)
    }

    virtual EE infer_output_tensors_size(Vec<TensorDesc> inDims, Vec<TensorDesc>* outDims) override
    {
        UNUSED(inDims);
        UNUSED(outDims);
        CHECK_STATUS(NOT_SUPPORTED);
        return NOT_SUPPORTED;
    }

    virtual EE infer_output_tensors_size(Vec<TensorDesc> inDims, Vec<TensorDesc>* outDims, Vec<GCLMemDesc>* gclmemInputDesc, Vec<GCLMemDesc>* gclmemOutputDesc) override
    {
        auto inDim    = inDims[0];
        auto gridDim  = inDims[1];
        DataType dt;
        DataFormat df;
        U32 width;
        U32 height;
        U32 numChannels;
        U32 num;
        CHECK_STATUS(tensor4dGet(inDim, &dt, &df, &num, &numChannels, &height, &width));
        TensorDesc inputDesc = tensor4df(dt,     df, num, numChannels, height, width);
        TensorDesc guideDesc = tensor4df(DT_F16, df, num, 1,           height, width);

        CHECK_STATUS(tensor4dGet(gridDim, &dt, &df, &num, &numChannels, &height, &width));
        TensorDesc gridDesc = tensor4df(dt, df, num, numChannels, height, width);

        BilateralSliceApplyDesc bilateralSliceApplyDesc = BilateralSliceApply::create_BilateralSliceApplyDesc(this->coefficient_len, this->has_offset, this->mode);
        U32 stride[3] = {0, 0, 0};
        U32 offset[3] = {0, 0, 0};
        GCLMemDesc tmpDesc = gcl_mem_desc(stride, offset, DT_U8, DF_NCWHC4);
        GCLMemDesc gclmemDesc[3];
        gclmemDesc[0] = tmpDesc;
        gclmemDesc[1] = tmpDesc;
        gclmemDesc[2] = tmpDesc;
        this->oclExtInfo.maliInfo.gclmemInputDesc = gclmemDesc;
        this->oclExtInfo.maliInfo.gclmemOutputDesc = &((*gclmemOutputDesc)[0]);
        CHECK_STATUS(bilateral_slice_apply_infer_output_size(inputDesc, guideDesc, gridDesc, bilateralSliceApplyDesc, &((*outDims)[0]), this->schedule, &this->oclExtInfo));
        (*gclmemInputDesc)[0] = gclmemDesc[0];//inputDesc
        (*gclmemInputDesc)[1] = gclmemDesc[1];//gridDesc
        if(this->mode == BSliceApply_NULL)(*gclmemInputDesc)[2] = gclmemDesc[2];//guideDesc
        return SUCCESS;
    }

    virtual U32 infer_tmp_memory_size()override
    {
        TensorDesc inputDesc = (this->inputTensors[0]).desc;
        TensorDesc gridDesc  = (this->inputTensors[1]).desc;
        DataFormat df;
        U32 in, ih, iw;
        tensorSelectGet(inputDesc, NULL, &df, &in, NULL, &ih, &iw);
        TensorDesc guideDesc = tensor4df(DT_F16, df, in, 1, ih, iw);

        BilateralSliceApplyDesc bilateralSliceApplyDesc = BilateralSliceApply::create_BilateralSliceApplyDesc(this->coefficient_len, this->has_offset, this->mode);
        U32 bytes = 0;
        CHECK_STATUS(bilateral_slice_apply_infer_forward_tmp_bytes(inputDesc, guideDesc, gridDesc, bilateralSliceApplyDesc, &bytes, this->schedule, &this->oclExtInfo));
        return bytes;
    }

};

#endif //_BILATERAL_SLICE_APPLY_OCL_H
