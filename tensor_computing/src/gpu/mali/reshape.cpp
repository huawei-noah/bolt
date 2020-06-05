// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


#include "sys.h"
#include "type.h"
#include "tensor_desc.h"
#include "error.h"
#include "gpu/mali/tensor_computing_mali.h"
#include "gpu/mali/fp16/reshape_mali_fp16.h"
#include "gpu/mali/infer_gclmem_desc_mali.h"

EE reshape_infer_output_size_mali(TensorDesc   inputDesc,
                                  TensorDesc*  outputDesc,
                                  I32*         dims,
                                  I32          shapeSize,
                                  GCLMemDesc_t gclmemInputDesc,
                                  GCLMemDesc_t gclmemOutputDesc){
    /*tensorDesc record cpu org data format info*/
    /*gclmemDesc record gpu trans data format info*/
    I32 dimTran[4] = {1, 1, 1, 1};
    U32 factor = 1;
    U32 count  = 0;
    for(I32 i = 0; i < shapeSize; i++){
        I32 value = dims[i];
        if(value == 0)  value = inputDesc.dims[3 - i];
        if(value == -1) {
            value = 0;
            count++;
        } else {
            factor *=value;
        }
        dimTran[3 - i] = value;
    }

    for(I32 i = 0; i < 4; i++) {
        if(dimTran[i] == 0) dimTran[i] = tensorNumElements(inputDesc) / factor;
    }
    

    if(inputDesc.df == DF_NCHW) {
        DataType   idt;
        DataFormat idf;
        U32 iw, ih, ic, in;
        tensorSelectGet(inputDesc,  &idt, &idf, &in, &ic, &ih, &iw);
        if(shapeSize == 2 || shapeSize == 4) {
            if(dimTran[2] != (I32)ic) CHECK_STATUS(NOT_SUPPORTED);//gpu use ncwhc4, if reshape on axis c, need to reset data
            if(outputDesc) {
                *outputDesc = inputDesc;
                (*outputDesc).dims[0] = dimTran[0];
                (*outputDesc).dims[1] = dimTran[1];
                (*outputDesc).dims[2] = dimTran[2];
                (*outputDesc).dims[3] = dimTran[3];
            }
            CHECK_STATUS(infer_gclmem_desc_ncwhc4(iw, ih, ic, 0, 0, dimTran[0], dimTran[1], dimTran[2], idt, idt, gclmemInputDesc, gclmemOutputDesc));
        }
        if(shapeSize == 3) {
            U32 m = dimTran[3];
            U32 k = dimTran[2];
            U32 t = dimTran[1];
            if(outputDesc) *outputDesc = tensor3df(idt, DF_MKT, m, k, t);
            U32 ogw, ogh, ogc;
            map_nlp_mkt_to_ncwhc4(m, k, t, &ogw, &ogh, &ogc);
            CHECK_STATUS(infer_gclmem_desc_nchw(iw, ih, ic, 0, 0, 0, 0, 0, idt, idt, gclmemInputDesc, NULL));
            CHECK_STATUS(infer_gclmem_desc_ncwhc4(0, 0, 0, 0, 0, ogw, ogh, ogc * 4, idt, idt, NULL, gclmemOutputDesc));
        }
        return SUCCESS;
    }

    if(inputDesc.df == DF_MKT) {
         DataType idt;
         U32 m, k, t;
         get_nlp_mkt_val(inputDesc, &idt, &m, &k, &t);
         if(outputDesc) *outputDesc = tensor4df(idt, DF_NCHW, dimTran[3], dimTran[2], dimTran[1], dimTran[0]);
         U32 igw, igh, igc;
         map_nlp_mkt_to_ncwhc4(m, k, t, &igw, &igh, &igc);
         if((I32)igh != dimTran[0]) CHECK_STATUS(NOT_MATCH);
         if((I32)igc != (dimTran[1] * dimTran[2] + 3) / 4) CHECK_STATUS(NOT_MATCH);
         CHECK_STATUS(infer_gclmem_desc_ncwhc4(igw, igh, igc * 4, 0, 0, 0, 0, 0, idt, idt, gclmemInputDesc, NULL));
         CHECK_STATUS(infer_gclmem_desc_nchw(0, 0, 0, 0, 0, dimTran[0], dimTran[1], dimTran[2], idt, idt, NULL, gclmemOutputDesc));
         return SUCCESS;
    }
    return NOT_SUPPORTED;
}

inline EE reshape_checkpara_mali(GCLHandle_t handle, 
                                 TensorDesc  inputDesc,
                                 GCLMem_t    input,
                                 TensorDesc  outputDesc,
                                 GCLMem_t    output) {

    if(handle == nullptr || nullptr == input || nullptr == output) return NULL_POINTER;
    if(inputDesc.df  != DF_NCHW && inputDesc.df  != DF_MKT) return NOT_SUPPORTED;
    if(outputDesc.df != DF_NCHW && outputDesc.df != DF_MKT) return NOT_SUPPORTED;
    if(output->desc.memFormat != DF_NCWHC4 && output->desc.memFormat != DF_NCHW) return NOT_SUPPORTED;
    if(input->desc.offset[0] != 0 || input->desc.offset[1] != 0) return NOT_SUPPORTED;
    return SUCCESS; 
}

EE reshape_mali(GCLHandle_t handle, 
                TensorDesc  inputDesc,
                GCLMem_t    input,
                TensorDesc  outputDesc,
                GCLMem_t    output) {
    EE ret = SUCCESS;
    CHECK_STATUS(reshape_checkpara_mali(handle, inputDesc, input, outputDesc, output));
    switch(inputDesc.dt){
        case DT_F16:{
            ret = reshape_mali_fp16(handle, inputDesc, input, outputDesc, output);
            break;
        }
        case DT_I8:{
            ret = NOT_SUPPORTED;
            break;
        }
        default:
            ret = NOT_SUPPORTED;
            break;
    }
    return ret;
}



