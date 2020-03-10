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
#include "gpu/mali/fp16/pooling_mali_fp16.h"
#include "gpu/mali/infer_gclmem_desc_mali.h"

EE pooling_infer_output_size_mali(TensorDesc   inputDesc, 
                                  PoolingDesc  poolingDesc,
                                  TensorDesc*  outputDesc,
                                  GCLMemDesc_t gclmemInputDesc,
                                  GCLMemDesc_t gclmemOutputDesc){
    /*tensorDesc record cpu org data format info*/
    /*gclmemDesc record gpu trans data format info*/
    DataType   idt;
    DataFormat idf;
    U32 iw, ih, ic, in;
    U32 ow, oh;
    U32 kw, kh, sw, sh, pw, ph, pr, pb;
    tensorSelectGet(inputDesc,  &idt, &idf, &in, &ic, &ih, &iw);
    pw = poolingDesc.padding_left;
    pr = poolingDesc.padding_right;
    ph = poolingDesc.padding_top;
    pb = poolingDesc.padding_bottom;
    kw = poolingDesc.kernelSize_w;
    kh = poolingDesc.kernelSize_h;
    sw = poolingDesc.stride_w;
    sh = poolingDesc.stride_h;
    if(kw != kh || sw != sh) CHECK_STATUS(NOT_SUPPORTED);
    if(pw != ph || ph != pb || pw != pr) CHECK_STATUS( NOT_SUPPORTED);
    switch (poolingDesc.rm){
        case CEIL: {
            ow = (U32)(ceil((double(iw + 2 * pw - kw) / sw))) + 1;
            oh = (U32)(ceil((double(ih + 2 * ph - kh) / sh))) + 1;
            break;
        }
        case FLOOR: {
            ow = (U32)(floor((double(iw + 2 * pw - kw) / sw))) + 1;
            oh = (U32)(floor((double(ih + 2 * ph - kh) / sh))) + 1;
            break;
        }
        default: {
            CHECK_STATUS(NOT_SUPPORTED);
        }
    }

    *outputDesc = tensor4df(idt, idf, in, ic, oh, ow);
    CHECK_STATUS(infer_gclmem_desc_ncwhc4(iw, ih, ic, pw, ph, ow, oh, ic, idt, idt, gclmemInputDesc, gclmemOutputDesc));
    return SUCCESS;
}
EE pooling_mali(GCLHandle_t   handle,
                TensorDesc     inputDesc, 
                const GCLMem_t input,
                PoolingDesc    poolingDesc, 
                const void*    scale,
                TensorDesc     outputDesc,
                GCLMem_t       output){
    UNUSED(scale);               
    EE ret = SUCCESS;
    switch(inputDesc.dt){
        case DT_F16:{
            ret = pooling_mali_fp16(handle, inputDesc, input, poolingDesc, outputDesc, output);
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



