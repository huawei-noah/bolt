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
#include "gpu/mali/fp16/depthwise_convolution_mali_fp16.h"
#include "gpu/mali/infer_gclmem_desc_mali.h"

EE depthwise_convolution_infer_output_size_mali(TensorDesc           inputDesc,
                                                TensorDesc           filterDesc,
                                                ConvolutionDesc      convDesc,
                                                TensorDesc*          outputDesc,
                                                GCLMemDesc_t         gclmemInputDesc,
                                                GCLMemDesc_t         gclmemOutputDesc,
                                                ForwardRunInfoMali_t forwardRunInfo){
    UNUSED(forwardRunInfo);
    DataType   idt, fdt;
    DataFormat idf, fdf;
    U32 iw, ih, ic, in;
    U32 fw, fh, fc, fn;
    U32 ow, oh;
    U32 sw, sh, pw, ph, dw, dh, pr, pb;
    tensorSelectGet(inputDesc,  &idt, &idf, &in, &ic, &ih, &iw);
    tensorSelectGet(filterDesc, &fdt, &fdf, &fn, &fc, &fh, &fw);
    pw = convDesc.padding_left;
    pr = convDesc.padding_right;
    ph = convDesc.padding_top;
    pb = convDesc.padding_bottom;
    sw = convDesc.stride_w;
    sh = convDesc.stride_h;
    dw = convDesc.dilatedRate_w;
    dh = convDesc.dilatedRate_h;
    if (fw < 1  || fh < 1)        return NOT_SUPPORTED;
    if (dw != 1 || dh != 1)       return NOT_SUPPORTED;
    if (pw != ph || sw != sh)     return NOT_SUPPORTED;
    if (pb != ph || pr != pw)     return NOT_SUPPORTED;
    if ((fn & 3) != 0)     return NOT_SUPPORTED;
    ow = (iw + 2 * pw - fw) / sw + 1;
    oh = (ih + 2 * ph - fh) / sh + 1;
    *outputDesc = tensor4df(idt, idf, in, fn, oh, ow);
    
    U32 iw_align, item_w, ext_w;
    if(idf == DF_NCHW){
        if(sw == 1) item_w = (fw == 5) ? 4 : 8;
        if(sw == 2) item_w = 4;
        if(ow < item_w) item_w = ow;
        ext_w = fw / 2;
        iw_align = (ow + item_w - 1) / item_w * item_w;
        iw_align = iw_align * sw;
        CHECK_STATUS(infer_gclmem_desc_ncwhc4(iw_align, ih, ic, ext_w, ph, ow, oh, fn, idt, idt, gclmemInputDesc, gclmemOutputDesc));
        return SUCCESS;
    }
    return NOT_SUPPORTED;
}

EE depthwise_convolution_infer_forward_algorithm_mali(GCLHandle_t          handle,
                                                      TensorDesc           inputDesc, 
                                                      TensorDesc           filterDesc, 
                                                      TensorDesc           outputDesc,
                                                      ConvolutionDesc      convDesc,
                                                      ConvolutionPolicy    policy, 
                                                      ActivationMode       depthwisewiseActivationMode,
                                                      ActivationMode       pointwiseActivationMode,
                                                      ForwardRunInfoMali_t forwardRunInfo){
    EE ret = SUCCESS;
    switch(inputDesc.dt){
        case DT_F16:{
           ret = depthwise_convolution_infer_forward_algorithm_mali_fp16(handle, inputDesc, filterDesc, convDesc, outputDesc, policy, 
                   depthwisewiseActivationMode, pointwiseActivationMode, forwardRunInfo);
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

EE depthwise_convolution_transform_filter_bytes_mali(TensorDesc            filterDesc, 
                                                     ForwardRunInfoMali_t  forwardRunInfo,
                                                     GCLMemDesc_t          gclmemFilterDesc,
                                                     U32*                  bytes){
    EE ret = SUCCESS;
    switch(filterDesc.dt){
        case DT_F16:{
           ret = depthwise_convolution_transform_filter_bytes_mali_fp16(filterDesc, forwardRunInfo, gclmemFilterDesc, bytes);
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

EE depthwise_convolution_transform_filter_mali(GCLHandle_t          handle,
                                               TensorDesc           filterDesc,
                                               GCLMem_t             filter,
                                               ForwardRunInfoMali_t forwardRunInfo,
                                               TensorDesc*          fltmemDesc,
                                               GCLMem_t             fltmem){
    EE ret = SUCCESS;
    switch(filterDesc.dt){
        case DT_F16:{
           ret = depthwise_convolution_transform_filter_mali_fp16(handle, filterDesc, filter, forwardRunInfo, fltmemDesc, fltmem);
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

EE depthwise_convolution_infer_forward_tmp_bytes_mali(TensorDesc            inputDesc, 
                                                      TensorDesc            filterDesc, 
                                                      TensorDesc            outputDesc,
                                                      ConvolutionDesc       convDesc, 
                                                      ForwardRunInfoMali_t  forwardRunInfo,
                                                      U32*                  bytes){
    EE ret = SUCCESS;
    switch(inputDesc.dt){
        case DT_F16:{
            ret = depthwise_convolution_infer_forward_tmp_bytes_mali_fp16(inputDesc, filterDesc, outputDesc, convDesc, forwardRunInfo, bytes);
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
EE depthwise_convolution_mali(GCLHandle_t          handle,
                              TensorDesc           inputDesc, 
                              const GCLMem_t       input,
                              TensorDesc           filterDesc, 
                              const GCLMem_t       filter,
                              ConvolutionDesc      convDesc,
                              ForwardRunInfoMali_t forwardRunInfo,
                              TensorDesc           biasDesc, 
                              const GCLMem_t       bias,
                              U32                  tmpBytes, 
                              GCLMem_t             tmpBuf,
                              TensorDesc           outputDesc, 
                              GCLMem_t             output,
                              ActivationMode       depthwisewiseActivationMode,
                              ActivationMode       pointwiseActivationMode){
    EE ret = SUCCESS;
    switch(inputDesc.dt){
        case DT_F16:{
            ret = depthwise_convolution_mali_fp16(handle, inputDesc, input, filterDesc, filter, convDesc, forwardRunInfo, biasDesc, bias, tmpBytes, tmpBuf, outputDesc, output, 
                    depthwisewiseActivationMode, pointwiseActivationMode);
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

