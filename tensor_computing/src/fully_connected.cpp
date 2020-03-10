// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), 
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, 
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE 
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR 
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


#include <string.h>

#include "tensor_computing.h"
#include "blas-enhance.h"
#ifdef _USE_MALI
#include "gpu/mali/tensor_computing_mali.h"
#endif

// input format: NCHW|NCHWC8|NORMAL
// weight(filter) format: NORMAL
// result format: NORMAL

inline EE fully_connected_infer_output_size_cpu(TensorDesc inputDesc, TensorDesc filterDesc, TensorDesc *outputDesc)
{
    if(outputDesc == nullptr)
        CHECK_STATUS(NULL_POINTER);

    DataType idt, fdt;
    DataFormat idf, fdf;
    U32 in, ic, ih, iw;
    U32 fh, fw;
    if (tensorIs2d(inputDesc)) {
        CHECK_STATUS(tensor2dfGet(inputDesc, &idt, &idf, &in, &iw));
        ic = ih = 1;
    }
    else if (tensorIs4d(inputDesc)) {
        CHECK_STATUS(tensor4dGet(inputDesc, &idt, &idf, &in, &ic, &ih, &iw));
        if (idf != DF_NCHW && idf != DF_NCHWC8)
            CHECK_STATUS(NOT_MATCH);
    }
    else
        CHECK_STATUS(NOT_MATCH);

    CHECK_REQUIREMENT(tensorIs2d(filterDesc));
    CHECK_STATUS(tensor2dfGet(filterDesc, &fdt, &fdf, &fh, &fw));
    if (fdf != DF_NORMAL)
        CHECK_STATUS(NOT_MATCH);

    if (fw != ic * ih * iw)
        CHECK_STATUS(NOT_MATCH);

    *outputDesc = tensor2df(idt, DF_NORMAL, in, fh);
    return SUCCESS;
}

EE fully_connected_infer_output_size(TensorDesc inputDesc, TensorDesc filterDesc, TensorDesc *outputDesc, Arch arch, ExtInfo_t extInfo)
{
#ifdef _USE_MALI
    if(arch == MALI){
        CHECK_STATUS(fully_connected_infer_output_size_mali(inputDesc, filterDesc, outputDesc, extInfo->maliInfo.gclmemInputDesc, extInfo->maliInfo.gclmemOutputDesc));
    } else {
#endif
        UNUSED(arch);
        UNUSED(extInfo);
        CHECK_STATUS(fully_connected_infer_output_size_cpu(inputDesc, filterDesc, outputDesc));
#ifdef _USE_MALI    
    }    
#endif    
    return SUCCESS;
}


EE fully_connected_infer_forward_tmp_bytes(TensorDesc inputDesc, TensorDesc filterDesc, U32 *bytes, Arch arch)
{
#ifndef _USE_MALI
    UNUSED(arch);
#endif    
    EE ret = SUCCESS;
#ifdef _USE_MALI
    if(arch == MALI){
        CHECK_STATUS(fully_connected_infer_forward_tmp_bytes_mali(inputDesc, filterDesc, bytes));
    } else {
#endif
        if(bytes == nullptr) CHECK_STATUS(NULL_POINTER);
        DataType idt;
        DataFormat idf;
        U32 in, ic, ih, iw;
        if (tensorIs2d(inputDesc)) {
            CHECK_STATUS(tensor2dfGet(inputDesc, &idt, &idf, &in, &iw));
            ic = ih = 1;
        }
        else if (tensorIs4d(inputDesc)) {
            CHECK_STATUS(tensor4dGet(inputDesc, &idt, &idf, &in, &ic, &ih, &iw));
        }
        else {
            return NOT_MATCH;
        }

        if(in != 1){
                // call gemm
            TensorDesc in_desc = tensor2df(idt, DF_NORMAL, in, ic*ih*iw);
            ret = matrix_matrix_multiply_tmp_bytes(in_desc, filterDesc, bytes, arch);
        }
        else{
                // call gemv
            TensorDesc in_desc = tensor1d(idt, ic*ih*iw);
            ret = matrix_vector_multiply_tmp_bytes(filterDesc, in_desc, bytes, arch);
        }
#ifdef _USE_MALI    
    }    
#endif    
    return ret;
}


EE fully_connected_transform_filter_bytes(TensorDesc filterDesc, U32* bytes, Arch arch, ExtInfo_t extInfo)
{
#ifndef _USE_MALI
    UNUSED(arch);
    UNUSED(extInfo);
#endif    
#ifdef _USE_MALI
    if(arch == MALI){
        CHECK_STATUS(fully_connected_transform_filter_bytes_mali(filterDesc, extInfo->maliInfo.gclmemFilterDesc, bytes));
    } else {
#endif
        if (bytes == nullptr) CHECK_STATUS(NULL_POINTER);
        *bytes = tensorNumBytes(filterDesc);
#ifdef _USE_MALI    
    }    
#endif    
    return SUCCESS;
}


template<typename T>
EE fully_connected_transform_filter_kernel(TensorDesc inputDesc, TensorDesc filterDesc, const void* filter,
    TensorDesc *ftmDesc, void* filterTransformed)
{
    if (filter == nullptr || ftmDesc == nullptr || filterTransformed == nullptr) {
        CHECK_STATUS(NULL_POINTER);
    }
    
    DataType idt, fdt;
    DataFormat idf, fdf;
    U32 in, ic, ih, iw;
    U32 fh, fw;
    if (tensorIs2d(inputDesc)) {
        CHECK_STATUS(tensor2dfGet(inputDesc, &idt, &idf, &in, &iw));
        ic = ih = 1;
    }
    else if (tensorIs4d(inputDesc)){
        CHECK_STATUS(tensor4dGet(inputDesc, &idt, &idf, &in, &ic, &ih, &iw));
    }
    else
        CHECK_STATUS(NOT_MATCH);
    CHECK_STATUS(tensor2dfGet(filterDesc, &fdt, &fdf, &fh, &fw));

    if (fw != ic*ih*iw)
        CHECK_STATUS(NOT_MATCH);
    bool need_transpose = false;
    if (in > 1)
        need_transpose = true;

    if (idf == DF_NCHW || idf == DF_NORMAL) {
        if(need_transpose){
            T *f_ptr   = (T *)filter;
            T *ftm_ptr = (T *)filterTransformed;
            for(U32 h = 0; h < fh; h++){
                for(U32 w = 0; w < fw; w++){
                    U32 f_index   = h * fw + w;
                    U32 ftm_index = w * fh + h;
                    ftm_ptr[ftm_index] = f_ptr[f_index];
                }
            }
        }
        else{
            memcpy(filterTransformed, filter, tensorNumBytes(filterDesc));
        }
    }
    if (idf == DF_NCHWC8) {
        U32 align = 8;
        U32 ic_new = ic / align;
        T *f_ptr   = (T *)filter;
        T *ftm_ptr = (T *)filterTransformed;
        for (U32 h = 0; h < fh; h++) {
            for(U32 w = 0; w < fw; w++){
                U32 i_n   = w / (ic * ih * iw);
                U32 remain = w % (ic * ih * iw);
                U32 i_c = remain / (ih * iw);
                remain   = remain % (ih * iw);
                U32 i_h = remain / iw;
                U32 i_w = remain % iw;
                U32 i_c_outer = i_c / align;
                U32 i_c_inner = i_c % align;
                U32 h_new = h;
                U32 w_new = (((i_n * ic_new + i_c_outer) * ih + i_h) * iw + i_w) * align + i_c_inner;
                U32 ld = fw;
                if (need_transpose) {
                    U32 tmp = h_new;
                    h_new = w_new;
                    w_new = tmp;
                    ld = fh;
                }
                U32 f_index   = h * fw + w;
                U32 ftm_index = h_new * ld + w_new;
                ftm_ptr[ftm_index] = f_ptr[f_index];
            }
        }
    }

    DataFormat fdf_after = fdf;
    U32 fh_after = fh;
    U32 fw_after = fw;
    if (need_transpose) {
        fh_after = fw;
        fw_after = fh;
    }
    *ftmDesc = tensor2df(fdt, fdf_after, fh_after, fw_after);
    return SUCCESS;
}

EE fully_connected_transform_filter(TensorDesc inputDesc, TensorDesc filterDesc, const void* filter,
    TensorDesc *ftmDesc, void* filterTransformed, Arch arch, ExtInfo_t extInfo)
{
#ifndef _USE_MALI
    UNUSED(arch);
    UNUSED(extInfo);
#endif    
    EE ret = SUCCESS;
#ifdef _USE_MALI
    if(arch == MALI){
        CHECK_STATUS(fully_connected_transform_filter_mali(extInfo->maliInfo.handle, filterDesc, (GCLMem_t)filter, ftmDesc, (GCLMem_t)filterTransformed));
    } else {
#endif
        switch(filterDesc.dt) {
#ifdef _USE_FP16
            case DT_F16: {
                 ret = fully_connected_transform_filter_kernel<F16>(inputDesc, filterDesc, filter, ftmDesc, filterTransformed);
                 break;
            }
#endif
#ifdef _USE_FP32
            case DT_F32: {
                 ret = fully_connected_transform_filter_kernel<F32>(inputDesc, filterDesc, filter, ftmDesc, filterTransformed);
                 break;
            }
#endif
            default:
                 ret = NOT_SUPPORTED;
                 break;
            }
#ifdef _USE_MALI
    }
#endif
    return ret;
}

EE fully_connected(TensorDesc inputDesc, const void* input,
    TensorDesc filterDesc, const void* filter,
    void* tmp, U32 bytes,
    TensorDesc outputDesc, void* output,
    TensorDesc biasDesc, const void* bias,
    Arch arch, ExtInfo_t extInfo)
{
    EE ret = SUCCESS;
#ifndef _USE_MALI
    UNUSED(arch);
    UNUSED(extInfo);
#endif    
#ifdef _USE_MALI
    if(arch == MALI){
        CHECK_STATUS(fully_connected_mali(extInfo->maliInfo.handle, inputDesc, (GCLMem_t)input,
                                          filterDesc, (GCLMem_t) filter,
                                          biasDesc,   (GCLMem_t) bias,
                                          bytes,      (GCLMem_t) tmp,
                                          outputDesc, (GCLMem_t) output));
    } else {
#endif
        if(input == nullptr || filter == nullptr || output == nullptr) CHECK_STATUS(NULL_POINTER);

        U32 in, ic, ih, iw;
        U32 oh, ow;
        U32 fh, fw, bw;
        DataType idt, fdt, odt, bdt;
        DataFormat idf, fdf, odf;
        if (tensorIs2d(inputDesc)) {
            CHECK_STATUS(tensor2dfGet(inputDesc, &idt, &idf, &in, &iw));
            ic = ih = 1;
        }
        else if (tensorIs4d(inputDesc)){
            CHECK_STATUS(tensor4dGet(inputDesc, &idt, &idf, &in, &ic, &ih, &iw));
        }
        else
            CHECK_STATUS(NOT_MATCH);

        CHECK_REQUIREMENT(tensorIs2d(filterDesc));
        CHECK_STATUS(tensor2dfGet(filterDesc, &fdt, &fdf, &fh, &fw));
        CHECK_STATUS(tensor2dfGet(outputDesc, &odt, &odf, &oh, &ow));

        if (bias != nullptr) {
            CHECK_STATUS(tensor1dGet(biasDesc, &bdt, &bw));

            if(bw != ow){
                CHECK_STATUS(NOT_MATCH);
            } else {
                U8 *outArray = (U8*)output;
                U32 size = tensorNumBytes(biasDesc);
                for (U32 i = 0; i < in; i++) {
                     memcpy(outArray + i*size, bias, size);
                }
            }
        }

        if (in == 1) {
            TensorDesc vectorDesc = tensor1d(idt, ic*ih*iw);
            TensorDesc resultDesc = tensor1d(odt, ow);
            ret = matrix_vector_multiply(filterDesc, filter, vectorDesc, input, bytes, tmp, resultDesc, output, arch);
        } else {
            if (idf == DF_TRANSPOSE || fdf == DF_TRANSPOSE) CHECK_STATUS(NOT_MATCH);
            TensorDesc in_desc  = tensor2df(idt, DF_NORMAL, in, ic*ih*iw);
            ret = matrix_matrix_multiply(in_desc, input, filterDesc, filter, bytes, tmp, outputDesc, output, arch);
        }
#ifdef _USE_MALI
    }
#endif
    return ret;
}
