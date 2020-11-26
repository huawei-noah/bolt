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
#include "types.h"
#include "tensor_desc.h"
#include "error.h"
#include "gpu/mali/tensor_computing_mali.h"
#include "gpu/mali/fp16/concat_mali_fp16.h"

EE concat_infer_output_size_mali(std::vector<TensorDesc> inputDesc,
    ConcatParamSpec p,
    TensorDesc *outputDesc,
    GCLMemDesc_t gclmemInputDesc,
    GCLMemDesc_t gclmemOutputDesc)
{
    /*tensorDesc record cpu org data format info*/
    /*gclmemDesc record gpu trans data format info*/
    if (outputDesc) {
        *outputDesc = inputDesc[0];
    }
    U32 sumDimSize = 0;
    I32 dim = inputDesc[0].nDims;
    int concatDim = p.axis;
    concatDim = (concatDim + dim) % dim;
    concatDim = dim - 1 - concatDim;
    for (auto p : inputDesc) {
        if (inputDesc[0].df != p.df) {
            CHECK_STATUS(NOT_SUPPORTED);
        }
    }
    if (inputDesc[0].df == DF_MKT) {
        concatDim = 1 - concatDim;
    }
    for (U32 i = 0; i < inputDesc.size(); i++) {
        sumDimSize += inputDesc[i].dims[concatDim];
    }

    if (outputDesc) {
        *outputDesc = inputDesc[0];
        (*outputDesc).dims[concatDim] = sumDimSize;
    }

    if (gclmemInputDesc && gclmemOutputDesc) {
        DataType idt;
        DataFormat idf;
        U32 iw, ih, ic, in;
        for (U32 i = 0; i < inputDesc.size(); i++) {
            tensorSelectGet(inputDesc[i], &idt, &idf, &in, &ic, &ih, &iw);
            CHECK_STATUS(infer_gclmem_desc_ncwhc4(
                iw, ih, ic, 0, 0, iw, ih, ic, idt, idt, &gclmemInputDesc[i], gclmemOutputDesc));
        }
        U32 s0 = gclmemOutputDesc->stride[0];
        U32 s1 = gclmemOutputDesc->stride[1];
        U32 s2 = gclmemOutputDesc->stride[2];
        if (inputDesc[0].df == DF_NCHW) {
            if (concatDim == 0) {
                s1 = sumDimSize;
            } else if (concatDim == 1) {
                s0 = sumDimSize;
            } else if (concatDim == 2) {
                s2 = (sumDimSize + 3) / 4;
            } else {
                CHECK_STATUS(NOT_SUPPORTED);
            }
        }
        if (inputDesc[0].df == DF_MKT || inputDesc[0].df == DF_MTK) {
            if (concatDim == 0) {
                s2 = (sumDimSize + 3) / 4;
            } else if (concatDim == 1) {
                s0 = sumDimSize;
            } else {
                CHECK_STATUS(NOT_SUPPORTED);
            }
        }
        gclmemOutputDesc->stride[0] = s0;
        gclmemOutputDesc->stride[1] = s1;
        gclmemOutputDesc->stride[2] = s2;
        gclmemOutputDesc->num = s0 * s1 * s2 * 4;
        gclmemOutputDesc->byteSize = s0 * s1 * s2 * 4 * bytesOf(idt);
    }
    return SUCCESS;
}

inline EE concat_checkpara_mali(GCLHandle_t handle,
    std::vector<TensorDesc> inputDesc,
    std::vector<void *> input,
    ConcatParamSpec p,
    TensorDesc outputDesc,
    GCLMem_t output)
{
    if (handle == nullptr || nullptr == output) {
        return NULL_POINTER;
    }
    if (input.size() < 1) {
        return NOT_MATCH;
    }
    for (auto it : inputDesc) {
        if (it.df != outputDesc.df) {
            return NOT_MATCH;
        }
    }
    if (outputDesc.df != DF_NCHW && outputDesc.df != DF_MKT && outputDesc.df != DF_MTK) {
        return NOT_SUPPORTED;
    }
    for (auto it : input) {
        GCLMem_t ptr = (GCLMem_t)it;
        if (ptr == nullptr) {
            return NULL_POINTER;
        }
        if (ptr->desc.memFormat != output->desc.memFormat) {
            return NOT_SUPPORTED;
        }
    }
    return SUCCESS;
}

EE concat_infer_forward_tmp_bytes_mali(std::vector<TensorDesc> inputDesc, U32 *bytes)
{
    EE ret = SUCCESS;
    switch (inputDesc[0].dt) {
        case DT_F16: {
            ret = concat_infer_forward_tmp_bytes_mali_fp16(inputDesc, bytes);
            break;
        }
        default:
            ret = NOT_SUPPORTED;
            break;
    }
    return ret;
}

EE concat_mali(GCLHandle_t handle,
    std::vector<TensorDesc> inputDesc,
    std::vector<void *> input,
    GCLMem_t inputScale,
    ConcatParamSpec p,
    GCLMem_t tmpbuf,
    TensorDesc outputDesc,
    GCLMem_t output,
    GCLMem_t outputScale)
{
    UNUSED(inputScale);
    UNUSED(outputScale);
    EE ret = SUCCESS;
    CHECK_STATUS(concat_checkpara_mali(handle, inputDesc, input, p, outputDesc, output));
    switch (inputDesc[0].dt) {
        case DT_F16: {
            ret = concat_mali_fp16(handle, inputDesc, input, outputDesc, output, tmpbuf, p.axis);
            break;
        }
        default:
            ret = NOT_SUPPORTED;
            break;
    }
    return ret;
}
