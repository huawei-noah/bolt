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
    if (outputDesc == nullptr || gclmemInputDesc == nullptr || gclmemOutputDesc == nullptr) {
        CHECK_STATUS(NULL_POINTER);
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
    for (U32 i = 0; i < inputDesc.size(); i++) {
        sumDimSize += inputDesc[i].dims[concatDim];
    }

    *outputDesc = inputDesc[0];
    (*outputDesc).dims[concatDim] = sumDimSize;

    bool use_nchw = true;
    for (U32 i = 0; i < inputDesc.size(); i++) {
        if (gclmemInputDesc[i].memFormat == DF_NCWHC4) {
            use_nchw = false;
            break;
        }
    }

    DataType idt;
    DataFormat idf;
    U32 iw, ih, ic, in;
    for (U32 i = 0; i < inputDesc.size(); i++) {
        tensorSelectGet(inputDesc[i], &idt, &idf, &in, &ic, &ih, &iw);
        if (use_nchw || (gclmemInputDesc[i].memFormat == DF_NCHW 
            && gclmemInputDesc[i].byteSize >= 0)) {
            CHECK_STATUS(infer_gclmem_desc_nchw(
                (iw + 3) / 4 * 4, ih, ic, 0, 0, 0, 0, 0, idt, idt, &gclmemInputDesc[i], NULL));
        } else {
            CHECK_STATUS(infer_gclmem_desc_ncwhc4(
                iw, ih, ic, 0, 0, 0, 0, 0, idt, idt, &gclmemInputDesc[i], NULL));
        }
    }

    U32 ow, oh, oc;
    tensorSelectGet((*outputDesc), NULL, NULL, NULL, &oc, &oh, &ow);
    if (use_nchw) {
        CHECK_STATUS(
            infer_gclmem_desc_nchw(0, 0, 0, 0, 0, ow, oh, oc, idt, idt, NULL, gclmemOutputDesc));
    } else {
        CHECK_STATUS(
            infer_gclmem_desc_ncwhc4(0, 0, 0, 0, 0, ow, oh, oc, idt, idt, NULL, gclmemOutputDesc));
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
        CHECK_STATUS(NULL_POINTER);
    }
    if (input.size() < 1) {
        CHECK_STATUS(NOT_MATCH);
    }
    for (auto it : inputDesc) {
        if (it.df != outputDesc.df) {
            CHECK_STATUS(NOT_MATCH);
        }
    }
    for (auto it : input) {
        if (it == nullptr) {
            CHECK_STATUS(NULL_POINTER);
        }
    }
    return SUCCESS;
}

EE concat_infer_forward_tmp_bytes_mali(std::vector<TensorDesc> inputDesc, 
    std::vector<GCLMemDesc> gclmemInputDesc,
    U32 *bytes)
{
    EE ret = SUCCESS;
    switch (inputDesc[0].dt) {
        case DT_F16: {
            ret = concat_infer_forward_tmp_bytes_mali_fp16(inputDesc, gclmemInputDesc, bytes);
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
