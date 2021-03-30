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
#include "gpu/mali/fp16/eltwise_mali_fp16.h"

EE eltwise_infer_output_size_mali(std::vector<TensorDesc> inputDesc,
    TensorDesc *outputDesc,
    GCLMemDesc_t gclmemInputDesc,
    GCLMemDesc_t gclmemOutputDesc)
{
    /*tensorDesc record cpu org data format info*/
    /*gclmemDesc record gpu trans data format info*/
    if (outputDesc == nullptr || gclmemInputDesc == nullptr || gclmemOutputDesc == nullptr) {
        CHECK_STATUS(NULL_POINTER);
    }
    U32 arrayDimMax = 0;
    bool sameDesc = eltwise_same_desc(inputDesc, &arrayDimMax);
    *outputDesc = inputDesc[arrayDimMax];

    DataType idt;
    DataFormat idf;
    U32 iw, ih, ic, in, it;
    tensorSelectGet(inputDesc[arrayDimMax], &idt, &idf, &in, &ic, &ih, &iw, &it);

    if (sameDesc) {
        U32 size = inputDesc.size();
        bool useNCHW = true;
        for (U32 i = 0; i < size; i++) {
            if (gclmemInputDesc[i].memFormat != DF_NCHW && gclmemInputDesc[i].byteSize != 0) {
                useNCHW = false;
                break;
            }
        }
        if (useNCHW) {
            CHECK_STATUS(infer_gclmem_desc_nchw_3d(
                0, 0, 0, 0, 0, 0, 0, iw, ih, ic, it, in, idt, idt, NULL, gclmemOutputDesc));
            for (U32 i = 0; i < size; i++) {
                CHECK_STATUS(infer_gclmem_desc_nchw_3d(
                    iw, ih, ic, it, in, 0, 0, 0, 0, 0, 0, 0, idt, idt, &gclmemInputDesc[i], NULL));
            }
        } else {
            CHECK_STATUS(infer_gclmem_desc_ncwhc4_3d(
                0, 0, 0, 0, 0, 0, 0, iw, ih, ic, it, in, idt, idt, NULL, gclmemOutputDesc));
            for (U32 i = 0; i < size; i++) {
                CHECK_STATUS(infer_gclmem_desc_ncwhc4_3d(
                    iw, ih, ic, it, in, 0, 0, 0, 0, 0, 0, 0, idt, idt, &gclmemInputDesc[i], NULL));
            }
        }
        return SUCCESS;
    } else {
        if (inputDesc.size() > 2) {
            CHECK_STATUS(NOT_SUPPORTED);
        }
        DataFormat imf[2];
        DataFormat omf;
        U32 ibytes[2];
        imf[0] = gclmemInputDesc[arrayDimMax].memFormat;
        imf[1] = gclmemInputDesc[1 - arrayDimMax].memFormat;
        ibytes[0] = gclmemInputDesc[arrayDimMax].byteSize;
        ibytes[1] = gclmemInputDesc[1 - arrayDimMax].byteSize;

        if (imf[0] == DF_NCHW || ibytes[0] == 0) {
            CHECK_STATUS(infer_gclmem_desc_nchw_3d(iw, ih, ic, it, in, 0, 0, iw, ih, ic, it, in,
                idt, idt, &gclmemInputDesc[arrayDimMax], gclmemOutputDesc));
        } else {
            CHECK_STATUS(infer_gclmem_desc_ncwhc4_3d(iw, ih, ic, it, in, 0, 0, iw, ih, ic, it, in,
                idt, idt, &gclmemInputDesc[arrayDimMax], gclmemOutputDesc));
        }

        tensorSelectGet(inputDesc[1 - arrayDimMax], &idt, NULL, &in, &ic, &ih, &iw, &it);
        if (imf[1] == DF_NCHW || ibytes[1] == 0) {
            CHECK_STATUS(infer_gclmem_desc_nchw_3d(iw, ih, ic, it, in, 0, 0, iw, ih, ic, it, in,
                idt, idt, &gclmemInputDesc[1 - arrayDimMax], NULL));
        } else {
            CHECK_STATUS(infer_gclmem_desc_ncwhc4_3d(iw, ih, ic, it, in, 0, 0, iw, ih, ic, it, in,
                idt, idt, &gclmemInputDesc[1 - arrayDimMax], NULL));
        }
        return SUCCESS;
    }
}

inline EE eltwise_checkpara_mali(GCLHandle_t handle,
    std::vector<TensorDesc> inputDesc,
    std::vector<void *> input,
    EltwiseParamSpec eltwiseDesc,
    TensorDesc outputDesc,
    GCLMem_t output)
{
    if (handle == nullptr || nullptr == output) {
        CHECK_STATUS(NULL_POINTER);
    }
    for (auto it : input) {
        GCLMem_t ptr = (GCLMem_t)it;
        if (ptr == nullptr) {
            CHECK_STATUS(NULL_POINTER);
        }
    }
    EltwiseMode eltwiseMode = eltwiseDesc.elt_mode;
    U32 arrayDimMax = 0;
    bool sameDesc = eltwise_same_desc(inputDesc, &arrayDimMax);
    if (sameDesc) {
        for (auto it : input) {
            if (((GCLMem_t)(it))->desc.memFormat != output->desc.memFormat) {
                CHECK_STATUS(NOT_SUPPORTED);
            }
        }
        for (auto it : inputDesc) {
            if (it.nDims != outputDesc.nDims) {
                CHECK_STATUS(NOT_SUPPORTED);
            }
            for (U32 i = 0; i < it.nDims; i++) {
                if (it.dims[i] != outputDesc.dims[i]) {
                    CHECK_STATUS(NOT_SUPPORTED);
                }
            }
        }
    } else {
        if (inputDesc.size() > 2) {
            CHECK_STATUS(NOT_SUPPORTED);
        }
        GCLMem_t iMaxInput = (GCLMem_t)input[arrayDimMax];
        if (iMaxInput->desc.memFormat != output->desc.memFormat) {
            CHECK_STATUS(NOT_SUPPORTED);
        }
    }
    if (eltwiseMode != ELTWISE_MAX && eltwiseMode != ELTWISE_MIN && eltwiseMode != ELTWISE_SUM &&
        eltwiseMode != ELTWISE_SUB && eltwiseMode != ELTWISE_PROD && eltwiseMode != ELTWISE_DIV) {
        CHECK_STATUS(NOT_SUPPORTED);
    }
    return SUCCESS;
}

EE eltwise_infer_forward_tmp_bytes_mali(
    std::vector<TensorDesc> inputDesc, std::vector<GCLMemDesc> gclmemInputDesc, U32 *bytes)
{
    EE ret = SUCCESS;
    switch (inputDesc[0].dt) {
        case DT_F16: {
            ret = eltwise_infer_forward_tmp_bytes_mali_fp16(inputDesc, gclmemInputDesc, bytes);
            break;
        }
        default:
            ret = NOT_SUPPORTED;
            break;
    }
    return ret;
}

EE eltwise_mali(GCLHandle_t handle,
    std::vector<TensorDesc> inputDesc,
    std::vector<void *> input,
    EltwiseParamSpec eltwiseDesc,
    GCLMem_t tmpbuf,
    TensorDesc outputDesc,
    GCLMem_t output)
{
    EE ret = SUCCESS;
    CHECK_STATUS(eltwise_checkpara_mali(handle, inputDesc, input, eltwiseDesc, outputDesc, output));
    switch (inputDesc[0].dt) {
        case DT_F16: {
            ret = eltwise_mali_fp16(
                handle, inputDesc, input, tmpbuf, outputDesc, output, eltwiseDesc);
            break;
        }
        default:
            ret = NOT_SUPPORTED;
            break;
    }
    return ret;
}
