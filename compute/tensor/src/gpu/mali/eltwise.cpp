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
#include "gpu/mali/fp16/eltwise_mali_fp16.h"

inline void gcl_mem_desc_align(U32 size, DataType dt, GCLMemDesc_t desc)
{
    U32 s0 = desc[0].stride[0];
    U32 s1 = desc[0].stride[1];
    U32 s2 = desc[0].stride[2];
    U32 off0 = desc[0].offset[0];
    U32 off1 = desc[0].offset[1];
    U32 off2 = desc[0].offset[2];
    for (U32 i = 1; i < size; i++) {
        s0 = (s0 >= desc[i].stride[0]) ? s0 : desc[i].stride[0];
        s1 = (s1 >= desc[i].stride[1]) ? s1 : desc[i].stride[1];
        s2 = (s2 >= desc[i].stride[2]) ? s2 : desc[i].stride[2];
        off0 = (off0 >= desc[i].offset[0]) ? off0 : desc[i].offset[0];
        off1 = (off1 >= desc[i].offset[1]) ? off1 : desc[i].offset[1];
        off2 = (off2 >= desc[i].offset[2]) ? off2 : desc[i].offset[2];
    }
    U32 num = s0 * s1 * s2 * 4;
    U32 byteSize = num * bytesOf(dt);
    for (U32 i = 0; i < size; i++) {
        desc[i].stride[0] = s0;
        desc[i].stride[1] = s1;
        desc[i].stride[2] = s2;
        desc[i].offset[0] = off0;
        desc[i].offset[1] = off1;
        desc[i].offset[2] = off2;
        desc[i].num = num;
        desc[i].byteSize = byteSize;
    }
}
EE eltwise_infer_output_size_mali(std::vector<TensorDesc> inputDesc,
    TensorDesc *outputDesc,
    GCLMemDesc_t gclmemInputDesc,
    GCLMemDesc_t gclmemOutputDesc)
{
    /*tensorDesc record cpu org data format info*/
    /*gclmemDesc record gpu trans data format info*/
    U32 arrayDimMax = 0;
    bool sameDesc = eltwise_same_desc(inputDesc, &arrayDimMax);
    if (outputDesc) {
        *outputDesc = inputDesc[arrayDimMax];
    }

    DataType idt;
    DataFormat idf;
    U32 iw, ih, ic, in;
    tensorSelectGet(inputDesc[arrayDimMax], &idt, &idf, &in, &ic, &ih, &iw);

    if (sameDesc) {
        U32 size = inputDesc.size();
        if (gclmemInputDesc) {
            bool inputIsModelInput = true;
            bool inputIsAllNCHW = true;

            for (U32 i = 0; i < size; i++) {
                if (gclmemInputDesc[i].byteSize > 0) {
                    inputIsModelInput = false;
                }
                if (gclmemInputDesc[i].memFormat != DF_NCHW) {
                    inputIsAllNCHW = false;
                }
            }

            if (inputIsAllNCHW && !inputIsModelInput) {
                CHECK_STATUS(infer_gclmem_desc_nchw(
                    iw, ih, ic, 0, 0, iw, ih, ic, idt, idt, gclmemInputDesc, gclmemOutputDesc));
                for (U32 i = 0; i < size; i++) {
                    DataType tdt;
                    U32 tw, th, tc;
                    tensorSelectGet(inputDesc[i], &tdt, NULL, NULL, &tc, &th, &tw);
                    CHECK_STATUS(infer_gclmem_desc_nchw(
                        tw, th, tc, 0, 0, tw, th, tc, tdt, tdt, &gclmemInputDesc[i], NULL));
                }
            } else {
                for (U32 i = 0; i < size; i++) {
                    DataType tdt;
                    U32 tw, th, tc;
                    tensorSelectGet(inputDesc[i], &tdt, NULL, NULL, &tc, &th, &tw);
                    CHECK_STATUS(infer_gclmem_desc_ncwhc4(
                        tw, th, tc, 0, 0, tw, th, tc, tdt, tdt, &gclmemInputDesc[i], NULL));
                }
                CHECK_STATUS(infer_gclmem_desc_ncwhc4(
                    iw, ih, ic, 0, 0, iw, ih, ic, idt, idt, NULL, gclmemOutputDesc));
            }
            gcl_mem_desc_align(size, idt, gclmemInputDesc);
        }
        return SUCCESS;
    } else {
        if (inputDesc.size() > 2) {
            CHECK_STATUS(NOT_SUPPORTED);
        }
        if (gclmemInputDesc) {
            CHECK_STATUS(infer_gclmem_desc_ncwhc4(iw, ih, ic, 0, 0, iw, ih, ic, idt, idt,
                &gclmemInputDesc[arrayDimMax], gclmemOutputDesc));
            tensorSelectGet(inputDesc[1 - arrayDimMax], &idt, NULL, &in, &ic, &ih, &iw);
            if (gclmemInputDesc[1 - arrayDimMax].byteSize == 0 ||
                gclmemInputDesc[1 - arrayDimMax].memFormat == DF_NCHW) {
                CHECK_STATUS(infer_gclmem_desc_nchw(iw, ih, ic, 0, 0, iw, ih, ic, idt, idt,
                    &gclmemInputDesc[1 - arrayDimMax], NULL));
            } else {
                CHECK_STATUS(infer_gclmem_desc_ncwhc4(iw, ih, ic, 0, 0, iw, ih, ic, idt, idt,
                    &gclmemInputDesc[1 - arrayDimMax], NULL));
            }
        }
        return SUCCESS;
    }
    return NOT_SUPPORTED;
}

inline EE eltwise_checkpara_mali(GCLHandle_t handle,
    std::vector<TensorDesc> inputDesc,
    std::vector<void *> input,
    EltwiseParamSpec eltwiseDesc,
    TensorDesc outputDesc,
    GCLMem_t output)
{
    if (handle == nullptr || nullptr == output) {
        return NULL_POINTER;
    }
    for (auto it : input) {
        GCLMem_t ptr = (GCLMem_t)it;
        if (ptr == nullptr) {
            return NULL_POINTER;
        }
    }
    EltwiseMode eltwiseMode = eltwiseDesc.elt_mode;
    U32 arrayDimMax = 0;
    bool sameDesc = eltwise_same_desc(inputDesc, &arrayDimMax);
    if (sameDesc) {
        for (auto it : input) {
            if (((GCLMem_t)(it))->desc.memFormat != output->desc.memFormat) {
                return NOT_SUPPORTED;
            }
        }
        for (auto it : inputDesc) {
            // if(it.df != outputDesc.df)           return NOT_SUPPORTED;
            if (it.dims[0] != outputDesc.dims[0]) {
                return NOT_SUPPORTED;
            }
            if (it.dims[1] != outputDesc.dims[1]) {
                return NOT_SUPPORTED;
            }
            if (it.dims[2] != outputDesc.dims[2]) {
                return NOT_SUPPORTED;
            }
            if (it.dims[3] != outputDesc.dims[3]) {
                return NOT_SUPPORTED;
            }
        }
    } else {
        if (inputDesc.size() > 2) {
            CHECK_STATUS(NOT_SUPPORTED);
        }
    }
    if (outputDesc.df != DF_NCHW && outputDesc.df != DF_MKT) {
        return NOT_SUPPORTED;
    }
    if (eltwiseMode != ELTWISE_SUM && eltwiseMode != ELTWISE_MAX && eltwiseMode != ELTWISE_PROD) {
        return NOT_SUPPORTED;
    }
    return SUCCESS;
}

EE eltwise_mali(GCLHandle_t handle,
    std::vector<TensorDesc> inputDesc,
    std::vector<void *> input,
    EltwiseParamSpec eltwiseDesc,
    TensorDesc outputDesc,
    GCLMem_t output)
{
    EE ret = SUCCESS;
    CHECK_STATUS(eltwise_checkpara_mali(handle, inputDesc, input, eltwiseDesc, outputDesc, output));
    switch (inputDesc[0].dt) {
        case DT_F16: {
            ret = eltwise_mali_fp16(handle, inputDesc, input, outputDesc, output, eltwiseDesc);
            break;
        }
        default:
            ret = NOT_SUPPORTED;
            break;
    }
    return ret;
}
