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
#include "gpu/mali/fp16/argmax_mali_fp16.h"

EE argmax_infer_output_size_mali(TensorDesc inputDesc,
    ArgMaxParamSpec p,
    TensorDesc *outputDesc,
    GCLMemDesc_t gclmemInputDesc,
    GCLMemDesc_t gclmemOutputDesc)
{
    /*tensorDesc record cpu org data format info*/
    /*gclmemDesc record gpu trans data format info*/
    int axis = p.axis;
    TensorDesc desc = inputDesc;
    if (axis < 0) {
        axis += inputDesc.nDims;
    }
    axis = inputDesc.nDims - 1 - axis;
    for (int i = axis; i < (I32)(inputDesc.nDims) - 1; i++) {
        desc.dims[i] = desc.dims[i + 1];
    }
    desc.nDims = inputDesc.nDims - 1;
    desc.dt = DT_U32;
    if (outputDesc) {
        *outputDesc = desc;
    }

    if (gclmemInputDesc || gclmemOutputDesc) {
        U32 iw, ih, ic;
        U32 ow, oh, oc;
        U32 inDims = inputDesc.nDims;
        U32 onDims = desc.nDims;
        DataType idt = inputDesc.dt;
        DataType odt = desc.dt;
        iw = inputDesc.dims[0];
        ih = (inDims > 1) ? inputDesc.dims[1] : 1;
        ic = (inDims > 2) ? inputDesc.dims[2] : 1;
        ow = desc.dims[0];
        oh = (onDims > 1) ? desc.dims[1] : 1;
        oc = (onDims > 2) ? desc.dims[2] : 1;
        U32 iw_align = (axis == 0) ? (iw + 7) / 8 * 8 : iw;
        U32 ih_align = (axis == 1) ? (iw + 7) / 8 * 8 : ih;
        U32 ic_align = (axis == 2) ? (iw + 7) / 8 * 8 : ic;
        bool need_pad = false;
        if (iw_align != iw || ih_align != ih || ic_align != ic) {
            need_pad = true;
        }
        CHECK_STATUS(infer_gclmem_desc_nchw(iw_align, ih_align, ic_align, 0, 0, ow, oh, oc, idt,
            odt, gclmemInputDesc, gclmemOutputDesc, need_pad));
    }
    return SUCCESS;
}

inline EE argmax_checkpara_mali(GCLHandle_t handle, GCLMem_t input, GCLMem_t tmpbuf, GCLMem_t output)
{
    if (handle == nullptr || input == nullptr || output == nullptr || tmpbuf == nullptr) {
        return NULL_POINTER;
    }
    if (input->desc.memFormat != output->desc.memFormat || input->desc.memFormat != DF_NCHW) {
        return NOT_SUPPORTED;
    }
    return SUCCESS;
}

EE argmax_infer_forward_tmp_bytes_mali(
    TensorDesc inputDesc, ArgMaxParamSpec p, TensorDesc outputDesc, U32 *bytes)
{
    EE ret = SUCCESS;
    switch (inputDesc.dt) {
        case DT_F16: {
            ret = argmax_infer_forward_tmp_bytes_mali_fp16(inputDesc, p.axis, outputDesc, bytes);
            break;
        }
        default:
            ret = NOT_SUPPORTED;
            break;
    }
    return ret;
}

EE argmax_mali(GCLHandle_t handle,
    TensorDesc inputDesc,
    GCLMem_t input,
    ArgMaxParamSpec p,
    GCLMem_t tmpbuf,
    TensorDesc outputDesc,
    GCLMem_t output)
{
    EE ret = SUCCESS;
    CHECK_STATUS(argmax_checkpara_mali(handle, input, tmpbuf, output));
    switch (inputDesc.dt) {
        case DT_F16: {
            ret = argmax_mali_fp16(handle, inputDesc, input, p.axis, tmpbuf, outputDesc, output);
            break;
        }
        default:
            ret = NOT_SUPPORTED;
            break;
    }
    return ret;
}
