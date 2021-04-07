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
#include "gpu/mali/fp16/pooling_mali_fp16.h"

EE pooling_infer_output_size_mali(TensorDesc inputDesc,
    PoolingParamSpec poolingParamSpec,
    TensorDesc *outputDesc,
    GCLMemDesc_t gclmemInputDesc,
    GCLMemDesc_t gclmemOutputDesc)
{
    /*tensorDesc record cpu org data format info*/
    /*gclmemDesc record gpu trans data format info*/
    DataType idt;
    DataFormat idf;
    U32 iw, ih, ic, in, it;
    U32 ow, oh, ot;
    U32 kw, kh, kt, sw, sh, st, pl, pt, pr, pb, pt_b, pt_a;
    U32 inDims;
    tensorSelectGet(inputDesc, &idt, &idf, &in, &ic, &ih, &iw, &it);
    inDims = inputDesc.nDims;
    pl = poolingParamSpec.padding_left;
    pr = poolingParamSpec.padding_right;
    pt = poolingParamSpec.padding_top;
    pb = poolingParamSpec.padding_bottom;
    pt_b = poolingParamSpec.padding_before;
    pt_a = poolingParamSpec.padding_after;
    kw = poolingParamSpec.kernel_w;
    kh = poolingParamSpec.kernel_h;
    kt = poolingParamSpec.kernel_t;
    sw = poolingParamSpec.stride_w;
    sh = poolingParamSpec.stride_h;
    st = poolingParamSpec.stride_t;
    if (st == 0) {
        st = 1;
    }
    switch (poolingParamSpec.rm) {
        case CEIL: {
            ow = (U32)(ceil((double(iw + pl + pr - kw) / sw))) + 1;
            oh = (U32)(ceil((double(ih + pt + pb - kh) / sh))) + 1;
            ot = (U32)(ceil((double(it + pt_b + pt_a - kt) / st))) + 1;
            break;
        }
        case FLOOR: {
            ow = (U32)(floor((double(iw + pl + pr - kw) / sw))) + 1;
            oh = (U32)(floor((double(ih + pb + pt - kh) / sh))) + 1;
            ot = (U32)(floor((double(it + pt_b + pt_a - kt) / st))) + 1;
            break;
        }
        case TF_SAME: {
            ow = (U32)(ceil((double(iw) / sw)));
            oh = (U32)(ceil((double(ih) / sh)));
            ot = (U32)(ceil((double(it) / st)));
            break;
        }
        case TF_VALID: {
            ow = (U32)(ceil((double(iw - kw + 1) / sw)));
            oh = (U32)(ceil((double(ih - kh + 1) / sh)));
            ot = (U32)(ceil((double(it - kt + 1) / st)));
            break;
        }
        default: {
            return NOT_SUPPORTED;
        }
    }
    U32 iw_align, ih_align;
    ih_align = ih + pt + pb;
    ih_align = ih_align - pt * 2;
    iw_align = iw + pl + pr;
    iw_align = iw_align - pl * 2;

    if (inDims == 5) {
        *outputDesc = tensor5df(idt, idf, in, ic, ot, oh, ow);
    } else {
        *outputDesc = tensor4df(idt, idf, in, ic, oh, ow);
        it = 1;
        ot = 1;
    }

    ic = ALIGN(ic, 4);
    CHECK_STATUS(infer_gclmem_desc_ncwhc4_3d(iw_align, ih_align, ic, it, in, pl, pt, ow, oh, ic, ot,
        in, idt, idt, gclmemInputDesc, gclmemOutputDesc));
    return SUCCESS;
}
EE pooling_mali(GCLHandle_t handle,
    TensorDesc inputDesc,
    const GCLMem_t input,
    PoolingParamSpec poolingParamSpec,
    const void *scale,
    GCLMem_t temp,
    TensorDesc outputDesc,
    GCLMem_t output)
{
    UNUSED(scale);
    EE ret = SUCCESS;
    switch (inputDesc.dt) {
        case DT_F16: {
            ret = pooling_mali_fp16(
                handle, inputDesc, input, poolingParamSpec, outputDesc, output, temp);
            break;
        }
        default:
            ret = NOT_SUPPORTED;
            break;
    }
    return ret;
}

EE pooling_infer_forward_tmp_bytes_mali(
    TensorDesc inputDesc, U32 *bytes, ForwardRunInfoMali_t forwardRunInfo)
{
    EE ret = SUCCESS;
    switch (inputDesc.dt) {
        case DT_F16: {
            ret = pooling_infer_forward_tmp_bytes_mali_fp16(inputDesc, bytes, forwardRunInfo);
            break;
        }
        default:
            ret = NOT_SUPPORTED;
            break;
    }
    return ret;
}
