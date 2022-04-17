// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _DEPTHWISE_CONVOLUTION_MALI_FP16
#define _DEPTHWISE_CONVOLUTION_MALI_FP16

#include "gpu/mali/fp16/tensor_computing_fp16.h"

inline void calDepthwisePaddingVal(TensorDesc inputDesc,
    ConvolutionParamSpec convParamSpec,
    U32 edge_align,
    U32 *pl,
    U32 *pr,
    U32 *pt,
    U32 *pb)
{
    U32 fh = convParamSpec.kernel_h;
    U32 sh = convParamSpec.stride_h;
    U32 dh = convParamSpec.dilatedRate_h;
    U32 fhd = (fh - 1) * dh + 1;
    U32 ih = inputDesc.dims[1];
    U32 plv = convParamSpec.pad_left;
    U32 prv = convParamSpec.pad_right;
    U32 ptv = convParamSpec.pad_top;
    U32 pbv = edge_align * sh + (fhd / 2) * 2 - ptv - ih;
    if (pbv < convParamSpec.pad_bottom) {
        pbv = convParamSpec.pad_bottom;
    }
    *pl = plv;
    *pr = prv;
    *pt = ptv;
    *pb = pbv;
}

EE depthwise_convolution_transform_filter_bytes_mali_fp16(
    TensorDesc filterDesc, ForwardRunInfoMali_t forwardRunInfo, TensorDesc *ftmDesc);

EE depthwise_convolution_transform_filter_mali_fp16(GCLHandle_t handle,
    TensorDesc filterDesc,
    GCLMem_t filter,
    ForwardRunInfoMali_t forwardRunInfo,
    TensorDesc *fltmemDesc,
    GCLMem_t fltmem);

EE depthwise_convolution_infer_forward_tmp_bytes_mali_fp16(TensorDesc inputDesc,
    TensorDesc filterDesc,
    TensorDesc outputDesc,
    ConvolutionParamSpec convParamSpec,
    ForwardRunInfoMali_t forwardRunInfo,
    U32 *bytes);

EE depthwise_convolution_mali_fp16(GCLHandle_t handle,
    TensorDesc inputDesc,
    const GCLMem_t input,
    TensorDesc filterDesc,
    const GCLMem_t filter,
    ConvolutionParamSpec convParamSpec,
    ForwardRunInfoMali_t forwardRunInfo,
    TensorDesc biasDesc,
    const GCLMem_t bias,
    U32 tmpBytes,
    GCLMem_t tmpBuf,
    TensorDesc outputDesc,
    GCLMem_t output,
    ActivationMode depthwiseActivationMode);
#endif
