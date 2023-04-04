// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _CONVOLUTION_MALI_FP16
#define _CONVOLUTION_MALI_FP16

#include "gpu/mali/fp16/tensor_computing_fp16.h"
#include "gpu/mali/cl/kernel_option/common_opt.h"
inline bool useNchwCalMode(DataFormat idf, U32 fw, U32 ic, U32 dw, U32 dh)
{
    bool useNchwMode = false;
    bool qualCommDev = check_qualcomm_device();
    char rc = ic & 3;
    if (idf == DF_NCHW && dw == 1 && dh == 1 && fw <= 7) {
        if (!qualCommDev || rc != 0) {
            useNchwMode = true;
        }
    }
    return useNchwMode;
}

inline bool useGemvCalMode(
    TensorDesc inputDesc, ConvolutionParamSpec convParamSpec, GCLMemType imt, GCLMemType omt)
{
    bool useGemvCalMode = false;
    U32 iw, ih, it;
    U32 fw = convParamSpec.kernel_w;
    U32 fh = convParamSpec.kernel_h;
    U32 ft = convParamSpec.kernel_t;
    U32 sw = convParamSpec.stride_w;
    U32 sh = convParamSpec.stride_h;
    U32 dw = convParamSpec.dilatedRate_w;
    U32 dh = convParamSpec.dilatedRate_h;
    tensorSelectGet(inputDesc, NULL, NULL, NULL, NULL, &ih, &iw, &it);
    if (fw * fh * ft * iw * ih * it * sw * sh * dw * dh == 1 && imt == GCL_MEM_BUF &&
        omt == GCL_MEM_BUF) {
        useGemvCalMode = true;
    }
    return useGemvCalMode;
}

inline void calPaddingVal(TensorDesc inputDesc,
    TensorDesc filterDesc,
    ConvolutionParamSpec convParamSpec,
    U32 w_align,
    U32 h_align,
    U32 n_align,
    bool useNchwMode,
    U32 *pl,
    U32 *pr,
    U32 *pt,
    U32 *pb,
    U32 *pa,
    U32 *pf)
{
    U32 iw, ih, ic, it, in;
    tensorSelectGet(inputDesc, NULL, NULL, &in, &ic, &ih, &iw, &it);
    U32 plv, prv, ptv, pbv, pav;
    U32 fh = convParamSpec.kernel_h;
    U32 sh = convParamSpec.stride_h;
    U32 dh = convParamSpec.dilatedRate_h;
    U32 fhd = (fh - 1) * dh + 1;
    h_align *= sh;
    plv = convParamSpec.pad_left;
    ptv = convParamSpec.pad_top;
    if (useNchwMode) {
        U32 fw = convParamSpec.kernel_w;
        U32 sw = convParamSpec.stride_w;
        U32 dw = convParamSpec.dilatedRate_w;
        U32 fwd = (fw - 1) * dw + 1;
        w_align *= sw;
        prv = w_align + (fwd / 2 * 2) - plv - iw;
        if (prv < convParamSpec.pad_right) {
            prv = convParamSpec.pad_right;
        }
        pbv = h_align + (fhd / 2 * 2) - ptv - ih;
        if (pbv < convParamSpec.pad_bottom) {
            pbv = convParamSpec.pad_bottom;
        }
    } else {
        prv = convParamSpec.pad_right;
        pbv = h_align + (fhd / 2 * 2) - ptv - ih;
        if (pbv < convParamSpec.pad_bottom) {
            pbv = convParamSpec.pad_bottom;
        }
        ic = (ic + 3) / 4;
    }
    pav = (n_align - in) * ic * it;
    *pl = plv;
    *pr = prv;
    *pt = ptv;
    *pb = pbv;
    *pa = pav;
    *pf = 0;
}

EE convolution_transform_filter_bytes_mali_fp16(
    TensorDesc filterDesc, ForwardRunInfoMali_t forwardRunInfo, TensorDesc *ftmDesc);

EE convolution_transform_filter_mali_fp16(GCLHandle_t handle,
    TensorDesc filterDesc,
    GCLMem_t filter,
    ForwardRunInfoMali_t forwardRunInfo,
    TensorDesc *fltmemDesc,
    GCLMem_t fltmem,
    GCLMem_t tmp);

EE convolution_infer_forward_tmp_bytes_mali_fp16(TensorDesc inputDesc,
    TensorDesc filterDesc,
    TensorDesc outputDesc,
    ConvolutionParamSpec convParamSpec,
    ForwardRunInfoMali_t forwardRunInfo,
    U32 *bytes);

EE convolution_mali_fp16(GCLHandle_t handle,
    TensorDesc inputDesc,
    const GCLMem_t input,
    TensorDesc filterDesc,
    const GCLMem_t filter,
    ConvolutionParamSpec convParamSpec,
    ForwardRunInfoMali_t forwardRunInfo,
    TensorDesc biasDesc,
    const GCLMem_t bias,
    U32 tmpBytes,
    std::vector<GCLMem_t> tmpBuf,
    TensorDesc outputDesc,
    GCLMem_t output,
    ActivationParamSpec activationMode);
#endif
