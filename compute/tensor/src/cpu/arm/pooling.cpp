// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "cpu/arm/tensor_computing_arm.h"
#ifdef _USE_FP32
#include "cpu/arm/fp32/tensor_computing_fp32.h"
#endif
#ifdef _USE_FP16
#include "cpu/arm/fp16/tensor_computing_fp16.h"
#endif
#ifdef _USE_INT8
#include "cpu/arm/int8/tensor_computing_int8.h"
#endif
typedef EE (*ArmPoolingFunction)(const I32 &tstart,
    const I32 &tend,
    const I32 &hstart,
    const I32 &hend,
    const I32 &wstart,
    const I32 &wend,
    const I32 &poolSize,
    const I32 &kernelSize,
    const U8 *input,
    const I32 &it,
    const I32 &ih,
    const I32 &iw,
    U8 *output,
    void *scale);

EE pooling_arm(TensorDesc inputDesc,
    const void *input,
    PoolingParamSpec p,
    void *scale,
    TensorDesc outputDesc,
    void *output)
{
    if (nullptr == input || nullptr == output) {
        return NULL_POINTER;
    }
    DataType idt, odt;
    DataFormat idf, odf;
    U32 in, ic, it, ih, iw;
    U32 on, oc, ot, oh, ow;
    if (tensorIs4d(inputDesc)) {
        CHECK_STATUS(tensor4dGet(inputDesc, &idt, &idf, &in, &ic, &ih, &iw));
        CHECK_STATUS(tensor4dGet(outputDesc, &odt, &odf, &on, &oc, &oh, &ow));
        it = ot = 1;
        p.pad_before = p.pad_after = 0;
        p.kernel_t = p.stride_t = 1;
    } else if (tensorIs5d(inputDesc)) {
        CHECK_STATUS(tensor5dGet(inputDesc, &idt, &idf, &in, &ic, &it, &ih, &iw));
        CHECK_STATUS(tensor5dGet(outputDesc, &odt, &odf, &on, &oc, &ot, &oh, &ow));
    } else {
        return NOT_SUPPORTED;
    }
    EE ret = SUCCESS;
    if (idt != odt) {
        ret = NOT_MATCH;
    }
    if (in != on || ic != oc) {
        ret = NOT_MATCH;
    }
    if (idf != DF_NCHWC8 || odf != idf) {
        ret = NOT_MATCH;
    }
    if (p.pad_before >= p.kernel_t || p.pad_top >= p.kernel_h || p.pad_left >= p.kernel_w) {
        return NOT_SUPPORTED;
    }

    ic /= 8;
    ArmPoolingFunction func = nullptr;
    if (p.mode == POOLING_MAX) {
        switch (idt) {
#ifdef _USE_FP32
            case DT_F32:
                func = pooling_c8_fp32<POOLING_MAX>;
                break;
#endif
#ifdef _USE_FP16
            case DT_F16:
                func = pooling_c8_fp16<POOLING_MAX>;
                break;
#endif
#ifdef _USE_INT8
            case DT_I8:
                func = pooling_c8_int8<POOLING_MAX>;
                break;
#endif
            default:
                return NOT_SUPPORTED;
        }
    } else if (p.mode == POOLING_MEAN) {
        switch (idt) {
#ifdef _USE_FP32
            case DT_F32:
                func = pooling_c8_fp32<POOLING_MEAN>;
                break;
#endif
#ifdef _USE_FP16
            case DT_F16:
                func = pooling_c8_fp16<POOLING_MEAN>;
                break;
#endif
#ifdef _USE_INT8
            case DT_I8:
                func = pooling_c8_int8<POOLING_MEAN>;
                break;
#endif
            default:
                return NOT_SUPPORTED;
        }
    } else {
        return NOT_SUPPORTED;
    }

#ifdef _USE_OPENMP
#pragma omp parallel num_threads(OMP_NUM_THREADS)
#endif
    {
        int kernelSize = p.kernel_t * p.kernel_h * p.kernel_w;
#ifdef _USE_OPENMP
#pragma omp for
#endif
        for (U32 o = 0; o < in * ic; o++) {
            U32 n = o / ic;
            U32 c = o % ic;
            const U8 *src = (const U8 *)input + o * it * ih * iw * 8 * bytesOf(idt);
            U8 *dst = (U8 *)output + o * ot * oh * ow * 8 * bytesOf(idt);
            for (U32 t = 0; t < ot; t++) {
                int tstart = t * (int)p.stride_t - (int)p.pad_before;
                int tend = UNI_MIN(tstart + p.kernel_t, it);
                tstart = UNI_MAX(tstart, 0);
                for (U32 h = 0; h < oh; h++) {
                    int hstart = h * (int)p.stride_h - (int)p.pad_top;
                    int hend = UNI_MIN(hstart + p.kernel_h, ih);
                    hstart = UNI_MAX(hstart, 0);
                    for (U32 w = 0; w < ow; w++, dst += 8 * bytesOf(idt)) {
                        int wstart = w * (int)p.stride_w - (int)p.pad_left;
                        int wend = UNI_MIN(wstart + p.kernel_w, iw);
                        wstart = UNI_MAX(wstart, 0);
                        int poolSize = kernelSize;
                        if (!p.count_include_pad) {
                            poolSize = (tend - tstart) * (hend - hstart) * (wend - wstart);
                        }
                        ret = func(tstart, tend, hstart, hend, wstart, wend, kernelSize, poolSize,
                            src, it, ih, iw, dst, scale);
                    }
                }
            }
        }
    }
    return ret;
}

EE pooling_bp_arm(
    TensorDesc inputDesc, const void *input, PoolingParamSpec p, TensorDesc outputDesc, void *output)
{
    EE ret = SUCCESS;
    if (nullptr == input || nullptr == output) {
        ret = NULL_POINTER;
    }
    DataType idt, odt;
    DataFormat idf, odf;
    U32 in = 0, ic = 0, ih = 0, iw = 0, on = 0, oc = 0, oh = 0, ow = 0;
    CHECK_STATUS(tensor4dGet(inputDesc, &idt, &idf, &in, &ic, &ih, &iw));
    CHECK_STATUS(tensor4dGet(outputDesc, &odt, &odf, &on, &oc, &oh, &ow));

    if (idt != odt) {
        ret = NOT_MATCH;
    }
    if (in != on || ic != oc) {
        ret = NOT_MATCH;
    }
    if (idf != DF_NCHWC8 || odf != idf) {
        ret = NOT_MATCH;
    }
    if (p.pad_top >= p.kernel_h || p.pad_left >= p.kernel_w) {
        ret = NOT_SUPPORTED;
    }

    UNI_MEMSET(output, 0, tensorNumBytes(outputDesc));
    ic /= 8;
#ifdef _USE_OPENMP
#pragma omp parallel num_threads(OMP_NUM_THREADS)
#endif
    {
        int poolSize = p.kernel_t * p.kernel_h * p.kernel_w;
#ifdef _USE_OPENMP
#pragma omp for
#endif
        for (U32 o = 0; o < in * ic; o++) {
            U32 n = o / ic;
            U32 c = o % ic;
            const U8 *src = (const U8 *)input + o * ih * iw * 8 * bytesOf(idt);
            U8 *dst = (U8 *)output + o * oh * ow * 8 * bytesOf(idt);
            for (U32 h = 0; h < ih; h++) {
                int hstart = (int)h * (int)p.stride_h - (int)p.pad_top;
                int hend = UNI_MIN(hstart + p.kernel_h, oh);
                hstart = UNI_MAX(hstart, 0);
                for (U32 w = 0; w < iw; w++, src += 8 * bytesOf(idt)) {
                    int wstart = (int)w * (int)p.stride_w - (int)p.pad_left;
                    int wend = UNI_MIN(wstart + p.kernel_w, ow);
                    wstart = UNI_MAX(wstart, 0);
                    if (!p.count_include_pad) {
                        poolSize = (hend - hstart) * (wend - wstart);
                    }
                    switch (idt) {
#ifdef _USE_FP32
                        case DT_F32:
                            ret = pooling_bp_c8_fp32((const F32 *)src, hstart, hend, wstart, wend,
                                poolSize, (F32 *)dst, ow, p);
                            break;
#endif
                        default:
                            ret = NOT_SUPPORTED;
                            break;
                    }
                }
            }
        }
    }

    return ret;
}
