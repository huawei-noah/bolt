// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include <float.h>
#include "error.h"

#include "cpu/general/tensor_computing_general.h"

template <typename T>
EE pooling(T *input,
    T *output,
    I32 in,
    I32 ic,
    I32 it,
    I32 ih,
    I32 iw,
    I32 ot,
    I32 oh,
    I32 ow,
    I32 stride_t,
    I32 stride_h,
    I32 stride_w,
    I32 padding_before,
    I32 padding_after,
    I32 padding_top,
    I32 padding_bottom,
    I32 padding_left,
    I32 padding_right,
    I32 kernel_t,
    I32 kernel_h,
    I32 kernel_w,
    PoolingMode pm,
    RoundMode rm,
    I32 alignSize,
    F32 minValue)
{
    CHECK_REQUIREMENT(ic % alignSize == 0);
    ic = ic / alignSize;

    EE ret = SUCCESS;
    for (I32 n = 0; n < in; n++) {
        for (I32 c = 0; c < ic; c++) {
            for (I32 j = 0; j < alignSize; j++) {
                for (I32 t = 0; t < ot; t++) {
                    for (I32 h = 0; h < oh; h++) {
                        for (I32 w = 0; w < ow; w++) {
                            int tstart = t * stride_t - padding_before;
                            int hstart = h * stride_h - padding_top;
                            int wstart = w * stride_w - padding_left;
                            int tend = tstart + kernel_t;
                            int hend = hstart + kernel_h;
                            int wend = wstart + kernel_w;
                            tstart = UNI_MAX(tstart, 0);
                            hstart = UNI_MAX(hstart, 0);
                            wstart = UNI_MAX(wstart, 0);
                            tend = UNI_MIN(tend, it);
                            hend = UNI_MIN(hend, ih);
                            wend = UNI_MIN(wend, iw);
                            float poolSize = (tend - tstart) * (hend - hstart) * (wend - wstart);
                            T value;
                            switch (pm) {
                                case POOLING_MAX:
                                    value = minValue;
                                    break;
                                case POOLING_MEAN:
                                    value = 0;
                                    break;
                                default:
                                    return NOT_SUPPORTED;
                            }
                            U32 out_off =
                                ((((n * ic + c) * ot + t) * oh + h) * ow + w) * alignSize + j;
                            for (int z = tstart; z < tend; z++) {
                                for (int x = hstart; x < hend; x++) {
                                    for (int y = wstart; y < wend; y++) {
                                        U32 in_off = ((((n * ic + c) * it + z) * ih + x) * iw + y) *
                                                alignSize +
                                            j;
                                        switch (pm) {
                                            case POOLING_MAX:
                                                value = (value > input[in_off]) ? value
                                                                                : input[in_off];
                                                break;
                                            case POOLING_MEAN:
                                                value += input[in_off];
                                                break;
                                            default:
                                                ret = NOT_SUPPORTED;
                                                break;
                                        }
                                    }
                                }
                            }
                            switch (pm) {
                                case POOLING_MAX:
                                    break;
                                case POOLING_MEAN:
                                    value = value / poolSize;
                                    break;
                                default:
                                    ret = NOT_SUPPORTED;
                                    break;
                            }

                            output[out_off] = value;
                        }
                    }
                }
            }
        }
    }
    return ret;
}

EE pooling_general(
    TensorDesc inputDesc, const void *input, PoolingParamSpec p, TensorDesc outputDesc, void *output)
{
    if (nullptr == input || nullptr == output) {
        CHECK_STATUS(NULL_POINTER);
    }
    DataType idt, odt;
    DataFormat idf, odf;
    U32 in, ic, it, ih, iw;
    U32 on, oc, ot, oh, ow;
    if (tensorIs4d(inputDesc)) {
        CHECK_STATUS(tensor4dGet(inputDesc, &idt, &idf, &in, &ic, &ih, &iw));
        CHECK_STATUS(tensor4dGet(outputDesc, &odt, &odf, &on, &oc, &oh, &ow));
        it = ot = 1;
    } else if (tensorIs5d(inputDesc)) {
        CHECK_STATUS(tensor5dGet(inputDesc, &idt, &idf, &in, &ic, &it, &ih, &iw));
        CHECK_STATUS(tensor5dGet(outputDesc, &odt, &odf, &on, &oc, &ot, &oh, &ow));
    } else {
        return NOT_SUPPORTED;
    }

    if (in != on || ic != oc || idf != DF_NCHWC8 || odf != idf) {
        CHECK_STATUS(NOT_MATCH);
    }
    EE ret = SUCCESS;
    switch (idt) {
#ifdef _USE_FP32
        case DT_F32:
            ret = pooling((F32 *)input, (F32 *)output, in, ic, it, ih, iw, ot, oh, ow, p.stride_t,
                p.stride_h, p.stride_w, p.padding_before, p.padding_after, p.padding_top,
                p.padding_bottom, p.padding_left, p.padding_right, p.kernel_t, p.kernel_h,
                p.kernel_w, p.mode, p.rm, 8, -FLT_MAX);
            break;
#endif
#ifdef _USE_FP16
        case DT_F16:
            ret = pooling((F16 *)input, (F16 *)output, in, ic, it, ih, iw, ot, oh, ow, p.stride_t,
                p.stride_h, p.stride_w, p.padding_before, p.padding_after, p.padding_top,
                p.padding_bottom, p.padding_left, p.padding_right, p.kernel_t, p.kernel_h,
                p.kernel_w, p.mode, p.rm, 8, -UNI_F16_MAX);
            break;
#endif
        default:
            ret = NOT_SUPPORTED;
            break;
    }
    return ret;
}
