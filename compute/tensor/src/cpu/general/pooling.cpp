// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "cpu/general/tensor_computing_general.h"

template <typename T1, typename T2>
EE pooling(DataType idt,
    T1 *input,
    T1 *output,
    I32 in,
    I32 ic,
    I32 it,
    I32 ih,
    I32 iw,
    I32 ot,
    I32 oh,
    I32 ow,
    PoolingParamSpec p,
    I32 alignSize,
    F32 minValue,
    void *scale)
{
    CHECK_REQUIREMENT(ic % alignSize == 0);
    ic = ic / alignSize;
    float poolSize = p.kernel_t * p.kernel_h * p.kernel_w;

#ifdef _USE_INT8
    F32 *inputScale = (F32 *)scale;
    F32 *outputScale = inputScale + 1;
    I32 shift = 65536;
    I32 factor = shift / poolSize;
    if (p.mode == POOLING_MAX) {
        *outputScale = *inputScale;
    } else {
        *outputScale = *inputScale * factor * poolSize / (F32)shift;
    }
#endif

    EE ret = SUCCESS;
    for (I32 n = 0; n < in; n++) {
        for (I32 c = 0; c < ic; c++) {
            for (I32 j = 0; j < alignSize; j++) {
                for (I32 t = 0; t < ot; t++) {
                    for (I32 h = 0; h < oh; h++) {
                        for (I32 w = 0; w < ow; w++) {
                            int tstart = t * p.stride_t - p.pad_before;
                            int hstart = h * p.stride_h - p.pad_top;
                            int wstart = w * p.stride_w - p.pad_left;
                            int tend = tstart + p.kernel_t;
                            int hend = hstart + p.kernel_h;
                            int wend = wstart + p.kernel_w;
                            tstart = UNI_MAX(tstart, 0);
                            hstart = UNI_MAX(hstart, 0);
                            wstart = UNI_MAX(wstart, 0);
                            tend = UNI_MIN(tend, it);
                            hend = UNI_MIN(hend, ih);
                            wend = UNI_MIN(wend, iw);
                            if (!p.count_include_pad) {
                                poolSize = (tend - tstart) * (hend - hstart) * (wend - wstart);
                            }
                            T1 maxVal = 0;
                            T2 meanVal = 0;
                            switch (p.mode) {
                                case POOLING_MAX:
                                    maxVal = minValue;
                                    break;
                                case POOLING_MEAN:
                                    meanVal = 0;
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
                                        switch (p.mode) {
                                            case POOLING_MAX:
                                                maxVal = (maxVal > input[in_off]) ? maxVal
                                                                                  : input[in_off];
                                                break;
                                            case POOLING_MEAN:
                                                meanVal += input[in_off];
                                                break;
                                            default:
                                                ret = NOT_SUPPORTED;
                                                break;
                                        }
                                    }
                                }
                            }
                            switch (p.mode) {
                                case POOLING_MAX:
                                    output[out_off] = maxVal;
                                    break;
                                case POOLING_MEAN:
                                    if (idt == DT_I8 || idt == DT_U8_Q) {
#ifdef _USE_INT8
                                        I32 factor = shift /
                                            ((tend - tstart) * (hend - hstart) * (wend - wstart));
                                        output[out_off] = ((I32)meanVal * factor) >> 16;
#endif
                                    } else {
                                        output[out_off] = meanVal / poolSize;
                                    }
                                    break;
                                default:
                                    ret = NOT_SUPPORTED;
                                    break;
                            }
                        }
                    }
                }
            }
        }
    }
    return ret;
}

EE pooling_general(TensorDesc inputDesc,
    const void *input,
    PoolingParamSpec p,
    void *scale,
    TensorDesc outputDesc,
    void *output)
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

    if (in != on || ic != oc || odf != idf) {
        CHECK_STATUS(NOT_MATCH);
    }
    I32 alignSize = 1;
    if (idf == DF_NCHWC4) {
        alignSize = 4;
    } else if (idf == DF_NCHWC8) {
        alignSize = 8;
    } else if (idf == DF_NCHWC16) {
        alignSize = 16;
    }
    EE ret = SUCCESS;
    switch (idt) {
#ifdef _USE_FP32
        case DT_F32:
            ret = pooling<F32, F32>(idt, (F32 *)input, (F32 *)output, in, ic, it, ih, iw, ot, oh,
                ow, p, alignSize, -FLT_MAX, scale);
            break;
#endif
#ifdef _USE_FP16
        case DT_F16:
            ret = pooling<F16, F16>(idt, (F16 *)input, (F16 *)output, in, ic, it, ih, iw, ot, oh,
                ow, p, alignSize, -UNI_F16_MAX, scale);
            break;
#endif
#ifdef _USE_INT8
        case DT_I8:
            ret = pooling<INT8, I32>(idt, (INT8 *)input, (INT8 *)output, in, ic, it, ih, iw, ot, oh,
                ow, p, alignSize, -UNI_F16_MAX, scale);
            break;
        case DT_U8_Q:
            ret = pooling<UINT8, I32>(idt, (UINT8 *)input, (UINT8 *)output, in, ic, it, ih, iw, ot,
                oh, ow, p, alignSize, -UNI_F16_MAX, scale);
            break;
#endif
        default:
            ret = NOT_SUPPORTED;
            break;
    }
    return ret;
}
