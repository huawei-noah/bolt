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
EE pooling_bp(T *input,
    T *output,
    U32 in,
    U32 ic,
    U32 ih,
    U32 iw,
    U32 strideH,
    U32 strideW,
    U32 paddingT,
    U32 paddingL,
    U32 kernelH,
    U32 kernelW,
    PoolingMode pm,
    U32 oh,
    U32 ow,
    U32 alignSize)
{
    UNUSED(pm);
    CHECK_REQUIREMENT(ic % alignSize == 0);
    ic = ic / alignSize;

    for (U32 n = 0; n < in; n++) {
        for (U32 c = 0; c < ic; c++) {
            for (U32 j = 0; j < alignSize; j++) {
                for (I32 h = 0; h < (I32)ih; h++) {
                    for (I32 w = 0; w < (I32)iw; w++) {
                        int hstart = int(h * strideH - paddingT);
                        int wstart = int(w * strideW - paddingL);
                        int hend = hstart + kernelH;
                        int wend = wstart + kernelW;
                        hstart = (hstart < 0) ? 0 : hstart;
                        wstart = (wstart < 0) ? 0 : wstart;
                        hend = (hend > (int)oh) ? oh : hend;
                        wend = (wend > (int)ow) ? ow : wend;
                        float poolSize = (hend - hstart) * (wend - wstart);
                        for (int x = hstart; x < hend; x++) {
                            for (int y = wstart; y < wend; y++) {
                                U32 in_off = ((((n * ic + c) * ih) + h) * iw + w) * alignSize + j;
                                U32 out_off = ((((n * ic + c) * oh) + x) * ow + y) * alignSize + j;
                                output[out_off] += input[in_off] / poolSize;
                            }
                        }
                    }
                }
            }
        }
    }
    return SUCCESS;
}

EE pooling_bp_general(TensorDesc inputDesc,
    const void *input,
    PoolingParamSpec poolingParamSpec,
    TensorDesc outputDesc,
    void *output)
{
    if (nullptr == input || nullptr == output) {
        CHECK_STATUS(NULL_POINTER);
    }
    DataType idt, odt;
    DataFormat idf, odf;
    U32 in = 0, ic = 0, ih = 0, iw = 0, on = 0, oc = 0, oh = 0, ow = 0;
    CHECK_STATUS(tensor4dGet(inputDesc, &idt, &idf, &in, &ic, &ih, &iw));
    CHECK_STATUS(tensor4dGet(outputDesc, &odt, &odf, &on, &oc, &oh, &ow));

    if (in != on || ic != oc) {
        CHECK_STATUS(NOT_MATCH);
    }
    if (idf != DF_NCHWC8 || odf != idf) {
        CHECK_STATUS(NOT_MATCH);
    }

    U32 strideH = poolingParamSpec.stride_h;
    U32 strideW = poolingParamSpec.stride_w;
    U32 paddingT = poolingParamSpec.padding_top;
    U32 paddingL = poolingParamSpec.padding_left;
    U32 kernelSizeH = poolingParamSpec.kernel_h;
    U32 kernelSizeW = poolingParamSpec.kernel_w;

    EE ret = SUCCESS;
    switch (idt) {
#ifdef _USE_FP32
        case DT_F32:
            ret = pooling_bp((F32 *)input, (F32 *)output, in, ic, ih, iw, strideH, strideW,
                paddingT, paddingL, kernelSizeH, kernelSizeW, poolingParamSpec.mode, oh, ow, 8);
            break;
#endif
        default:
            ret = NOT_SUPPORTED;
    }
    return ret;
}