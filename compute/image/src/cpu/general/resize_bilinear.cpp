// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "cpu/general/image_general.h"
#include "uni.h"

template <typename IT, typename OT>
EE resize_bilinear(TensorDesc inputDesc, IT *inArray, TensorDesc outputDesc, OT *outArray)
{
    DataType idt, odt;
    DataFormat idf, odf;
    U32 in, ic, ih, iw;
    U32 on, oc, oh, ow;
    CHECK_STATUS(tensor4dGet(inputDesc, &idt, &idf, &in, &ic, &ih, &iw));
    CHECK_STATUS(tensor4dGet(outputDesc, &odt, &odf, &on, &oc, &oh, &ow));

    U32 ic_align = 1, oc_align = 1;
    if (idf == DF_NCHWC8) {
        ic_align = 8;
    }
    if (odf == DF_NCHWC8) {
        oc_align = 8;
    }

    F32 strideH = ((F32)ih) / oh;
    F32 strideW = ((F32)iw) / ow;
    ic /= ic_align;
    oc /= oc_align;
    U32 srcTL, srcTR, srcBL, srcBR;
    for (U32 n = 0, dst = 0; n < on; n++) {
        for (U32 c = 0; c < oc; c++) {
            for (U32 h = 0; h < oh; h++) {
                F32 hC = strideH * h;
                U32 hT = floor(hC);
                U32 hB = ceil(hC);
                if (hB == hT) {
                    hB = hT + 1;
                }
                U32 hBB = UNI_MIN(hB, ih - 1);
                for (U32 w = 0; w < ow; w++) {
                    F32 wC = strideW * w;
                    U32 wL = floor(wC);
                    U32 wR = ceil(wC);
                    if (wL == wR) {
                        wR = wL + 1;
                    }
                    U32 wRR = UNI_MIN(wR, iw - 1);
                    F32 factorTL = (hB - hC) * (wR - wC);
                    F32 factorTR = (hB - hC) * (wC - wL);
                    F32 factorBL = (hC - hT) * (wR - wC);
                    F32 factorBR = (hC - hT) * (wC - wL);

                    for (U32 i = 0; i < oc_align; i++, dst++) {
                        U32 cc = c * oc_align + i;
                        if (idf == DF_NCHWC8) {
                            U32 cc1 = cc / ic_align;
                            U32 cc2 = cc % ic_align;
                            srcTL = (((n * ic + cc1) * ih + hT) * iw + wL) * ic_align + cc2;
                            srcTR = (((n * ic + cc1) * ih + hT) * iw + wRR) * ic_align + cc2;
                            srcBL = (((n * ic + cc1) * ih + hBB) * iw + wL) * ic_align + cc2;
                            srcBR = (((n * ic + cc1) * ih + hBB) * iw + wRR) * ic_align + cc2;
                        } else {
                            srcTL = ((n * ic + cc) * ih + hT) * iw + wL;
                            srcTR = ((n * ic + cc) * ih + hT) * iw + wRR;
                            srcBL = ((n * ic + cc) * ih + hBB) * iw + wL;
                            srcBR = ((n * ic + cc) * ih + hBB) * iw + wRR;
                        }
                        outArray[dst] = inArray[srcTL] * factorTL + inArray[srcTR] * factorTR +
                            inArray[srcBL] * factorBL + inArray[srcBR] * factorBR;
                    }
                }
            }
        }
    }
    return SUCCESS;
}

EE resize_bilinear_general(TensorDesc inputDesc, void *input, TensorDesc outputDesc, void *output)
{
    EE ret = NOT_SUPPORTED;
    switch (inputDesc.dt) {
#ifdef _USE_FP16
        case DT_F16: {
            ret = resize_bilinear<F16, F16>(inputDesc, (F16 *)input, outputDesc, (F16 *)output);
            break;
        }
#endif
#ifdef _USE_FP32
        case DT_F32: {
            ret = resize_bilinear<F32, F32>(inputDesc, (F32 *)input, outputDesc, (F32 *)output);
            break;
        }
#endif
        case DT_U8: {
#ifdef _USE_FP16
            if (DT_F16 == outputDesc.dt) {
                ret = resize_bilinear<U8, F16>(inputDesc, (U8 *)input, outputDesc, (F16 *)output);
            }
#endif
#ifdef _USE_FP32
            if (DT_F32 == outputDesc.dt) {
                ret = resize_bilinear<U8, F32>(inputDesc, (U8 *)input, outputDesc, (F32 *)output);
            }
#endif
            break;
        }
        default:
            break;
    }
    return ret;
}
