// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), 
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, 
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE 
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR 
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


#include <cmath>
#include <cstring>
#include "type.h"
#include "tensor_desc.h"
#include "error.h"
#include "image.h"
#include "cpu/general/image_general.h"

template<typename IT, typename OT>
EE resize_bilinear(TensorDesc inputDesc, IT* inArray,
                        TensorDesc outputDesc, OT* outArray)
{
    DataType idt, odt;
    DataFormat idf, odf;
    U32 in, ic, ih, iw;
    U32 on, oc, oh, ow;
    CHECK_STATUS(tensor4dGet(inputDesc, &idt, &idf, &in, &ic, &ih, &iw));
    CHECK_STATUS(tensor4dGet(outputDesc, &odt, &odf, &on, &oc, &oh, &ow));

    if (idf == DF_NCHWC8) {
        CHECK_STATUS(from_nchwc8_to_nchw<IT>(&inputDesc, inArray));
    }
    CHECK_STATUS(tensor4dGet(inputDesc, &idt, &idf, &in, &ic, &ih, &iw));
    if (idf != DF_NCHW && idf != DF_RGB) {
        CHECK_STATUS(NOT_MATCH);
    }

    F32 strideH = (F32)(ih - 1) / (F32)(oh - 1);
    F32 strideW = (F32)(iw - 1) / (F32)(ow - 1);

    for (U32 n = 0; n < on; n++) {
        for (U32 c = 0; c < oc; c++) {
            I32 outBase = n*oc*oh*ow + c*oh*ow;
            I32 inBase = n*oc*ih*iw + c*ih*iw;
            for (U32 h = 0; h < oh; h++) {
                for (U32 w = 0; w < ow; w++) {
                    if (h == 0 && w == 0) {
                        outArray[outBase] = inArray[inBase];
                        continue;
                    }
                    if (h == 0 && w == ow - 1) {
                        outArray[outBase + w] = inArray[inBase + iw - 1];
                        continue;
                    }
                    if (h == oh - 1 && w == 0) {
                        outArray[outBase + h * ow] = inArray[inBase + (ih - 1) * iw];
                        continue;
                    }
                    if (h == oh - 1 && w == ow - 1) {
                        outArray[outBase + h * ow + w] = inArray[inBase + (ih - 1) * iw + iw - 1];
                        continue;
                    }

                    F32 hC = strideH * h;
                    F32 wC = strideW * w;

                    I32 hT = floor(hC);
                    I32 hB = ceil(hC);
                    I32 wL = floor(wC);
                    I32 wR = ceil(wC);

                    if (hT == hB && wL == wR) {
                        outArray[outBase + h * ow + w] = inArray[inBase + hT * iw + wL];
                    } else if (hT == hB) {
                        outArray[outBase + h * ow + w] = inArray[inBase + hT * iw + wL] * (wR - wC) + inArray[inBase + hT * iw + wR] * (wC - wL);
                    } else if (wL == wR) {
                        outArray[outBase + h * ow + w] = inArray[inBase + hT * iw + wL] * (hB - hC) + inArray[inBase + hB * iw + wL] * (hC - hT);
                    } else {
                        F32 factorTL = (hB - hC) * (wR - wC);
                        F32 factorTR = (hB - hC) * (wC - wL);
                        F32 factorBL = (hC - hT) * (wR - wC);
                        F32 factorBR = (hC - hT) * (wC - wL);

                        outArray[outBase + h * ow + w] = inArray[inBase + hT * iw + wL] * factorTL;
                        outArray[outBase + h * ow + w] += inArray[inBase + hT * iw + wR] * factorTR;
                        outArray[outBase + h * ow + w] += inArray[inBase + hB * iw + wL] * factorBL;
                        outArray[outBase + h * ow + w] += inArray[inBase + hB * iw + wR] * factorBR;
                    }
                }
            }
        }
    }

    if (odf == DF_NCHWC8) {
        outputDesc.df = DF_NCHW;
        CHECK_STATUS(from_nchw_to_nchwc8<OT>(&outputDesc, outArray));
    }
    return SUCCESS;
}

EE resize_bilinear_general(TensorDesc inputDesc, void* input,
            TensorDesc outputDesc, void* output)
{
    EE ret = SUCCESS;
    switch (inputDesc.dt) {
#ifdef _USE_FP16
        case DT_F16: {
            ret = resize_bilinear<F16, F16>(inputDesc, (F16*)input,
                                    outputDesc, (F16*)output);
            break;
        }
#endif
#ifdef _USE_FP32
        case DT_F32: {
            ret = resize_bilinear<F32, F32>(inputDesc, (F32*)input,
                                    outputDesc, (F32*)output);
            break;
        }
#endif
        case DT_U8: {
#ifdef _USE_FP16
            if (DT_F16 == outputDesc.dt) {
                ret = resize_bilinear<U8, F16>(inputDesc, (U8*)input,
                                        outputDesc, (F16*)output);
            }
#endif
#ifdef _USE_FP32
            if (DT_F32 == outputDesc.dt) {
                ret = resize_bilinear<U8, F32>(inputDesc, (U8*)input,
                                        outputDesc, (F32*)output);
            }
#endif
            break;
        }
        default:
            return NOT_SUPPORTED;
    }
    return ret;
}
