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

EE padding_general(TensorDesc inputDesc,
    const void *input,
    PadParamSpec padParamSpec,
    TensorDesc outputDesc,
    void *output)
{
    DataType idt, odt;
    DataFormat idf, odf;
    U32 in = 0, ic = 0, ih = 0, iw = 0, on = 0, oc = 0, oh = 0, ow = 0;
    CHECK_STATUS(tensor4dGet(inputDesc, &idt, &idf, &in, &ic, &ih, &iw));
    CHECK_STATUS(tensor4dGet(outputDesc, &odt, &odf, &on, &oc, &oh, &ow));
    CHECK_REQUIREMENT(in == on);
    CHECK_REQUIREMENT(ic == oc);
    U32 alignSize = 1;
    if (idf == DF_NCHWC8) {
        alignSize = 8;
    }
    ic /= alignSize;
    oc /= alignSize;
    for (U32 n = 0; n < in; n++) {
        for (U32 c = 0; c < ic; c++) {
            for (U32 h = 0; h < ih; h++) {
                const U8 *inPtr =
                    (const U8 *)input + (((n * ic + c) * ih + h) * iw) * alignSize * bytesOf(idt);
                U8 *outPtr = (U8 *)output +
                    (((n * oc + c) * oh + (padParamSpec.top + h)) * ow) * alignSize * bytesOf(odt);
                if (padParamSpec.pad_mode == Pad_Constant) {
                    memset(outPtr, 0, padParamSpec.left * alignSize * bytesOf(odt));
                    outPtr += padParamSpec.left * alignSize * bytesOf(odt);
                    memcpy(outPtr, inPtr, iw * alignSize * bytesOf(idt));
                    outPtr += iw * alignSize * bytesOf(odt);
                    memset(outPtr, 0, padParamSpec.right * alignSize * bytesOf(odt));
                } else {
                    for (U32 w = 0; w < padParamSpec.left; w++) {
                        U32 index = 0;
                        if (padParamSpec.pad_mode == Pad_Reflect) {
                            index = (padParamSpec.left - w) * alignSize * bytesOf(idt);
                        } else if (padParamSpec.pad_mode == Pad_Symmetric) {
                            index = (padParamSpec.left - w - 1) * alignSize * bytesOf(idt);
                        }
                        memcpy(outPtr, inPtr + index, alignSize * bytesOf(idt));
                        outPtr += alignSize * bytesOf(idt);
                    }
                    memcpy(outPtr, inPtr, iw * alignSize * bytesOf(idt));
                    outPtr += iw * alignSize * bytesOf(odt);
                    for (U32 w = 0; w < padParamSpec.right; w++) {
                        U32 index = (iw - 1) * alignSize * bytesOf(idt);
                        if (padParamSpec.pad_mode == Pad_Reflect) {
                            index = (iw - w - 2) * alignSize * bytesOf(idt);
                        } else if (padParamSpec.pad_mode == Pad_Symmetric) {
                            index = (iw - w - 1) * alignSize * bytesOf(idt);
                        }
                        memcpy(outPtr, inPtr + index, alignSize * bytesOf(idt));
                        outPtr += alignSize * bytesOf(idt);
                    }
                }
            }
            U8 *outPtr = (U8 *)output + (((n * oc + c) * oh) * ow) * alignSize * bytesOf(odt);
            for (U32 h = 0; h < padParamSpec.top; h++) {
                U32 index = h * ow * alignSize * bytesOf(odt);
                if (padParamSpec.pad_mode == Pad_Constant) {
                    memset(outPtr + index, 0, ow * alignSize * bytesOf(odt));
                } else if (padParamSpec.pad_mode == Pad_Edge) {
                    memcpy(outPtr + index,
                        outPtr + (padParamSpec.top * ow * alignSize * bytesOf(odt)),
                        ow * alignSize * bytesOf(odt));
                } else if (padParamSpec.pad_mode == Pad_Reflect) {
                    memcpy(outPtr + index,
                        outPtr +
                            ((padParamSpec.top + padParamSpec.top - h) * ow * alignSize *
                                bytesOf(odt)),
                        ow * alignSize * bytesOf(odt));
                } else if (padParamSpec.pad_mode == Pad_Symmetric) {
                    memcpy(outPtr + index,
                        outPtr +
                            ((padParamSpec.top + padParamSpec.top - h - 1) * ow * alignSize *
                                bytesOf(odt)),
                        ow * alignSize * bytesOf(odt));
                } else {
                    return NOT_SUPPORTED;
                }
            }
            for (U32 h = 0; h < padParamSpec.bottom; h++) {
                U32 index = (padParamSpec.top + ih + h) * ow * alignSize * bytesOf(odt);
                if (padParamSpec.pad_mode == Pad_Constant) {
                    memset(outPtr + index, 0, ow * alignSize * bytesOf(odt));
                } else if (padParamSpec.pad_mode == Pad_Edge) {
                    memcpy(outPtr + index,
                        outPtr + ((padParamSpec.top + ih - 1) * ow * alignSize * bytesOf(odt)),
                        ow * alignSize * bytesOf(odt));
                } else if (padParamSpec.pad_mode == Pad_Reflect) {
                    // memcpy(outPtr+index, outPtr+((padParamSpec.top+ih-2-h)*ow*alignSize*bytesOf(odt)), ow*alignSize*bytesOf(odt));
                    memcpy(outPtr + index,
                        outPtr +
                            ((padParamSpec.top + ih - 1 - padParamSpec.bottom + h) * ow *
                                alignSize * bytesOf(odt)),
                        ow * alignSize * bytesOf(odt));
                } else if (padParamSpec.pad_mode == Pad_Symmetric) {
                    memcpy(outPtr + index,
                        outPtr + ((padParamSpec.top + ih - 1 - h) * ow * alignSize * bytesOf(odt)),
                        ow * alignSize * bytesOf(odt));
                } else {
                    return NOT_SUPPORTED;
                }
            }
        }
    }
    return SUCCESS;
}
