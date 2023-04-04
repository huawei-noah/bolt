// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "cpu/tensor_computing_cpu.h"

EE padding_infer_output_size_cpu(TensorDesc inputDesc, PadParamSpec p, TensorDesc *outputDesc)
{
    if (nullptr == outputDesc) {
        CHECK_STATUS(NULL_POINTER);
    }
    DataType idt = DT_F32;
    DataFormat idf = DF_NCHW;
    U32 in = 0, ic = 0, ih = 0, iw = 0;
    if (tensorIs2d(inputDesc)) {
        CHECK_STATUS(tensor2dGet(inputDesc, &idt, &idf, &in, &ic));
        ih = 1;
    } else if (tensorIs3d(inputDesc)) {
        CHECK_STATUS(tensor3dGet(inputDesc, &idt, &idf, &in, &ic, &ih));
        iw = 1;
    } else if (tensorIs4d(inputDesc)) {
        CHECK_STATUS(tensor4dGet(inputDesc, &idt, &idf, &in, &ic, &ih, &iw));
    } else {
        return NOT_SUPPORTED;
    }
    int out_n = in;
    int out_c = ic + p.front + p.back;
    int out_h = ih + p.top + p.bottom;
    int out_w = iw + p.left + p.right;
    if (tensorIs2d(inputDesc)) {
        *outputDesc = tensor2df(idt, idf, out_n, out_c);
    } else if (tensorIs3d(inputDesc)) {
        *outputDesc = tensor3df(idt, idf, out_n, out_c, out_h);
    } else if (tensorIs4d(inputDesc)) {
        *outputDesc = tensor4df(idt, idf, out_n, out_c, out_h, out_w);
    }
    return SUCCESS;
}

EE padding_cpu(
    TensorDesc inputDesc, const void *input, PadParamSpec p, TensorDesc outputDesc, void *output)
{
    DataType idt, odt;
    DataFormat idf, odf;
    U32 in = 0, ic = 0, ih = 0, iw = 0, on = 0, oc = 0, oh = 0, ow = 0;
    if (tensorIs2d(inputDesc)) {
        CHECK_STATUS(tensor2dGet(inputDesc, &idt, &idf, &in, &ic));
        CHECK_STATUS(tensor2dGet(outputDesc, &odt, &odf, &on, &oc));
        ih = oh = iw = ow = 1;
    } else if (tensorIs3d(inputDesc)) {
        CHECK_STATUS(tensor3dGet(inputDesc, &idt, &idf, &in, &ic, &ih));
        CHECK_STATUS(tensor3dGet(outputDesc, &odt, &odf, &on, &oc, &oh));
        iw = ow = 1;
    } else if (tensorIs4d(inputDesc)) {
        CHECK_STATUS(tensor4dGet(inputDesc, &idt, &idf, &in, &ic, &ih, &iw));
        CHECK_STATUS(tensor4dGet(outputDesc, &odt, &odf, &on, &oc, &oh, &ow));
    } else {
        return NOT_SUPPORTED;
    }
    CHECK_REQUIREMENT(in == on);
    U32 alignSize = 1;
    if (idf == DF_NCHWC8) {
        alignSize = 8;
        if (p.front % 8 != 0 || p.back % 8 != 0) {
            UNI_ERROR_LOG("try to pad in channel dimension, input layout is nchwc8, but "
                          "padding(%d,%d) mod 8 != 0\n",
                p.front, p.back);
        } else {
            p.front /= 8;
            p.back /= 8;
        }
    }
    ic /= alignSize;
    oc /= alignSize;
    for (U32 n = 0; n < in; n++) {
        for (U32 c = 0; c < ic; c++) {
            for (U32 h = 0; h < ih; h++) {
                const U8 *src =
                    (const U8 *)input + (((n * ic + c) * ih + h) * iw) * alignSize * bytesOf(idt);
                U8 *dst = (U8 *)output +
                    (((n * oc + (p.front + c)) * oh + (p.top + h)) * ow) * alignSize * bytesOf(odt);
                if (p.pad_mode == PAD_CONSTANT) {
                    UNI_INIT(p.left * alignSize, odt, p.constant_value, dst);
                    dst += p.left * alignSize * bytesOf(odt);
                    UNI_MEMCPY(dst, src, iw * alignSize * bytesOf(idt));
                    dst += iw * alignSize * bytesOf(odt);
                    UNI_INIT(p.right * alignSize, odt, p.constant_value, dst);
                } else {
                    for (U32 w = 0; w < p.left; w++) {
                        U32 index = 0;
                        if (p.pad_mode == PAD_REFLECT) {
                            index = (p.left - w) * alignSize * bytesOf(idt);
                        } else if (p.pad_mode == PAD_SYMMETRIC) {
                            index = (p.left - w - 1) * alignSize * bytesOf(idt);
                        }
                        UNI_MEMCPY(dst, src + index, alignSize * bytesOf(idt));
                        dst += alignSize * bytesOf(idt);
                    }
                    UNI_MEMCPY(dst, src, iw * alignSize * bytesOf(idt));
                    dst += iw * alignSize * bytesOf(odt);
                    for (U32 w = 0; w < p.right; w++) {
                        U32 index = (iw - 1) * alignSize * bytesOf(idt);
                        if (p.pad_mode == PAD_REFLECT) {
                            index = (iw - w - 2) * alignSize * bytesOf(idt);
                        } else if (p.pad_mode == PAD_SYMMETRIC) {
                            index = (iw - w - 1) * alignSize * bytesOf(idt);
                        }
                        UNI_MEMCPY(dst, src + index, alignSize * bytesOf(idt));
                        dst += alignSize * bytesOf(idt);
                    }
                }
            }
            U8 *dst = (U8 *)output + (((n * oc + c) * oh) * ow) * alignSize * bytesOf(odt);
            for (U32 h = 0; h < p.top; h++) {
                U32 index = h * ow * alignSize * bytesOf(odt);
                if (p.pad_mode == PAD_CONSTANT) {
                    UNI_INIT(ow * alignSize, odt, p.constant_value, dst + index);
                } else if (p.pad_mode == PAD_EDGE) {
                    UNI_MEMCPY(dst + index, dst + (p.top * ow * alignSize * bytesOf(odt)),
                        ow * alignSize * bytesOf(odt));
                } else if (p.pad_mode == PAD_REFLECT) {
                    UNI_MEMCPY(dst + index,
                        dst + ((p.top + p.top - h) * ow * alignSize * bytesOf(odt)),
                        ow * alignSize * bytesOf(odt));
                } else if (p.pad_mode == PAD_SYMMETRIC) {
                    UNI_MEMCPY(dst + index,
                        dst + ((p.top + p.top - h - 1) * ow * alignSize * bytesOf(odt)),
                        ow * alignSize * bytesOf(odt));
                } else {
                    return NOT_SUPPORTED;
                }
            }
            for (U32 h = 0; h < p.bottom; h++) {
                U32 index = (p.top + ih + h) * ow * alignSize * bytesOf(odt);
                if (p.pad_mode == PAD_CONSTANT) {
                    UNI_INIT(ow * alignSize, odt, p.constant_value, dst + index);
                } else if (p.pad_mode == PAD_EDGE) {
                    UNI_MEMCPY(dst + index, dst + ((p.top + ih - 1) * ow * alignSize * bytesOf(odt)),
                        ow * alignSize * bytesOf(odt));
                } else if (p.pad_mode == PAD_REFLECT) {
                    UNI_MEMCPY(dst + index,
                        dst + ((p.top + ih - 2 - h) * ow * alignSize * bytesOf(odt)),
                        ow * alignSize * bytesOf(odt));
                } else if (p.pad_mode == PAD_SYMMETRIC) {
                    UNI_MEMCPY(dst + index,
                        dst + ((p.top + ih - 1 - h) * ow * alignSize * bytesOf(odt)),
                        ow * alignSize * bytesOf(odt));
                } else {
                    return NOT_SUPPORTED;
                }
            }
        }

        U8 *dst = (U8 *)output + (((n * oc) * oh) * ow) * alignSize * bytesOf(odt);
        for (U32 c = 0; c < p.front; c++) {
            U32 index = c * oh * ow * alignSize * bytesOf(odt);
            if (p.pad_mode == PAD_CONSTANT) {
                UNI_INIT(oh * ow * alignSize, odt, p.constant_value, dst + index);
            } else if (p.pad_mode == PAD_EDGE) {
                UNI_MEMCPY(dst + index, dst + (p.front * oh * ow * alignSize * bytesOf(odt)),
                    oh * ow * alignSize * bytesOf(odt));
            } else if (p.pad_mode == PAD_REFLECT) {
                UNI_MEMCPY(dst + index,
                    dst + ((p.front + p.front - c) * oh * ow * alignSize * bytesOf(odt)),
                    oh * ow * alignSize * bytesOf(odt));
            } else if (p.pad_mode == PAD_SYMMETRIC) {
                UNI_MEMCPY(dst + index,
                    dst + ((p.front + p.front - c - 1) * oh * ow * alignSize * bytesOf(odt)),
                    oh * ow * alignSize * bytesOf(odt));
            } else {
                return NOT_SUPPORTED;
            }
        }

        for (U32 c = 0; c < p.back; c++) {
            U32 index = (p.front + ic + c) * oh * ow * alignSize * bytesOf(odt);
            if (p.pad_mode == PAD_CONSTANT) {
                UNI_INIT(oh * ow * alignSize, odt, p.constant_value, dst + index);
            } else if (p.pad_mode == PAD_EDGE) {
                UNI_MEMCPY(dst + index,
                    dst + ((p.front + ic - 1) * oh * ow * alignSize * bytesOf(odt)),
                    oh * ow * alignSize * bytesOf(odt));
            } else if (p.pad_mode == PAD_REFLECT) {
                UNI_MEMCPY(dst + index,
                    dst + ((p.front + ic - 2 - c) * oh * ow * alignSize * bytesOf(odt)),
                    oh * ow * alignSize * bytesOf(odt));
            } else if (p.pad_mode == PAD_SYMMETRIC) {
                UNI_MEMCPY(dst + index,
                    dst + ((p.front + ic - 1 - c) * oh * ow * alignSize * bytesOf(odt)),
                    oh * ow * alignSize * bytesOf(odt));
            } else {
                return NOT_SUPPORTED;
            }
        }
    }
    return SUCCESS;
}
