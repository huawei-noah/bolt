// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _H_TENSOR_TRANSPOSE
#define _H_TENSOR_TRANSPOSE

#include "tensor_desc.h"
#include "uni.h"
#include "affinity_policy.h"

template <typename T>
inline static EE transformToNCHWKernel(
    TensorDesc inputDesc, const T *input, TensorDesc outputDesc, T *output)
{
    DataType idt, odt;
    DataFormat idf, odf;
    U32 in, ic, ih, iw, on, oc, oh, ow;
    if (tensorIs2d(inputDesc)) {
        CHECK_STATUS(tensor2dGet(inputDesc, &idt, &idf, &in, &iw));
        ic = 1;
        ih = 1;
    } else if (tensorIs3d(inputDesc)) {
        if (inputDesc.df == DF_NHWC) {
            CHECK_STATUS(tensor3dGet(inputDesc, &idt, &idf, &in, &ih, &iw));
            ic = 1;
        } else {
            CHECK_STATUS(tensor3dGet(inputDesc, &idt, &idf, &in, &ic, &ih));
            iw = 1;
        }
    } else if (tensorIs4d(inputDesc)) {
        CHECK_STATUS(tensor4dGet(inputDesc, &idt, &idf, &in, &ic, &ih, &iw));
    } else {
        UNI_ERROR_LOG("not support transform %d-dim tensor to NCHW format.\n", (int)inputDesc.nDims);
        return NOT_SUPPORTED;
    }
    if (tensorIs2d(outputDesc)) {
        CHECK_STATUS(tensor2dGet(outputDesc, &odt, &odf, &on, &oc));
        oh = ow = 1;
    } else if (tensorIs3d(outputDesc)) {
        CHECK_STATUS(tensor3dGet(outputDesc, &odt, &odf, &on, &oc, &oh));
        ow = 1;
    } else if (tensorIs4d(outputDesc)) {
        CHECK_STATUS(tensor4dGet(outputDesc, &odt, &odf, &on, &oc, &oh, &ow));
    } else {
        UNI_ERROR_LOG("not support transform to %d-dim NCHW tensor.\n", (int)outputDesc.nDims);
        return NOT_SUPPORTED;
    }
    CHECK_REQUIREMENT(idt == odt);
    EE ret = SUCCESS;
    switch (idf) {
        case DF_NORMAL:
        case DF_MTK:
        case DF_NCHW: {
            if (in == on && ic == oc && ih == oh && iw == ow) {
                if (output != input) {
                    UNI_MEMCPY(output, input, tensorNumBytes(outputDesc));
                }
            } else {
                U32 tileSize = UNI_MIN(iw, ow) * bytesOf(idt);
                for (U32 n = 0; n < on && n < in; n++) {
                    for (U32 c = 0; c < oc && c < ic; c++) {
                        for (U32 h = 0; h < oh && h < ih; h++) {
                            U32 srcIndex = ((n * ic + c) * ih + h) * iw;
                            U32 dstIndex = ((n * oc + c) * oh + h) * ow;
                            UNI_MEMCPY(output + dstIndex, input + srcIndex, tileSize);
                        }
                    }
                }
            }
            break;
        }
        case DF_NCHWC8: {
            U32 cx = 8;
            U32 ic_a = ic / cx;
            U32 minH = UNI_MIN(oh, ih);
            U32 minC = UNI_MIN(oc, ic);
            for (U32 n = 0; n < on && n < in; n++) {
#ifdef _USE_OPENMP
#pragma omp parallel for num_threads(OMP_NUM_THREADS)
#endif
                for (U32 c = 0; c < minC; c++) {
                    for (U32 h = 0; h < minH; h++) {
                        for (U32 w = 0; w < ow && w < iw; w++) {
                            U32 c_a = c / cx;
                            U32 c_b = c % cx;
                            U32 srcIndex = (((n * ic_a + c_a) * ih + h) * iw + w) * cx + c_b;
                            U32 dstIndex = ((n * oc + c) * oh + h) * ow + w;
                            // support channel cut
                            output[dstIndex] = input[srcIndex];
                        }
                    }
                }
            }
            break;
        }
        case DF_NCHWC16: {
            U32 ic16 = ic / 16;
            for (U32 n = 0; n < in; ++n) {
                U32 c = 0;
                for (; c < ic16; ++c) {
                    for (U32 h = 0; h < ih; ++h) {
                        for (U32 w = 0; w < iw; ++w) {
                            for (U32 cc = 0; cc < 16; ++cc) {
                                output[n * ic * ih * iw + (c * 16 + cc) * ih * iw + h * iw + w] =
                                    input[n * ic * ih * iw + c * 16 * ih * iw + (h * iw + w) * 16 + cc];
                            }
                        }
                    }
                }
                c *= 16;
                while (c < ic) {
                    U32 cx = ic - c;
                    cx = (cx == 12) ? 8 : cx;
                    for (U32 h = 0; h < ih; ++h) {
                        for (U32 w = 0; w < iw; ++w) {
                            for (U32 cc = 0; cc < cx; ++cc) {
                                output[n * ic * ih * iw + (c + cc) * ih * iw + h * iw + w] =
                                    input[n * ic * ih * iw + c * ih * iw + (h * iw + w) * cx + cc];
                            }
                        }
                    }
                    c += cx;
                }
            }
            break;
        }
        case DF_NHWCN8: {
            in /= 8;
            for (U32 n = 0; n < in; n++) {
                for (U32 h = 0; h < oh && h < ih; h++) {
                    for (U32 w = 0; w < ow && w < iw; w++) {
                        for (U32 c = 0; c < oc && c < ic; c++) {
                            for (U32 n8 = 0; n8 < 8; n8++) {
                                U32 srcIndex = (((n * ih + h) * iw + w) * ic + c) * 8 + n8;
                                U32 dstIndex = (((n * 8 + n8) * oc + c) * oh + h) * ow + w;
                                output[dstIndex] = input[srcIndex];
                            }
                        }
                    }
                }
            }
            break;
        }
        case DF_NHWC: {
            for (U32 n = 0; n < on && n < in; n++) {
                for (U32 c = 0; c < oc && c < ic; c++) {
                    for (U32 h = 0; h < oh && h < ih; h++) {
                        for (U32 w = 0; w < ow && w < iw; w++) {
                            U32 srcIndex = ((n * ih + h) * iw + w) * ic + c;
                            U32 dstIndex = ((n * oc + c) * oh + h) * ow + w;
                            output[dstIndex] = input[srcIndex];
                        }
                    }
                }
            }
            break;
        }
        default: {
            UNI_ERROR_LOG(
                "not support transform %s format to NCHW format.\n", DataFormatName()[idf]);
            ret = NOT_SUPPORTED;
            break;
        }
    }
    return ret;
}

inline EE transformToNCHW(
    TensorDesc inputDesc, const void *input, TensorDesc outputDesc, void *output)
{
    if (nullptr == input || nullptr == output) {
        CHECK_STATUS(NULL_POINTER);
    }
    EE ret = NOT_SUPPORTED;
    switch (inputDesc.dt) {
#ifdef _USE_FP32
        case DT_F32: {
            ret = transformToNCHWKernel<F32>(inputDesc, (F32 *)input, outputDesc, (F32 *)output);
            break;
        }
#endif
#ifdef _USE_FP16
        case DT_F16: {
            ret = transformToNCHWKernel<F16>(inputDesc, (F16 *)input, outputDesc, (F16 *)output);
            break;
        }
#endif
#ifdef _USE_INT8
        case DT_I8: {
            ret = transformToNCHWKernel<INT8>(inputDesc, (INT8 *)input, outputDesc, (INT8 *)output);
            break;
        }
        case DT_U8_Q: {
            ret = transformToNCHWKernel<UINT8>(
                inputDesc, (UINT8 *)input, outputDesc, (UINT8 *)output);
            break;
        }
#endif
        default: {
            UNI_ERROR_LOG("not support transform %s type tensor.\n", DataTypeName()[inputDesc.dt]);
            break;
        }
    }
    return ret;
}

template <typename T>
inline static EE transformToNHWCKernel(
    TensorDesc inputDesc, const T *input, TensorDesc outputDesc, T *output)
{
    DataType idt, odt;
    DataFormat idf, odf;
    U32 in, ic, ih, iw, on, oc, oh, ow;
    if (tensorIs2d(inputDesc)) {
        CHECK_STATUS(tensor2dGet(inputDesc, &idt, &idf, &in, &iw));
        ic = 1;
        ih = 1;
    } else if (tensorIs3d(inputDesc)) {
        CHECK_STATUS(tensor3dGet(inputDesc, &idt, &idf, &in, &ih, &iw));
        ic = 1;
    } else if (tensorIs4d(inputDesc)) {
        CHECK_STATUS(tensor4dGet(inputDesc, &idt, &idf, &in, &ic, &ih, &iw));
    } else {
        UNI_ERROR_LOG("not support transform %d-dim tensor to NHWC format\n", (int)inputDesc.nDims);
        return NOT_SUPPORTED;
    }
    if (tensorIs4d(outputDesc)) {
        CHECK_STATUS(tensor4dGet(outputDesc, &odt, &odf, &on, &oc, &oh, &ow));
    } else {
        UNI_ERROR_LOG("not support transform to %d-dim NHWC tensor.\n", (int)outputDesc.nDims);
        return NOT_SUPPORTED;
    }
    U32 size = tensorNumElements(outputDesc);
    U32 ihiw = ih * iw;
    EE ret = SUCCESS;
    switch (idf) {
        case DF_NHWC: {
            CHECK_REQUIREMENT(tensorNumElements(inputDesc) == size);
            if (input != output) {
                UNI_MEMCPY(output, input, tensorNumBytes(inputDesc));
            }
            break;
        }
        case DF_NORMAL:
        case DF_MTK:
        case DF_NCHW: {
            CHECK_REQUIREMENT(tensorNumElements(inputDesc) == size);
            for (U32 o = 0, srcIndex = 0; o < in; o++) {
                for (U32 cc = 0; cc < ic; cc++) {
                    for (U32 hw = 0; hw < ihiw; hw++, srcIndex++) {
                        U32 dstIndex = (o * ihiw + hw) * ic + cc;
                        output[dstIndex] = input[srcIndex];
                    }
                }
            }
            break;
        }
        case DF_NCHWC8:
        case DF_NCHWC16: {
            U32 align = (idf == DF_NCHWC16) ? 16 : 8;
            CHECK_REQUIREMENT(ic % align == 0);
            ic /= align;
            for (U32 n = 0, srcIndex = 0; n < in; n++) {
                for (U32 c = 0; c < ic; c++) {
                    for (U32 hw = 0; hw < ihiw; hw++) {
                        for (U32 cx = 0; cx < align; cx++, srcIndex++) {
                            U32 dstIndex = ((n * ihiw + hw) * ic + c) * align + cx;
                            output[dstIndex] = input[srcIndex];
                        }
                    }
                }
            }
            break;
        }
        default: {
            UNI_ERROR_LOG(
                "not support transform %s format tensor to NHWC format\n", DataFormatName()[idf]);
            ret = NOT_SUPPORTED;
            break;
        }
    }
    return ret;
}

inline EE transformToNHWC(
    TensorDesc inputDesc, const void *input, TensorDesc outputDesc, void *output)
{
    if (nullptr == input || nullptr == output) {
        return NULL_POINTER;
    }
    EE ret = NOT_SUPPORTED;
    switch (inputDesc.dt) {
#ifdef _USE_FP32
        case DT_F32: {
            ret = transformToNHWCKernel<F32>(inputDesc, (F32 *)input, outputDesc, (F32 *)output);
            break;
        }
#endif
#ifdef _USE_FP16
        case DT_F16: {
            ret = transformToNHWCKernel<F16>(inputDesc, (F16 *)input, outputDesc, (F16 *)output);
            break;
        }
#endif
#ifdef _USE_INT8
        case DT_I8: {
            ret = transformToNHWCKernel<INT8>(inputDesc, (INT8 *)input, outputDesc, (INT8 *)output);
            break;
        }
#endif
        default: {
            UNI_ERROR_LOG("not support transform %s type tensor.\n", DataTypeName()[inputDesc.dt]);
            break;
        }
    }
    return ret;
}

inline EE transformNCHWC16ToNCHWC8(
    TensorDesc inputDesc, const void *input, TensorDesc outputDesc, void *output)
{
    if (input == NULL || output == NULL) {
        CHECK_STATUS(NULL_POINTER);
    }
    DataType idt, odt;
    DataFormat idf, odf;
    U32 in, ic, ih, iw, on, oc, oh, ow;
    if (tensorIs2d(inputDesc)) {
        if (input != output) {
            UNI_MEMCPY(output, input, tensorNumBytes(inputDesc));
        }
        return SUCCESS;
    } else if (tensorIs3d(inputDesc)) {
        CHECK_STATUS(tensor3dGet(inputDesc, &idt, &idf, &in, &ic, &ih));
        CHECK_STATUS(tensor3dGet(outputDesc, &odt, &odf, &on, &oc, &oh));
        iw = ow = 1;
    } else {
        CHECK_STATUS(tensor4dGet(inputDesc, &idt, &idf, &in, &ic, &ih, &iw));
        CHECK_STATUS(tensor4dGet(outputDesc, &odt, &odf, &on, &oc, &oh, &ow));
    }
    CHECK_REQUIREMENT(in == on && idf == DF_NCHWC16 && odf == DF_NCHWC8 && idt == odt);
    int elementSize = bytesOf(idt);
    const U8 *inputPtr = (const U8 *)input;
    U8 *outputPtr = (U8 *)output;
    oc /= 8;
    for (U32 n = 0; n < on && n < in; n++) {
        for (U32 c = 0; c < oc; c += 2) {
            for (U32 h = 0; h < oh && h < ih; h++) {
                for (U32 w = 0; w < ow && w < iw; w++) {
                    for (U32 c8 = 0; c8 < 2 && c8 + c < oc; ++c8) {
                        U32 srcIndex =
                            n * ic * ih * iw + c * ih * iw * 8 + (h * iw + w) * 16 + c8 * 8;
                        U32 dstIndex = n * ic * ih * iw + (c + c8) * ih * iw * 8 + (h * iw + w) * 8;
                        UNI_MEMCPY(outputPtr + dstIndex * elementSize,
                            inputPtr + srcIndex * elementSize, elementSize * 8);
                    }
                }
            }
        }
    }
    return SUCCESS;
}

inline EE transformNCHWToNCHWC8(
    TensorDesc inputDesc, const void *input, TensorDesc outputDesc, void *output)
{
    if (input == NULL || output == NULL) {
        CHECK_STATUS(NULL_POINTER);
    }
    DataType idt, odt;
    DataFormat idf, odf;
    U32 in, ic, ih, iw, on, oc, oh, ow;
    if (tensorIs2d(inputDesc)) {
        if (input != output) {
            UNI_MEMCPY(output, input, tensorNumBytes(inputDesc));
        }
        return SUCCESS;
    } else if (tensorIs3d(inputDesc)) {
        CHECK_STATUS(tensor3dGet(inputDesc, &idt, &idf, &in, &ic, &ih));
        CHECK_STATUS(tensor3dGet(outputDesc, &odt, &odf, &on, &oc, &oh));
        iw = ow = 1;
    } else {
        CHECK_STATUS(tensor4dGet(inputDesc, &idt, &idf, &in, &ic, &ih, &iw));
        CHECK_STATUS(tensor4dGet(outputDesc, &odt, &odf, &on, &oc, &oh, &ow));
    }
    CHECK_REQUIREMENT(in == on && idf != DF_NCHWC8 && odf == DF_NCHWC8 && idt == odt);
    int elementSize = bytesOf(idt);
    const U8 *inputPtr = (const U8 *)input;
    U8 *outputPtr = (U8 *)output;
    oc /= 8;
    for (U32 n = 0; n < on && n < in; n++) {
        for (U32 c = 0; c < oc; c++) {
            for (U32 h = 0; h < oh && h < ih; h++) {
                for (U32 w = 0; w < ow && w < iw; w++) {
                    for (U32 c8 = 0, c_i = c * 8; c8 < 8; c8++, c_i++) {
                        U32 dstIndex = ((((n * oc + c) * oh + h) * ow + w) * 8 + c8) * elementSize;
                        // support channel padding
                        if (c_i < ic) {
                            U32 srcIndex = (((n * ic + c_i) * ih + h) * iw + w) * elementSize;
                            UNI_MEMCPY(outputPtr + dstIndex, inputPtr + srcIndex, elementSize);
                        } else {
                            UNI_MEMSET(outputPtr + dstIndex, 0, elementSize);
                        }
                    }
                }
            }
        }
    }
    return SUCCESS;
}

inline EE transformNHWCToNCHWC8(
    TensorDesc inputDesc, const void *input, TensorDesc outputDesc, void *output)
{
    if (input == NULL || output == NULL) {
        CHECK_STATUS(NULL_POINTER);
    }
    DataType idt, odt;
    DataFormat idf, odf;
    U32 in, ic, ih, iw, on, oc, oh, ow;
    CHECK_STATUS(tensor4dGet(inputDesc, &idt, &idf, &in, &ic, &ih, &iw));
    CHECK_STATUS(tensor4dGet(outputDesc, &odt, &odf, &on, &oc, &oh, &ow));
    CHECK_REQUIREMENT(in == on && idf == DF_NHWC && odf == DF_NCHWC8 && idt == odt);
    int elementSize = bytesOf(idt);
    const U8 *inputPtr = (const U8 *)input;
    U8 *outputPtr = (U8 *)output;
    oc /= 8;
    for (U32 n = 0; n < on && n < in; n++) {
        for (U32 c = 0; c < oc; c++) {
            for (U32 h = 0; h < oh && h < ih; h++) {
                for (U32 w = 0; w < ow && w < iw; w++) {
                    for (U32 c8 = 0, c_i = c * 8; c8 < 8; c8++, c_i++) {
                        U32 dstIndex = ((((n * oc + c) * oh + h) * ow + w) * 8 + c8) * elementSize;
                        // support channel padding
                        if (c_i < ic) {
                            U32 srcIndex = (((n * ih + h) * iw + w) * ic + c_i) * elementSize;
                            UNI_MEMCPY(outputPtr + dstIndex, inputPtr + srcIndex, elementSize);
                        } else {
                            UNI_MEMSET(outputPtr + dstIndex, 0, elementSize);
                        }
                    }
                }
            }
        }
    }
    return SUCCESS;
}

inline EE transformNCHWC8ToNCHWC8ByGroup(
    TensorDesc inputDesc, const void *input, int group, TensorDesc outputDesc, void *output)
{
    U32 inputSize = tensorNumElements(inputDesc);
    U32 outputSize = tensorNumElements(outputDesc);
    if (group <= 1 || inputSize == outputSize) {
        if (input != output) {
            UNI_MEMCPY(output, input, outputSize);
        }
        return SUCCESS;
    }

    U32 channelAlignSize = 8;
    DataType idt, odt;
    DataFormat idf, odf;
    U32 in, ic, ih, iw;
    U32 on, oc, oh, ow;
    CHECK_STATUS(tensor4dGet(inputDesc, &idt, &idf, &in, &ic, &ih, &iw));
    CHECK_STATUS(tensor4dGet(outputDesc, &odt, &odf, &on, &oc, &oh, &ow));
    CHECK_REQUIREMENT(idt == odt);
    CHECK_REQUIREMENT(idf == DF_NCHWC8 && odf == DF_NCHWC8);
    U32 icg = ic / group;
    U32 ocg = oc / group;
    U32 ict = ic / channelAlignSize;
    U32 oct = oc / channelAlignSize;
    U32 elementSize = bytesOf(idt);
    for (U32 n = 0; n < in; n++) {
        for (I32 g = 0, od = 0; g < group; g++) {
            for (U32 c = 0; c < ocg; c++, od++) {
                U32 id = g * icg + c;
                U32 id_a = id / channelAlignSize;
                U32 od_a = od / channelAlignSize;
                U32 id_b = id % channelAlignSize;
                U32 od_b = od % channelAlignSize;
                for (U32 h = 0; h < oh; h++) {
                    for (U32 w = 0; w < ow; w++) {
                        U32 dstIndex =
                            ((((n * oct + od_a) * oh + h) * ow + w) * channelAlignSize + od_b) *
                            elementSize;
                        if (h < ih && w < iw) {
                            U32 srcIndex =
                                ((((n * ict + id_a) * ih + h) * iw + w) * channelAlignSize + id_b) *
                                elementSize;
                            UNI_MEMCPY(
                                (U8 *)output + dstIndex, (const U8 *)input + srcIndex, elementSize);
                        } else {
                            UNI_MEMSET((U8 *)output + dstIndex, 0, elementSize);
                        }
                    }
                }
            }
        }
    }
    return SUCCESS;
}

template <typename T>
inline static EE transformToNCHWC16Kernel(
    TensorDesc inputDesc, const T *input, TensorDesc outputDesc, T *output)
{
    DataType idt, odt;
    DataFormat idf, odf;
    U32 in, ic, ih, iw, on, oc, oh, ow;
    if (tensorIs2d(inputDesc)) {
        CHECK_STATUS(tensor2dGet(inputDesc, &idt, &idf, &in, &iw));
        ic = 1;
        ih = 1;
    } else if (tensorIs3d(inputDesc)) {
        if (inputDesc.df == DF_NHWC) {
            CHECK_STATUS(tensor3dGet(inputDesc, &idt, &idf, &in, &ih, &iw));
            ic = 1;
        } else {
            CHECK_STATUS(tensor3dGet(inputDesc, &idt, &idf, &in, &ic, &ih));
            iw = 1;
        }
    } else if (tensorIs4d(inputDesc)) {
        CHECK_STATUS(tensor4dGet(inputDesc, &idt, &idf, &in, &ic, &ih, &iw));
    } else {
        UNI_ERROR_LOG(
            "not support transform %d-dim tensor to NCHWC16 format\n", (int)inputDesc.nDims);
        return NOT_SUPPORTED;
    }
    if (tensorIs3d(outputDesc)) {
        CHECK_STATUS(tensor3dGet(outputDesc, &odt, &odf, &on, &oc, &oh));
        ow = 1;
    } else if (tensorIs4d(outputDesc)) {
        CHECK_STATUS(tensor4dGet(outputDesc, &odt, &odf, &on, &oc, &oh, &ow));
    } else {
        UNI_ERROR_LOG("not support transform to %d-dim NCHWC16 tensor\n", (int)outputDesc.nDims);
        return NOT_SUPPORTED;
    }
    CHECK_REQUIREMENT(idt == odt);
    EE ret = SUCCESS;
    switch (idf) {
        case DF_NORMAL:
        case DF_MTK:
        case DF_NCHW: {
            U32 ic16 = ic / 16;
            for (U32 n = 0; n < in; ++n) {
                U32 c = 0;
                for (; c < ic16; ++c) {
                    for (U32 h = 0; h < ih; ++h) {
                        for (U32 w = 0; w < iw; ++w) {
                            for (U32 cc = 0; cc < 16; ++cc) {
                                output[n * ic * ih * iw + c * 16 * ih * iw + (h * iw + w) * 16 + cc] =
                                    input[n * ic * ih * iw + (c * 16 + cc) * ih * iw + h * iw + w];
                            }
                        }
                    }
                }
                c *= 16;
                while (c < ic) {
                    U32 cx = ic - c;
                    cx = (cx == 12) ? 8 : cx;
                    for (U32 h = 0; h < ih; ++h) {
                        for (U32 w = 0; w < iw; ++w) {
                            for (U32 cc = 0; cc < cx; ++cc) {
                                output[n * ic * ih * iw + c * ih * iw + (h * iw + w) * cx + cc] =
                                    input[n * ic * ih * iw + (c + cc) * ih * iw + h * iw + w];
                            }
                        }
                    }
                    c += cx;
                }
            }
            break;
        }
        case DF_NCHWC8: {
            U32 ic16 = ic / 16;
            for (U32 n = 0; n < in; ++n) {
                U32 c = 0;
                for (; c < ic16; ++c) {
                    for (U32 h = 0; h < ih; ++h) {
                        for (U32 w = 0; w < iw; ++w) {
                            for (U32 cc = 0; cc < 16; cc += 8) {
                                for (U32 c8 = 0; c8 < 8; ++c8) {
                                    output[n * ic * ih * iw + c * 16 * ih * iw + (h * iw + w) * 16 +
                                        cc + c8] = input[n * ic * ih * iw +
                                        (c * 16 + cc) * ih * iw + (h * iw + w) * 8 + c8];
                                }
                            }
                        }
                    }
                }
                c *= 16;
                while (c < ic) {
                    U32 cx = ic - c;
                    cx = (cx == 12) ? 8 : cx;
                    for (U32 h = 0; h < ih; ++h) {
                        for (U32 w = 0; w < iw; ++w) {
                            for (U32 cc = 0; cc < cx; cc += 8) {
                                for (U32 c8 = 0; c8 < 8; ++c8) {
                                    output[n * ic * ih * iw + c * ih * iw + (h * iw + w) * 8 + cc + c8] =
                                        input[n * ic * ih * iw + (c + cc) * ih * iw +
                                            (h * iw + w) * 8 + c8];
                                }
                            }
                        }
                    }
                    c += cx;
                }
            }
            break;
        }
        default: {
            UNI_ERROR_LOG(
                "not support transform %s format to NCHWC16 format\n", DataFormatName()[idf]);
            ret = NOT_SUPPORTED;
            break;
        }
    }
    return ret;
}

inline EE transformToNCHWC16(
    TensorDesc inputDesc, const void *input, TensorDesc outputDesc, void *output)
{
    if (nullptr == input || nullptr == output) {
        return NULL_POINTER;
    }
    EE ret = NOT_SUPPORTED;
    switch (inputDesc.dt) {
#ifdef _USE_FP32
        case DT_F32: {
            ret = transformToNCHWC16Kernel<F32>(inputDesc, (F32 *)input, outputDesc, (F32 *)output);
            break;
        }
#endif
#ifdef _USE_INT8
        case DT_U8_Q: {
            ret = transformToNCHWC16Kernel<UINT8>(
                inputDesc, (UINT8 *)input, outputDesc, (UINT8 *)output);
            break;
        }
#endif
        default: {
            UNI_ERROR_LOG("not support transform %s type tensor.\n", DataTypeName()[inputDesc.dt]);
            break;
        }
    }
    return ret;
}

inline EE transformFormat(
    TensorDesc inputDesc, const void *input, TensorDesc outputDesc, void *output)
{
    EE ret = NOT_SUPPORTED;
    if (outputDesc.df == DF_NCHW || outputDesc.df == DF_MTK || outputDesc.df == DF_NORMAL) {
        ret = transformToNCHW(inputDesc, input, outputDesc, output);
    } else if (outputDesc.df == DF_NCHWC8) {
        if (inputDesc.df == DF_NORMAL) {
            UNI_MEMCPY(output, input, tensorNumBytes(inputDesc));
            ret = SUCCESS;
        } else if (inputDesc.df == DF_NCHW || inputDesc.df == DF_MTK || inputDesc.df == DF_NORMAL) {
            ret = transformNCHWToNCHWC8(inputDesc, input, outputDesc, output);
        } else if (inputDesc.df == DF_NHWC) {
            ret = transformNHWCToNCHWC8(inputDesc, input, outputDesc, output);
        } else if (inputDesc.df == DF_NCHWC8) {
            ret = transformNCHWC8ToNCHWC8ByGroup(inputDesc, input, 1, outputDesc, output);
        } else if (inputDesc.df == DF_NCHWC16) {
            ret = transformNCHWC16ToNCHWC8(inputDesc, input, outputDesc, output);
        } else {
            UNI_ERROR_LOG("layout transpose can not support transform from %s format "
                          "to NCHWC8 format.\n",
                DataFormatName()[inputDesc.df]);
        }
    } else if (outputDesc.df == DF_NCHWC16) {
        ret = transformToNCHWC16(inputDesc, input, outputDesc, output);
    } else if (outputDesc.df == DF_NHWC) {
        ret = transformToNHWC(inputDesc, input, outputDesc, output);
    } else {
        UNI_ERROR_LOG("layout transpose can not support transform to %s format.\n",
            DataFormatName()[outputDesc.df]);
    }
    return ret;
}

inline EE transposeFilter(
    TensorDesc inputDesc, const void *input, TensorDesc outputDesc, void *output)
{
    if (input == NULL || output == NULL) {
        CHECK_STATUS(NULL_POINTER);
    }
    DataType idt, odt;
    DataFormat idf, odf;
    U32 in, ic, ih, iw, on, oc, oh, ow;
    if (tensorIs4d(inputDesc) && tensorIs4d(outputDesc)) {
        CHECK_STATUS(tensor4dGet(inputDesc, &idt, &idf, &in, &ic, &ih, &iw));
        CHECK_STATUS(tensor4dGet(outputDesc, &odt, &odf, &on, &oc, &oh, &ow));
    } else {
        UNI_ERROR_LOG("currently only support to transpose 4-dim filter.\n");
        return NOT_SUPPORTED;
    }
    CHECK_REQUIREMENT(idf == odf);
    const U8 *src = (const U8 *)input;
    U8 *dst = (U8 *)output;
    EE ret = SUCCESS;
    switch (idf) {
        case DF_NHWCN8: {
            CHECK_REQUIREMENT(in % 8 == 0);
            in /= 8;
            U32 hwMax = ih * iw - 1;
            U32 innerSize = bytesOf(idt) * ic * 8;
            for (U32 o = 0; o < in; o++) {
                for (U32 hw = 0; hw < ih * iw; hw++) {
                    U32 srcIndex = o * ih * iw * innerSize + hw * innerSize;
                    U32 dstIndex = o * ih * iw * innerSize + (hwMax - hw) * innerSize;
                    UNI_MEMCPY(dst + dstIndex, src + srcIndex, innerSize);
                }
            }
            break;
        }
        default: {
            UNI_ERROR_LOG(
                "currently not support to transpose %s format filter.\n", DataFormatName()[idf]);
            ret = NOT_SUPPORTED;
            break;
        }
    }
    return ret;
}
#endif
