// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include <string.h>
#include <bitset>
#include "tensor_desc.h"

void UNI_memcpy(void *dst, const void *src, int size)
{
    if (src == dst || size <= 0 || dst == nullptr || src == nullptr) {
        return;
    }
    memcpy(dst, src, size);
}

void UNI_init(U32 num, DataType dt, F32 val, void *dst)
{
    switch (dt) {
#ifdef _USE_FP16
        case DT_F16: {
            F16 v = val;
            F16 *arr = (F16 *)dst;
            for (U32 i = 0; i < num; i++) {
                arr[i] = v;
            }
            break;
        }
#endif
        default: {
            CHECK_STATUS(NOT_SUPPORTED);
        }
    }
}

void transformFromFloat(DataType dataType, float *src, void *dst, int num, float scale)
{
    switch (dataType) {
        case DT_F32: {
            UNI_memcpy(dst, src, sizeof(float) * num);
            break;
        }
        case DT_U32: {
            U32 *ptr = (U32 *)dst;
            for (int i = 0; i < num; i++) {
                ptr[i] = src[i];
            }
            break;
        }
        case DT_I32: {
            I32 *ptr = (I32 *)dst;
            for (int i = 0; i < num; i++) {
                ptr[i] = src[i];
            }
            break;
        }
#ifdef __aarch64__
        case DT_F16: {
            F16 *ptr = (F16 *)dst;
            for (int i = 0; i < num; i++) {
                ptr[i] = src[i];
            }
            break;
        }
        case DT_F16_8Q: {
            F16 *ptr = (F16 *)dst;
            for (int i = 0; i < num; i++) {
                ptr[i] = src[i];
            }
            break;
        }
#endif
        case DT_I8: {
            INT8 *ptr = (INT8 *)dst;
            for (int i = 0; i < num; i++) {
                ptr[i] = src[i] * scale;
            }
            break;
        }
        case DT_U8: {
            U8 *ptr = (U8 *)dst;
            for (int i = 0; i < num; i++) {
                ptr[i] = src[i];
            }
            break;
        }
        default: {
            UNI_ERROR_LOG("not unsupport transform float to %d type data\n", dataType);
            break;
        }
    }
}

void transformToFloat(DataType dataType, void *src, float *dst, int num, float scale)
{
    switch (dataType) {
        case DT_F32: {
            UNI_memcpy(dst, src, sizeof(float) * num);
            break;
        }
        case DT_U32: {
            U32 *ptr = (U32 *)src;
            for (int i = 0; i < num; i++) {
                dst[i] = ptr[i];
            }
            break;
        }
        case DT_I32: {
            I32 *ptr = (I32 *)src;
            for (int i = 0; i < num; i++) {
                dst[i] = ptr[i];
            }
            break;
        }
#ifdef __aarch64__
        case DT_F16: {
            F16 *ptr = (F16 *)src;
            for (int i = 0; i < num; i++) {
                dst[i] = ptr[i];
            }
            break;
        }
        case DT_F16_8Q: {
            F16 *ptr = (F16 *)src;
            for (int i = 0; i < num; i++) {
                dst[i] = ptr[i];
            }
            break;
        }
#endif
        case DT_I8: {
            INT8 *ptr = (INT8 *)src;
            for (int i = 0; i < num; i++) {
                dst[i] = ptr[i] / scale;
            }
            break;
        }
        case DT_U8: {
            U8 *ptr = (U8 *)src;
            for (int i = 0; i < num; i++) {
                dst[i] = ptr[i];
            }
            break;
        }
        case DT_BIN01: {
            BIN8 *ptr = (BIN8 *)src;
            for (int i = 0; i < num; i++) {
                std::bitset<8> Val(((BIN8 *)ptr)[i / 8]);
                if (Val.test(7 - (i % 8))) {
                    dst[i] = 1.0;
                } else {
                    dst[i] = 0;
                }
            }
            break;
        }
        case DT_BIN11: {
            BIN8 *ptr = (BIN8 *)src;
            for (int i = 0; i < num; i++) {
                std::bitset<8> Val(((BIN8 *)ptr)[i / 8]);
                if (Val.test(7 - (i % 8))) {
                    dst[i] = 1.0;
                } else {
                    dst[i] = -1.0;
                }
            }
            break;
        }
        default: {
            UNI_ERROR_LOG("not unsupport transform %d type data to float\n", dataType);
            break;
        }
    }
}

template <typename T>
static void transformToNCHWKernel(
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
        UNI_ERROR_LOG("not support transform %d-dim tensor to NCHW format\n", (int)inputDesc.nDims);
        return;
    }
    CHECK_STATUS(tensor4dGet(outputDesc, &odt, &odf, &on, &oc, &oh, &ow));
    U32 ihiw = ih * iw;
    U32 size = tensorNumElements(outputDesc);
    switch (idf) {
        case DF_NCHW: {
            CHECK_REQUIREMENT(tensorNumElements(inputDesc) == size);
            if (output != input) {
                memcpy(output, input, size);
            }
            break;
        }
        case DF_NCHWC8: {
            CHECK_REQUIREMENT(ic % 8 == 0);
            ic /= 8;
            for (U32 n = 0, srcIndex = 0; n < in; n++) {
                for (U32 c = 0; c < ic; c++) {
                    for (U32 hw = 0; hw < ihiw; hw++) {
                        for (U32 c8 = 0; c8 < 8; c8++, srcIndex++) {
                            U32 c_o = c * 8 + c8;
                            // support channel cut
                            if (c_o < oc) {
                                U32 dstIndex = (n * oc + c_o) * ihiw + hw;
                                output[dstIndex] = input[srcIndex];
                            }
                        }
                    }
                }
            }
            break;
        }
        case DF_NHWCN8: {
            CHECK_REQUIREMENT(in % 8 == 0);
            in /= 8;
            for (U32 o = 0, srcIndex = 0; o < in; o++) {
                for (U32 hw = 0; hw < ihiw; hw++) {
                    for (U32 c = 0; c < ic; c++) {
                        for (U32 o8 = 0; o8 < 8; o8++, srcIndex++) {
                            U32 dstIndex = ((o * 8 + o8) * ic + c) * ihiw + hw;
                            output[dstIndex] = input[srcIndex];
                        }
                    }
                }
            }
            break;
        }
        case DF_NHWC: {
            CHECK_REQUIREMENT(tensorNumElements(inputDesc) == size);
            for (U32 o = 0, srcIndex = 0; o < in; o++) {
                for (U32 hw = 0; hw < ihiw; hw++) {
                    for (U32 cc = 0; cc < ic; cc++, srcIndex++) {
                        U32 dstIndex = (o * ic + cc) * ihiw + hw;
                        output[dstIndex] = input[srcIndex];
                    }
                }
            }
            break;
        }
        default: {
            UNI_ERROR_LOG("not support transform %d format tensor to NCHW format\n", idf);
        }
    }
}

EE transformToNCHW(TensorDesc inputDesc, const void *input, TensorDesc outputDesc, void *output)
{
    if (nullptr == input || nullptr == output) {
        return NULL_POINTER;
    }
    switch (inputDesc.dt) {
#ifdef _USE_FP32
        case DT_F32: {
            transformToNCHWKernel<F32>(inputDesc, (F32 *)input, outputDesc, (F32 *)output);
            break;
        }
#endif
#ifdef _USE_FP16
        case DT_F16: {
            transformToNCHWKernel<F16>(inputDesc, (F16 *)input, outputDesc, (F16 *)output);
            break;
        }
#endif
#ifdef _USE_INT8
        case DT_I8: {
            transformToNCHWKernel<INT8>(inputDesc, (INT8 *)input, outputDesc, (INT8 *)output);
            break;
        }
#endif
        default: {
            return NOT_SUPPORTED;
        }
    }
    return SUCCESS;
}

template <typename T>
static void transformToNHWCKernel(
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
        return;
    }
    CHECK_STATUS(tensor4dGet(outputDesc, &odt, &odf, &on, &oc, &oh, &ow));
    U32 size = tensorNumElements(outputDesc);
    U32 ihiw = ih * iw;
    switch (idf) {
        case DF_NHWC: {
            CHECK_REQUIREMENT(tensorNumElements(inputDesc) == size);
            if (input != output) {
                memcpy(output, input, size);
            }
            break;
        }
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
        case DF_NCHWC8: {
            CHECK_REQUIREMENT(ic % 8 == 0);
            ic /= 8;
            for (U32 n = 0, srcIndex = 0; n < in; n++) {
                for (U32 c = 0; c < ic; c++) {
                    for (U32 hw = 0; hw < ihiw; hw++) {
                        for (U32 c8 = 0; c8 < 8; c8++, srcIndex++) {
                            U32 dstIndex = ((n * ihiw + hw) * ic + c) * 8 + c8;
                            output[dstIndex] = input[srcIndex];
                        }
                    }
                }
            }
            break;
        }
        default: {
            UNI_ERROR_LOG("not support transform %d format tensor to NHWC format\n", idf);
        }
    }
}

EE transformToNHWC(TensorDesc inputDesc, const void *input, TensorDesc outputDesc, void *output)
{
    if (nullptr == input || nullptr == output) {
        return NULL_POINTER;
    }
    switch (inputDesc.dt) {
#ifdef _USE_FP32
        case DT_F32: {
            transformToNHWCKernel<F32>(inputDesc, (F32 *)input, outputDesc, (F32 *)output);
            break;
        }
#endif
#ifdef _USE_FP16
        case DT_F16: {
            transformToNHWCKernel<F16>(inputDesc, (F16 *)input, outputDesc, (F16 *)output);
            break;
        }
#endif
#ifdef _USE_INT8
        case DT_I8: {
            transformToNHWCKernel<INT8>(inputDesc, (INT8 *)input, outputDesc, (INT8 *)output);
            break;
        }
#endif
        default: {
            return NOT_SUPPORTED;
        }
    }
    return SUCCESS;
}

EE transformNCHWToNCHWC8(
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
    CHECK_REQUIREMENT(in == on && idf == DF_NCHW && odf == DF_NCHWC8 && idt == odt && ic <= oc &&
        ih == oh && iw == ow);
    int elementSize = bytesOf(idt);
    oc /= 8;
    U32 ohow = oh * ow;
    const U8 *inputPtr = (const U8 *)input;
    U8 *outputPtr = (U8 *)output;
    for (U32 n = 0, dstIndex = 0; n < on; n++) {
        for (U32 c = 0; c < oc; c++) {
            for (U32 hw = 0; hw < ohow; hw++) {
                for (U32 c8 = 0; c8 < 8; c8++, dstIndex += elementSize) {
                    U32 c_i = c * 8 + c8;
                    // support channel padding
                    if (c_i < ic) {
                        U32 srcIndex = ((n * ic + c_i) * ohow + hw) * elementSize;
                        memcpy(outputPtr + dstIndex, inputPtr + srcIndex, elementSize);
                    } else {
                        memset(outputPtr + dstIndex, 0, elementSize);
                    }
                }
            }
        }
    }
    return SUCCESS;
}

EE transformNCHWC8ToNCHWC8ByGroup(
    TensorDesc inputDesc, const void *input, int group, TensorDesc outputDesc, void *output)
{
    U32 inputSize = tensorNumElements(inputDesc);
    U32 outputSize = tensorNumElements(outputDesc);
    if (group <= 1 || inputSize == outputSize) {
        if (input != output) {
            memcpy(output, input, outputSize);
        }
        return SUCCESS;
    }

    U32 channelAlignSize = 8;
    DataType dtBefore, dtAfter;
    DataFormat dfBefore, dfAfter;
    U32 batch, channelBefore, hBefore, wBefore;
    U32 batchAfter, channelAfter, hAfter, wAfter;
    CHECK_STATUS(
        tensor4dGet(inputDesc, &dtBefore, &dfBefore, &batch, &channelBefore, &hBefore, &wBefore));
    CHECK_STATUS(
        tensor4dGet(outputDesc, &dtAfter, &dfAfter, &batchAfter, &channelAfter, &hAfter, &wAfter));
    CHECK_REQUIREMENT(dtBefore == dtAfter);
    CHECK_REQUIREMENT(dfBefore == DF_NCHWC8 && dfAfter == DF_NCHWC8);
    CHECK_REQUIREMENT(batch == batchAfter);
    CHECK_REQUIREMENT(hBefore == hAfter);
    CHECK_REQUIREMENT(wBefore == wAfter);
    U32 channelGroupSizeBefore = channelBefore / group;
    U32 channelGroupSizeAfter = channelAfter / group;
    U32 channelTileSizeBefore = channelBefore / channelAlignSize;
    U32 channelTileSizeAfter = channelAfter / channelAlignSize;
    U32 elementSize = bytesOf(dtBefore);
    U32 hw = hBefore * wBefore;
    for (U32 n = 0; n < batch; n++) {
        for (I32 g = 0, channelIdAfter = 0; g < group; g++) {
            for (U32 c = 0; c < channelGroupSizeAfter; c++, channelIdAfter++) {
                U32 channelIdBefore = g * channelGroupSizeBefore + c;
                U32 channelTileBefore = channelIdBefore / channelAlignSize;
                U32 channelTileAfter = channelIdAfter / channelAlignSize;
                U32 channelLocalBefore = channelIdBefore % channelAlignSize;
                U32 channelLocalAfter = channelIdAfter % channelAlignSize;
                U32 indexBefore =
                    (((n * channelTileSizeBefore + channelTileBefore) * hw) * channelAlignSize +
                        channelLocalBefore) *
                    elementSize;
                U32 indexAfter =
                    (((n * channelTileSizeAfter + channelTileAfter) * hw) * channelAlignSize +
                        channelLocalAfter) *
                    elementSize;
                U32 stepSize = channelAlignSize * elementSize;
                U32 indexBeforeUpper = indexBefore + stepSize * hw;
                while (indexBefore < indexBeforeUpper) {
                    memcpy((U8 *)output + indexAfter, (const U8 *)input + indexBefore, elementSize);
                    indexBefore += stepSize;
                    indexAfter += stepSize;
                }
            }
        }
    }
    return SUCCESS;
}

EE transposeFilter(TensorDesc inputDesc, const void *input, TensorDesc outputDesc, void *output)
{
    if (input == NULL || output == NULL) {
        CHECK_STATUS(NULL_POINTER);
    }
    DataType idt, odt;
    DataFormat idf, odf;
    U32 in, ic, ih, iw, on, oc, oh, ow;
    CHECK_STATUS(tensor4dGet(inputDesc, &idt, &idf, &in, &ic, &ih, &iw));
    CHECK_STATUS(tensor4dGet(outputDesc, &odt, &odf, &on, &oc, &oh, &ow));
    CHECK_REQUIREMENT(idf == odf);
    const U8 *inputPtr = (const U8 *)input;
    U8 *outputPtr = (U8 *)output;

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
                    memcpy(outputPtr + dstIndex, inputPtr + srcIndex, innerSize);
                }
            }
            break;
        }
        default: {
            CHECK_STATUS(NOT_SUPPORTED);
        }
    }
    return SUCCESS;
}

EE array_transpose(DataType dt,
    U32 *inputDims,
    const void *input,
    U32 *outputDims,
    void *output,
    U32 *transposeDims,
    int dimsNum)
{
    U32 sizeInner = 1;
    I32 sizeInnerIndex = 0;
    for (I32 i = dimsNum - 1; i >= 0; i--) {
        if ((I32)transposeDims[i] == i) {
            sizeInner *= inputDims[dimsNum - 1 - i];
            sizeInnerIndex++;
        } else {
            break;
        }
    }
    U32 inputSize = 1, outputSize = 1;
    for (int i = 0; i < dimsNum; i++) {
        inputSize *= inputDims[i];
        outputSize *= outputDims[i];
    }
    CHECK_REQUIREMENT(inputSize == outputSize);
    outputSize = outputSize / sizeInner;

    std::vector<U32> inputLocalIndex(dimsNum);
    const U8 *inputPtr = (const U8 *)input;
    U8 *outputPtr = (U8 *)output;
    U32 tileSize = sizeInner * bytesOf(dt);
    for (U32 i = 0; i < outputSize; i++) {
        U32 outputIndex = i;
        for (I32 j = sizeInnerIndex; j < dimsNum; j++) {
            U32 value = outputIndex % outputDims[j];
            outputIndex /= outputDims[j];
            inputLocalIndex[dimsNum - 1 - transposeDims[dimsNum - 1 - j]] = value;
        }
        U32 inputIndex = 0;
        for (I32 j = dimsNum - 1; j > sizeInnerIndex; j--) {
            inputIndex = (inputIndex + inputLocalIndex[j]) * inputDims[j - 1];
        }
        inputIndex += inputLocalIndex[sizeInnerIndex];
        memcpy(outputPtr + i * tileSize, inputPtr + inputIndex * tileSize, tileSize);
    }

    return SUCCESS;
}

EE array_transpose_naive(DataType dt,
    U32 *inputDims,
    const void *input,
    U32 *outputDims,
    void *output,
    U32 *transposeDims,
    int dimsNum)
{
    if (dimsNum <= 1) {
        return SUCCESS;
    }
    U32 inputSize = 1, outputSize = 1;
    for (int i = 0; i < dimsNum; i++) {
        inputSize *= inputDims[i];
        outputSize *= outputDims[i];
    }
    std::vector<U32> inputLocalIndex(dimsNum);
    const U8 *inputPtr = (const U8 *)input;
    U8 *outputPtr = (U8 *)output;
    U32 tileSize = bytesOf(dt);
    for (U32 i = 0; i < outputSize; i++) {
        U32 outputIndex = i;
        for (I32 j = 0; j < dimsNum; j++) {
            U32 value = outputIndex % outputDims[j];
            outputIndex /= outputDims[j];
            inputLocalIndex[dimsNum - 1 - transposeDims[dimsNum - 1 - j]] = value;
        }
        U32 inputIndex = 0;
        for (I32 j = dimsNum - 1; j > 0; j--) {
            inputIndex = (inputIndex + inputLocalIndex[j]) * inputDims[j - 1];
        }
        inputIndex += inputLocalIndex[0];
        memcpy(outputPtr + i * tileSize, inputPtr + inputIndex * tileSize, tileSize);
    }

    return SUCCESS;
}
