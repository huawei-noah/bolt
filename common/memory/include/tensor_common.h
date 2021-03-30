// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _H_ARRAY_COMMON
#define _H_ARRAY_COMMON

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
    CHECK_REQUIREMENT(idt == odt);
    switch (idf) {
        case DF_NCHW: {
            if (in == on && ic == oc && ih == oh && iw == ow) {
                if (output != input) {
                    memcpy(output, input, tensorNumBytes(outputDesc));
                }
            } else {
                U32 tileSize = UNI_MIN(iw, ow) * bytesOf(idt);
                for (U32 n = 0; n < on && n < in; n++) {
                    for (U32 c = 0; c < oc && c < ic; c++) {
                        for (U32 h = 0; h < oh && h < ih; h++) {
                            U32 srcIndex = ((n * ic + c) * ih + h) * iw;
                            U32 dstIndex = ((n * oc + c) * oh + h) * ow;
                            memcpy(output + dstIndex, input + srcIndex, tileSize);
                        }
                    }
                }
            }
            break;
        }
        case DF_NCHWC8: {
            U32 ic_a = ic / 8;
            for (U32 n = 0; n < on && n < in; n++) {
                for (U32 c = 0; c < oc && c < ic; c++) {
                    for (U32 h = 0; h < oh && h < ih; h++) {
                        for (U32 w = 0; w < ow && w < iw; w++) {
                            U32 c_a = c / 8;
                            U32 c_b = c % 8;
                            U32 srcIndex = (((n * ic_a + c_a) * ih + h) * iw + w) * 8 + c_b;
                            U32 dstIndex = ((n * oc + c) * oh + h) * ow + w;
                            // support channel cut
                            output[dstIndex] = input[srcIndex];
                        }
                    }
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
            UNI_ERROR_LOG("not support transform %d format to NCHW format\n", idf);
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
                memcpy(output, input, tensorNumBytes(inputDesc));
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
    CHECK_REQUIREMENT(in == on && idf == DF_NCHW && odf == DF_NCHWC8 && idt == odt);
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
                            memcpy(outputPtr + dstIndex, inputPtr + srcIndex, elementSize);
                        } else {
                            memset(outputPtr + dstIndex, 0, elementSize);
                        }
                    }
                }
            }
        }
    }
    return SUCCESS;
}

EE transformNHWCToNCHWC8(
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
                            memcpy(outputPtr + dstIndex, inputPtr + srcIndex, elementSize);
                        } else {
                            memset(outputPtr + dstIndex, 0, elementSize);
                        }
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
                            memcpy(
                                (U8 *)output + dstIndex, (const U8 *)input + srcIndex, elementSize);
                        } else {
                            memset((U8 *)output + dstIndex, 0, elementSize);
                        }
                    }
                }
            }
        }
    }
    return SUCCESS;
}

EE transformFormat(TensorDesc inputDesc, const void *input, TensorDesc outputDesc, void *output)
{
    EE ret = NOT_SUPPORTED;
    if (outputDesc.df == DF_NCHW) {
        ret = transformToNCHW(inputDesc, input, outputDesc, output);
    } else if (outputDesc.df == DF_NCHWC8) {
        if (inputDesc.df == DF_NCHW) {
            ret = transformNCHWToNCHWC8(inputDesc, input, outputDesc, output);
        } else if (inputDesc.df == DF_NHWC) {
            ret = transformNHWCToNCHWC8(inputDesc, input, outputDesc, output);
        } else if (inputDesc.df == DF_NCHWC8) {
            ret = transformNCHWC8ToNCHWC8ByGroup(inputDesc, input, 1, outputDesc, output);
        } else {
            UNI_ERROR_LOG("layout transpose cat not support transform from %d format "
                          "to NCHWC8 "
                          "format.\n",
                inputDesc.df);
        }
    } else {
        UNI_ERROR_LOG("layout transpose cat not support transform to %d format.\n", outputDesc.df);
    }
    return ret;
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
    int inputDimsNum,
    int outputDimsNum)
{
    U32 inputSize = 1, outputSize = 1;
    for (int i = 0; i < inputDimsNum; i++) {
        inputSize *= inputDims[i];
    }
    for (int i = 0; i < outputDimsNum; i++) {
        outputSize *= outputDims[i];
    }
    CHECK_REQUIREMENT(inputSize == outputSize);

    U32 sizeInner = 1;
    I32 sizeInnerIndex = 0;
    for (I32 i = outputDimsNum - 1; i >= 0; i--) {
        if ((I32)transposeDims[i] == i) {
            sizeInner *= inputDims[inputDimsNum - 1 - i];
            sizeInnerIndex++;
        } else {
            break;
        }
    }
    outputSize = outputSize / sizeInner;

    std::vector<U32> inputLocalIndex(inputDimsNum, 0);
    const U8 *inputPtr = (const U8 *)input;
    U8 *outputPtr = (U8 *)output;
    U32 tileSize = sizeInner * bytesOf(dt);
    for (U32 i = 0; i < outputSize; i++) {
        U32 outputIndex = i;
        for (I32 j = sizeInnerIndex; j < outputDimsNum; j++) {
            U32 value = outputIndex % outputDims[j];
            outputIndex /= outputDims[j];
            inputLocalIndex[inputDimsNum - 1 - transposeDims[outputDimsNum - 1 - j]] = value;
        }
        U32 inputIndex = 0;
        for (I32 j = inputDimsNum - 1; j > sizeInnerIndex; j--) {
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
