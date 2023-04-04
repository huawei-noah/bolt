// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include <set>
#include <vector>
#include "tensor_transpose.h"
#include "uni.h"
#include "secure_c_wrapper.h"
#include "affinity_policy.h"

void array_transpose_naive(U32 elementSize,
    U32 *inputDims,
    const void *input,
    U32 *outputDims,
    void *output,
    U32 *transposeDims,
    I32 dimsNum)
{
    if (dimsNum <= 1) {
        return;
    }
    U32 inputSize = 1, outputSize = 1;
    for (I32 i = 0; i < dimsNum; i++) {
        inputSize *= inputDims[i];
        outputSize *= outputDims[i];
    }
    const char *inputPtr = (const char *)input;
    char *outputPtr = (char *)output;
#ifdef _USE_OPENMP
#pragma omp parallel num_threads(OMP_NUM_THREADS)
#endif
    {
        std::vector<U32> inputLocalIndex(dimsNum);
#ifdef _USE_OPENMP
#pragma omp for
#endif
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
            UNI_MEMCPY(
                outputPtr + i * elementSize, inputPtr + inputIndex * elementSize, elementSize);
        }
    }
}

template <I32 branch, typename T>
static void inner_transpose_template(U32 tileSize,
    U32 *inputDims,
    const T *input,
    U32 *outputDims,
    T *output,
    U32 *transposeDims,
    I32 inputDimsNum,
    I32 outputDimsNum,
    U32 outputSize,
    I32 sizeInnerIndex)
{
#ifdef _USE_OPENMP
#pragma omp parallel num_threads(OMP_NUM_THREADS)
#endif
    {
        std::vector<U32> inputLocalIndex(inputDimsNum);
#ifdef _USE_OPENMP
#pragma omp for
#endif
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
            if (branch == 0) {
                *(output + i) = *(input + inputIndex);
            } else {
                UNI_MEMCPY(output + i * tileSize, input + inputIndex * tileSize, tileSize);
            }
        }
    }
}

typedef U32 (*transF4W)(
    U32 n, U32 c, U32 h, U32 w, U32 in, U32 ic, U32 ih, U32 iw);

typedef U32 (*transF5W)(
    U32 t, U32 n, U32 c, U32 h, U32 w, U32 it, U32 in, U32 ic, U32 ih, U32 iw);

inline U32 fNWCH(U32 n, U32 c, U32 h, U32 w, U32 in, U32 ic, U32 ih, U32 iw) {
    return n * iw * ih * ic + w * ih * ic + c * ih + h;
}

inline U32 fNWHC(U32 n, U32 c, U32 h, U32 w, U32 in, U32 ic, U32 ih, U32 iw) {
    return n * iw * ih * ic + w * ih * ic + h * ic + c;
}

inline U32 fNHWC(U32 n, U32 c, U32 h, U32 w, U32 in, U32 ic, U32 ih, U32 iw) {
    return n * ih * iw * ic + h * iw * ic + w * ic + c;
}

inline U32 fNCWH(U32 n, U32 c, U32 h, U32 w, U32 in, U32 ic, U32 ih, U32 iw) {
    return n * ih * iw * ic + c * iw * ih + w * ih + h;
}

inline U32 fNHCW(U32 n, U32 c, U32 h, U32 w, U32 in, U32 ic, U32 ih, U32 iw) {
    return n * ih * ic + h * ic + c;
}

inline U32 fCTHNW(U32 t, U32 n, U32 c, U32 h, U32 w, U32 it, U32 in, U32 ic, U32 ih, U32 iw) {
    return ((c * it + t) * ih + h) * in + n;
}


inline U32 fTHNCW(U32 t, U32 n, U32 c, U32 h, U32 w, U32 it, U32 in, U32 ic, U32 ih, U32 iw) {
    return ((t * ih + h) * in + n) * ic + c;
}

template <typename T>
inline void transByF4W(U32 in, U32 ic, U32 ih, U32 iw, const T *input, T *output, transF4W f) {
    for (U32 n = 0; n < in; ++n) {
        for (U32 w = 0; w < iw; ++w) {
            for (U32 h = 0; h < ih; ++h) {
                for (U32 c = 0; c < ic; ++c) {
                    U32 iidx = n * ic * ih * iw + c * ih * iw + h * iw + w;
                    U32 oidx = f(n, c, h, w, in, ic, ih, iw);
                    output[oidx] = input[iidx];
                }
            }
        }
    }
}

template <typename T>
inline void transByFCopyW4W(U32 in, U32 ic, U32 ih, U32 iw, const T *input, T *output, transF4W f) {
    for (U32 n = 0; n < in; ++n) {
        for (U32 h = 0; h < ih; ++h) {
            for (U32 c = 0; c < ic; ++c) {
                U32 iidx = (n * ic * ih + c * ih + h) * iw;
                U32 oidx = f(n, c, h, 0, in, ic, ih, iw) * iw;
                UNI_MEMCPY(output + oidx, input + iidx, iw * sizeof(T));
            }
        }
    }
}

template <typename T>
inline void transByFCopyW5W(U32 it, U32 in, U32 ic, U32 ih, U32 iw, const T *input, T *output, transF5W f) {
    for (U32 t = 0; t < it; ++t) {
        for (U32 n = 0; n < in; ++n) {
            for (U32 h = 0; h < ih; ++h) {
                for (U32 c = 0; c < ic; ++c) {
                    U32 iidx = (((t * in + n) * ic + c) * ih + h) * iw;
                    U32 oidx = f(t, n, c, h, 0, it, in, ic, ih, iw) * iw;
                    UNI_MEMCPY(output + oidx, input + iidx, iw * sizeof(T));
                }
            }
        }
    }
}

template <typename T>
inline void trans2W(I32 ih, I32 iw, const T *input, T *output) {
    I32 h = 0;
#ifdef _USE_X86
    if (sizeof(T) == 4) {
        __m256i vindex = _mm256_set_epi32(iw * 7, iw * 6, iw * 5, iw * 4, iw * 3, iw * 2, iw, 0);
        for (h = 0; h < ih - 7; h += 8) {
            for (I32 w = 0; w < iw; ++w) {
                _mm256_storeu_ps((F32 *)output + w * ih + h,
                    _mm256_i32gather_ps((const F32 *)input + h * iw + w, vindex, 4));
            }
        }
    }
#endif
    for (; h < ih; ++h) {
        for (I32 w = 0; w < iw; ++w) {
            output[w * ih + h] = input[h * iw + w];
        }
    }
}

template <typename T>
static bool transposeKernel(U32 *inputDims,
    const T *input,
    T *output,
    U32 key)
{
    if (nullptr == input || nullptr == output) {
        CHECK_STATUS(NULL_POINTER);
    }
    if (key == 0x0321) {
        transByF4W<T>(inputDims[3], inputDims[2], inputDims[1], inputDims[0], input, output, fNWHC);
    } else if (key == 0x0312) {
        transByF4W<T>(inputDims[3], inputDims[2], inputDims[1], inputDims[0], input, output, fNWCH);
    } else if (key == 0x0213) {
        transByFCopyW4W<T>(inputDims[3], inputDims[2], inputDims[1], inputDims[0], input, output, fNHCW);
    } else if (key == 0x0231) {
        transByF4W<T>(inputDims[3], inputDims[2], inputDims[1], inputDims[0], input, output, fNHWC);
    } else if (key == 0x0132) {
        U32 ihiw = inputDims[1] * inputDims[0];
        for (U32 n = 0; n < inputDims[2] * inputDims[3]; ++n) {
            trans2W<T>(inputDims[1], inputDims[0], input + n * ihiw, output + n * ihiw);
        }
    } else if (key == 0x20314) {
        transByFCopyW5W<T>(inputDims[4], inputDims[3], inputDims[2], inputDims[1], inputDims[0], input, output, fCTHNW);
    } else {
        return false;
    }

    return true;
}

static bool accTransposeCase(U32 elementSize,
    U32 *inputDims,
    const void *input,
    void *output,
    U32 *transposeDims,
    I32 inputDimsNum)
{
    bool transposed = false;
    U32* paddingDims = transposeDims;
    U32* paddingInputDims = inputDims;
    U32 paddingDimsNum = inputDimsNum;
    U32 buffer[8];
    if (inputDimsNum < 4) {
        paddingDims = buffer;
        paddingInputDims = buffer + 4;
        for (I32 i = 0; i < 4; ++i) {
            if (i < (4 - inputDimsNum)) {
                paddingDims[i] = i;
            } else {
                paddingDims[i] = transposeDims[i - (4 - inputDimsNum)] + 4 - inputDimsNum;
            }
            if (i < inputDimsNum) {
                paddingInputDims[i] = inputDims[i];
            } else {
                paddingInputDims[i] = 1;
            }
        }
        paddingDimsNum = 4;
    }
    U32 key = 0;
    for (U32 i = 0; i < paddingDimsNum; ++i) {
        key = key | (paddingDims[i] << ((paddingDimsNum - i - 1) * 4));
    }
    std::set<U32> accCase = {0x0321, 0x0312, 0x0213, 0x0231, 0x0132, 0x20314};
    if (accCase.count(key)) {
        switch (elementSize) {
            case 4: {
                transposed = transposeKernel<U32>(paddingInputDims, (const U32 *)input, (U32 *)output, key);
                break;
            }
            case 2: {
                transposed = transposeKernel<U16>(paddingInputDims, (const U16*)input, (U16 *)output, key);
                break;
            }
            case 1: {
                transposed = transposeKernel<U8>(paddingInputDims, (const U8 *)input, (U8 *)output, key);
                break;
            }
            default:
                transposed = false;
                break;
        }
    }
    return transposed;
}

void array_transpose(U32 elementSize,
    U32 *inputDims,
    const void *input,
    U32 *outputDims,
    void *output,
    U32 *transposeDims,
    I32 inputDimsNum,
    I32 outputDimsNum)
{
    if (inputDimsNum == outputDimsNum) {
        bool transposed = accTransposeCase(
            elementSize, inputDims, input, output, transposeDims, inputDimsNum);
        if (transposed) {
            return;
        }
    }

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
    I32 tileSize = elementSize * sizeInner;
    I32 in = inputDims[inputDimsNum - 1], ihiw = 0, ic = 0;
    if (outputDimsNum - sizeInnerIndex == 3 && transposeDims[0] == 0 && transposeDims[1] == 2 &&
        transposeDims[2] == 1) {
        ic = inputDims[inputDimsNum - 2];
        ihiw = inputDims[inputDimsNum - 3];
    }
    if (outputDimsNum - sizeInnerIndex == 4 && transposeDims[0] == 0 && transposeDims[1] == 2 &&
        transposeDims[2] == 3 && transposeDims[3] == 1) {
        ic = inputDims[inputDimsNum - 2];
        ihiw = inputDims[inputDimsNum - 3] * inputDims[inputDimsNum - 4];
    }
    if (ic > 0 && ihiw > 0 && input != output) {
#ifdef _USE_OPENMP
#pragma omp parallel for num_threads(OMP_NUM_THREADS)
#endif
        for (I32 o = 0; o < in * ihiw; o++) {
            I32 n = o / ihiw;
            I32 hw = o % ihiw;
            U8 *dst = (U8 *)output + o * ic * tileSize;
            for (I32 c = 0; c < ic; c++, dst += tileSize) {
                const U8 *src = (const U8 *)input + ((n * ic + c) * ihiw + hw) * tileSize;
                UNI_MEMCPY(dst, src, tileSize);
            }
        }
        return;
    }

    U32 inputSize = 1, outputSize = 1;
    for (I32 i = 0; i < inputDimsNum; i++) {
        inputSize *= inputDims[i];
    }
    for (I32 i = 0; i < outputDimsNum; i++) {
        outputSize *= outputDims[i];
    }
    CHECK_REQUIREMENT(inputSize == outputSize);
    outputSize = outputSize / sizeInner;

    const char *inputPtr = (const char *)input;
    char *outputPtr = (char *)output;
    if (sizeInner == 1 && elementSize == 4) {
        inner_transpose_template<0, U32>(elementSize, inputDims, (const U32 *)input, outputDims,
            (U32 *)output, transposeDims, inputDimsNum, outputDimsNum, outputSize, sizeInnerIndex);
    } else if (sizeInner == 1 && elementSize == 2) {
        inner_transpose_template<0, U16>(elementSize, inputDims, (const U16 *)input, outputDims,
            (U16 *)output, transposeDims, inputDimsNum, outputDimsNum, outputSize, sizeInnerIndex);
    } else {
        inner_transpose_template<1, U8>(tileSize, inputDims, (const U8 *)input, outputDims,
            (U8 *)output, transposeDims, inputDimsNum, outputDimsNum, outputSize, sizeInnerIndex);
    }
}

template <typename T>
static EE transformToNCHWKernel(
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
    U32 minH = UNI_MIN(oh, ih);
    U32 minC = UNI_MIN(oc, ic);
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
#ifdef _USE_OPENMP
#pragma omp parallel num_threads(OMP_NUM_THREADS)
#endif
                {
                    U32 minH = UNI_MIN(oh, ih);
                    for (U32 n = 0; n < on && n < in; n++) {
#ifdef _USE_OPENMP
#pragma omp for nowait
#endif
                        for (U32 id = 0; id < minC * minH; id++) {
                            U32 c = id / minH;
                            U32 h = id % minH;
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
            U32 minW = UNI_MIN(ow, iw);
            U32 minN = UNI_MIN(on, in);
#ifdef _USE_OPENMP
#pragma omp parallel num_threads(OMP_NUM_THREADS)
#endif
            {
                for (U32 n = 0; n < minN; ++n) {
                    for (U32 c = 0; c < minC; c += 8) {
#ifdef _USE_OPENMP
#pragma omp for nowait
#endif
                        for (U32 h = 0; h < minH; ++h) {
                            for (U32 w = 0; w < minW; w++) {
                                for (U32 c8 = 0; c8 < 8 && c + c8 < minC; ++c8) {
                                    U32 srcIndex = (n * ic + c) * ih * iw + (h * iw + w) * cx + c8;
                                    U32 dstIndex = ((n * oc + c + c8) * oh + h) * ow + w;
                                    output[dstIndex] = input[srcIndex];
                                }
                            }
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

EE transformToNCHW(
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
#ifndef _USE_LITE
        case DT_I8: {
            ret = transformToNCHWKernel<INT8>(inputDesc, (INT8 *)input, outputDesc, (INT8 *)output);
            break;
        }
        case DT_U8:
        case DT_U8_Q: {
            ret = transformToNCHWKernel<UINT8>(
                inputDesc, (UINT8 *)input, outputDesc, (UINT8 *)output);
            break;
        }
        case DT_I32: {
            ret = transformToNCHWKernel<I32>(inputDesc, (I32 *)input, outputDesc, (I32 *)output);
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
static EE transformToNHWCKernel(
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

EE transformToNHWC(
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
        case DT_U8_Q:
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

EE transformNCHWC16ToNCHWC8(
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
            U32 cx = (c + 2 <= oc)? 16: 8;
            for (U32 h = 0; h < oh && h < ih; h++) {
                for (U32 w = 0; w < ow && w < iw; w++) {
                    for (U32 c8 = 0; (c8 < 2) && (c8 + c < oc); ++c8) {
                        U32 srcIndex =
                            n * ic * ih * iw + c * ih * iw * 8 + (h * iw + w) * cx + c8 * 8;
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

template <typename T>
static void transformNCHWToNCHWC8Kernel(
    U32 in, U32 ic, U32 ih, U32 iw, U32 on, U32 oc, U32 oh, U32 ow, U32 val, const T *input, T *output)
{
    oc /= 8;
    for (U32 n = 0; n < on && n < in; n++) {
        for (U32 c = 0; c < oc; c++) {
            for (U32 h = 0; h < oh && h < ih; h++) {
                for (U32 w = 0; w < ow && w < iw; w++) {
                    for (U32 c8 = 0, c_i = c * 8; c8 < 8; c8++, c_i++) {
                        U32 dstIndex = (((n * oc + c) * oh + h) * ow + w) * 8 + c8;
                        // support channel padding
                        if (c_i < ic) {
                            U32 srcIndex = ((n * ic + c_i) * ih + h) * iw + w;
                            output[dstIndex] = input[srcIndex];
                        } else {
                            output[dstIndex] = val;
                        }
                    }
                }
            }
        }
    }
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
    switch (idt) {
        case DT_F32: {
            transformNCHWToNCHWC8Kernel<F32>(
                in, ic, ih, iw, on, oc, oh, ow, 0, (const F32 *)input, (F32 *)output);
            break;
        }
        case DT_I32: {
            transformNCHWToNCHWC8Kernel<I32>(
                in, ic, ih, iw, on, oc, oh, ow, 0, (const I32 *)input, (I32 *)output);
            break;
        }
#ifdef _USE_FP16
        case DT_F16: {
            transformNCHWToNCHWC8Kernel<F16>(
                in, ic, ih, iw, on, oc, oh, ow, 0, (const F16 *)input, (F16 *)output);
            break;
        }
#endif
        case DT_I8: {
            transformNCHWToNCHWC8Kernel<INT8>(
                in, ic, ih, iw, on, oc, oh, ow, 0, (const INT8 *)input, (INT8 *)output);
            break;
        }
        case DT_U8_Q: {
            transformNCHWToNCHWC8Kernel<UINT8>(
                in, ic, ih, iw, on, oc, oh, ow, 128, (const UINT8 *)input, (UINT8 *)output);
            break;
        }
        default:
            return NOT_SUPPORTED;
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

EE transformNCHWC8ToNCHWC8ByGroup(
    TensorDesc inputDesc, const void *input, int group, TensorDesc outputDesc, void *output)
{
    U32 inputSize = tensorNumBytes(inputDesc);
    U32 outputSize = tensorNumBytes(outputDesc);
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
#ifdef _USE_OPENMP
#pragma omp parallel num_threads(OMP_NUM_THREADS)
#endif
    {
        for (U32 n = 0; n < in; n++) {
#ifdef _USE_OPENMP
#pragma omp for nowait
#endif
            for (I32 g = 0; g < group; g++) {
                for (U32 c = 0; c < ocg; c++) {
                    U32 id = g * icg + c;
                    U32 od = g * ocg + c;
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
                                    ((((n * ict + id_a) * ih + h) * iw + w) * channelAlignSize +
                                        id_b) *
                                    elementSize;
                                UNI_MEMCPY((U8 *)output + dstIndex, (const U8 *)input + srcIndex,
                                    elementSize);
                            } else {
                                UNI_MEMSET((U8 *)output + dstIndex, 0, elementSize);
                            }
                        }
                    }
                }
            }
        }
    }
    return SUCCESS;
}

template <typename T>
static EE transformToNCHWC16Kernel(
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

EE transformToNCHWC16(
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

EE transformFormat(
    TensorDesc inputDesc, const void *input, TensorDesc outputDesc, void *output)
{
    EE ret = NOT_SUPPORTED;
    if (outputDesc.df == DF_NCHW || outputDesc.df == DF_MTK || outputDesc.df == DF_NORMAL) {
        ret = transformToNCHW(inputDesc, input, outputDesc, output);
    } else if (outputDesc.df == DF_NCHWC8) {
        if (inputDesc.nDims == 2) {
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
