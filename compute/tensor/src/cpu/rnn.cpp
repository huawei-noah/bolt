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
#include "cpu/tensor_computing_cpu.h"
#ifdef _USE_GENERAL
#include "cpu/general/tensor_computing_general.h"
#endif
#ifdef _USE_X86
#include "cpu/x86/tensor_computing_x86.h"
#endif
#ifdef _USE_NEON
#include "cpu/arm/tensor_computing_arm.h"
#endif
#include "blas_enhance.h"

template <typename T>
static EE rnn_transform_filter(TensorDesc filterDesc,
    const T *filterArray,
    RNNParamSpec rnnParamSpec,
    TensorDesc *ftmDesc,
    T *ftmArray)
{
    if (nullptr == filterArray || nullptr == ftmDesc || nullptr == ftmArray) {
        CHECK_STATUS(NULL_POINTER);
    }
    DataType fdt;
    DataFormat fdf;
    U32 fn, fk, ftm_n, ftm_k;
    CHECK_STATUS(tensor2dGet(filterDesc, &fdt, &fdf, &fn, &fk));
    U32 alignSize = 32;
    EE ret = SUCCESS;
    switch (fdf) {
        case DF_NKN32: {
            ftm_n = fn;
            ftm_k = fk;
            break;
        }
        case DF_NK: {
            // NK => NKN32
            if (fn % alignSize != 0) {
                return NOT_MATCH;
            }
            ftm_n = fn / alignSize;
            ftm_k = fk;
            for (U32 n = 0; n < ftm_n; n++) {
                for (U32 k = 0; k < ftm_k; k++) {
                    for (U32 n32 = 0; n32 < alignSize; n32++) {
                        ftmArray[n * ftm_k * alignSize + k * alignSize + n32] =
                            filterArray[(n * alignSize + n32) * ftm_k + k];
                    }
                }
            }
            break;
        }
        default:
            ret = NOT_MATCH;
            break;
    }
    *ftmDesc = tensor2df(fdt, DF_NKN32, fn, fk);
    return ret;
}

static EE rnn_transform_filter_cpu_kernel(TensorDesc filterDesc,
    const void *filterArray,
    RNNParamSpec rnnParamSpec,
    TensorDesc *ftmDesc,
    void *ftmArray)
{
    EE ret = SUCCESS;
    switch (filterDesc.dt) {
#ifdef _USE_FP32
        case DT_F32: {
            ret = rnn_transform_filter<F32>(
                filterDesc, (const F32 *)filterArray, rnnParamSpec, ftmDesc, (F32 *)ftmArray);
            break;
        }
#endif
#ifdef _USE_FP16
        case DT_F16: {
            ret = rnn_transform_filter<F16>(
                filterDesc, (const F16 *)filterArray, rnnParamSpec, ftmDesc, (F16 *)ftmArray);
            break;
        }
#endif
        default:
            ret = NOT_SUPPORTED;
            break;
    }
    return ret;
}

EE rnn_transform_filter_cpu(const TensorDesc *filterDesc,
    const void **filterArray,
    RNNParamSpec rnnParamSpec,
    TensorDesc *ftmDesc,
    void **ftmArray)
{
    int num1 = rnnParamSpec.biDirection ? 2 : 1;
    int num2 = rnnParamSpec.numProjection > 0 ? 2 : 1;
    EE ret = SUCCESS;
    for (int i = 0; i < num1 * num2; i++) {
        ret = rnn_transform_filter_cpu_kernel(
            filterDesc[i], filterArray[i], rnnParamSpec, &ftmDesc[i], ftmArray[i]);
    }
    return ret;
}

EE rnn_transform_filter_bytes_cpu(
    const TensorDesc *filterDesc, RNNParamSpec rnnParamSpec, U32 *bytes)
{
    if (nullptr == bytes) {
        CHECK_STATUS(NULL_POINTER);
    }
    int num1 = rnnParamSpec.biDirection ? 2 : 1;
    int num2 = rnnParamSpec.numProjection > 0 ? 2 : 1;
    for (int i = 0; i < num1 * num2; i++) {
        bytes[i] = tensorNumBytes(filterDesc[i]);
    }
    return SUCCESS;
}

EE rnncell_infer_forward_tmp_bytes_cpu(TensorDesc inputDesc,
    TensorDesc filterDesc,
    TensorDesc outputDesc,
    RNNParamSpec rnnParamSpec,
    U32 *bytes,
    Arch arch)
{
    UNUSED(outputDesc);
    if (nullptr == bytes) {
        CHECK_STATUS(NULL_POINTER);
    }
    DataType idt;
    DataFormat idf;
    U32 batch, xDim;
    CHECK_STATUS(tensor2dGet(inputDesc, &idt, &idf, &batch, &xDim));
    U32 hDim = rnnParamSpec.numOutput;
    U32 column = (rnnParamSpec.numProjection > 0) ? rnnParamSpec.numProjection
                                                  : rnnParamSpec.numOutput;
    *bytes = (hDim + xDim + column * 4) * bytesOf(idt);
    return SUCCESS;
}

EE rnn_infer_forward_tmp_bytes_cpu(TensorDesc inputDesc,
    TensorDesc filterDesc,
    TensorDesc outputDesc,
    RNNParamSpec rnnParamSpec,
    U32 *bytes,
    Arch arch)
{
    UNUSED(filterDesc);
    UNUSED(outputDesc);
    if (nullptr == bytes) {
        CHECK_STATUS(NULL_POINTER);
    }
    DataType idt;
    DataFormat idf;
    U32 batch, step, xDim;
    CHECK_STATUS(tensor3dGet(inputDesc, &idt, &idf, &batch, &step, &xDim));
    U32 hDim = rnnParamSpec.numOutput;
    TensorDesc xDesc = tensor2df(idt, DF_NORMAL, batch, xDim);
    CHECK_STATUS(rnncell_infer_forward_tmp_bytes_cpu(
        xDesc, filterDesc, outputDesc, rnnParamSpec, bytes, arch));
    U32 column = (rnnParamSpec.numProjection > 0) ? rnnParamSpec.numProjection
                                                  : rnnParamSpec.numOutput;
    *bytes += batch * (column + hDim) * bytesOf(idt);
    return SUCCESS;
}

EE rnncell_cpu(TensorDesc xDesc,
    const void *currentX,
    const TensorDesc *filterDesc,
    const void **filter,
    const TensorDesc *biasDesc,
    const void **bias,
    void *state,
    RNNParamSpec rnnParamSpec,
    U32 batchStrideX,
    U32 batchStrideH,
    U32 tmpBytes,
    void *tmp,
    TensorDesc hDesc,
    void *currentH,
    Arch arch)
{
    EE ret = NOT_SUPPORTED;
    if (IS_GENERAL(arch)) {
#ifdef _USE_GENERAL
        ret = rnncell_general(xDesc, currentX, filterDesc, filter, biasDesc, bias, state, tmpBytes,
            tmp, rnnParamSpec, batchStrideX, batchStrideH, hDesc, currentH);
#endif
#ifdef _USE_X86
    } else if (IS_X86_AVX2(arch)) {
        ret = rnncell_x86(xDesc, currentX, filterDesc, filter, biasDesc, bias, state, tmpBytes, tmp,
            rnnParamSpec, batchStrideX, batchStrideH, hDesc, currentH, arch);
#endif
#ifdef _USE_NEON
    } else if (IS_ARM(arch)) {
        ret = rnncell_arm(xDesc, currentX, filterDesc, filter, biasDesc, bias, state, tmpBytes, tmp,
            rnnParamSpec, batchStrideX, batchStrideH, hDesc, currentH, arch);
#endif
    }
    return ret;
}

EE rnn_cpu(TensorDesc inputDesc,
    const void *input,
    const TensorDesc *filterDesc,
    const void **filter,
    const TensorDesc *biasDesc,
    const void **bias,
    RNNParamSpec rnnParamSpec,
    U32 tmpBytes,
    void *tmp,
    TensorDesc outputDesc,
    void *output,
    Arch arch)
{
    UNUSED(outputDesc);

    if (nullptr == input || nullptr == filter || nullptr == bias || nullptr == tmp ||
        nullptr == output) {
        CHECK_STATUS(NULL_POINTER);
    }

    DataType idt;
    DataFormat idf;
    U32 batch, step, xDim;
    int num1 = rnnParamSpec.biDirection ? 2 : 1;
    CHECK_STATUS(tensor3dGet(inputDesc, &idt, &idf, &batch, &step, &xDim));
    U32 hDim = rnnParamSpec.numOutput;
    U32 column = (rnnParamSpec.numProjection > 0) ? rnnParamSpec.numProjection
                                                  : rnnParamSpec.numOutput;

    U8 *cellState = (U8 *)tmp;
    U8 *tmpArray = cellState + batch * (column + hDim) * bytesOf(idt);
    U32 batchStrideX = step * xDim;
    U32 batchStrideH = step * hDim * num1;
    TensorDesc xDesc = tensor2df(idt, DF_NORMAL, batch, xDim);
    TensorDesc hDesc = tensor2df(idt, DF_NORMAL, batch, hDim);

    memset(cellState, 0, batch * (column + hDim) * bytesOf(idt));
    for (U32 t = 0; t < step; t++) {
        const U8 *currentX = (const U8 *)input + t * xDim * bytesOf(idt);
        U8 *currentH = (U8 *)output + t * hDim * num1 * bytesOf(idt);
        CHECK_STATUS(rnncell_cpu(xDesc, currentX, filterDesc, filter, biasDesc, bias, cellState,
            rnnParamSpec, batchStrideX, batchStrideH, tmpBytes, tmpArray, hDesc, currentH, arch));
    }

    if (rnnParamSpec.biDirection) {
        memset(cellState, 0, batch * (column + hDim) * bytesOf(idt));
        int num2 = (rnnParamSpec.numProjection > 0) ? 2 : 1;
        for (I32 t = step - 1; t >= 0; t--) {
            const U8 *currentX = (const U8 *)input + t * xDim * bytesOf(idt);
            U8 *currentH = (U8 *)output + t * hDim * num1 * bytesOf(idt) + hDim * bytesOf(idt);
            CHECK_STATUS(rnncell_cpu(xDesc, currentX, &filterDesc[num2], &filter[num2],
                &biasDesc[num2], &bias[num2], cellState, rnnParamSpec, batchStrideX, batchStrideH,
                tmpBytes, tmpArray, hDesc, currentH, arch));
        }
    }
    return SUCCESS;
}
