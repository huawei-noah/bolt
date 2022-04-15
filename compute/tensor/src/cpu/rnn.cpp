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
void transformNK2NKN32(const T *src, U32 stride, T *dst, U32 N, U32 K)
{
    for (U32 n = 0; n < N / 32; ++n) {
        for (U32 k = 0; k < K; ++k) {
            for (U32 i = 0; i < 32; ++i) {
                dst[(n * K + k) * 32 + i] = src[(n * 32 + i) * stride + k];
            }
        }
    }
}

template <typename T>
static EE rnn_transform_filter(TensorDesc filterDesc,
    const T *filterArray,
    RNNParamSpec rnnParamSpec,
    TensorDesc *ftmDesc,
    T *ftmArray,
    float *scale,
    DataFormat ftmDataFormat,
    Arch arch)
{
    if (nullptr == filterArray || nullptr == ftmDesc || nullptr == ftmArray) {
        CHECK_STATUS(NULL_POINTER);
    }
    DataType fdt;
    DataFormat fdf;
    U32 fn, fk;
    CHECK_STATUS(tensor2dGet(filterDesc, &fdt, &fdf, &fn, &fk));
    EE ret = SUCCESS;
    if (fdf == ftmDataFormat) {
        *ftmDesc = tensor2df(fdt, ftmDataFormat, fn, fk);
        return ret;
    }
    if (fdf != DF_NK) {
        return NOT_MATCH;
    }
    if (fn % 32 != 0) {
        UNI_ERROR_LOG(
            "RNN/LSTM/GRU currently only support hidden states(%u) mod 32 = 0 case.\n", fn);

        return NOT_MATCH;
    }
    U32 hDim;
    if (rnnParamSpec.num_projection > 0) {
        hDim = rnnParamSpec.num_projection;
    } else {
        hDim = rnnParamSpec.num_outputs;
    }
    U32 xDim = fk - rnnParamSpec.num_outputs;
    *ftmDesc = tensor2df(fdt, ftmDataFormat, fn, fk);
    switch (ftmDataFormat) {
        case DF_NKN32: {
            // NK => NKN32
            if (rnnParamSpec.mode == RNN_GRU_LBR) {
                transformNK2NKN32<T>(filterArray, fk, ftmArray, fn / 3 * 2, fk);
                U32 offset = fn / 3 * 2 * fk;
                transformNK2NKN32<T>(filterArray + offset, fk, ftmArray + offset, fn / 3, xDim);
                transformNK2NKN32<T>(filterArray + offset + xDim, fk,
                    ftmArray + offset + fn / 3 * xDim, fn / 3, hDim);
            } else {
                transformNK2NKN32<T>(filterArray, fk, ftmArray, fn, fk);
            }
            break;
        }
        case DF_NKNx_NKN32: {
            // NK => NKNx_NKN32
            std::vector<T> filterTmp(fn * UNI_MAX(xDim, hDim));
            for (U32 n = 0; n < fn; ++n) {
                UNI_MEMCPY(filterTmp.data() + n * xDim, filterArray + n * fk, xDim * sizeof(T));
            }
            TensorDesc mmmDesc = tensor2df(fdt, DF_TRANSPOSE, fn, xDim);
            matrix_matrix_multiply_transform_rhs(
                mmmDesc, filterTmp.data(), &mmmDesc, ftmArray, arch);

            if (0) {
#if defined(_USE_INT8) && defined(_USE_ULTRA_OPTIMIZATION)
            } else if (arch == X86_AVX512 && rnnParamSpec.mode == RNN_LSTM &&
                rnnParamSpec.num_projection == 0) {
                for (U32 n = 0; n < fn; ++n) {
                    UNI_MEMCPY(
                        filterTmp.data() + n * hDim, filterArray + n * fk + xDim, hDim * sizeof(T));
                }
                TensorDesc mvmDesc = tensor2df(fdt, DF_NORMAL, fn, hDim);
                TensorDesc mvmQuantDesc = tensor2df(DT_I8, DF_NORMAL, fn, hDim);
                TensorDesc mvmTransDesc;
                std::vector<INT8> filterQuant(fn * hDim);
                CHECK_STATUS(quantize_cpu(
                    mvmDesc, filterTmp.data(), &mvmQuantDesc, filterQuant.data(), scale, arch));
                CHECK_STATUS(matrix_vector_multiply_transform_weight(
                    mvmQuantDesc, filterQuant.data(), &mvmTransDesc, ftmArray + fn * xDim, arch));
#endif
            } else {
                transformNK2NKN32(
                    filterArray + xDim, fk, ftmArray + fn * xDim, fn, rnnParamSpec.num_outputs);
            }
            break;
        }
        default:
            ret = NOT_MATCH;
            break;
    }
    return ret;
}

static EE rnn_transform_filter_cpu_kernel(TensorDesc filterDesc,
    const void *filterArray,
    RNNParamSpec rnnParamSpec,
    TensorDesc *ftmDesc,
    void *ftmArray,
    float *scale,
    DataFormat ftmDataFormat,

    Arch arch)
{
    EE ret = SUCCESS;
    switch (filterDesc.dt) {
#ifdef _USE_FP32
        case DT_F32: {
            ret = rnn_transform_filter<F32>(filterDesc, (const F32 *)filterArray, rnnParamSpec,
                ftmDesc, (F32 *)ftmArray, scale, ftmDataFormat, arch);
            break;
        }
#endif
#ifdef _USE_FP16
        case DT_F16: {
            ret = rnn_transform_filter<F16>(filterDesc, (const F16 *)filterArray, rnnParamSpec,
                ftmDesc, (F16 *)ftmArray, scale, ftmDataFormat, arch);
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
    void **ftmArray,
    float *scale,
    Arch arch)
{
    int num1 = rnnParamSpec.bi_direction ? 2 : 1;
    int num2 = rnnParamSpec.num_projection > 0 ? 2 : 1;
    EE ret = SUCCESS;
    DataFormat ftmDataFormat;
    for (int i = 0; i < num1 * num2; i++) {
        if (((i % 2 == 0) || (num2 == 1)) && (rnnParamSpec.steps >= 0)) {
            ftmDataFormat = DF_NKNx_NKN32;
        } else {
            ftmDataFormat = DF_NKN32;
        }
        CHECK_STATUS(rnn_transform_filter_cpu_kernel(filterDesc[i], filterArray[i], rnnParamSpec,
            &ftmDesc[i], ftmArray[i], scale + i, ftmDataFormat, arch));
    }
    return ret;
}

EE rnn_transform_filter_bytes_cpu(
    const TensorDesc *filterDesc, RNNParamSpec rnnParamSpec, U32 *bytes)
{
    if (nullptr == bytes) {
        CHECK_STATUS(NULL_POINTER);
    }
    int num1 = rnnParamSpec.bi_direction ? 2 : 1;
    int num2 = rnnParamSpec.num_projection > 0 ? 2 : 1;
    for (int i = 0; i < num1 * num2; i++) {
        bytes[i] = tensorNumBytes(filterDesc[i]);
        // x86 need to add offset for U8 type, bytes = bias_length(fn) * size(int)
        if (rnnParamSpec.mode == RNN_LSTM) {
            bytes[i] += filterDesc[i].dims[1] * sizeof(I32);
        }
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
    U32 hDim = rnnParamSpec.num_outputs;
    U32 column = (rnnParamSpec.num_projection > 0) ? rnnParamSpec.num_projection
                                                   : rnnParamSpec.num_outputs;
    EE ret = SUCCESS;
    U32 factor = 0;
    switch (rnnParamSpec.mode) {
        case RNN_LSTM:
            factor = 4;
            break;
        case RNN_GRU:
            factor = 3;
            break;
        case RNN_GRU_LBR:
            factor = 3;
            break;
        default:
            ret = NOT_SUPPORTED;
            break;
    }

    *bytes = (hDim + xDim + column * factor) * bytesOf(idt);
    // for input quantization
    *bytes += (hDim + xDim) * bytesOf(DT_I8);
    return ret;
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
    DataType idt = inputDesc.dt;
    DataFormat idf = inputDesc.df;
    U32 batch = inputDesc.dims[inputDesc.nDims - 1];
    U32 step = inputDesc.dims[inputDesc.nDims - 2];
    U32 xDim = inputDesc.dims[inputDesc.nDims - 3];
    for (U32 i = 0; i < inputDesc.nDims - 3; ++i) {
        xDim *= inputDesc.dims[i];
    }
    U32 hDim = rnnParamSpec.num_outputs;
    TensorDesc xDesc = tensor2df(idt, DF_NORMAL, batch, xDim);
    CHECK_STATUS(rnncell_infer_forward_tmp_bytes_cpu(
        xDesc, filterDesc, outputDesc, rnnParamSpec, bytes, arch));
    U32 column = (rnnParamSpec.num_projection > 0) ? rnnParamSpec.num_projection
                                                   : rnnParamSpec.num_outputs;
    EE ret = SUCCESS;
    U32 factor = 0;
    switch (rnnParamSpec.mode) {
        case RNN_LSTM:
            factor = 4;
            break;
        case RNN_GRU:
            factor = 3;
            break;
        case RNN_GRU_LBR:
            factor = 3;
            break;
        default:
            ret = NOT_SUPPORTED;
            break;
    }

    int num1 = rnnParamSpec.bi_direction ? 2 : 1;
    *bytes += batch * ((column + hDim) * num1 + column * factor) * bytesOf(idt);
    if (idf == DF_NCHWC8) {
        *bytes += tensorNumBytes(inputDesc);
    }
    if (rnnParamSpec.steps >= 0) {
        // Intermediate gate result
        *bytes += batch * step * column * factor * bytesOf(idt);
        // mmm tmp buffer
        *bytes += UNI_MAX(batch * step * xDim, xDim * column) * bytesOf(idt);
        *bytes += 32;
    }
    // for input quantization
    *bytes += (hDim + xDim) * bytesOf(DT_I8);
    return ret;
}

EE rnncell_cpu(TensorDesc xDesc,
    const void *currentX,
    const TensorDesc *filterDesc,
    const void **filter,
    const TensorDesc *biasDesc,
    const void **bias,
    float *scale,
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
    } else if (IS_X86(arch)) {
        ret = rnncell_x86(xDesc, currentX, filterDesc, filter, biasDesc, bias, scale, state,
            tmpBytes, tmp, rnnParamSpec, batchStrideX, batchStrideH, hDesc, currentH, arch);
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
    float *scale,
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

    DataType fdt;
    DataFormat fdf;
    U32 fk, fn;
    int num1 = rnnParamSpec.bi_direction ? 2 : 1;
    CHECK_STATUS(tensor2dGet(filterDesc[0], &fdt, &fdf, &fn, &fk));
    if (fdf != DF_NKNx_NKN32) {
        CHECK_STATUS(NOT_MATCH);
    }
    DataType idt = inputDesc.dt;
    DataFormat idf = inputDesc.df;
    U32 batch = inputDesc.dims[inputDesc.nDims - 1];
    U32 step = inputDesc.dims[inputDesc.nDims - 2];
    U32 xDim = inputDesc.dims[inputDesc.nDims - 3];
    for (U32 i = 0; i < inputDesc.nDims - 3; ++i) {
        xDim *= inputDesc.dims[i];
    }

    const void *inputTmp = input;
    if (idf == DF_NCHWC8) {
        TensorDesc tmpDesc = inputDesc;
        tmpDesc.df = DF_NCHW;
        transformToNCHW(inputDesc, input, tmpDesc, tmp);
        inputTmp = tmp;
        tmp = (U8 *)tmp + tensorNumBytes(tmpDesc);
    }

    U32 hDim = rnnParamSpec.num_outputs;
    I32 column = (rnnParamSpec.num_projection > 0) ? rnnParamSpec.num_projection
                                                   : rnnParamSpec.num_outputs;
    U8 bytesOfIdt = bytesOf(idt);
    U32 batchStrideX = step * xDim;
    U32 batchStrideH = step * hDim * num1;
    TensorDesc xDesc = tensor2df(idt, DF_NORMAL, batch, 0);
    TensorDesc hDesc = tensor2df(idt, DF_NORMAL, batch, hDim);

    U8 *cellState = (U8 *)tmp;
    U8 *tmpArray = cellState + batch * num1 * (column + hDim) * bytesOfIdt;
    U8 *intermediateH = tmpArray + batch * step * xDim * bytesOfIdt;
    U8 *InterGate = intermediateH + fn * bytesOfIdt;

    const U8 *mmmFilter;
    mmmFilter = (const U8 *)filter[0];

    U32 tileSize = fn * bytesOfIdt;
    for (U32 m = 0; m < batch; m++) {
        for (U32 t = 0; t < step; ++t) {
            UNI_MEMCPY(InterGate + (m * step + t) * tileSize, bias[0], tileSize);
        }
    }

    TensorDesc inDesc = tensor2df(idt, DF_NORMAL, batch * step, xDim);
    TensorDesc mmmFilterDesc;
    TensorDesc outDesc = tensor2df(idt, DF_NORMAL, batch * step, fn);
    if (IS_GENERAL(arch)) {
        mmmFilterDesc = tensor2df(fdt, DF_TRANSPOSE, fn, xDim);
    } else {
        mmmFilterDesc = tensor2df(fdt, targetFormat4MatrixB(fdt), xDim, fn);
    }
    CHECK_STATUS(matrix_matrix_multiply(inDesc, inputTmp, mmmFilterDesc, mmmFilter,
        batch * step * xDim * bytesOfIdt, tmpArray, outDesc, InterGate, nullptr, arch));

    const void *useFilter[2] = {(const void *)(mmmFilter + fn * xDim * bytesOfIdt), nullptr};
    const void *useBias[2] = {nullptr, nullptr};
    if (rnnParamSpec.num_projection > 0) {
        useFilter[1] = filter[1];
    }
    if (rnnParamSpec.mode == RNN_GRU_LBR) {
        useBias[1] = bias[1];
    }
    TensorDesc useFilterDesc = tensor2df(fdt, DF_NKN32, fn, hDim);
    for (U32 t = 0; t < step; t++) {
        U8 *currentH = (U8 *)output + t * hDim * num1 * bytesOfIdt;
        useBias[0] = (void *)(InterGate + t * fn * bytesOfIdt);
        CHECK_STATUS(rnncell_cpu(xDesc, nullptr, &useFilterDesc, useFilter, biasDesc, useBias,
            scale, cellState, rnnParamSpec, batchStrideX, batchStrideH, tmpBytes, intermediateH,
            hDesc, currentH, arch));
    }

    if (rnnParamSpec.bi_direction) {
        int fCount = (rnnParamSpec.num_projection > 0) ? 2 : 1;
        int bCount = (rnnParamSpec.mode == RNN_GRU_LBR) ? 2 : 1;
        mmmFilter = (const U8 *)filter[fCount];
        for (U32 m = 0; m < batch; m++) {
            for (U32 t = 0; t < step; ++t) {
                UNI_MEMCPY(InterGate + (m * step + t) * tileSize, bias[bCount], tileSize);
            }
        }
        CHECK_STATUS(matrix_matrix_multiply(inDesc, inputTmp, mmmFilterDesc, mmmFilter,
            step * xDim * bytesOfIdt, tmpArray, outDesc, InterGate, nullptr, arch));
        useFilter[0] = mmmFilter + fn * xDim * bytesOfIdt;
        if (rnnParamSpec.num_projection > 0) {
            useFilter[1] = filter[fCount + 1];
        }
        if (rnnParamSpec.mode == RNN_GRU_LBR) {
            useBias[1] = bias[bCount + 1];
        }
        cellState += (column + hDim) * bytesOfIdt;
        for (I32 t = step - 1; t >= 0; t--) {
            U8 *currentH = (U8 *)output + (t * hDim * num1 + hDim) * bytesOfIdt;
            useBias[0] = (void *)(InterGate + t * fn * bytesOfIdt);
            CHECK_STATUS(rnncell_cpu(xDesc, nullptr, &useFilterDesc, useFilter, biasDesc, useBias,
                scale + fCount, cellState, rnnParamSpec, batchStrideX, batchStrideH, tmpBytes,
                intermediateH, hDesc, currentH, arch));
        }
    }
    return SUCCESS;
}
