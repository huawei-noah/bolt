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

#include "cpu/arm/tensor_computing_arm.h"
#ifdef _USE_FP32
#include "cpu/arm/fp32/tensor_computing_fp32.h"
#endif
#ifdef _USE_FP16
#include "cpu/arm/fp16/tensor_computing_fp16.h"
#endif
#include "blas-enhance.h"

template<typename T>
EE lstm_transform_filter(TensorDesc filterDesc, const T* filterArray, LSTMDesc lstmDesc, TensorDesc *ftmDesc, T* ftmArray)
{
    if (nullptr == filterArray || nullptr == ftmDesc || nullptr == ftmArray)
        CHECK_STATUS(NULL_POINTER);
    DataType fdt;
    DataFormat fdf;
    U32 fn, fk, ftm_n, ftm_k;
    CHECK_STATUS(tensor2dfGet(filterDesc, &fdt, &fdf, &fn, &fk));
    U32 alignSize = 32;
    switch(fdf) {
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
                        ftmArray[n*ftm_k*alignSize + k*alignSize + n32] = filterArray[(n*alignSize+n32)*ftm_k + k];
                    }
                }
            }
            break;
        }
        default:
            return NOT_MATCH;
    }
    if (lstmDesc.numProjection > 0) {
        U32 offset = fn * fk;
        if (lstmDesc.numOutput % alignSize != 0) {
            return NOT_MATCH;
        }
        U32 row = lstmDesc.numOutput / alignSize;
        U32 col = lstmDesc.numProjection;
        for (U32 n = 0; n < row; n++) {
            for (U32 k = 0; k < col; k++) {
                for (U32 n32 = 0; n32 < alignSize; n32++) {
                    ftmArray[offset+n*col*alignSize + k*alignSize + n32] = filterArray[offset+(n*alignSize+n32)*col + k];
                }
            }
        }
    }
    *ftmDesc = tensor2df(fdt, DF_NKN32, fn, fk);
    return SUCCESS;
}

EE lstm_transform_filter_arm_kernel(TensorDesc filterDesc, const void* filterArray, LSTMDesc lstmDesc, TensorDesc *ftmDesc, void* ftmArray)
{
    EE ret = SUCCESS;
    switch (filterDesc.dt) {
#ifdef _USE_FP32
        case DT_F32: {
            ret = lstm_transform_filter<F32>(filterDesc, (const F32*)filterArray, lstmDesc, ftmDesc, (F32*)ftmArray);
            break;
        }
#endif
#ifdef _USE_FP16
        case DT_F16: {
            ret = lstm_transform_filter<F16>(filterDesc, (const F16*)filterArray, lstmDesc, ftmDesc, (F16*)ftmArray);
            break;
        }
#endif
        default:
            ret = NOT_SUPPORTED;
            break;
    }
    return ret;
}

EE lstm_transform_filter_arm(TensorDesc filterDesc, const void* filterArray, LSTMDesc lstmDesc, TensorDesc *ftmDesc, void* ftmArray)
{
    EE ret = SUCCESS;
    U32 bytes = tensorNumBytes(filterDesc) + bytesOf(filterDesc.dt) * lstmDesc.numProjection * lstmDesc.numOutput;
    int num = lstmDesc.biDirection ? 2 : 1;
    for (int i = 0; i < num; i++) {
        const U8* filterArrayPtr = (const U8*)filterArray + i * bytes;
        U8* ftmArrayPtr = (U8*)ftmArray + i * bytes;
        ret = lstm_transform_filter_arm_kernel(filterDesc, filterArrayPtr, lstmDesc, ftmDesc, ftmArrayPtr);
    }
    return ret;
}

EE lstm_transform_filter_bytes_arm(TensorDesc filterDesc, LSTMDesc lstmDesc, U32* bytes)
{
    if (nullptr == bytes)
        CHECK_STATUS(NULL_POINTER);
    *bytes = tensorNumBytes(filterDesc) + bytesOf(filterDesc.dt) * lstmDesc.numProjection * lstmDesc.numOutput;
    int num = lstmDesc.biDirection ? 2 : 1;
    *bytes *= num;
    return SUCCESS;
}

EE lstmcell_infer_forward_tmp_bytes_arm(TensorDesc inputDesc, TensorDesc filterDesc, TensorDesc outputDesc, LSTMDesc lstmDesc, U32 *bytes, Arch arch)
{
    UNUSED(outputDesc);
    if (nullptr == bytes)
        CHECK_STATUS(NULL_POINTER);
    DataType idt;
    DataFormat idf;
    U32 batch, xDim;
    CHECK_STATUS(tensor2dfGet(inputDesc, &idt, &idf, &batch, &xDim));
    U32 hDim = lstmDesc.numOutput;
    U32 column = (lstmDesc.numProjection > 0) ? lstmDesc.numProjection : lstmDesc.numOutput;
    TensorDesc projectionMatrixDesc = tensor2df(filterDesc.dt, DF_NORMAL, lstmDesc.numProjection, lstmDesc.numOutput);
    TensorDesc projectionVectorDesc = tensor1d(filterDesc.dt, lstmDesc.numProjection);
    CHECK_STATUS(matrix_vector_multiply_tmp_bytes(projectionMatrixDesc, projectionVectorDesc, bytes, arch));
    *bytes += (hDim + xDim + column * 4) * bytesOf(idt);
    return SUCCESS;
}

EE lstm_infer_forward_tmp_bytes_arm(TensorDesc inputDesc, TensorDesc filterDesc, TensorDesc outputDesc, LSTMDesc lstmDesc, U32 *bytes, Arch arch)
{
    UNUSED(filterDesc);
    UNUSED(outputDesc);
    if (nullptr == bytes)
        CHECK_STATUS(NULL_POINTER);
    DataType idt;
    DataFormat idf;
    U32 batch, step, xDim;
    CHECK_STATUS(tensor3dGet(inputDesc, &idt, &idf, &batch, &step, &xDim));
    U32 hDim = lstmDesc.numOutput;
    TensorDesc xDesc = tensor2df(idt, DF_NORMAL, batch, xDim);
    CHECK_STATUS(lstmcell_infer_forward_tmp_bytes_arm(xDesc, filterDesc, outputDesc, lstmDesc, bytes, arch));
    U32 column = (lstmDesc.numProjection > 0) ? lstmDesc.numProjection : lstmDesc.numOutput;
    *bytes += batch * (column + hDim) * bytesOf(idt);
    return SUCCESS;
}

EE lstmcell_arm(TensorDesc xDesc, const void* currentX,
    TensorDesc filterDesc, const void* filter,
    TensorDesc biasDesc, const void* bias,
    void *state,
    U32 tmpBytes, void *tmp,
    LSTMDesc lstmDesc, U32 batchStrideX, U32 batchStrideH,
    TensorDesc hDesc, void* output,
    Arch arch)
{
    EE ret = SUCCESS;
    switch (xDesc.dt) {
#ifdef _USE_FP32
        case DT_F32: {
            ret = lstmcell_fp32(xDesc, currentX, 
                                filterDesc, filter,
                                biasDesc, bias,
                                state,
                                tmpBytes, tmp,
                                lstmDesc, batchStrideX, batchStrideH,
                                hDesc, output,
                                arch);
            break;
        }
#endif
#ifdef _USE_FP16
        case DT_F16: {
            ret = lstmcell_fp16(xDesc, currentX,
                                filterDesc, filter,
                                biasDesc, bias,
                                state,
                                tmpBytes, tmp,
                                lstmDesc, batchStrideX, batchStrideH,
                                hDesc, output,
                                arch);
            break;
        }
#endif
        default:
            ret = NOT_SUPPORTED;
            break;
    }
    return ret;
}

EE lstm_arm(TensorDesc inputDesc, const void* input,
    TensorDesc filterDesc, const void* filter,
    TensorDesc biasDesc, const void* bias,
    U32 tmpBytes, void* tmp,
    LSTMDesc lstmDesc,
    TensorDesc outputDesc, void* output,
    Arch arch)
{
    UNUSED(outputDesc);

    if (nullptr == input
        || nullptr == filter
        || nullptr == bias
        || nullptr == tmp
        || nullptr == output)
        CHECK_STATUS(NULL_POINTER);

    DataType idt;
    DataFormat idf;
    U32 batch, step, xDim;
    int num = lstmDesc.biDirection ? 2 : 1;
    CHECK_STATUS(tensor3dGet(inputDesc, &idt, &idf, &batch, &step, &xDim));
    U32 hDim = lstmDesc.numOutput;
    U32 column = (lstmDesc.numProjection > 0) ? lstmDesc.numProjection : lstmDesc.numOutput;

    U8 *cellState = (U8*)tmp;
    U8 *tmpArray = cellState + batch * (column + hDim) * bytesOf(idt);
    U32 batchStrideX = step * xDim;
    U32 batchStrideH = step * hDim * num;
    TensorDesc xDesc = tensor2df(idt, DF_NORMAL, batch, xDim);
    TensorDesc hDesc = tensor2df(idt, DF_NORMAL, batch, hDim);

    memset(cellState, 0, batch * (column + hDim) * bytesOf(idt));
    for (U32 t = 0; t < step; t++) {
        const U8* currentX = (const U8*)input + t * xDim * bytesOf(idt);
        U8 *currentH = (U8*)output + t * hDim * num * bytesOf(idt);
        CHECK_STATUS(lstmcell_arm(xDesc, currentX,
              filterDesc, filter,
              biasDesc, bias,
              cellState,
              tmpBytes, tmpArray,
              lstmDesc, batchStrideX, batchStrideH,
              hDesc, currentH,
              arch));
    }

    if (lstmDesc.biDirection) {
        memset(cellState, 0, batch * (column + hDim) * bytesOf(idt));
        U32 filterBytes = tensorNumBytes(filterDesc) + bytesOf(filterDesc.dt) * lstmDesc.numProjection * lstmDesc.numOutput;
        U32 biasBytes = tensorNumBytes(biasDesc);
        const U8* filterPtr = (const U8*)filter + filterBytes;
        const U8* biasPtr = (const U8*)bias + biasBytes;
        for (I32 t = step-1; t >= 0; t--) {
            const U8* currentX = (const U8*)input + t * xDim * bytesOf(idt);
            U8 *currentH = (U8*)output + t * hDim * num * bytesOf(idt) + hDim * bytesOf(idt);
            CHECK_STATUS(lstmcell_arm(xDesc, currentX,
                  filterDesc, filterPtr,
                  biasDesc, biasPtr,
                  cellState,
                  tmpBytes, tmpArray,
                  lstmDesc, batchStrideX, batchStrideH,
                  hDesc, currentH,
                  arch));
        }
    }
    return SUCCESS;
}
