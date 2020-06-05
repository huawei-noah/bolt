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
#include <math.h>

#include "cpu/general/tensor_computing_general.h"
#include "cpu/general/general_functions.h"

template<typename T>
void mvm_nkn32(U32 fn, U32 fk, const T* filterArray, T* input, T* output) {
    for (U32 i = 0; i < fn; i++) {
        for (U32 j = 0; j < 32; j++) {
            U32 n = i * 32 + j;
            F32 value = 0;
            for (U32 k = 0; k < fk; k++) {
                value += input[k] * filterArray[(i * fk + k) * 32 + j];
            }
            output[n] += value;
        }
    }
}

template<typename T>
EE lstmcell(TensorDesc xDesc, const void* currentX,
    TensorDesc filterDesc, const void* filter,
    TensorDesc biasDesc, const void* bias,
    void *state,
    U32 tmpBytes, void *tmp,
    LSTMDesc lstmDesc, U32 batchStrideX, U32 batchStrideH,
    TensorDesc hDesc, void* output)
{
    UNUSED(biasDesc);
    UNUSED(tmpBytes);
    if (nullptr == currentX
        || nullptr == filter
        || nullptr == bias
        || nullptr == state
        || nullptr == tmp
        || nullptr == output)
        CHECK_STATUS(NULL_POINTER);

    DataType idt, fdt, odt;
    DataFormat idf, fdf, odf;
    U32 in, ix;
    U32 on, oh;
    U32 fk, fn;
    CHECK_STATUS(tensor2dfGet(xDesc, &idt, &idf, &in, &ix));
    CHECK_STATUS(tensor2dfGet(filterDesc, &fdt, &fdf, &fn, &fk));
    CHECK_STATUS(tensor2dfGet(hDesc, &odt, &odf, &on, &oh));
    if(fdf != DF_NKN32) {
        CHECK_STATUS(NOT_MATCH);
    }

    U32 batch = in;
    U32 xDim  = ix;
    U32 hDim  = lstmDesc.numOutput;
    I32 column = (lstmDesc.numProjection > 0) ? lstmDesc.numProjection : lstmDesc.numOutput;
    F32 forgetBias = lstmDesc.forgetBias;
    ActivationMode activationMode = lstmDesc.activationMode;
    if (activationMode != ACTIVATION_TANH)
        CHECK_STATUS(NOT_SUPPORTED);

    if (!(idt == fdt && idt == odt)) {
        CHECK_STATUS(NOT_MATCH);
    }

    const T *currentXArray   = (const T*)currentX;
    const T *filterArray     = (const T*)filter;
    const T *biasArray       = (const T*)bias;
    const T *projectionArray = (const T*)filter + (fn * fk);
    T *lastStateArray = (T*)state;
    T *lastHArray     = lastStateArray + column;
    T *tmpArray       = (T*)tmp;
    T *currentStateArray = (T*)state;
    T *currentHArray     = currentStateArray + column;
    T *outputArray       = (T*)output;
    T *xhArray           = tmpArray;
    T *intermediateH     = xhArray + (xDim + hDim);
    U32 lastStateStride    = column + hDim;
    U32 lastHStride        = column + hDim;
    U32 currentStateStride = column + hDim;
    U32 currentHStride     = column + hDim;
    for (U32 m = 0; m < batch; m++) {
        T *lastBatchH = lastHArray + m * lastHStride;
        memcpy(xhArray, currentXArray+m*batchStrideX, xDim*sizeof(T));
        memcpy(xhArray+xDim, lastBatchH, hDim*sizeof(T));

        // MVM
        memcpy(intermediateH, biasArray, column * 4 * sizeof(T));
        mvm_nkn32<T>(fn/32, fk, filterArray, xhArray, intermediateH);

        T *out_i = intermediateH;
        T *out_g = out_i + column;
        T *out_f = out_i + column * 2;
        T *out_o = out_i + column * 3;
        T *lastBatchState = lastStateArray + m * lastStateStride;
        T *currentBatchState = currentStateArray + m * currentStateStride;
        T *currentBatchH = currentHArray + m * currentHStride;
        T *currentOutput = outputArray + m * batchStrideH;
        T* tmpState, *tmpHH, *tmpH;
        if (lstmDesc.zoneoutCell == 0) {
            tmpState = currentBatchState;
        } else {
            tmpState = out_i;
        }
        if (lstmDesc.zoneoutOutput != 0) {
            tmpHH = out_g;
            tmpH = out_f;
        } else {
            if (lstmDesc.numProjection > 0) {
                tmpHH = out_g;
                tmpH = out_f;
            } else {
                tmpHH = currentBatchH;
                tmpH = currentBatchH;
            }
        }

        for (I32 h = 0; h < column; h++) {
            F32 C_s = lastBatchState[h];
            F32 I_s = 1.0 / (1.0 + exp(-out_i[h]));
            F32 G_s = tanh(out_g[h]);
            F32 F_s = 1.0 / (1.0 + exp(-(out_f[h] + forgetBias)));
            F32 O_s = 1.0 / (1.0 + exp(-out_o[h]));
            C_s = C_s * F_s + I_s * G_s;
            F32 value = O_s * tanh(C_s);
            tmpState[h] = C_s;
            tmpHH[h] = value;
        }

        if (lstmDesc.zoneoutCell != 0) {
            array_scale<T>(tmpState, tmpState, column, 1-lstmDesc.zoneoutCell, 0);
            array_scale<T>(lastBatchState, lastBatchState, column, lstmDesc.zoneoutCell, 0);
            array_add<T>(tmpState, lastBatchState, currentBatchState, column);
        }
        if (lstmDesc.zoneoutOutput != 0) {
            array_scale<T>(tmpHH, tmpH, column, 1-lstmDesc.zoneoutOutput, 0);
            array_scale<T>(lastBatchH, lastBatchH, column, lstmDesc.zoneoutOutput, 0);
            array_add<T>(tmpH, lastBatchH, currentBatchH, column);
        }

        if (lstmDesc.numProjection > 0) {
            memset(currentBatchH, 0, sizeof(T) * hDim);
            mvm_nkn32(hDim/32, lstmDesc.numProjection, projectionArray, tmpHH, currentBatchH);
            tmpHH = currentBatchH;
        }
        memcpy(currentOutput, tmpHH, sizeof(T) * hDim);
    }
    return SUCCESS;
}

EE lstm(TensorDesc inputDesc, const void* input,
    TensorDesc filterDesc, const void* filter,
    TensorDesc biasDesc, const void* bias,
    U32 tmpBytes, void* tmp,
    LSTMDesc lstmDesc,
    TensorDesc outputDesc, void* output)
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
        CHECK_STATUS(lstmcell_general(xDesc, currentX,
              filterDesc, filter,
              biasDesc, bias,
              cellState,
              tmpBytes, tmpArray,
              lstmDesc, batchStrideX, batchStrideH,
              hDesc, currentH));
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
            CHECK_STATUS(lstmcell_general(xDesc, currentX,
                  filterDesc, filterPtr,
                  biasDesc, biasPtr,
                  cellState,
                  tmpBytes, tmpArray,
                  lstmDesc, batchStrideX, batchStrideH,
                  hDesc, currentH));
        }
    }
    return SUCCESS;
}

EE lstmcell_general(TensorDesc xDesc, const void* currentX,
    TensorDesc filterDesc, const void* filter,
    TensorDesc biasDesc, const void* bias,
    void *state,
    U32 tmpBytes, void *tmp,
    LSTMDesc lstmDesc, U32 batchStrideX, U32 batchStrideH,
    TensorDesc hDesc, void* output)
{
    EE ret = SUCCESS;
    switch (xDesc.dt) {
#ifdef _USE_FP16
        case DT_F16:
            ret = lstmcell<F16>(xDesc, currentX, filterDesc, filter, biasDesc, bias,
                                state, tmpBytes, tmp, lstmDesc, batchStrideX, batchStrideH, hDesc, output);
            break;
#endif
#ifdef _USE_FP32
        case DT_F32:
            ret = lstmcell<F32>(xDesc, currentX, filterDesc, filter, biasDesc, bias,
                                state, tmpBytes, tmp, lstmDesc, batchStrideX, batchStrideH, hDesc, output);
            break;
#endif
        default:
            ret = NOT_SUPPORTED;
            break;
    }
    return ret;
}

EE lstm_general(TensorDesc inputDesc, const void* input,
    TensorDesc filterDesc, const void* filter,
    TensorDesc biasDesc, const void* bias,
    U32 tmpBytes, void* tmp,
    LSTMDesc lstmDesc,
    TensorDesc outputDesc, void* output)
{
    EE ret = SUCCESS;
    switch (inputDesc.dt) {
#ifdef _USE_FP16
        case DT_F16:
            ret = lstm(inputDesc, input, filterDesc, filter, biasDesc, bias,
                       tmpBytes, tmp, lstmDesc, outputDesc, output);
            break;
#endif
#ifdef _USE_FP32
        case DT_F32:
            ret = lstm(inputDesc, input, filterDesc, filter, biasDesc, bias,
                       tmpBytes, tmp, lstmDesc, outputDesc, output);
            break;
#endif
        default:
            ret = NOT_SUPPORTED;
            break;
    }
    return ret;
}
