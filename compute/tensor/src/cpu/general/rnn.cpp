// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include <math.h>

#include "cpu/general/tensor_computing_general.h"
#include "cpu/general/general_functions.h"

template <typename T>
static void mvm_nkn32_template(U32 fn, U32 fk, const T *filterArray, T *input, T *output)
{
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

template <typename T>
static EE lstmcell(TensorDesc xDesc,
    const void *currentX,
    const TensorDesc *filterDesc,
    const void **filter,
    const TensorDesc *biasDesc,
    const void **bias,
    void *state,
    U32 tmpBytes,
    void *tmp,
    RNNParamSpec rnnParamSpec,
    U32 batchStrideX,
    U32 batchStrideH,
    TensorDesc hDesc,
    void *output)
{
    UNUSED(biasDesc);
    UNUSED(tmpBytes);
    if (nullptr == filter || nullptr == bias || nullptr == state || nullptr == tmp ||
        nullptr == output) {
        CHECK_STATUS(NULL_POINTER);
    }

    DataType idt, fdt, odt;
    DataFormat idf, fdf, odf;
    U32 in, ix;
    U32 on, oh;
    U32 fk, fn;
    CHECK_STATUS(tensor2dGet(xDesc, &idt, &idf, &in, &ix));
    CHECK_STATUS(tensor2dGet(filterDesc[0], &fdt, &fdf, &fn, &fk));
    CHECK_STATUS(tensor2dGet(hDesc, &odt, &odf, &on, &oh));
    if (fdf != DF_NKN32) {
        CHECK_STATUS(NOT_MATCH);
    }

    U32 batch = in;
    U32 xDim = ix;
    U32 hDim = rnnParamSpec.num_outputs;
    I32 column = (rnnParamSpec.num_projection > 0) ? rnnParamSpec.num_projection
                                                   : rnnParamSpec.num_outputs;
    int num1 = rnnParamSpec.bi_direction ? 2 : 1;
    U32 steps = batchStrideH / hDim / num1;
    if (rnnParamSpec.activation_type != ACTIVATION_TANH) {
        CHECK_STATUS(NOT_SUPPORTED);
    }

    if (!(idt == fdt && idt == odt)) {
        CHECK_STATUS(NOT_MATCH);
    }

    const T *currentXArray = (const T *)currentX;
    T *lastStateArray = (T *)state;
    T *lastHArray = lastStateArray + column;
    T *tmpArray = (T *)tmp;
    T *currentStateArray = (T *)state;
    T *currentHArray = currentStateArray + column;
    T *outputArray = (T *)output;
    T *xhArray = tmpArray;
    T *intermediateH = xhArray + (xDim + hDim);
    U32 lastStateStride = column + hDim;
    U32 lastHStride = column + hDim;
    U32 currentStateStride = column + hDim;
    U32 currentHStride = column + hDim;
    for (U32 m = 0; m < batch; m++) {
        T *lastBatchH = lastHArray + m * lastHStride;
        if (xDim > 0) {
            UNI_MEMCPY(xhArray, currentXArray + m * batchStrideX, xDim * sizeof(T));
            UNI_MEMCPY(xhArray + xDim, lastBatchH, hDim * sizeof(T));
        } else {
            intermediateH = tmpArray;
            xhArray = lastBatchH;
        }

        // MVM
        const T *mBias = (const T *)bias[0] + m * steps * column * 4;
        UNI_MEMCPY(intermediateH, mBias, column * 4 * sizeof(T));
        mvm_nkn32_template<T>(fn / 32, fk, (const T *)filter[0], xhArray, intermediateH);

        T *out_i = intermediateH;
        T *out_g = out_i + column;
        T *out_f = out_i + column * 2;
        T *out_o = out_i + column * 3;
        T *lastBatchState = lastStateArray + m * lastStateStride;
        T *currentBatchState = currentStateArray + m * currentStateStride;
        T *currentBatchH = currentHArray + m * currentHStride;
        T *currentOutput = outputArray + m * batchStrideH;
        T *tmpState, *tmpHH, *tmpH;
        if (rnnParamSpec.zoneout_cell == 0) {
            tmpState = currentBatchState;
        } else {
            tmpState = out_i;
        }
        if (rnnParamSpec.num_projection > 0) {
            tmpHH = out_g;
            tmpH = currentOutput;
        } else {
            tmpHH = currentOutput;
            tmpH = out_g;
        }

        for (I32 h = 0; h < column; h++) {
            F32 C_s = lastBatchState[h];
            F32 I_s = 1.0 / (1.0 + exp(-out_i[h]));
            F32 G_s = tanh(out_g[h]);
            F32 F_s = 1.0 / (1.0 + exp(-(out_f[h] + rnnParamSpec.forget_bias)));
            F32 O_s = 1.0 / (1.0 + exp(-out_o[h]));
            C_s = C_s * F_s + I_s * G_s;
            F32 value = O_s * tanh(C_s);
            tmpState[h] = C_s;
            tmpHH[h] = value;
        }

        if (rnnParamSpec.zoneout_cell != 0) {
            array_scale_template<T>(tmpState, tmpState, column, 1 - rnnParamSpec.zoneout_cell, 0);
            array_scale_template<T>(
                lastBatchState, lastBatchState, column, rnnParamSpec.zoneout_cell, 0);
            array_add_template<T>(tmpState, lastBatchState, currentBatchState, column);
        }

        if (rnnParamSpec.num_projection > 0) {
            UNI_MEMSET(tmpH, 0, sizeof(T) * hDim);
            mvm_nkn32_template<T>(
                hDim / 32, rnnParamSpec.num_projection, (const T *)filter[1], tmpHH, tmpH);
        }
        if (rnnParamSpec.zoneout_output != 0) {
            if (rnnParamSpec.num_projection > 0) {
                array_scale_template<T>(tmpH, out_f, hDim, 1 - rnnParamSpec.zoneout_output, 0);
            } else {
                array_scale_template<T>(tmpHH, out_f, hDim, 1 - rnnParamSpec.zoneout_output, 0);
            }
            array_scale_template<T>(lastBatchH, lastBatchH, hDim, rnnParamSpec.zoneout_output, 0);
            array_add_template<T>(out_f, lastBatchH, currentBatchH, hDim);
        } else {
            UNI_MEMCPY(currentBatchH, currentOutput, sizeof(T) * hDim);
        }
    }
    return SUCCESS;
}

template <typename T>
static EE grucell(TensorDesc xDesc,
    const void *currentX,
    const TensorDesc *filterDesc,
    const void **filter,
    const TensorDesc *biasDesc,
    const void **bias,
    void *state,
    U32 tmpBytes,
    void *tmp,
    RNNParamSpec rnnParamSpec,
    U32 batchStrideX,
    U32 batchStrideH,
    TensorDesc hDesc,
    void *output)
{
    UNUSED(biasDesc);
    UNUSED(tmpBytes);
    if (nullptr == filter || nullptr == bias || nullptr == state || nullptr == tmp ||
        nullptr == output) {
        CHECK_STATUS(NULL_POINTER);
    }

    DataType idt, fdt, odt;
    DataFormat idf, fdf, odf;
    U32 in, ix;
    U32 on, oh;
    U32 fk, fn;
    CHECK_STATUS(tensor2dGet(xDesc, &idt, &idf, &in, &ix));
    CHECK_STATUS(tensor2dGet(filterDesc[0], &fdt, &fdf, &fn, &fk));
    CHECK_STATUS(tensor2dGet(hDesc, &odt, &odf, &on, &oh));
    if (fdf != DF_NKN32) {
        CHECK_STATUS(NOT_MATCH);
    }

    U32 batch = in;
    U32 xDim = ix;
    U32 hDim = rnnParamSpec.num_outputs;
    I32 column = hDim;
    int num1 = rnnParamSpec.bi_direction ? 2 : 1;
    U32 steps = batchStrideH / hDim / num1;
    if (rnnParamSpec.activation_type != ACTIVATION_TANH) {
        CHECK_STATUS(NOT_SUPPORTED);
    }

    if (!(idt == fdt && idt == odt)) {
        CHECK_STATUS(NOT_MATCH);
    }

    const T *currentXArray = (const T *)currentX;
    T *lastHArray = (T *)state;
    T *tmpArray = (T *)tmp;
    T *currentHArray = (T *)state;
    T *outputArray = (T *)output;
    T *xhArray = tmpArray;
    T *intermediateH = xhArray + (xDim + hDim);
    U32 lastHStride = hDim;
    U32 currentHStride = hDim;
    for (U32 m = 0; m < batch; m++) {
        T *lastBatchH = lastHArray + m * lastHStride;
        T *currentBatchH = currentHArray + m * currentHStride;
        T *currentOutput = outputArray + m * batchStrideH;
        if (xDim > 0) {
            UNI_MEMCPY(xhArray, currentXArray + m * batchStrideX, xDim * sizeof(T));
            UNI_MEMCPY(xhArray + xDim, lastBatchH, hDim * sizeof(T));
        } else {
            intermediateH = tmpArray;
            xhArray = lastBatchH;
            UNI_MEMCPY(currentOutput, lastBatchH, hDim * sizeof(T));
        }
        const T *mBias = (const T *)bias[0] + m * steps * column * 3;
        UNI_MEMCPY(intermediateH, mBias, column * 2 * sizeof(T));
        mvm_nkn32_template<T>(column * 2 / 32, fk, (const T *)filter[0], xhArray, intermediateH);
        T *out_z = intermediateH;
        T *out_r = out_z + column;
        T *out_h = out_r + column;

        for (I32 h = 0; h < column; h++) {
            out_r[h] = 1.0 / (1.0 + exp(-out_r[h]));
        }

        if (rnnParamSpec.mode == RNN_GRU_LBR) {
            T *h_x_b = (T *)mBias + column * 2;
            T *h_h_b = (T *)bias[1];
            UNI_MEMCPY(out_h, h_h_b, column * sizeof(T));
            mvm_nkn32_template<T>(column / 32, hDim,
                (const T *)filter[0] + column * 2 * fk + column * xDim, xhArray + xDim, out_h);
            array_mul_template<T>(out_r, out_h, out_h, hDim);
            if (xDim > 0) {
                UNI_MEMCPY(out_r, h_x_b, column * sizeof(T));
                mvm_nkn32_template<T>(
                    column / 32, xDim, (const T *)filter[0] + column * 2 * fk, xhArray, out_r);
                h_x_b = out_r;
            }
            array_add_template<T>(h_x_b, out_h, out_h, hDim);
        } else {
            array_mul_template<T>(out_r, xhArray + xDim, xhArray + xDim, hDim);
            UNI_MEMCPY(out_h, (const T *)mBias + column * 2, column * sizeof(T));
            mvm_nkn32_template<T>(
                column / 32, fk, (const T *)filter[0] + column * 2 * fk, xhArray, out_h);
        }
        for (I32 h = 0; h < column; h++) {
            out_z[h] = 1.0 / (1.0 + exp(-out_z[h]));
            out_h[h] = tanh(out_h[h]);
        }
        if (xDim > 0) {
            array_mul_template<T>(out_z, lastBatchH, out_r, column);
        } else {
            array_mul_template<T>(out_z, currentOutput, out_r, column);
        }
        array_scale_template<T>(out_z, out_z, column, -1, 1);
        array_mul_template<T>(out_z, out_h, out_h, column);
        array_add_template<T>(out_r, out_h, currentOutput, column);
        UNI_MEMCPY(currentBatchH, currentOutput, sizeof(T) * hDim);
    }
    return SUCCESS;
}

template <typename T>
static EE rnncell(TensorDesc xDesc,
    const void *currentX,
    const TensorDesc *filterDesc,
    const void **filter,
    const TensorDesc *biasDesc,
    const void **bias,
    void *state,
    U32 tmpBytes,
    void *tmp,
    RNNParamSpec rnnParamSpec,
    U32 batchStrideX,
    U32 batchStrideH,
    TensorDesc hDesc,
    void *output)
{
    EE ret = SUCCESS;
    switch (rnnParamSpec.mode) {
        case RNN_GRU_LBR:
        case RNN_GRU: {
            ret = grucell<T>(xDesc, currentX, filterDesc, filter, biasDesc, bias, state, tmpBytes,
                tmp, rnnParamSpec, batchStrideX, batchStrideH, hDesc, output);
            break;
        }
        case RNN_LSTM: {
            ret = lstmcell<T>(xDesc, currentX, filterDesc, filter, biasDesc, bias, state, tmpBytes,
                tmp, rnnParamSpec, batchStrideX, batchStrideH, hDesc, output);
            break;
        }
        default:
            ret = NOT_SUPPORTED;
            break;
    }
    return ret;
}

EE rnncell_general(TensorDesc xDesc,
    const void *currentX,
    const TensorDesc *filterDesc,
    const void **filter,
    const TensorDesc *biasDesc,
    const void **bias,
    void *state,
    U32 tmpBytes,
    void *tmp,
    RNNParamSpec rnnParamSpec,
    U32 batchStrideX,
    U32 batchStrideH,
    TensorDesc hDesc,
    void *output)
{
    EE ret = SUCCESS;
    switch (xDesc.dt) {
#ifdef _USE_FP16
        case DT_F16:
            ret = rnncell<F16>(xDesc, currentX, filterDesc, filter, biasDesc, bias, state, tmpBytes,
                tmp, rnnParamSpec, batchStrideX, batchStrideH, hDesc, output);
            break;
#endif
#ifdef _USE_FP32
        case DT_F32:
            ret = rnncell<F32>(xDesc, currentX, filterDesc, filter, biasDesc, bias, state, tmpBytes,
                tmp, rnnParamSpec, batchStrideX, batchStrideH, hDesc, output);
            break;
#endif
        default:
            ret = NOT_SUPPORTED;
            break;
    }
    return ret;
}
