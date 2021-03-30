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
#include "cpu/arm/fp32/tensor_computing_fp32.h"
#include "cpu/arm/fp32/mvm_nkn32.h"

EE lstmcell_fp32(TensorDesc xDesc,
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
    void *output,
    Arch arch)
{
    UNUSED(biasDesc);
    UNUSED(tmpBytes);
    UNUSED(arch);
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
    fn /= 32;

    U32 batch = in;
    I32 xDim = ix;
    I32 hDim = rnnParamSpec.numOutput;
    I32 column = (rnnParamSpec.numProjection > 0) ? rnnParamSpec.numProjection
                                                  : rnnParamSpec.numOutput;
    if (!(idt == DT_F32 && fdt == DT_F32 && odt == DT_F32)) {
        CHECK_STATUS(NOT_MATCH);
    }
    if (!(4 * column == (I32)fn * 32 && (ix + oh) == fk && in == on)) {
        CHECK_STATUS(NOT_MATCH);
    }
    F32 forgetBias = rnnParamSpec.forgetBias;
    ActivationMode activationMode = rnnParamSpec.activationMode;
    if (activationMode != ACTIVATION_TANH) {
        CHECK_STATUS(NOT_SUPPORTED);
    }

    const F32 *currentXArray = (const F32 *)currentX;
    F32 *lastStateArray = (F32 *)state;
    F32 *lastHArray = lastStateArray + column;
    F32 *tmpArray = (F32 *)tmp;
    F32 *currentStateArray = (F32 *)state;
    F32 *currentHArray = currentStateArray + column;
    F32 *outputArray = (F32 *)output;
    F32 *xhArray = tmpArray;
    F32 *intermediateH = xhArray + (xDim + hDim);
    U32 lastStateStride = column + hDim;
    U32 lastHStride = column + hDim;
    U32 currentStateStride = column + hDim;
    U32 currentHStride = column + hDim;
    float32x4_t forgetBiasVector = vdupq_n_f32(forgetBias);
    for (U32 m = 0; m < batch; m++) {
        F32 *lastBatchH = lastHArray + m * lastHStride;
        if (xDim > 0) {
            memcpy(xhArray, currentXArray + m * batchStrideX, xDim * sizeof(F32));
            memcpy(xhArray + xDim, lastBatchH, hDim * sizeof(F32));
        } else {
            intermediateH = tmpArray;
            xhArray = lastBatchH;
        }

        memcpy(intermediateH, bias[0], column * 4 * sizeof(F32));
        mvm_nkn32(fn, fk, (const F32 *)filter[0], xhArray, intermediateH);
        F32 *out_i = intermediateH;
        F32 *out_g = out_i + column;
        F32 *out_f = out_i + column * 2;
        F32 *out_o = out_i + column * 3;

        F32 *lastBatchState = lastStateArray + m * lastStateStride;
        F32 *currentBatchState = currentStateArray + m * currentStateStride;
        F32 *currentBatchH = currentHArray + m * currentHStride;
        F32 *currentOutput = outputArray + m * batchStrideH;

        F32 *tmpState, *tmpHH, *tmpH;
        if (rnnParamSpec.zoneoutCell == 0) {
            tmpState = currentBatchState;
        } else {
            tmpState = out_i;
        }
        if (rnnParamSpec.numProjection > 0) {
            tmpHH = out_g;
            tmpH = currentOutput;
        } else {
            tmpHH = currentOutput;
            tmpH = out_g;
        }

        I32 h = 0;
        for (; h < column - 3; h += 4) {
            float32x4_t out_i_v = vld1q_f32(out_i + h);
            float32x4_t out_g_v = vld1q_f32(out_g + h);
            float32x4_t out_f_v = vld1q_f32(out_f + h);
            float32x4_t out_o_v = vld1q_f32(out_o + h);
            float32x4_t C_v = vld1q_f32(lastBatchState + h);
            float32x4_t I_v = vsigmoidq_f32(out_i_v);
            float32x4_t F_v = vsigmoidq_f32(vaddq_f32(out_f_v, forgetBiasVector));
            float32x4_t O_v = vsigmoidq_f32(out_o_v);
            float32x4_t G_v = vtanhq_f32(out_g_v);
            C_v = vaddq_f32(vmulq_f32(C_v, F_v), vmulq_f32(I_v, G_v));
            float32x4_t out_hidden_v = vmulq_f32(O_v, vtanhq_f32(C_v));
            vst1q_f32(tmpState + h, C_v);
            vst1q_f32(tmpHH + h, out_hidden_v);
        }
        for (; h < column; h++) {
            F32 C_s = lastBatchState[h];
            F32 I_s = 1.0 / (1.0 + exp(-out_i[h]));
            F32 F_s = 1.0 / (1.0 + exp(-(out_f[h] + forgetBias)));
            F32 O_s = 1.0 / (1.0 + exp(-out_o[h]));
            F32 G_s = tanh(out_g[h]);
            C_s = C_s * F_s + I_s * G_s;
            F32 value = O_s * tanh(C_s);
            tmpState[h] = C_s;
            tmpHH[h] = value;
        }
        if (rnnParamSpec.zoneoutCell != 0) {
            array_scale_f32(tmpState, tmpState, column, 1 - rnnParamSpec.zoneoutCell, 0);
            array_scale_f32(lastBatchState, lastBatchState, column, rnnParamSpec.zoneoutCell, 0);
            array_add_f32(tmpState, lastBatchState, currentBatchState, column);
        }

        if (rnnParamSpec.numProjection > 0) {
            memset(tmpH, 0, sizeof(F32) * hDim);
            mvm_nkn32(hDim / 32, rnnParamSpec.numProjection, (const F32 *)filter[1], tmpHH, tmpH);
        }
        if (rnnParamSpec.zoneoutOutput != 0) {
            if (rnnParamSpec.numProjection > 0) {
                array_scale_f32(tmpH, out_f, hDim, 1 - rnnParamSpec.zoneoutOutput, 0);
            } else {
                array_scale_f32(tmpHH, out_f, hDim, 1 - rnnParamSpec.zoneoutOutput, 0);
            }
            array_scale_f32(lastBatchH, lastBatchH, hDim, rnnParamSpec.zoneoutOutput, 0);
            array_add_f32(out_f, lastBatchH, currentBatchH, hDim);
        } else {
            memcpy(currentBatchH, currentOutput, sizeof(F32) * hDim);
        }
    }
    return SUCCESS;
}
