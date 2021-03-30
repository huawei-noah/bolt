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
#include "cpu/arm/fp16/tensor_computing_fp16.h"
#include "cpu/arm/fp16/mvm_nkn32.h"

EE lstmcell_fp16(TensorDesc xDesc,
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
    if (!(idt == DT_F16 && fdt == DT_F16 && odt == DT_F16)) {
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

    const F16 *currentXArray = (const F16 *)currentX;
    F16 *lastStateArray = (F16 *)state;
    F16 *lastHArray = lastStateArray + column;
    F16 *tmpArray = (F16 *)tmp;
    F16 *currentStateArray = (F16 *)state;
    F16 *currentHArray = currentStateArray + column;
    F16 *outputArray = (F16 *)output;
    F16 *xhArray = tmpArray;
    F16 *intermediateH = xhArray + (xDim + hDim);
    U32 lastStateStride = column + hDim;
    U32 lastHStride = column + hDim;
    U32 currentStateStride = column + hDim;
    U32 currentHStride = column + hDim;
    float16x8_t forgetBiasVector = vdupq_n_f16(forgetBias);
    for (U32 m = 0; m < batch; m++) {
        F16 *lastBatchH = lastHArray + m * lastHStride;
        if (xDim > 0) {
            memcpy(xhArray, currentXArray + m * batchStrideX, xDim * sizeof(F16));
            memcpy(xhArray + xDim, lastBatchH, hDim * sizeof(F16));
        } else {
            intermediateH = tmpArray;
            xhArray = lastBatchH;
        }

        memcpy(intermediateH, bias[0], column * 4 * sizeof(F16));
        mvm_nkn32(fn, fk, (const F16 *)filter[0], xhArray, intermediateH);

        F16 *out_i = intermediateH;
        F16 *out_g = out_i + column;
        F16 *out_f = out_i + column * 2;
        F16 *out_o = out_i + column * 3;

        F16 *lastBatchState = lastStateArray + m * lastStateStride;
        F16 *currentBatchState = currentStateArray + m * currentStateStride;
        F16 *currentBatchH = currentHArray + m * currentHStride;
        F16 *currentOutput = outputArray + m * batchStrideH;

        F16 *tmpState, *tmpHH, *tmpH;
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
        for (; h < column - 7; h += 8) {
            float16x8_t out_i_v = vld1q_f16(out_i + h);
            float16x8_t out_g_v = vld1q_f16(out_g + h);
            float16x8_t out_f_v = vld1q_f16(out_f + h);
            float16x8_t out_o_v = vld1q_f16(out_o + h);
            float16x8_t C_v = vld1q_f16(lastBatchState + h);
            float16x8_t I_v = vsigmoidq_f16(out_i_v);
            float16x8_t F_v = vsigmoidq_f16(vaddq_f16(out_f_v, forgetBiasVector));
            float16x8_t O_v = vsigmoidq_f16(out_o_v);
            float16x8_t G_v = vtanhq_f16(out_g_v);
            C_v = vaddq_f16_f32(vmulq_f16(C_v, F_v), vmulq_f16(I_v, G_v));
            float16x8_t out_hidden_v = vmulq_f16(O_v, vtanhq_f16(C_v));
            vst1q_f16(tmpState + h, C_v);
            vst1q_f16(tmpHH + h, out_hidden_v);
        }
        for (; h < column; h++) {
            F16 C_s = lastBatchState[h];
            F16 I_s = 1.0 / (1.0 + exp(-out_i[h]));
            F16 F_s = 1.0 / (1.0 + exp(-(out_f[h] + forgetBias)));
            F16 O_s = 1.0 / (1.0 + exp(-out_o[h]));
            F16 G_s = tanh(out_g[h]);
            C_s = C_s * F_s + I_s * G_s;
            F16 value = O_s * tanh(C_s);
            tmpState[h] = C_s;
            tmpHH[h] = value;
        }
        if (rnnParamSpec.zoneoutCell != 0) {
            array_scale_f16(tmpState, tmpState, column, 1 - rnnParamSpec.zoneoutCell, 0);
            array_scale_f16(lastBatchState, lastBatchState, column, rnnParamSpec.zoneoutCell, 0);
            array_add_f16(tmpState, lastBatchState, currentBatchState, column);
        }

        if (rnnParamSpec.numProjection > 0) {
            memset(tmpH, 0, sizeof(F16) * hDim);
            mvm_nkn32(hDim / 32, rnnParamSpec.numProjection, (const F16 *)filter[1], tmpHH, tmpH);
        }
        if (rnnParamSpec.zoneoutOutput != 0) {
            if (rnnParamSpec.numProjection > 0) {
                array_scale_f16(tmpH, out_f, hDim, 1 - rnnParamSpec.zoneoutOutput, 0);
            } else {
                array_scale_f16(tmpHH, out_f, hDim, 1 - rnnParamSpec.zoneoutOutput, 0);
            }
            array_scale_f16(lastBatchH, lastBatchH, hDim, rnnParamSpec.zoneoutOutput, 0);
            array_add_f16(out_f, lastBatchH, currentBatchH, hDim);
        } else {
            memcpy(currentBatchH, currentOutput, sizeof(F16) * hDim);
        }
    }
    return SUCCESS;
}
