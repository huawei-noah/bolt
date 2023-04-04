// Copyright (C) 2023. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "cpu/arm/int8/tensor_computing_int8.h"
#include "cpu/arm/fp16/tensor_computing_fp16.h"
#include "cpu/arm/fp16/mvm_nkn32.h"
#include "cpu/arm/tensor_computing_arm.h"
#include "cpu/tensor_computing_cpu.h"
#include "blas_enhance.h"

EE lstmcell_int8(TensorDesc xDesc,
    const void *currentX,
    const TensorDesc *filterDesc,
    const void **filter,
    const TensorDesc *biasDesc,
    const void **bias,
    F32 *scale,
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
    I32 hDim = rnnParamSpec.num_outputs;
    I32 column = (rnnParamSpec.num_projection > 0) ? rnnParamSpec.num_projection
                                                   : rnnParamSpec.num_outputs;
    U32 steps = batchStrideH / hDim / (rnnParamSpec.bi_direction + 1);
    if (!(idt == DT_F16 && ((fdt == DT_F16) || (fdt == DT_I8)) && odt == DT_F16)) {
        CHECK_STATUS(NOT_MATCH);
    }
    if (!(4 * column == (I32)fn * 32 && (ix + oh) == fk && in == on)) {
        CHECK_STATUS(NOT_MATCH);
    }
    F32 forgetBias = rnnParamSpec.forget_bias;
    if (rnnParamSpec.activation_type != ACTIVATION_TANH) {
        CHECK_STATUS(NOT_SUPPORTED);
    }

    const F16 *currentXArray = (const F16 *)currentX;

    F16 *stateC = (F16 *)state;
    F16 *stateH = (F16 *)state + column;
    U32 stride = column + hDim;

    F16 *outputArray = (F16 *)output;
    F16 *xhArray = (F16 *)tmp;
    F16 *intermediateH = xhArray + (xDim + hDim);
    INT8 *quant = (INT8 *)(intermediateH + fn * 32);
    I32 *tmpOut = (I32 *)(quant + fk);
    float16x8_t forgetBiasVector = vdupq_n_f16(forgetBias);
    for (U32 m = 0; m < batch; m++) {
        F16 *curStateC = stateC + m * stride;
        F16 *curStateH = stateH + m * stride;
        F16 *currentOutput = outputArray + m * batchStrideH;
        const F16 *mBias = (const F16 *)bias[0] + m * steps * column * 4;

        if (xDim > 0) {
            UNI_MEMCPY(xhArray, currentXArray + m * batchStrideX, xDim * sizeof(F16));
            UNI_MEMCPY(xhArray + xDim, curStateH, hDim * sizeof(F16));
        } else {
            intermediateH = (F16 *)tmp;
            xhArray = curStateH;
        }

        TensorDesc cDesc = tensor1d(DT_I32, fn * 32);
        TensorDesc aDesc =
            tensor2df(DT_I8, matrix_vector_multiply_weight_format(DT_I8), fn * 32, fk);
        TensorDesc b1Desc = tensor1d(DT_I8, fk);
        F32 iScale = scale[0];
        TensorDesc b0Desc = tensor1d(DT_F16, fk);
        CHECK_STATUS(quantize_cpu(b0Desc, xhArray, &b1Desc, quant, &iScale, arch, 1));
        F32 oScale = iScale * scale[1];

        UNI_MEMSET(tmpOut, 0, sizeof(I32) * fn * 32);
        CHECK_STATUS(matrix_vector_multiply(aDesc, filter[0], b1Desc, quant, tmpBytes,
            quant + fk, cDesc, tmpOut, &oScale, arch));

        TensorDesc dDesc = tensor1d(DT_F16, fn * 32);
        dequantize_arm(cDesc, tmpOut, &oScale, biasDesc[0], (void *)mBias, dDesc, intermediateH);

        F16 *out_i = intermediateH;
        F16 *out_g = out_i + column;
        F16 *out_f = out_i + column * 2;
        F16 *out_o = out_i + column * 3;

        F16 *actOutC = curStateC;
        F16 *actOutH = currentOutput;
        F16 *projOut = currentOutput;
        if (rnnParamSpec.zoneout_cell != 0) {
            actOutC = out_i;
        }
        if (rnnParamSpec.num_projection > 0) {
            actOutH = out_g;
        }

        I32 h = 0;
        for (; h < column - 7; h += 8) {
            float16x8_t out_i_v = vld1q_f16(out_i + h);
            float16x8_t out_g_v = vld1q_f16(out_g + h);
            float16x8_t out_f_v = vld1q_f16(out_f + h);
            float16x8_t out_o_v = vld1q_f16(out_o + h);
            float16x8_t C_v = vld1q_f16(curStateC + h);
            float16x8_t I_v = vsigmoidq_f16(out_i_v);
            float16x8_t F_v = vsigmoidq_f16(vaddq_f16(out_f_v, forgetBiasVector));
            float16x8_t O_v = vsigmoidq_f16(out_o_v);
            float16x8_t G_v = vtanhq_f16(out_g_v);
            C_v = vaddq_f16_f32(vmulq_f16(C_v, F_v), vmulq_f16(I_v, G_v));
            float16x8_t out_hidden_v = vmulq_f16(O_v, vtanhq_f16(C_v));
            vst1q_f16(actOutC + h, C_v);
            vst1q_f16(actOutH + h, out_hidden_v);
        }
        for (; h < column; h++) {
            F16 C_s = curStateC[h];
            F16 I_s = 1.0 / (1.0 + exp(-out_i[h]));
            F16 F_s = 1.0 / (1.0 + exp(-(out_f[h] + forgetBias)));
            F16 O_s = 1.0 / (1.0 + exp(-out_o[h]));
            F16 G_s = tanh(out_g[h]);
            C_s = C_s * F_s + I_s * G_s;
            F16 value = O_s * tanh(C_s);
            actOutC[h] = C_s;
            actOutH[h] = value;
        }
        if (rnnParamSpec.zoneout_cell != 0) {
            array_scale_f16(actOutC, actOutC, column, 1 - rnnParamSpec.zoneout_cell, 0);
            array_scale_f16(curStateC, curStateC, column, rnnParamSpec.zoneout_cell, 0);
            array_add_f16(actOutC, curStateC, curStateC, column);
        }

        if (rnnParamSpec.num_projection > 0) {
            UNI_MEMSET(projOut, 0, sizeof(F16) * hDim);
            mvm_nkn32(hDim / 32, rnnParamSpec.num_projection, (const F16 *)filter[1], actOutH, projOut);
        }
        if (rnnParamSpec.zoneout_output != 0) {
            array_scale_f16(projOut, out_f, hDim, 1 - rnnParamSpec.zoneout_output, 0);
            array_scale_f16(curStateH, curStateH, hDim, rnnParamSpec.zoneout_output, 0);
            array_add_f16(out_f, curStateH, curStateH, hDim);
        } else {
            UNI_MEMCPY(curStateH, currentOutput, sizeof(F16) * hDim);
        }
    }
    return SUCCESS;
}