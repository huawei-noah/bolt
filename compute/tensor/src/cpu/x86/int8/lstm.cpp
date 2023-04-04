// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "cpu/x86/int8/tensor_computing_int8.h"
#include "cpu/x86/fp32/x86_functions_fp32.h"
#include "cpu/x86/fp32/mvm_nkn32.h"
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
    if (!(idt == DT_F32 && ((fdt == DT_I8) || (fdt == DT_F32)) && odt == DT_F32)) {
        CHECK_STATUS(NOT_MATCH);
    }
    if (!(4 * column == (I32)fn * 32 && (ix + oh) == fk && in == on)) {
        CHECK_STATUS(NOT_MATCH);
    }
    F32 forgetBias = rnnParamSpec.forget_bias;
    if (rnnParamSpec.activation_type != ACTIVATION_TANH) {
        CHECK_STATUS(NOT_SUPPORTED);
    }

    const F32 *currentXArray = (const F32 *)currentX;

    F32 *stateC = (F32 *)state;
    F32 *stateH = (F32 *)state + column;
    U32 stride = column + hDim;

    F32 *outputArray = (F32 *)output;
    F32 *xhArray = (F32 *)tmp;
    F32 *intermediateH = xhArray + (xDim + hDim);
    UINT8 *quant = (UINT8 *)(intermediateH + fn * 32);
    __m256 forgetBiasVector = _mm256_set1_ps(forgetBias);

    for (U32 m = 0; m < batch; m++) {
        F32 *curStateC = stateC + m * stride;
        F32 *curStateH = stateH + m * stride;
        F32 *currentOutput = outputArray + m * batchStrideH;
        const F32 *mBias = (const F32 *)bias[0] + m * steps * column * 4;

        if (xDim > 0) {
            UNI_MEMCPY(xhArray, currentXArray + m * batchStrideX, xDim * sizeof(F32));
            UNI_MEMCPY(xhArray + xDim, curStateH, hDim * sizeof(F32));
        } else {
            intermediateH = (F32 *)tmp;
            xhArray = curStateH;
        }
        TensorDesc cDesc = tensor1d(DT_F32, fn * 32);
        TensorDesc aDesc =
            tensor2df(DT_I8, matrix_vector_multiply_weight_format(DT_I8), fn * 32, fk);
        TensorDesc b1Desc = tensor1d(DT_U8_Q, fk);
        F32 iScale = scale[0];
        TensorDesc b0Desc = tensor1d(DT_F32, fk);
        CHECK_STATUS(quantize_cpu(b0Desc, xhArray, &b1Desc, quant, &iScale, arch, 1));
        F32 oScale = iScale * scale[1];

        UNI_MEMSET(intermediateH, 0, sizeof(F32) * fn * 32);
        I32 *offset = (I32 *)((INT8 *)filter[0] + fn * 32 * UNI_ALIGN(fk, 8));
        CHECK_STATUS(matrix_vector_multiply(aDesc, filter[0], b1Desc, quant, tmpBytes,
            offset, cDesc, intermediateH, &oScale, arch));

        array_add_f32(intermediateH, mBias, intermediateH, fn * 32);

        F32 *out_i = (F32 *)intermediateH;
        F32 *out_g = out_i + column;
        F32 *out_f = out_i + column * 2;
        F32 *out_o = out_i + column * 3;

        F32 *actOutC = curStateC;
        F32 *actOutH = currentOutput;
        F32 *projOut = currentOutput;
        if (rnnParamSpec.zoneout_cell != 0) {
            actOutC = out_i;
        }
        if (rnnParamSpec.num_projection > 0) {
            actOutH = out_g;
        }

        I32 h = 0;
        for (; h < column - 7; h += 8) {
            __m256 out_i_v = _mm256_loadu_ps(out_i + h);
            __m256 out_g_v = _mm256_loadu_ps(out_g + h);
            __m256 out_f_v = _mm256_loadu_ps(out_f + h);
            __m256 out_o_v = _mm256_loadu_ps(out_o + h);
            __m256 C_v = _mm256_loadu_ps(curStateC + h);
            __m256 I_v = _mm256_sigmod_ps(out_i_v);
            __m256 F_v = _mm256_sigmod_ps(_mm256_add_ps(out_f_v, forgetBiasVector));
            __m256 O_v = _mm256_sigmod_ps(out_o_v);
            __m256 G_v = _mm256_tanh_ps(out_g_v);
            C_v = _mm256_add_ps(_mm256_mul_ps(C_v, F_v), _mm256_mul_ps(I_v, G_v));
            __m256 out_hidden_v = _mm256_mul_ps(O_v, _mm256_tanh_ps(C_v));
            _mm256_storeu_ps(actOutC + h, C_v);
            _mm256_storeu_ps(actOutH + h, out_hidden_v);
        }
        for (; h < column; h++) {
            F32 C_s = curStateC[h];
            F32 I_s = 1.0 / (1.0 + exp(-out_i[h]));
            F32 F_s = 1.0 / (1.0 + exp(-(out_f[h] + forgetBias)));
            F32 O_s = 1.0 / (1.0 + exp(-out_o[h]));
            F32 G_s = tanh(out_g[h]);
            C_s = C_s * F_s + I_s * G_s;
            F32 value = O_s * tanh(C_s);
            actOutC[h] = C_s;
            actOutH[h] = value;
        }
        if (rnnParamSpec.zoneout_cell != 0) {
            array_scale_f32(actOutC, actOutC, column, 1 - rnnParamSpec.zoneout_cell, 0);
            array_scale_f32(curStateC, curStateC, column, rnnParamSpec.zoneout_cell, 0);
            array_add_f32(actOutC, curStateC, curStateC, column);
        }

        if (rnnParamSpec.num_projection > 0) {
            mvm_nkn32_with_bias(hDim / 32, rnnParamSpec.num_projection, (const F32 *)filter[1],
                actOutH, projOut, nullptr);
        }

        if (rnnParamSpec.zoneout_output != 0) {
            array_scale_f32(projOut, out_f, hDim, 1 - rnnParamSpec.zoneout_output, 0);
            array_scale_f32(curStateH, curStateH, hDim, rnnParamSpec.zoneout_output, 0);
            array_add_f32(out_f, curStateH, curStateH, hDim);
        } else {
            UNI_MEMCPY(curStateH, currentOutput, sizeof(F32) * hDim);
        }
    }
    return SUCCESS;
}
