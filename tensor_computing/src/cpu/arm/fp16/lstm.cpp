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
#include "blas-enhance.h"

void mvm_nkn32(U32 fn, U32 fk, const F16* filterArray, F16* input, F16* output) {
    for (U32 n = 0; n < fn; n++) {
        F16 *in = input;
        const F16 *f = filterArray + n*fk*32;
        __asm__ __volatile__(
            "ldr s0, [%[in]]\n"
            "ldr q1, [%[out]]\n"
            "ldr q2, [%[out], #16]\n"
            "ldr q3, [%[out], #32]\n"
            "ldr q4, [%[out], #48]\n"
            "mov x0, %[k]\n"
            "ldr q5, [%[f]]\n"
            "ldr q6, [%[f], #16]\n"
            "ldr q7, [%[f], #32]\n"
            "ldr q8, [%[f], #48]\n"
            "0:\n"
            "prfm pldl2strm, [%[f], #4096]\n"
            "prfm pldl1strm, [%[f], #1024]\n"
            "ldr d9, [%[f], #64]\n"
            "fmla v1.8h, v5.8h, v0.h[0]\n"
            "ldr x9, [%[f], #72]\n"
            "ins v9.d[1], x9\n"
            "ldr d10, [%[f], #80]\n"
            "fmla v2.8h, v6.8h, v0.h[0]\n"
            "ldr x10, [%[f], #88]\n"
            "ins v10.d[1], x10\n"
            "ldr d11, [%[f], #96]\n"
            "fmla v3.8h, v7.8h, v0.h[0]\n"
            "ldr x11, [%[f], #104]\n"
            "ins v11.d[1], x11\n"
            "ldr d12, [%[f], #112]\n"
            "fmla v4.8h, v8.8h, v0.h[0]\n"
            "ldr x12, [%[f], #120]\n"
            "ins v12.d[1], x12\n"

            "ldr d5, [%[f], #128]\n"
            "fmla v1.8h, v9.8h, v0.h[1]\n"
            "ldr x5, [%[f], #136]\n"
            "ins v5.d[1], x5\n"
            "ldr d6, [%[f], #144]\n"
            "fmla v2.8h, v10.8h, v0.h[1]\n"
            "ldr x6, [%[f], #152]\n"
            "ins v6.d[1], x6\n"
            "ldr d7, [%[f], #160]\n"
            "fmla v3.8h, v11.8h, v0.h[1]\n"
            "ldr x7, [%[f], #168]\n"
            "ins v7.d[1], x7\n"
            "ldr d8, [%[f], #176]\n"
            "fmla v4.8h, v12.8h, v0.h[1]\n"
            "ldr x8, [%[f], #184]\n"
            "add %[in], %[in], #4\n"
            "ins v8.d[1], x8\n"
            "add %[f], %[f], #128\n"
            "ldr s0, [%[in]]\n"
            "sub x0, x0, #2\n"

            "cmp x0, #3\n"
            "bgt 0b\n"
            "ldr  q9, [%[f], #64]\n"
            "ldr q10, [%[f], #80]\n"
            "ldr q11, [%[f], #96]\n"
            "ldr q12, [%[f], #112]\n"
            "fmla v1.8h,  v5.8h, v0.h[0]\n"
            "fmla v2.8h,  v6.8h, v0.h[0]\n"
            "fmla v3.8h,  v7.8h, v0.h[0]\n"
            "fmla v4.8h,  v8.8h, v0.h[0]\n"
            "fmla v1.8h,  v9.8h, v0.h[1]\n"
            "fmla v2.8h, v10.8h, v0.h[1]\n"
            "fmla v3.8h, v11.8h, v0.h[1]\n"
            "fmla v4.8h, v12.8h, v0.h[1]\n"
            "cmp x0, #3\n"
            "bne 1f\n"
            "ldr h0, [%[in], #4]\n"
            "ldr q5, [%[f], #128]\n"
            "ldr q6, [%[f], #144]\n"
            "ldr q7, [%[f], #160]\n"
            "ldr q8, [%[f], #176]\n"
            "fmla v1.8h,  v5.8h, v0.h[0]\n"
            "fmla v2.8h,  v6.8h, v0.h[0]\n"
            "fmla v3.8h,  v7.8h, v0.h[0]\n"
            "fmla v4.8h,  v8.8h, v0.h[0]\n"

            "1:\n"
            "str q1, [%[out]]\n"
            "str q2, [%[out], #16]\n"
            "str q3, [%[out], #32]\n"
            "str q4, [%[out], #48]\n"
            :[out]"+r"(output),
             [f]"+r"(f),
             [in]"+r"(in)
            :[k]"r"((I64)fk)
            :"memory", "cc", "x0", "x5", "x6", "x7", "x8", "x9", "x10", "x11", "x12",
                "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10",
                "v11", "v12"
        );
        output += 32;
    }
}

EE lstmcell_fp16(TensorDesc xDesc, const void* currentX,
    TensorDesc filterDesc, const void* filter,
    TensorDesc biasDesc, const void* bias,
    void *state,
    U32 tmpBytes, void *tmp,
    LSTMDesc lstmDesc, U32 batchStrideX, U32 batchStrideH,
    TensorDesc hDesc, void* output,
    Arch arch)
{
    UNUSED(biasDesc);
    UNUSED(tmpBytes);
    UNUSED(arch);
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
    fn /= 32;

    U32 batch = in;
    I32 xDim  = ix;
    I32 hDim  = lstmDesc.numOutput;
    I32 column = (lstmDesc.numProjection > 0) ? lstmDesc.numProjection : lstmDesc.numOutput;
    if (!(idt == DT_F16 && fdt == DT_F16 && odt == DT_F16)) {
        CHECK_STATUS(NOT_MATCH);
    }
    if (!(4*column == (I32)fn*32 && (ix+oh) == fk && in == on)) {
        CHECK_STATUS(NOT_MATCH);
    }
    F32 forgetBias = lstmDesc.forgetBias;
    ActivationMode activationMode = lstmDesc.activationMode;
    if (activationMode != ACTIVATION_TANH)
        CHECK_STATUS(NOT_SUPPORTED);

    const F16 *currentXArray   = (const F16*)currentX;
    const F16 *filterArray     = (const F16*)filter;
    const F16 *biasArray       = (const F16*)bias;
    const F16 *projectionArray = (const F16*)filter + (fn * 32 * fk);
    F16 *lastStateArray = (F16*)state;
    F16 *lastHArray     = lastStateArray + column;
    F16 *tmpArray       = (F16*)tmp;
    F16 *currentStateArray = (F16*)state;
    F16 *currentHArray     = currentStateArray + column;
    F16 *outputArray       = (F16*)output;
    F16 *xhArray           = tmpArray;
    F16 *intermediateH     = xhArray + (xDim + hDim);
    U32 lastStateStride    = column + hDim;
    U32 lastHStride        = column + hDim;
    U32 currentStateStride = column + hDim;
    U32 currentHStride     = column + hDim;
    float16x8_t forgetBiasVector = vdupq_n_f16(forgetBias);
    for (U32 m = 0; m < batch; m++) {
        F16 *lastBatchH = lastHArray + m * lastHStride;
        memcpy(xhArray, currentXArray+m*batchStrideX, xDim*sizeof(F16));
        memcpy(xhArray+xDim, lastBatchH, hDim*sizeof(F16));

        memcpy(intermediateH, biasArray, column * 4 * sizeof(F16));
        mvm_nkn32(fn, fk, filterArray, xhArray, intermediateH);

        F16 *out_i = intermediateH;
        F16 *out_g = out_i + column;
        F16 *out_f = out_i + column * 2;
        F16 *out_o = out_i + column * 3;

        F16 *lastBatchState = lastStateArray + m * lastStateStride;
        F16 *currentBatchState = currentStateArray + m * currentStateStride;
        F16 *currentBatchH = currentHArray + m * currentHStride;
        F16 *currentOutput = outputArray + m * batchStrideH;

        F16* tmpState, *tmpHH, *tmpH;
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

        I32 h = 0;
        for (; h < column-7; h+=8) {
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
        if (lstmDesc.zoneoutCell != 0) {
            array_scale_f16(tmpState, tmpState, column, 1-lstmDesc.zoneoutCell, 0);
            array_scale_f16(lastBatchState, lastBatchState, column, lstmDesc.zoneoutCell, 0);
            array_add_f16(tmpState, lastBatchState, currentBatchState, column);
        }
        if (lstmDesc.zoneoutOutput != 0) {
            array_scale_f16(tmpHH, tmpH, column, 1-lstmDesc.zoneoutOutput, 0);
            array_scale_f16(lastBatchH, lastBatchH, column, lstmDesc.zoneoutOutput, 0);
            array_add_f16(tmpH, lastBatchH, currentBatchH, column);
        }
        if (lstmDesc.numProjection > 0) {
            memset(currentBatchH, 0, sizeof(F16) * hDim);
            mvm_nkn32(hDim/32, lstmDesc.numProjection, projectionArray, tmpHH, currentBatchH);
            tmpHH = currentBatchH;
        }
        memcpy(currentOutput, tmpHH, sizeof(F16) * hDim);
    }
    return SUCCESS;
}
