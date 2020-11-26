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
#include "cpu/x86/fp32/tensor_computing_fp32.h"

void mvm_nkn32_with_bias(
    U32 fn, U32 fk, const F32 *filterArray, const F32 *input, F32 *output, const F32 *bias)
{
#ifdef _USE_OPENMP
#pragma omp parallel for num_threads(OMP_NUM_THREADS)
#endif
    for (U32 n = 0; n < fn; ++n) {
        const F32 *f = filterArray + n * fk * 32;
        F32 *out = output + n * 32;
        const F32 *b = bias + n * 32;
        if (bias == nullptr) {
            __asm__ __volatile__("vxorps %%ymm0, %%ymm0, %%ymm0                     \n\t"
                                 "vxorps %%ymm1, %%ymm1, %%ymm1                     \n\t"
                                 "vxorps %%ymm2, %%ymm2, %%ymm2                     \n\t"
                                 "vxorps %%ymm3, %%ymm3, %%ymm3                     \n\t"
                                 :
                                 :
                                 : "%ymm0", "%ymm1", "%ymm2", "%ymm3");
        } else {
            __asm__ __volatile__("vmovups (%0), %%ymm0                     \n\t"
                                 "vmovups 0x20(%0), %%ymm1                     \n\t"
                                 "vmovups 0x40(%0), %%ymm2                     \n\t"
                                 "vmovups 0x60(%0), %%ymm3                     \n\t"
                                 :
                                 : "r"(b)
                                 : "%ymm0", "%ymm1", "%ymm2", "%ymm3");
        }
        __asm__ __volatile__("mov %1, %%rax                                     \n\t"
                             "mov %3, %%ecx                                     \n\t"
                             "shr $3, %%ecx                                     \n\t"
                             "je 1f                                \n\t"
                             ".align 16                                         \n\t"
                             "0:                                      \n\t"

                             "vmovups (%0), %%ymm4                             \n\t"
                             "vmovups 0x20(%0), %%ymm5                         \n\t"
                             "vmovups 0x40(%0), %%ymm6                         \n\t"
                             "vmovups 0x60(%0), %%ymm7                         \n\t"
                             "vbroadcastss 0x0(%%rax), %%ymm8                     \n\t"
                             "vmovups 0x80(%0), %%ymm9                             \n\t"
                             "vmovups 0xA0(%0), %%ymm10                         \n\t"
                             "vmovups 0xC0(%0), %%ymm11                         \n\t"
                             "vmovups 0xE0(%0), %%ymm12                         \n\t"
                             "vbroadcastss 0x4(%%rax), %%ymm13                     \n\t"
                             "vfmadd231ps %%ymm8, %%ymm4, %%ymm0              \n\t"
                             "vfmadd231ps %%ymm8, %%ymm5, %%ymm1              \n\t"
                             "vfmadd231ps %%ymm8, %%ymm6, %%ymm2              \n\t"
                             "vfmadd231ps %%ymm8, %%ymm7, %%ymm3              \n\t"

                             "vmovups 0x100(%0), %%ymm4                             \n\t"
                             "vmovups 0x120(%0), %%ymm5                         \n\t"
                             "vmovups 0x140(%0), %%ymm6                         \n\t"
                             "vmovups 0x160(%0), %%ymm7                         \n\t"
                             "vbroadcastss 0x8(%%rax), %%ymm8                     \n\t"
                             "vfmadd231ps %%ymm13, %%ymm9, %%ymm0              \n\t"
                             "vfmadd231ps %%ymm13, %%ymm10, %%ymm1              \n\t"
                             "vfmadd231ps %%ymm13, %%ymm11, %%ymm2              \n\t"
                             "vfmadd231ps %%ymm13, %%ymm12, %%ymm3              \n\t"

                             "vmovups 0x180(%0), %%ymm9                              \n\t"
                             "vmovups 0x1A0(%0), %%ymm10                         \n\t"
                             "vmovups 0x1C0(%0), %%ymm11                         \n\t"
                             "vmovups 0x1E0(%0), %%ymm12                         \n\t"
                             "vbroadcastss 0xC(%%rax), %%ymm13                     \n\t"
                             "vfmadd231ps %%ymm8, %%ymm4, %%ymm0              \n\t"
                             "vfmadd231ps %%ymm8, %%ymm5, %%ymm1              \n\t"
                             "vfmadd231ps %%ymm8, %%ymm6, %%ymm2              \n\t"
                             "vfmadd231ps %%ymm8, %%ymm7, %%ymm3              \n\t"

                             "vmovups 0x200(%0), %%ymm4                             \n\t"
                             "vmovups 0x220(%0), %%ymm5                         \n\t"
                             "vmovups 0x240(%0), %%ymm6                         \n\t"
                             "vmovups 0x260(%0), %%ymm7                         \n\t"
                             "vbroadcastss 0x10(%%rax), %%ymm8                     \n\t"
                             "vfmadd231ps %%ymm13, %%ymm9 , %%ymm0              \n\t"
                             "vfmadd231ps %%ymm13, %%ymm10, %%ymm1              \n\t"
                             "vfmadd231ps %%ymm13, %%ymm11, %%ymm2              \n\t"
                             "vfmadd231ps %%ymm13, %%ymm12, %%ymm3              \n\t"

                             "vmovups 0x280(%0), %%ymm9                              \n\t"
                             "vmovups 0x2A0(%0), %%ymm10                         \n\t"
                             "vmovups 0x2C0(%0), %%ymm11                         \n\t"
                             "vmovups 0x2E0(%0), %%ymm12                         \n\t"
                             "vbroadcastss 0x14(%%rax), %%ymm13                     \n\t"
                             "vfmadd231ps %%ymm8, %%ymm4, %%ymm0              \n\t"
                             "vfmadd231ps %%ymm8, %%ymm5, %%ymm1              \n\t"
                             "vfmadd231ps %%ymm8, %%ymm6, %%ymm2              \n\t"
                             "vfmadd231ps %%ymm8, %%ymm7, %%ymm3              \n\t"

                             "vmovups 0x300(%0), %%ymm4                             \n\t"
                             "vmovups 0x320(%0), %%ymm5                         \n\t"
                             "vmovups 0x340(%0), %%ymm6                         \n\t"
                             "vmovups 0x360(%0), %%ymm7                         \n\t"
                             "vbroadcastss 0x18(%%rax), %%ymm8                     \n\t"
                             "vfmadd231ps %%ymm13, %%ymm9 , %%ymm0              \n\t"
                             "vfmadd231ps %%ymm13, %%ymm10, %%ymm1              \n\t"
                             "vfmadd231ps %%ymm13, %%ymm11, %%ymm2              \n\t"
                             "vfmadd231ps %%ymm13, %%ymm12, %%ymm3              \n\t"

                             "vmovups 0x380(%0), %%ymm9                              \n\t"
                             "vmovups 0x3A0(%0), %%ymm10                         \n\t"
                             "vmovups 0x3C0(%0), %%ymm11                         \n\t"
                             "vmovups 0x3E0(%0), %%ymm12                         \n\t"
                             "vbroadcastss 0x1C(%%rax), %%ymm13                     \n\t"
                             "vfmadd231ps %%ymm8, %%ymm4, %%ymm0              \n\t"
                             "vfmadd231ps %%ymm8, %%ymm5, %%ymm1              \n\t"
                             "vfmadd231ps %%ymm8, %%ymm6, %%ymm2              \n\t"
                             "vfmadd231ps %%ymm8, %%ymm7, %%ymm3              \n\t"

                             "vfmadd231ps %%ymm13, %%ymm9 , %%ymm0              \n\t"
                             "vfmadd231ps %%ymm13, %%ymm10, %%ymm1              \n\t"
                             "vfmadd231ps %%ymm13, %%ymm11, %%ymm2              \n\t"
                             "vfmadd231ps %%ymm13, %%ymm12, %%ymm3              \n\t"

                             "add $0x400, %0                                    \n\t"
                             "add $0x20, %%rax                                     \n\t"

                             "sub $1, %%ecx                                     \n\t"
                             "jg 0b                                    \n\t"
                             ".align 16                                         \n\t"
                             "1:                                  \n\t"

                             "mov %3, %%ecx                                     \n\t"
                             "and $7, %%ecx                                     \n\t"
                             "je 3f                         \n\t"
                             "2:                               \n\t"
                             "vmovups (%0), %%ymm4                             \n\t"
                             "vmovups 0x20(%0), %%ymm5                         \n\t"
                             "vmovups 0x40(%0), %%ymm6                         \n\t"
                             "vmovups 0x60(%0), %%ymm7                         \n\t"
                             "vbroadcastss (%%rax), %%ymm8                     \n\t"
                             "vfmadd231ps %%ymm8, %%ymm4, %%ymm0              \n\t"
                             "vfmadd231ps %%ymm8, %%ymm5, %%ymm1              \n\t"
                             "vfmadd231ps %%ymm8, %%ymm6, %%ymm2              \n\t"
                             "vfmadd231ps %%ymm8, %%ymm7, %%ymm3              \n\t"
                             "add $0x80, %0                                     \n\t"
                             "add $0x4, %%rax                                     \n\t"
                             "sub $1, %%ecx                                     \n\t"
                             "jg 2b                             \n\t"

                             "3:                           \n\t"
                             "vmovups %%ymm0,  (%2)                             \n\t"
                             "vmovups %%ymm1,  0x20(%2)                         \n\t"
                             "vmovups %%ymm2,  0x40(%2)                         \n\t"
                             "vmovups %%ymm3,  0x60(%2)                         \n\t"

                             : "+r"(f)
                             : "r"(input), "r"(out), "r"(fk)
                             : "%rax", "%ecx", "%ymm0", "%ymm1", "%ymm2", "%ymm3", "%ymm4", "%ymm5",
                             "%ymm6", "%ymm7", "%ymm8", "%ymm9", "%ymm10", "%ymm11", "%ymm12",
                             "%ymm13", "memory");
    }
}

EE rnncell_fp32(TensorDesc xDesc,
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
    if (nullptr == currentX || nullptr == filter || nullptr == bias || nullptr == state ||
        nullptr == tmp || nullptr == output) {
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
    __m256 forgetBiasVector = _mm256_set1_ps(forgetBias);
    for (U32 m = 0; m < batch; m++) {
        F32 *lastBatchH = lastHArray + m * lastHStride;
        memcpy(xhArray, currentXArray + m * batchStrideX, xDim * sizeof(F32));
        memcpy(xhArray + xDim, lastBatchH, hDim * sizeof(F32));
        mvm_nkn32_with_bias(
            fn, fk, (const F32 *)filter[0], xhArray, intermediateH, (const F32 *)bias[0]);

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
        for (; h < column - 7; h += 8) {
            __m256 out_i_v = _mm256_loadu_ps(out_i + h);
            __m256 out_g_v = _mm256_loadu_ps(out_g + h);
            __m256 out_f_v = _mm256_loadu_ps(out_f + h);
            __m256 out_o_v = _mm256_loadu_ps(out_o + h);
            __m256 C_v = _mm256_loadu_ps(lastBatchState + h);
            __m256 I_v = _mm256_sigmod_ps(out_i_v);
            __m256 F_v = _mm256_sigmod_ps(_mm256_add_ps(out_f_v, forgetBiasVector));
            __m256 O_v = _mm256_sigmod_ps(out_o_v);
            __m256 G_v = _mm256_tanh_ps(out_g_v);
            C_v = _mm256_add_ps(_mm256_mul_ps(C_v, F_v), _mm256_mul_ps(I_v, G_v));
            __m256 out_hidden_v = _mm256_mul_ps(O_v, _mm256_tanh_ps(C_v));
            _mm256_storeu_ps(tmpState + h, C_v);
            _mm256_storeu_ps(tmpHH + h, out_hidden_v);
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
            mvm_nkn32_with_bias(hDim / 32, rnnParamSpec.numProjection, (const F32 *)filter[1],
                tmpHH, tmpH, nullptr);
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
