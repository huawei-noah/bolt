// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _H_CONVOLUTION_GEMM
#define _H_CONVOLUTION_GEMM

#include "sys.h"
#include "uni.h"
#include "arm_functions_int8.h"

#ifdef _USE_INT8
template <typename OT>
EE convolution_gemm_A55(TensorDesc inputDesc,
    const void *input,
    F16 *inputScale,
    TensorDesc filterDesc,
    const void *filter,
    F16 *filterScale,
    ConvolutionParamSpec convParamSpec,
    TensorDesc biasDesc,
    const void *bias,
    U32 tmpBytes,
    void *tmp,
    TensorDesc outputDesc,
    void *output,
    F16 *outputScale,
    ActivationParamSpec am);

template <typename OT>
EE convolution_gemm_A76(TensorDesc inputDesc,
    const void *input,
    F16 *inputScale,
    TensorDesc filterDesc,
    const void *filter,
    F16 *filterScale,
    ConvolutionParamSpec convParamSpec,
    TensorDesc biasDesc,
    const void *bias,
    U32 tmpBytes,
    void *tmp,
    TensorDesc outputDesc,
    void *output,
    F16 *outputScale,
    ActivationParamSpec am);

inline EE convolution_gemm(TensorDesc inputDesc,
    const void *input,
    F16 *inputScale,
    TensorDesc filterDesc,
    const void *filter,
    F16 *filterScale,
    ConvolutionParamSpec convParamSpec,
    TensorDesc biasDesc,
    const void *bias,
    U32 tmpBytes,
    void *tmp,
    TensorDesc outputDesc,
    void *output,
    F16 *outputScale,
    ActivationParamSpec am,
    Arch arch)
{
    EE ret = SUCCESS;
    switch (arch) {
        case ARM_A55: {
            ret = convolution_gemm_A55<INT8>(inputDesc, input, inputScale, filterDesc, filter,
                filterScale, convParamSpec, biasDesc, bias, tmpBytes, tmp, outputDesc, output,
                outputScale, am);
            break;
        }
        case ARM_A76: {
            ret = convolution_gemm_A76<INT8>(inputDesc, input, inputScale, filterDesc, filter,
                filterScale, convParamSpec, biasDesc, bias, tmpBytes, tmp, outputDesc, output,
                outputScale, am);
            break;
        }
        default: {
            ret = NOT_SUPPORTED;
            break;
        }
    }
    return ret;
}

inline EE quantize_I32(U32 num_v, I32 *out_d, I32 factor, F32 scale, INT8 *out_q)
{
    // num_v is the number of q-form vectors (I32)
    I32 *arr_d = out_d;
    I32 fact = factor;
    INT8 *arr_q = out_q;
    U32 i28 = num_v / 28;  // The number of iterations, each handling 28 vectors

    if (i28 > 0) {
        __asm__ __volatile__("ld4 {v1.4s, v2.4s, v3.4s, v4.4s}, [%[out_d]], #64\n"
                             "ldr s0, [%[factor]]\n"
                             "ld4 {v5.4s, v6.4s, v7.4s, v8.4s}, [%[out_d]], #64\n"
                             "mov x1, %[i]\n"
                             "ld4 {v9.4s, v10.4s, v11.4s, v12.4s}, [%[out_d]], #64\n"
                             "dup v0.4s, v0.s[0]\n"
                             "ld4 {v13.4s, v14.4s, v15.4s, v16.4s}, [%[out_d]], #64\n"
                             "ld4 {v17.4s, v18.4s, v19.4s, v20.4s}, [%[out_d]], #64\n"
                             "ld4 {v21.4s, v22.4s, v23.4s, v24.4s}, [%[out_d]], #64\n"

                             "0:\n"
                             "ld4 {v25.4s, v26.4s, v27.4s, v28.4s}, [%[out_d]], #64\n"
                             "subs x1, x1, #1\n"

                             "mul v4.4s, v4.4s, v0.4s\n"
                             "mul v3.4s, v3.4s, v0.4s\n"
                             "mul v2.4s, v2.4s, v0.4s\n"
                             "mul v1.4s, v1.4s, v0.4s\n"

                             "mul v8.4s, v8.4s, v0.4s\n"
                             "sri v4.4s, v3.4s, #8\n"
                             "mul v7.4s, v7.4s, v0.4s\n"
                             "sri v2.4s, v1.4s, #8\n"
                             "mul v6.4s, v6.4s, v0.4s\n"
                             "mul v5.4s, v5.4s, v0.4s\n"
                             "sri v4.4s, v2.4s, #16\n"

                             "mul v12.4s, v12.4s, v0.4s\n"
                             "sri v8.4s, v7.4s, #8\n"
                             "mul v11.4s, v11.4s, v0.4s\n"
                             "sri v6.4s, v5.4s, #8\n"
                             "mul v10.4s, v10.4s, v0.4s\n"
                             "str q4, [%[out_q]], #16\n"
                             "ld4 {v1.4s, v2.4s, v3.4s, v4.4s}, [%[out_d]], #64\n"
                             "mul v9.4s, v9.4s, v0.4s\n"
                             "sri v8.4s, v6.4s, #16\n"

                             "mul v16.4s, v16.4s, v0.4s\n"
                             "sri v12.4s, v11.4s, #8\n"
                             "mul v15.4s, v15.4s, v0.4s\n"
                             "sri v10.4s, v9.4s, #8\n"
                             "mul v14.4s, v14.4s, v0.4s\n"
                             "str q8, [%[out_q]], #16\n"
                             "ld4 {v5.4s, v6.4s, v7.4s, v8.4s}, [%[out_d]], #64\n"
                             "mul v13.4s, v13.4s, v0.4s\n"
                             "sri v12.4s, v10.4s, #16\n"

                             "mul v20.4s, v20.4s, v0.4s\n"
                             "sri v16.4s, v15.4s, #8\n"
                             "mul v19.4s, v19.4s, v0.4s\n"
                             "sri v14.4s, v13.4s, #8\n"
                             "mul v18.4s, v18.4s, v0.4s\n"
                             "str q12, [%[out_q]], #16\n"
                             "ld4 {v9.4s, v10.4s, v11.4s, v12.4s}, [%[out_d]], #64\n"
                             "mul v17.4s, v17.4s, v0.4s\n"
                             "sri v16.4s, v14.4s, #16\n"

                             "mul v24.4s, v24.4s, v0.4s\n"
                             "sri v20.4s, v19.4s, #8\n"
                             "mul v23.4s, v23.4s, v0.4s\n"
                             "sri v18.4s, v17.4s, #8\n"
                             "mul v22.4s, v22.4s, v0.4s\n"
                             "str q16, [%[out_q]], #16\n"
                             "ld4 {v13.4s, v14.4s, v15.4s, v16.4s}, [%[out_d]], #64\n"
                             "mul v21.4s, v21.4s, v0.4s\n"
                             "sri v20.4s, v18.4s, #16\n"

                             "mul v28.4s, v28.4s, v0.4s\n"
                             "sri v24.4s, v23.4s, #8\n"
                             "mul v27.4s, v27.4s, v0.4s\n"
                             "sri v22.4s, v21.4s, #8\n"
                             "mul v26.4s, v26.4s, v0.4s\n"
                             "str q20, [%[out_q]], #16\n"
                             "ld4 {v17.4s, v18.4s, v19.4s, v20.4s}, [%[out_d]], #64\n"
                             "mul v25.4s, v25.4s, v0.4s\n"
                             "sri v24.4s, v22.4s, #16\n"

                             "sri v28.4s, v27.4s, #8\n"
                             "sri v26.4s, v25.4s, #8\n"
                             "str q24, [%[out_q]], #16\n"
                             "sri v28.4s, v26.4s, #16\n"
                             "ld4 {v21.4s, v22.4s, v23.4s, v24.4s}, [%[out_d]], #64\n"
                             "str q28, [%[out_q]], #16\n"
                             "bne 0b\n"
                             : [out_d] "+r"(arr_d), [out_q] "+r"(arr_q)
                             : [factor] "r"(&fact), [i] "r"((I64)i28)
                             : "memory", "cc", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8",
                             "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18",
                             "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28",
                             "x1");
        arr_d -= 96;  // Prefetched 24 extra vectors
    }

    U32 remainder = num_v - i28 * 28;

    if (remainder % 4) {
        for (U32 i = 0; i < 8; i++) {
            arr_q[i] = round_towards_zero(arr_d[i] * scale);
        }
        arr_d += 8;
        arr_q += 8;
        remainder -= 2;
    }

    switch (remainder) {
        case 24: {
            __asm__ __volatile__("ldr s0, [%[factor]]\n"
                                 "dup v0.4s, v0.s[0]\n"

                                 "ld4 {v1.4s, v2.4s, v3.4s, v4.4s}, [%[out_d]], #64\n"
                                 "ld4 {v5.4s, v6.4s, v7.4s, v8.4s}, [%[out_d]], #64\n"
                                 "ld4 {v9.4s, v10.4s, v11.4s, v12.4s}, [%[out_d]], #64\n"
                                 "ld4 {v13.4s, v14.4s, v15.4s, v16.4s}, [%[out_d]], #64\n"
                                 "ld4 {v17.4s, v18.4s, v19.4s, v20.4s}, [%[out_d]], #64\n"
                                 "ld4 {v21.4s, v22.4s, v23.4s, v24.4s}, [%[out_d]], #64\n"

                                 "mul v4.4s, v4.4s, v0.4s\n"
                                 "mul v3.4s, v3.4s, v0.4s\n"
                                 "mul v2.4s, v2.4s, v0.4s\n"
                                 "mul v1.4s, v1.4s, v0.4s\n"

                                 "mul v8.4s, v8.4s, v0.4s\n"
                                 "sri v4.4s, v3.4s, #8\n"
                                 "mul v7.4s, v7.4s, v0.4s\n"
                                 "sri v2.4s, v1.4s, #8\n"
                                 "mul v6.4s, v6.4s, v0.4s\n"
                                 "mul v5.4s, v5.4s, v0.4s\n"
                                 "sri v4.4s, v2.4s, #16\n"

                                 "mul v12.4s, v12.4s, v0.4s\n"
                                 "sri v8.4s, v7.4s, #8\n"
                                 "mul v11.4s, v11.4s, v0.4s\n"
                                 "sri v6.4s, v5.4s, #8\n"
                                 "mul v10.4s, v10.4s, v0.4s\n"
                                 "str q4, [%[out_q]], #16\n"
                                 "mul v9.4s, v9.4s, v0.4s\n"
                                 "sri v8.4s, v6.4s, #16\n"

                                 "mul v16.4s, v16.4s, v0.4s\n"
                                 "sri v12.4s, v11.4s, #8\n"
                                 "mul v15.4s, v15.4s, v0.4s\n"
                                 "sri v10.4s, v9.4s, #8\n"
                                 "mul v14.4s, v14.4s, v0.4s\n"
                                 "str q8, [%[out_q]], #16\n"
                                 "mul v13.4s, v13.4s, v0.4s\n"
                                 "sri v12.4s, v10.4s, #16\n"

                                 "mul v20.4s, v20.4s, v0.4s\n"
                                 "sri v16.4s, v15.4s, #8\n"
                                 "mul v19.4s, v19.4s, v0.4s\n"
                                 "sri v14.4s, v13.4s, #8\n"
                                 "mul v18.4s, v18.4s, v0.4s\n"
                                 "str q12, [%[out_q]], #16\n"
                                 "mul v17.4s, v17.4s, v0.4s\n"
                                 "sri v16.4s, v14.4s, #16\n"

                                 "mul v24.4s, v24.4s, v0.4s\n"
                                 "sri v20.4s, v19.4s, #8\n"
                                 "mul v23.4s, v23.4s, v0.4s\n"
                                 "sri v18.4s, v17.4s, #8\n"
                                 "mul v22.4s, v22.4s, v0.4s\n"
                                 "str q16, [%[out_q]], #16\n"
                                 "mul v21.4s, v21.4s, v0.4s\n"
                                 "sri v20.4s, v18.4s, #16\n"

                                 "sri v24.4s, v23.4s, #8\n"
                                 "sri v22.4s, v21.4s, #8\n"
                                 "str q20, [%[out_q]], #16\n"
                                 "sri v24.4s, v22.4s, #16\n"

                                 "str q24, [%[out_q]], #16\n"
                                 : [out_d] "+r"(arr_d), [out_q] "+r"(arr_q)
                                 : [factor] "r"(&fact)
                                 : "memory", "cc", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",
                                 "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17",
                                 "v18", "v19", "v20", "v21", "v22", "v23", "v24", "x1");
            break;
        }
        case 20: {
            __asm__ __volatile__("ldr s0, [%[factor]]\n"
                                 "dup v0.4s, v0.s[0]\n"

                                 "ld4 {v1.4s, v2.4s, v3.4s, v4.4s}, [%[out_d]], #64\n"
                                 "ld4 {v5.4s, v6.4s, v7.4s, v8.4s}, [%[out_d]], #64\n"
                                 "ld4 {v9.4s, v10.4s, v11.4s, v12.4s}, [%[out_d]], #64\n"
                                 "ld4 {v13.4s, v14.4s, v15.4s, v16.4s}, [%[out_d]], #64\n"
                                 "ld4 {v17.4s, v18.4s, v19.4s, v20.4s}, [%[out_d]], #64\n"

                                 "mul v4.4s, v4.4s, v0.4s\n"
                                 "mul v3.4s, v3.4s, v0.4s\n"
                                 "mul v2.4s, v2.4s, v0.4s\n"
                                 "mul v1.4s, v1.4s, v0.4s\n"

                                 "mul v8.4s, v8.4s, v0.4s\n"
                                 "sri v4.4s, v3.4s, #8\n"
                                 "mul v7.4s, v7.4s, v0.4s\n"
                                 "sri v2.4s, v1.4s, #8\n"
                                 "mul v6.4s, v6.4s, v0.4s\n"
                                 "mul v5.4s, v5.4s, v0.4s\n"
                                 "sri v4.4s, v2.4s, #16\n"

                                 "mul v12.4s, v12.4s, v0.4s\n"
                                 "sri v8.4s, v7.4s, #8\n"
                                 "mul v11.4s, v11.4s, v0.4s\n"
                                 "sri v6.4s, v5.4s, #8\n"
                                 "mul v10.4s, v10.4s, v0.4s\n"
                                 "str q4, [%[out_q]], #16\n"
                                 "mul v9.4s, v9.4s, v0.4s\n"
                                 "sri v8.4s, v6.4s, #16\n"

                                 "mul v16.4s, v16.4s, v0.4s\n"
                                 "sri v12.4s, v11.4s, #8\n"
                                 "mul v15.4s, v15.4s, v0.4s\n"
                                 "sri v10.4s, v9.4s, #8\n"
                                 "mul v14.4s, v14.4s, v0.4s\n"
                                 "str q8, [%[out_q]], #16\n"
                                 "mul v13.4s, v13.4s, v0.4s\n"
                                 "sri v12.4s, v10.4s, #16\n"

                                 "mul v20.4s, v20.4s, v0.4s\n"
                                 "sri v16.4s, v15.4s, #8\n"
                                 "mul v19.4s, v19.4s, v0.4s\n"
                                 "sri v14.4s, v13.4s, #8\n"
                                 "mul v18.4s, v18.4s, v0.4s\n"
                                 "str q12, [%[out_q]], #16\n"
                                 "mul v17.4s, v17.4s, v0.4s\n"
                                 "sri v16.4s, v14.4s, #16\n"

                                 "sri v20.4s, v19.4s, #8\n"
                                 "sri v18.4s, v17.4s, #8\n"
                                 "str q16, [%[out_q]], #16\n"
                                 "sri v20.4s, v18.4s, #16\n"

                                 "str q20, [%[out_q]], #16\n"
                                 : [out_d] "+r"(arr_d), [out_q] "+r"(arr_q)
                                 : [factor] "r"(&fact)
                                 : "memory", "cc", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",
                                 "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17",
                                 "v18", "v19", "v20", "x1");
            break;
        }
        case 16: {
            __asm__ __volatile__("ldr s0, [%[factor]]\n"
                                 "dup v0.4s, v0.s[0]\n"

                                 "ld4 {v1.4s, v2.4s, v3.4s, v4.4s}, [%[out_d]], #64\n"
                                 "ld4 {v5.4s, v6.4s, v7.4s, v8.4s}, [%[out_d]], #64\n"
                                 "ld4 {v9.4s, v10.4s, v11.4s, v12.4s}, [%[out_d]], #64\n"
                                 "ld4 {v13.4s, v14.4s, v15.4s, v16.4s}, [%[out_d]], #64\n"

                                 "mul v4.4s, v4.4s, v0.4s\n"
                                 "mul v3.4s, v3.4s, v0.4s\n"
                                 "mul v2.4s, v2.4s, v0.4s\n"
                                 "mul v1.4s, v1.4s, v0.4s\n"

                                 "mul v8.4s, v8.4s, v0.4s\n"
                                 "sri v4.4s, v3.4s, #8\n"
                                 "mul v7.4s, v7.4s, v0.4s\n"
                                 "sri v2.4s, v1.4s, #8\n"
                                 "mul v6.4s, v6.4s, v0.4s\n"
                                 "mul v5.4s, v5.4s, v0.4s\n"
                                 "sri v4.4s, v2.4s, #16\n"

                                 "mul v12.4s, v12.4s, v0.4s\n"
                                 "sri v8.4s, v7.4s, #8\n"
                                 "mul v11.4s, v11.4s, v0.4s\n"
                                 "sri v6.4s, v5.4s, #8\n"
                                 "mul v10.4s, v10.4s, v0.4s\n"
                                 "str q4, [%[out_q]], #16\n"
                                 "mul v9.4s, v9.4s, v0.4s\n"
                                 "sri v8.4s, v6.4s, #16\n"

                                 "mul v16.4s, v16.4s, v0.4s\n"
                                 "sri v12.4s, v11.4s, #8\n"
                                 "mul v15.4s, v15.4s, v0.4s\n"
                                 "sri v10.4s, v9.4s, #8\n"
                                 "mul v14.4s, v14.4s, v0.4s\n"
                                 "str q8, [%[out_q]], #16\n"
                                 "mul v13.4s, v13.4s, v0.4s\n"
                                 "sri v12.4s, v10.4s, #16\n"

                                 "sri v16.4s, v15.4s, #8\n"
                                 "sri v14.4s, v13.4s, #8\n"
                                 "str q12, [%[out_q]], #16\n"
                                 "sri v16.4s, v14.4s, #16\n"

                                 "str q16, [%[out_q]], #16\n"
                                 : [out_d] "+r"(arr_d), [out_q] "+r"(arr_q)
                                 : [factor] "r"(&fact)
                                 : "memory", "cc", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",
                                 "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "x1");
            break;
        }
        case 12: {
            __asm__ __volatile__("ldr s0, [%[factor]]\n"
                                 "dup v0.4s, v0.s[0]\n"

                                 "ld4 {v1.4s, v2.4s, v3.4s, v4.4s}, [%[out_d]], #64\n"
                                 "ld4 {v5.4s, v6.4s, v7.4s, v8.4s}, [%[out_d]], #64\n"
                                 "ld4 {v9.4s, v10.4s, v11.4s, v12.4s}, [%[out_d]], #64\n"

                                 "mul v4.4s, v4.4s, v0.4s\n"
                                 "mul v3.4s, v3.4s, v0.4s\n"
                                 "mul v2.4s, v2.4s, v0.4s\n"
                                 "mul v1.4s, v1.4s, v0.4s\n"

                                 "mul v8.4s, v8.4s, v0.4s\n"
                                 "sri v4.4s, v3.4s, #8\n"
                                 "mul v7.4s, v7.4s, v0.4s\n"
                                 "sri v2.4s, v1.4s, #8\n"
                                 "mul v6.4s, v6.4s, v0.4s\n"
                                 "mul v5.4s, v5.4s, v0.4s\n"
                                 "sri v4.4s, v2.4s, #16\n"

                                 "mul v12.4s, v12.4s, v0.4s\n"
                                 "sri v8.4s, v7.4s, #8\n"
                                 "mul v11.4s, v11.4s, v0.4s\n"
                                 "sri v6.4s, v5.4s, #8\n"
                                 "mul v10.4s, v10.4s, v0.4s\n"
                                 "str q4, [%[out_q]], #16\n"
                                 "mul v9.4s, v9.4s, v0.4s\n"
                                 "sri v8.4s, v6.4s, #16\n"

                                 "sri v12.4s, v11.4s, #8\n"
                                 "sri v10.4s, v9.4s, #8\n"
                                 "str q8, [%[out_q]], #16\n"
                                 "sri v12.4s, v10.4s, #16\n"

                                 "str q12, [%[out_q]], #16\n"
                                 : [out_d] "+r"(arr_d), [out_q] "+r"(arr_q)
                                 : [factor] "r"(&fact)
                                 : "memory", "cc", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",
                                 "v8", "v9", "v10", "v11", "v12", "x1");
            break;
        }
        case 8: {
            __asm__ __volatile__(
                "ldr s0, [%[factor]]\n"
                "dup v0.4s, v0.s[0]\n"

                "ld4 {v1.4s, v2.4s, v3.4s, v4.4s}, [%[out_d]], #64\n"
                "ld4 {v5.4s, v6.4s, v7.4s, v8.4s}, [%[out_d]], #64\n"

                "mul v4.4s, v4.4s, v0.4s\n"
                "mul v3.4s, v3.4s, v0.4s\n"
                "mul v2.4s, v2.4s, v0.4s\n"
                "mul v1.4s, v1.4s, v0.4s\n"

                "mul v8.4s, v8.4s, v0.4s\n"
                "sri v4.4s, v3.4s, #8\n"
                "mul v7.4s, v7.4s, v0.4s\n"
                "sri v2.4s, v1.4s, #8\n"
                "mul v6.4s, v6.4s, v0.4s\n"
                "mul v5.4s, v5.4s, v0.4s\n"
                "sri v4.4s, v2.4s, #16\n"

                "sri v8.4s, v7.4s, #8\n"
                "sri v6.4s, v5.4s, #8\n"
                "str q4, [%[out_q]], #16\n"
                "sri v8.4s, v6.4s, #16\n"

                "str q8, [%[out_q]], #16\n"
                : [out_d] "+r"(arr_d), [out_q] "+r"(arr_q)
                : [factor] "r"(&fact)
                : "memory", "cc", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "x1");
            break;
        }
        case 4: {
            __asm__ __volatile__("ldr s0, [%[factor]]\n"
                                 "dup v0.4s, v0.s[0]\n"

                                 "ld4 {v1.4s, v2.4s, v3.4s, v4.4s}, [%[out_d]], #64\n"

                                 "mul v4.4s, v4.4s, v0.4s\n"
                                 "mul v3.4s, v3.4s, v0.4s\n"
                                 "mul v2.4s, v2.4s, v0.4s\n"
                                 "mul v1.4s, v1.4s, v0.4s\n"

                                 "sri v4.4s, v3.4s, #8\n"
                                 "sri v2.4s, v1.4s, #8\n"
                                 "sri v4.4s, v2.4s, #16\n"

                                 "str q4, [%[out_q]], #16\n"
                                 : [out_d] "+r"(arr_d), [out_q] "+r"(arr_q)
                                 : [factor] "r"(&fact)
                                 : "memory", "cc", "v0", "v1", "v2", "v3", "v4", "x1");
            break;
        }
        case 0: {
            break;
        }
        default: {
            return UNKNOWN;
        }
    }
    return SUCCESS;
}
#endif
#endif
