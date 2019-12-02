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
#include <arm_neon.h>

#include "cpu/arm/arm_neon_expand.h"
#include "cpu/arm/tensor_computing_arm.h"

EE lstm_transform_filter_arm(TensorDesc filterDesc, const void* filter, TensorDesc *ftmDesc, void* filterTransformed, U32 x_dim, U32 h_dim)
{
    if (nullptr == ftmDesc || nullptr == filterTransformed)
        CHECK_STATUS_WITH_RETURN(NULL_POINTER);
    DataType fdt;
    DataFormat fdf;
    U32 fn, fk, ftm_n, ftm_k;
    CHECK_STATUS_WITH_RETURN(tensor2dfGet(filterDesc, &fdt, &fdf, &fn, &fk));
    if(fn != 4*h_dim || fk != (x_dim + h_dim))
        CHECK_STATUS_WITH_RETURN(NOT_MATCH);
    
    F16 *filterArray = (F16*)filter;
    F16 *ftmArray = (F16*)filterTransformed;
    switch(fdf) {
        case DF_NKN32: {
            // everything is ready
            ftm_n = fn;
            ftm_k = fk;
            break;
        }
        case DF_NK: {
            // NK => NKN32
            if (fn % 32 != 0) {
                return NOT_MATCH;
            }
            ftm_n = fn/32;
            ftm_k = fk;
            for (U32 n = 0; n < ftm_n; n++) {
                for (U32 k = 0; k < ftm_k; k++) {
                    for (U32 n32 = 0; n32 < 32; n32++) {
                        ftmArray[n*ftm_k*32 + k*32 + n32] = filterArray[(n*32+n32)*ftm_k + k];
                    }
                }
            }
            break;
        }
        case DF_8NK: {
            // combine 8 matrix into 1 NK => NKN32
            // assume the order of 8 matrix is: h_I, h_F, h_O, h_G, x_I, x_F, x_O, x_G
            if (h_dim % 8 != 0) {
                return NOT_MATCH;
            }
            ftm_n = 4*h_dim/32;
            ftm_k = h_dim + x_dim;
            for (U32 n = 0; n < ftm_n; n++) {
                for (U32 hk = 0; hk < h_dim; hk++) {
                    for (U32 n32 = 0; n32 < 32; n32++) {
                        ftmArray[n*ftm_k*32 + hk*32 + n32] = filterArray[(n*32+n32)*h_dim + hk];
                    }
                }
                for (U32 xk = 0; xk < x_dim; xk++) {
                    for (U32 n32 = 0; n32 < 32; n32++) {
                        ftmArray[n*ftm_k*32 + (h_dim+xk)*32 + n32] = filterArray[ftm_n*32*h_dim + (n*32+n32)*x_dim + xk];
                    }
                }
            }
            break;
        }
        default:
            return NOT_MATCH;
    }
    *ftmDesc = tensor2df(fdt, DF_NKN32, fn, fk);
    return SUCCESS;
}

EE lstm_transform_filter_bytes_arm(TensorDesc filterDesc, U32* bytes)
{
    if (nullptr == bytes)
        CHECK_STATUS_WITH_RETURN(NULL_POINTER);
    *bytes = tensorNumBytes(filterDesc);
    return SUCCESS;
}

EE lstm_infer_forward_tmp_bytes_arm(TensorDesc inputDesc, TensorDesc filterDesc, TensorDesc outputDesc, LSTMDesc lstmDesc, U32 *bytes)
{
    UNUSED(filterDesc);
    UNUSED(outputDesc);
    if (nullptr == bytes)
        CHECK_STATUS_WITH_RETURN(NULL_POINTER);
    DataType idt;
    DataFormat idf;
    U32 batch, step, x_dim;
    CHECK_STATUS_WITH_RETURN(tensor3dGet(inputDesc, &idt, &idf, &batch, &step, &x_dim));
    U32 h_dim = lstmDesc.num_output;
    *bytes = batch*h_dim + batch*(h_dim+x_dim) + batch*h_dim*4;
    switch(idt) {
        case DT_F16:
            *bytes *= sizeof(F16);
            break;
        default:
            return NOT_SUPPORTED;
    }
    return SUCCESS;
}


EE lstm_arm(TensorDesc inputDesc, const void* input, TensorDesc filterDesc, const void* filter,
    LSTMDesc lstmDesc, TensorDesc biasDesc, const void* bias, U32 tmpBytes, void* tmp,
    TensorDesc outputDesc, void* output)
{
    UNUSED(biasDesc);
    UNUSED(tmpBytes);
    if (nullptr == input || nullptr == filter || nullptr == bias || nullptr == tmp || nullptr == output)
        CHECK_STATUS_WITH_RETURN(NULL_POINTER);
    F16 *inArray = (F16*)input;
    F16 *filterArray = (F16*)filter;
    F16 *biasArray = (F16*)bias;
    F16 *outArray = (F16*)output;

    DataType idt, fdt, odt;
    DataFormat idf, fdf, odf;
    U32 in_b, in_t, in_x;
    U32 out_b, out_t, out_h;
    U32 fk, fn;
    CHECK_STATUS_WITH_RETURN(tensor3dGet(inputDesc, &idt, &idf, &in_b, &in_t, &in_x));
    CHECK_STATUS_WITH_RETURN(tensor2dfGet(filterDesc, &fdt, &fdf, &fn, &fk));
    if(fdf != DF_NKN32) {
        CHECK_STATUS_WITH_RETURN(NOT_MATCH);
    }
    fn /= 32;

    CHECK_STATUS_WITH_RETURN(tensor3dGet(outputDesc, &odt, &odf, &out_b, &out_t, &out_h));

    U32 h_dim = lstmDesc.num_output;
    U32 x_dim = in_x;
    U32 step = in_t;
    U32 batch = in_b;

    if (!(idt == DT_F16 && fdt == DT_F16 && odt == DT_F16)) {
        CHECK_STATUS_WITH_RETURN(NOT_MATCH);
    }

    if (fdf != DF_NKN32) {
        CHECK_STATUS_WITH_RETURN(NOT_MATCH);
    }

    if (!(h_dim == out_h && 4*h_dim == fn*32 && (in_x+out_h) == fk && in_b == out_b && in_t == out_t)) {
        CHECK_STATUS_WITH_RETURN(NOT_MATCH);
    }

    F16 *cell_state = (F16*)tmp;
    F16 *in_hx = cell_state + batch*h_dim;
    F16 *out_4h = in_hx + batch*(h_dim+x_dim);
    // initialize c_t, h_t
    memset(cell_state, 0, batch*h_dim);
    memset(in_hx, 0, batch*h_dim);

    // batch = 1 in common
    for (U32 m = 0; m < batch; m++) {
        for (U32 t = 0; t < step; t++) {
            memcpy(in_hx + m*(h_dim+x_dim), inArray + (m*step+t)*x_dim, x_dim*sizeof(F16));
            // MVM
            F16 *out = out_4h + m*fn*32;
            F16 *b = biasArray;
            F16 *in0 = in_hx + m*fk;

            for (U32 n = 0; n < fn; n++) {
                F16 *in = in0;
                F16 *f = filterArray + n*fk*32;
                __asm__ __volatile__(
                    "ldr s0, [%[in]]\n"
                    "ldr q1, [%[b]]\n"
                    "ldr q2, [%[b], #16]\n"
                    "ldr q3, [%[b], #32]\n"
                    "ldr q4, [%[b], #48]\n"
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
                    :[out]"+r"(out),
                     [f]"+r"(f),
                     [in]"+r"(in)
                    :[k]"r"(fk),
                     [b]"r"(b)
                    :"memory", "cc", "x0", "x5", "x6", "x7", "x8", "x9", "x10", "x11", "x12", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12"
                );
                out += 32;
                b += 32;
            }

            F16 *out_i = out_4h + m*fn*32;
            F16 *out_f = out_i + h_dim;
            F16 *out_o = out_i + h_dim*2;
            F16 *out_g = out_i + h_dim*3;
            F16 *cell = cell_state + m*h_dim;
            F16 *out_hidden = in_hx + m*h_dim;

            for (U32 h = 0; h < h_dim; h+=8) {
                float16x8_t out_i_v = vld1q_f16(out_i + h);
                float16x8_t out_f_v = vld1q_f16(out_f + h);
                float16x8_t out_o_v = vld1q_f16(out_o + h);
                float16x8_t out_g_v = vld1q_f16(out_g + h);
                float16x8_t C_v = vld1q_f16(cell + h);
                float16x8_t I_v = vsigmoidq_f16(out_i_v);
                float16x8_t F_v = vsigmoidq_f16(out_f_v);
                float16x8_t O_v = vsigmoidq_f16(out_o_v);
                float16x8_t G_v = vtanhq_f16(out_g_v);
                C_v = vaddq_f16(vmulq_f16(C_v, F_v), vmulq_f16(I_v, G_v));
                float16x8_t out_hidden_v = vmulq_f16(O_v, vtanhq_f16(C_v));
                vst1q_f16(cell + h, C_v);
                vst1q_f16(out_hidden + h, out_hidden_v);
                vst1q_f16(outArray + t*batch*h_dim + m*h_dim + h, out_hidden_v);
            }

        }
    }
    return SUCCESS;
}
