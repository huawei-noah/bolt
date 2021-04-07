// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "gpu/mali/fp16/multihead_attention_mali_fp16.h"

#define set_best_wkc(fc_bw, fc_bk, tn_bw, tn_bk, nt_bw, nt_bk, nt_bc, runInfo) \
    {                                                                          \
        U32 *best_w = runInfo->best_w;                                         \
        U32 *best_k = runInfo->best_k;                                         \
        U32 *best_c = runInfo->best_c;                                         \
        fc_bw[0] = best_w[0];                                                  \
        fc_bw[1] = best_w[3];                                                  \
        fc_bw[2] = best_w[4];                                                  \
        fc_bw[3] = best_w[5];                                                  \
        fc_bk[0] = best_k[0];                                                  \
        fc_bk[1] = best_k[3];                                                  \
        fc_bk[2] = best_k[4];                                                  \
        fc_bk[3] = best_k[5];                                                  \
        tn_bw = best_w[1];                                                     \
        tn_bk = best_k[1];                                                     \
        nt_bw = best_w[2];                                                     \
        nt_bk = best_k[2];                                                     \
        nt_bc = best_c[2];                                                     \
    }

#define set_mem_flag(ln_out_flag, fc_out_flag, tn_out_flag, nt_out_flag) \
    {                                                                    \
        ln_out_flag[0] = 2;                                              \
        fc_out_flag[0] = 0;                                              \
        tn_out_flag = 1;                                                 \
        nt_out_flag = 0;                                                 \
        fc_out_flag[1] = 1;                                              \
        ln_out_flag[1] = 0;                                              \
        fc_out_flag[2] = 2;                                              \
    }

#define get_subbuf_size(dt, ln_out_w, ln_out_h, ln_out_flag, fc_out_w, fc_out_h, fc_out_flag,       \
    tn_out_w, tn_out_h, tn_out_c, tn_out_flag, nt_out_w, nt_out_h, nt_out_c, nt_out_flag, sub_size) \
    {                                                                                               \
        U32 size;                                                                                   \
        for (U32 i = 0; i < 3; i++)                                                                 \
            sub_size[i] = 0;                                                                        \
        for (U32 i = 0; i < 2; i++) {                                                               \
            size = ln_out_w[i] * ln_out_h[i] * bytesOf(dt);                                         \
            if (size > sub_size[ln_out_flag[i]])                                                    \
                sub_size[ln_out_flag[i]] = size;                                                    \
        }                                                                                           \
        for (U32 i = 0; i < 3; i++) {                                                               \
            size = fc_out_w[i] * fc_out_h[i] * bytesOf(dt);                                         \
            if (size > sub_size[fc_out_flag[i]])                                                    \
                sub_size[fc_out_flag[i]] = size;                                                    \
        }                                                                                           \
        size = tn_out_w * tn_out_h * tn_out_c * bytesOf(dt);                                        \
        if (size > sub_size[tn_out_flag])                                                           \
            sub_size[tn_out_flag] = size;                                                           \
        size = nt_out_w * nt_out_h * nt_out_c * bytesOf(dt);                                        \
        if (size > sub_size[nt_out_flag])                                                           \
            sub_size[nt_out_flag] = size;                                                           \
    }

#define get_ln0_out_wh(t, k, fc_bw, ow, oh, useEltIn)                 \
    {                                                                 \
        ow = ALIGN(t, fc_bw[0]);                                      \
        if (!useEltIn[0])                                             \
            ow = (ow > ALIGN(t, fc_bw[1])) ? ow : ALIGN(t, fc_bw[1]); \
        oh = ALIGN(k, 4);                                             \
    }

#define get_fc0_out_wh(t, k, fc_bw, fc_bk, tn_bw, tn_bk, nt_bc, ow, oh) \
    {                                                                   \
        ow = ALIGN(t, fc_bw[0]);                                        \
        oh = ALIGN(k, fc_bk[0]);                                        \
        ow = (ow > ALIGN(t, tn_bw)) ? ow : ALIGN(t, tn_bw);             \
        ow = (ow > ALIGN(t, tn_bk)) ? ow : ALIGN(t, tn_bk);             \
        ow = (ow > ALIGN(t, nt_bc)) ? ow : ALIGN(t, nt_bc);             \
    }

#define get_tn_sf_out_whc(Aw, Bw, t, k, sliceLen, nt_bw, nt_bc, ow, oh, oc) \
    {                                                                       \
        ow = Bw;                                                            \
        oh = Aw;                                                            \
        oc = k / sliceLen;                                                  \
        ow = (ow > ALIGN(t, 4)) ? ow : ALIGN(t, 4);                         \
        ow = (ow > ALIGN(t, nt_bc)) ? ow : ALIGN(t, nt_bc);                 \
        oh = (oh > ALIGN(t, nt_bw)) ? oh : ALIGN(t, nt_bw);                 \
    }

#define get_nt_out_whc(Ah, Bh, t, k, sliceLen, fc_bw, ow, oh, oc) \
    {                                                             \
        ow = Bh;                                                  \
        oh = Ah;                                                  \
        oc = k / sliceLen;                                        \
        ow = (ow > ALIGN(t, fc_bw[1])) ? ow : ALIGN(t, fc_bw[1]); \
        if (sliceLen != oh)                                       \
            CHECK_STATUS(NOT_MATCH);                              \
    }

#define get_fc1_out_wh(Bw, t, k, fc_bw, fc_bk, ow, oh)            \
    {                                                             \
        ow = Bw;                                                  \
        oh = ALIGN(k, fc_bk[1]);                                  \
        ow = (ow > ALIGN(t, fc_bw[2])) ? ow : ALIGN(t, fc_bw[2]); \
        ow = (ow > ALIGN(t, fc_bw[3])) ? ow : ALIGN(t, fc_bw[3]); \
    }

#define get_fc2_out_wh(Bw, t, k, fc_bw, fc_bk, ow, oh)            \
    {                                                             \
        ow = Bw;                                                  \
        oh = ALIGN(k, fc_bk[2]);                                  \
        ow = (ow > ALIGN(t, fc_bw[3])) ? ow : ALIGN(t, fc_bw[3]); \
    }

inline void fill_zero_nchw(GCLHandle_t handle, U32 len, U32 offset, Mem buf)
{
    char kernelName[128];
    Kernel kernel;
    sprintf(kernelName, "fill_memory_zero_vec4_f16");
    U32 gs = (len + 3) / 4;
    U32 ls = 0;
    U32 dim = 1;
    CHECK_STATUS(gcl_create_kernel(handle, kernelName, &kernel));
    CHECK_STATUS(gcl_set_kernelArgs(kernel, len, offset, gs, buf));
    gcl_set_kernelVec(handle, kernel, dim, &gs, &ls, kernelName);
#ifdef _DEBUG
    CHECK_STATUS(gcl_run_kernel(handle, kernel, dim, &gs, &ls, kernelName));
#endif
}

inline void layer_norm(GCLHandle_t handle,
    U32 len,
    U32 on,
    U32 ih_str,
    U32 ic_str,
    U32 ih_off,
    U32 iw_off,
    U32 oh_str,
    Mem alpbuf,
    Mem betbuf,
    Mem in,
    Mem out,
    bool USE_C1 = false)
{
    U32 gs = len;
    U32 ls = 0;
    U32 dim = 1;
    float para = 1.0 / on;
    Kernel kernel;
    if (USE_C1) {
        CHECK_STATUS(gcl_create_kernel(handle, "normalization_c1", &kernel));
    } else {
        CHECK_STATUS(gcl_create_kernel(handle, "normalization", &kernel));
    }
    CHECK_STATUS(gcl_set_kernelArgs(
        kernel, len, ih_str, ic_str, ih_off, iw_off, oh_str, 0, 0, para, alpbuf, betbuf, in, out));
    gcl_set_kernelVec(handle, kernel, dim, &gs, &ls, "normalization_c1");
#ifdef _DEBUG
    CHECK_STATUS(gcl_run_kernel(handle, kernel, dim, &gs, &ls, "normalization_c1"));
#endif
}

inline void inner_product_c1(GCLHandle_t handle,
    U32 M,
    U32 N,
    U32 K,
    U32 ow_str,
    U32 ow,
    U32 oh,
    U32 item_w,
    U32 item_k,
    Mem A,
    Mem B,
    Mem bias,
    Mem C)
{
    /*output is c1*/
    U32 ow_align = ALIGN(ow, item_w);
    U32 oh_align = ALIGN(oh, item_k);
    U32 gs[2] = {ow_align / item_w, oh_align / item_k};
    U32 ls[2] = {0, 0};
    U32 dim = 2;
    Kernel kernel;
    char kernelName[128];
    sprintf(kernelName, "gemm_tn_%d%d", item_k, item_w);
    CHECK_STATUS(gcl_create_kernel(handle, kernelName, &kernel));
    CHECK_STATUS(
        gcl_set_kernelArgs(kernel, M, N, K, ow_str, oh, 0, 0, ow, oh, gs[0], gs[1], A, B, bias, C));
    gcl_set_kernelVec(handle, kernel, dim, gs, ls, kernelName);
#ifdef _DEBUG
    CHECK_STATUS(gcl_run_kernel(handle, kernel, dim, gs, ls, kernelName));
#endif
}

inline void inner_product_with_eltwise_c4(GCLHandle_t handle,
    U32 M,
    U32 N,
    U32 K,
    U32 ow_str,
    U32 ow,
    U32 oh,
    U32 item_w,
    U32 item_k,
    bool useLayerNormIn,
    U32 ew_str,
    Mem A,
    Mem B,
    Mem bias,
    Mem C,
    Mem elt)
{
    /*output is c4*/
    U32 ow_align = ALIGN(ow, item_w);
    U32 oh_align = ALIGN(oh, item_k);
    U32 gs[2] = {ow_align / item_w, oh_align / item_k};
    U32 ls[2] = {0, 0};
    U32 dim = 2;
    Kernel kernel;
    char kernelName[128];
    if (useLayerNormIn) {
        sprintf(kernelName, "gemm_tn_eltwise4_ncwhc4_%d%d", item_k, item_w);
        CHECK_STATUS(gcl_create_kernel(handle, kernelName, &kernel));
        CHECK_STATUS(gcl_set_kernelArgs(kernel, M, N, K, ow, 1, oh, ow_str, 1, ow_str, 0, 0, gs[0],
            gs[1], A, B, bias, C, ew_str, 1, ew_str, 0, 0, elt));
    } else {
        sprintf(kernelName, "gemm_tn_eltwise1_ncwhc4_%d%d", item_k, item_w);
        CHECK_STATUS(gcl_create_kernel(handle, kernelName, &kernel));
        CHECK_STATUS(gcl_set_kernelArgs(kernel, M, N, K, ow, 1, oh, ow_str, 1, ow_str, 0, 0, gs[0],
            gs[1], A, B, bias, C, ew_str, 0, 0, elt));
    }
    gcl_set_kernelVec(handle, kernel, dim, gs, ls, kernelName);
#ifdef _DEBUG
    CHECK_STATUS(gcl_run_kernel(handle, kernel, dim, gs, ls, kernelName));
#endif
}

inline void inner_product_ncwhc4(GCLHandle_t handle,
    U32 iw_str,
    U32 ic_str,
    U32 fn,
    U32 ow_str,
    U32 oh_off,
    U32 ow_off,
    U32 ow,
    U32 item_w,
    U32 item_k,
    ActivationMode activation,
    bool useEltwise,
    Mem in,
    Mem flt,
    Mem bias,
    Mem out,
    U32 ew_str,
    Mem elt)
{
    item_k = item_k >> 2;
    U32 ow_align = ALIGN(ow, item_w);
    U32 gs[3] = {1, ow_align / item_w, (fn + 3) / 4 / item_k};
    U32 ls[3] = {0, 0, 0};
    U32 dim = 3;
    Kernel kernel;
    char kernelName[128];
    char modeName[128];
    if (useEltwise) {
        strcpy(modeName, "eltwise4_");
    } else {
        switch (activation) {
            case ACTIVATION_RELU:
                strcpy(modeName, "relu_");
                break;
            case ACTIVATION_GELU:
                strcpy(modeName, "gelu_");
                break;
            case ACTIVATION_NULL:
                strcpy(modeName, "");
                break;
            default:
                CHECK_STATUS(NOT_SUPPORTED);
        }
    }
    sprintf(kernelName, "conv_direct_s%d_%s%d%d%d", 1, modeName, 1, item_w, item_k);
    CHECK_STATUS(gcl_create_kernel(handle, kernelName, &kernel));
    if (useEltwise) {
        CHECK_STATUS(gcl_set_kernelArgs(kernel, 1, iw_str, ic_str, 0, 0, 1, ow_str, oh_off, ow_off,
            ow, fn, 1, 0, 0, gs[0], gs[1], in, flt, bias, out, 1, ew_str, 0, 0, elt));
    } else {
        CHECK_STATUS(gcl_set_kernelArgs(kernel, 1, iw_str, ic_str, 0, 0, 1, ow_str, oh_off, ow_off,
            ow, fn, 1, 0, 0, gs[0], gs[1], in, flt, bias, out));
    }
    gcl_set_kernelVec(handle, kernel, dim, gs, ls, kernelName);
#ifdef _DEBUG
    CHECK_STATUS(gcl_run_kernel(handle, kernel, dim, gs, ls, kernelName));
#endif
}

inline void matmul_tn_c1(GCLHandle_t handle,
    U32 M,
    U32 N,
    U32 K,
    U32 ow_str,
    U32 A_str,
    U32 B_str,
    U32 C_str,
    U32 A_off,
    U32 B_off,
    U32 ow,
    U32 oh,
    U32 item_w,
    U32 item_k,
    U32 batch,
    float alp,
    float bet,
    Mem A,
    Mem B,
    Mem C)
{
    /*output is c1*/
    U32 ow_align = ALIGN(ow, item_w);
    U32 oh_align = ALIGN(oh, item_k);
    U32 gs[3] = {ow_align / item_w, oh_align / item_k, batch};
    U32 ls[3] = {0, 0, 0};
    U32 dim = 3;
    Kernel kernel;
    char kernelName[128];
    sprintf(kernelName, "gemm_tn_nobias_%d%d", item_k, item_w);
    CHECK_STATUS(gcl_create_kernel(handle, kernelName, &kernel));
    CHECK_STATUS(gcl_set_kernelArgs(kernel, M, N, K, ow_str, A_str, B_str, C_str, A_off, B_off, ow,
        oh, gs[0], gs[1], alp, bet, A, B, C));
    gcl_set_kernelVec(handle, kernel, dim, gs, ls, kernelName);
#ifdef _DEBUG
    CHECK_STATUS(gcl_run_kernel(handle, kernel, dim, gs, ls, kernelName));
#endif
}

inline void matmul_nt_c1(GCLHandle_t handle,
    U32 KA,
    U32 KB,
    U32 K,
    U32 ow_str,
    U32 A_str,
    U32 B_str,
    U32 C_str,
    U32 A_off,
    U32 B_off,
    U32 ow,
    U32 oh,
    U32 item_w,
    U32 item_k,
    U32 item_c,
    U32 batch,
    Mem A,
    Mem B,
    Mem C)
{
    /*output is c1*/
    U32 ow_align = ALIGN(ow, item_w);
    U32 oh_align = ALIGN(oh, item_k);
    U32 gs[3] = {ow_align / item_w, oh_align / item_k, batch};
    U32 ls[3] = {0, 0, 0};
    U32 dim = 3;
    Kernel kernel;
    char kernelName[128];
    sprintf(kernelName, "gemm_nt_nobias_%d%d%d", item_k, item_w, (item_c >> 1));
    CHECK_STATUS(gcl_create_kernel(handle, kernelName, &kernel));
    CHECK_STATUS(gcl_set_kernelArgs(kernel, KA, KB, K, ow_str, A_str, B_str, C_str, A_off, B_off,
        ow, oh, gs[0], gs[1], A, B, C));
    gcl_set_kernelVec(handle, kernel, dim, gs, ls, kernelName);
#ifdef _DEBUG
    CHECK_STATUS(gcl_run_kernel(handle, kernel, dim, gs, ls, kernelName));
#endif
}

inline void softmax_w(GCLHandle_t handle,
    U32 iw,
    U32 ih,
    U32 ic,
    U32 iw_str,
    U32 ih_str,
    U32 iw_off,
    U32 ih_off,
    U32 ow_str,
    U32 oh_str,
    U32 ow_off,
    U32 oh_off,
    Mem in,
    Mem out)
{
    U32 gs[2] = {ih, ic};
    U32 ls[2] = {0, 0};
    U32 dim = 2;
    U32 iwd4 = (iw + 3) >> 2;
    U32 iwe4 = ((iw & 3) == 0) ? 4 : (iw & 3);
    Kernel kernel;
    char kernelName[128];
    sprintf(kernelName, "softmax_nchw_w");
    CHECK_STATUS(gcl_create_kernel(handle, kernelName, &kernel));
    CHECK_STATUS(gcl_set_kernelArgs(kernel, iwd4, iwe4, iw_str, ih_str, iw_off, ih_off, ow_str,
        oh_str, ow_off, oh_off, gs[0], gs[1], in, out));
    gcl_set_kernelVec(handle, kernel, dim, gs, ls, kernelName);
#ifdef _DEBUG
    CHECK_STATUS(gcl_run_kernel(handle, kernel, dim, gs, ls, kernelName));
#endif
}

inline EE multihead_attention_core_mali_fp16(GCLHandle_t handle,
    TensorDesc inputDesc,
    const GCLMem_t input,
    std::vector<TensorDesc> filterDesc,
    std::vector<void *> filter,
    std::vector<TensorDesc> biasDesc,
    std::vector<void *> bias,
    std::vector<void *> layerNormAlpha,
    std::vector<void *> layerNormBeta,
    void *multiplyAlpha,
    void *multiplyBeta,
    U32 *firstFCSliceNum,
    U32 matmulSliceLen,
    std::vector<bool> eltwiseWithLayerNormIn,
    ActivationMode activation,
    U32 tmpBytes,
    GCLMem_t tmpBuf,
    TensorDesc outputDesc,
    GCLMem_t output,
    ForwardRunInfoMali_t forwardRunInfo)
{
    DataType dt;
    U32 m, k, t;
    get_nlp_mkt_val(inputDesc, &dt, &m, &k, &t);
    U32 ih_str, ic_str, ih_off, iw_off;
    get_gclmem_dim(input->desc, NULL, &ih_str, &ic_str, &iw_off, &ih_off);

    U32 oh_str, oc_str, oh_off, ow_off;
    get_gclmem_dim(output->desc, NULL, &oh_str, &oc_str, &ow_off, &oh_off);
    U32 fn[4];
    for (U32 i = 0; i < filterDesc.size(); i++) {
        tensorSelectGet(filterDesc[i], NULL, NULL, &fn[i], NULL, NULL, NULL);
    }

    U32 fc_bw[4];
    U32 fc_bk[4];
    U32 tn_bw, tn_bk;
    U32 nt_bw, nt_bk, nt_bc;
    set_best_wkc(fc_bw, fc_bk, tn_bw, tn_bk, nt_bw, nt_bk, nt_bc, forwardRunInfo);

    U32 ln_out_flag[2];
    U32 fc_out_flag[3];
    U32 tn_out_flag, nt_out_flag;
    set_mem_flag(ln_out_flag, fc_out_flag, tn_out_flag, nt_out_flag);

    U32 ln_out_w[2];
    U32 ln_out_h[2];
    U32 fc_out_w[3];
    U32 fc_out_h[3];
    U32 tn_out_w, tn_out_h, tn_out_c;
    U32 nt_out_w, nt_out_h, nt_out_c;

    get_ln0_out_wh(t, k, fc_bw, ln_out_w[0], ln_out_h[0], eltwiseWithLayerNormIn);
    get_fc0_out_wh(t, fn[0], fc_bw, fc_bk, tn_bw, tn_bk, nt_bc, fc_out_w[0], fc_out_h[0]);
    U32 Aw = ALIGN(t, tn_bk);
    U32 Bw = ALIGN(t, tn_bw);
    get_tn_sf_out_whc(
        Aw, Bw, t, firstFCSliceNum[0], matmulSliceLen, nt_bw, nt_bc, tn_out_w, tn_out_h, tn_out_c);

    U32 Ah = ALIGN(matmulSliceLen, nt_bk);
    U32 Bh = ALIGN(t, nt_bw);
    get_nt_out_whc(
        Ah, Bh, t, firstFCSliceNum[2], matmulSliceLen, fc_bw, nt_out_w, nt_out_h, nt_out_c);

    Bw = ALIGN(t, fc_bw[1]);
    get_fc1_out_wh(Bw, t, fn[1], fc_bw, fc_bk, fc_out_w[1], fc_out_h[1]);

    ln_out_w[1] = fc_out_w[1];
    ln_out_h[1] = fc_out_h[1];

    Bw = ALIGN(t, fc_bw[2]);
    get_fc2_out_wh(Bw, t, fn[2], fc_bw, fc_bk, fc_out_w[2], fc_out_h[2]);

    U32 offset = 0;
    U32 sub_size[3];
    get_subbuf_size(dt, ln_out_w, ln_out_h, ln_out_flag, fc_out_w, fc_out_h, fc_out_flag, tn_out_w,
        tn_out_h, tn_out_c, tn_out_flag, nt_out_w, nt_out_h, nt_out_c, nt_out_flag, sub_size);

    Mem ln_out_mem[2];
    Mem fc_out_mem[3];
    Mem tn_out_mem, nt_out_mem;
    Mem subBuf[3];
    CHECK_STATUS(gcl_create_sub_buffer(sub_size[0], &offset, tmpBuf, &subBuf[0]));
    CHECK_STATUS(gcl_create_sub_buffer(sub_size[1], &offset, tmpBuf, &subBuf[1]));
    CHECK_STATUS(gcl_create_sub_buffer(sub_size[2], &offset, tmpBuf, &subBuf[2]));

    for (U32 i = 0; i < 2; i++) {
        ln_out_mem[i] = subBuf[ln_out_flag[i]];
    }
    for (U32 i = 0; i < 3; i++) {
        fc_out_mem[i] = subBuf[fc_out_flag[i]];
    }
    tn_out_mem = subBuf[tn_out_flag];
    nt_out_mem = subBuf[nt_out_flag];

    /* STAGE0: layerNorm
     * INPUT  (X, 78) C4
     * OUTPUT (X, 312) C1 --> X align to best_w[0]
     */
    Mem stage0LNIn = input->mem;
    Mem stage0LNAlp = ((GCLMem_t)(layerNormAlpha[0]))->mem;
    Mem stage0LNBet = ((GCLMem_t)(layerNormBeta[0]))->mem;

    layer_norm(handle, t, k, ih_str, ic_str, ih_off, iw_off, ln_out_w[0], stage0LNAlp, stage0LNBet,
        stage0LNIn, ln_out_mem[0], true);

    /* STAGE1: InnerProduct
     * TN GEMM
     * weight(T)   (932, 312) * stage0LNOut(N) (X, 312)
     * GPU:
     *      weight W : 932 -> 312 * 3
     *      weight H:  312
     * OUTPUT:
     * mat_q: (X, 312) --> (Xq, 26, 12)
     * mat_k: (X, 312) --> (Xk, 26, 12)
     * mat_v: (X, 312) --> (Xv, 26, 12)
     * Xq = Xk = Xv

     * mat_q * mat_k(TN) --->mat_qk(Xk, Xq, 12)
     * mat_q  --> Xq X align to best_k[1]
     * mat_k  --> Xk X align to best_w[1]
     * mat_qk --> Xqk_w X align to best_c[0]
     * mat_qk --> Xqk_h X align to best_w[2]

     * mat_v * mat_qk(NT) -->mat_vqk(Xq, 7, 12)
     * mat_v --> Xv X  align to best_c[0];
     * mat_v --> 26 26 align to best_k[2](require 26 % best_k[2] = 0);

     * Stage1:
     * OUTPUT
     * dim0: max(Xq align best_k[1], Xk align best_w[1], Xv align to best_c[0])
     * dim1: 312 + 312 + 312
     * INPUT:
     * A(dim1 align to best_k[0], 312) B(X align to best_w[0])
     */

    U32 M = ((GCLMem_t)(filter[0]))->desc.stride[0];
    U32 K = ((GCLMem_t)(filter[0]))->desc.stride[1];
    U32 N = ln_out_w[0];
    Mem stage1MatA = ((GCLMem_t)(filter[0]))->mem;
    Mem stage1MatB = ln_out_mem[0];
    Mem stage1Bias = ((GCLMem_t)(bias[0]))->mem;
    if (N < ALIGN(t, nt_bc)) {
        U32 off = (firstFCSliceNum[0] + firstFCSliceNum[1]) * fc_out_w[0];
        U32 len = fc_out_w[0] * fc_out_h[0] - off;
        fill_zero_nchw(handle, len, off, fc_out_mem[0]);
    }
    inner_product_c1(handle, M, N, K, fc_out_w[0], t, M, fc_bw[0], fc_bk[0], stage1MatA, stage1MatB,
        stage1Bias, fc_out_mem[0]);

    /* Stage2: Matmul mat_q * mat_k
     * TN GEMM
     * INPUT: mat_q(Xq, 26, 12) mat_k (Xk, 26, 12);
     * Xq X align to best_k[1]
     * Xk X align to best_w[1]
     * Use stride Xmax
     * Output: mat_qk(Xqk_w, Xqk_h, 12)
     * Xqk_w X align to best_c[0](Xk)
     * Xqk_h X align to best_w[2](Xq)
     */

    M = fc_out_w[0];
    N = fc_out_w[0];
    K = matmulSliceLen;
    Mem stage2MatA = fc_out_mem[0];
    Mem stage2MatB = fc_out_mem[0];
    Aw = ALIGN(t, tn_bk);
    Bw = ALIGN(t, tn_bw);
    if (tn_out_w > Aw || tn_out_h > Bw) {
        U32 len = tn_out_w * tn_out_h * tn_out_c;
        fill_zero_nchw(handle, len, 0, tn_out_mem);
    }
    U32 A_str = matmulSliceLen * M;
    U32 B_str = matmulSliceLen * N;
    U32 C_str = tn_out_w * tn_out_h;
    U32 A_off = 0;
    U32 B_off = firstFCSliceNum[0] * fc_out_w[0];
    float *mulAlp = (float *)multiplyAlpha;
    float *mulBet = (float *)multiplyBeta;
    matmul_tn_c1(handle, M, N, K, tn_out_w, A_str, B_str, C_str, A_off, B_off, t, t, tn_bw, tn_bk,
        tn_out_c, *mulAlp, *mulBet, stage2MatA, stage2MatB, tn_out_mem);

    /* STAGE3: Softmax on w for mat_qk */
    softmax_w(handle, t, t, tn_out_c, tn_out_w, tn_out_h, 0, 0, tn_out_w, tn_out_h, 0, 0,
        tn_out_mem, tn_out_mem);

    /* STAGE4: Matmul mat_v * mat_qk
     * NT GEMM
     * INPUT: mat_v(Xv, 26, 12) mat_qk(Xqk_w, Xqk_h, 12)
     * Xv X align to best_c[0]
     * 26 align to best_k[2]
     * Xqk_w align to best_c[0]
     * Xqk_h align to best_w[2]
     * OUTPUT: mat_vqk(Xvqk, 26, 12)
     * Xvqk X align to best_w[3]
     * set 26 divided by best_k[2], for next step
     */
    U32 KA = fc_out_w[0];
    U32 KB = tn_out_w;
    Mem stage4MatA = fc_out_mem[0];
    Mem stage4MatB = tn_out_mem;
    K = ALIGN(t, nt_bc);
    A_str = KA * matmulSliceLen;
    B_str = tn_out_w * tn_out_h;
    C_str = nt_out_w * nt_out_h;
    A_off = (firstFCSliceNum[0] + firstFCSliceNum[1]) * KA;
    B_off = 0;
    matmul_nt_c1(handle, KA, KB, K, nt_out_w, A_str, B_str, C_str, A_off, B_off, t, matmulSliceLen,
        nt_bw, nt_bk, nt_bc, nt_out_c, stage4MatA, stage4MatB, nt_out_mem);

    /* STAGE5: Innerproduct
     * TN GEMM
     * weight(T) (312, 312) stage4MatC(Xvqk, 312)
     * weight w 312 align to best_k[3]
     * Xvqk align to best_w[3], use stride Xvqk_max_w
     * Output: stage5MatC
     * use ncwhc4 for layer normal
     * (Xi5, 312)
     * Xi5, X align to best_w[4]
     */

    M = ((GCLMem_t)filter[1])->desc.stride[0];
    K = ((GCLMem_t)filter[1])->desc.stride[1];
    N = nt_out_w;
    Mem stage5MatA = ((GCLMem_t)filter[1])->mem;
    Mem stage5MatB = nt_out_mem;
    Mem stage5Bias = ((GCLMem_t)bias[1])->mem;
    U32 ew_str = (eltwiseWithLayerNormIn[0]) ? ih_str : ln_out_w[0];
    Mem elt = (eltwiseWithLayerNormIn[0]) ? stage0LNIn : ln_out_mem[0];
    inner_product_with_eltwise_c4(handle, M, N, K, fc_out_w[1], t, fn[1], fc_bw[1], fc_bk[1],
        eltwiseWithLayerNormIn[0], ew_str, stage5MatA, stage5MatB, stage5Bias, fc_out_mem[1], elt);

    /* STAGE6: LayerNorm
     */
    Mem stage6LNAlp = ((GCLMem_t)(layerNormAlpha[1]))->mem;
    Mem stage6LNBet = ((GCLMem_t)(layerNormBeta[1]))->mem;
    layer_norm(handle, t, fn[1], fc_out_w[1], (fn[1] + 3) / 4, 0, 0, ln_out_w[1], stage6LNAlp,
        stage6LNBet, fc_out_mem[1], ln_out_mem[1]);

    /* STAGE7: Innerproduct with relu
     */
    Mem stage7Flt = ((GCLMem_t)filter[2])->mem;
    Mem stage7In = ln_out_mem[1];
    Mem stage7Bias = ((GCLMem_t)bias[2])->mem;
    inner_product_ncwhc4(handle, ln_out_w[1], (fn[1] + 3) / 4, fn[2], fc_out_w[2], 0, 0, t, fc_bw[2],
        fc_bk[2], activation, false, stage7In, stage7Flt, stage7Bias, fc_out_mem[2], 0, NULL);

    /*STAGE8: Innerproduct with eltwise
     */
    M = ((GCLMem_t)(filter[3]))->desc.stride[0];
    K = ((GCLMem_t)(filter[3]))->desc.stride[1];
    N = fc_out_w[2];
    Mem stage8Flt = ((GCLMem_t)filter[3])->mem;
    Mem stage8In = fc_out_mem[2];
    Mem stage8Bias = ((GCLMem_t)bias[3])->mem;
    ew_str = (eltwiseWithLayerNormIn[1]) ? fc_out_w[1] : ln_out_w[1];
    Mem elt2 = (eltwiseWithLayerNormIn[1]) ? fc_out_mem[1] : ln_out_mem[1];
    inner_product_ncwhc4(handle, fc_out_w[2], (fn[2] + 3) / 4, fn[3], oh_str, oh_off, ow_off, t,
        fc_bw[3], fc_bk[3], ACTIVATION_NULL, true, stage8In, stage8Flt, stage8Bias, output->mem,
        ew_str, elt2);
    return SUCCESS;
}

inline EE multihead_attention_checkpara_mali_fp16(
    TensorDesc inputDesc, std::vector<TensorDesc> filterDesc, TensorDesc outputDesc)
{
    if (inputDesc.dt != outputDesc.dt || inputDesc.dt != DT_F16) {
        return NOT_MATCH;
    }
    for (U32 i = 0; i < filterDesc.size(); i++) {
        if (filterDesc[i].dt != DT_F16) {
            return NOT_SUPPORTED;
        }
    }
    return SUCCESS;
}

EE multihead_attention_transform_filter_bytes_mali_fp16(std::vector<TensorDesc> filterDesc,
    GCLMemDesc_t gclmemFilterDesc,
    U32 *bytes,
    ForwardRunInfoMali_t forwardRunInfo)
{
    U32 fc_bk[4];
    U32 fc_bc[4];
    fc_bk[0] = forwardRunInfo->best_k[0];
    fc_bk[1] = forwardRunInfo->best_k[3];
    fc_bk[2] = forwardRunInfo->best_k[4];
    fc_bk[3] = forwardRunInfo->best_k[5];
    fc_bc[0] = forwardRunInfo->best_c[0];
    fc_bc[1] = forwardRunInfo->best_c[3];
    fc_bc[2] = forwardRunInfo->best_c[4];
    fc_bc[3] = forwardRunInfo->best_c[5];
    for (U32 i = 0; i < 2; i++) {
        U32 fn, fc, fh, fw;
        U32 s0, s1, s2;
        U32 num;
        DataType dt = filterDesc[i].dt;
        tensorSelectGet(filterDesc[i], NULL, NULL, &fn, &fc, &fh, &fw);
        if (fh != 1 || fw != 1) {
            CHECK_STATUS(NOT_MATCH);
        }
        s0 = ALIGN(fn, fc_bk[i]);
        s1 = ALIGN(fc, 4);
        s2 = 1;
        num = s0 * s1 * s2;
        gclmemFilterDesc[i].stride[0] = s0;
        gclmemFilterDesc[i].stride[1] = s1;
        gclmemFilterDesc[i].stride[2] = s2;
        gclmemFilterDesc[i].offset[0] = 0;
        gclmemFilterDesc[i].offset[1] = 0;
        gclmemFilterDesc[i].offset[2] = 0;
        gclmemFilterDesc[i].num = num;
        gclmemFilterDesc[i].memFormat = DF_NCHW;
        gclmemFilterDesc[i].byteSize = num * bytesOf(dt);
        gclmemFilterDesc[i].memType = GCL_MEM_BUF;
        gclmemFilterDesc[i].flags = CL_MEM_READ_WRITE;
        gclmemFilterDesc[i].host_ptr = NULL;
    }
    for (U32 i = 2; i < filterDesc.size(); i++) {
        U32 fn, fc, fh, fw;
        U32 s0, s1, s2;
        U32 num;
        DataType dt = filterDesc[i].dt;
        tensorSelectGet(filterDesc[i], NULL, NULL, &fn, &fc, &fh, &fw);
        if (fh != 1 || fw != 1) {
            CHECK_STATUS(NOT_MATCH);
        }
        s0 = fc_bk[i] >> 2;
        s1 = (fc + fc_bc[i] - 1) / fc_bc[i];
        s2 = (fn + fc_bk[i] - 1) / fc_bk[i];
        num = s0 * s1 * s2 * fc_bc[i] * fc_bk[i] / (fc_bk[i] >> 2);
        gclmemFilterDesc[i].stride[0] = s0;
        gclmemFilterDesc[i].stride[1] = s1;
        gclmemFilterDesc[i].stride[2] = s2;
        gclmemFilterDesc[i].offset[0] = 0;
        gclmemFilterDesc[i].offset[1] = 0;
        gclmemFilterDesc[i].offset[2] = 0;
        gclmemFilterDesc[i].num = num;
        gclmemFilterDesc[i].memFormat = DF_NCWHN4C4;
        gclmemFilterDesc[i].byteSize = num * bytesOf(dt);
        gclmemFilterDesc[i].memType = GCL_MEM_BUF;
        gclmemFilterDesc[i].flags = CL_MEM_READ_WRITE;
        gclmemFilterDesc[i].host_ptr = NULL;
    }
    return SUCCESS;
}

EE multihead_attention_transform_filter_mali_fp16(GCLHandle_t handle,
    std::vector<TensorDesc> filterDesc,
    std::vector<void *> filter,
    std::vector<TensorDesc> *fltmemDesc,
    std::vector<void *> fltmem,
    ForwardRunInfoMali_t forwardRunInfo)
{
    U32 fc_bk[4];
    U32 fc_bc[4];

    fc_bk[0] = forwardRunInfo->best_k[0];
    fc_bk[1] = forwardRunInfo->best_k[3];
    fc_bk[2] = forwardRunInfo->best_k[4];
    fc_bk[3] = forwardRunInfo->best_k[5];
    fc_bc[0] = forwardRunInfo->best_c[0];
    fc_bc[1] = forwardRunInfo->best_c[3];
    fc_bc[2] = forwardRunInfo->best_c[4];
    fc_bc[3] = forwardRunInfo->best_c[5];
    char kernelname[128];
    Kernel kernel;
    U32 gs[3];
    U32 ls[3] = {0, 0, 0};
    U32 dim = 3;
    U32 filterNum = filterDesc.size();
    if (filterNum != filter.size() || filterNum != fltmemDesc->size() || filterNum != fltmem.size()) {
        CHECK_STATUS(NOT_MATCH);
    }
    for (auto p : filterDesc) {
        fltmemDesc->push_back(p);
    }
    U32 fwh = 1;
    for (U32 i = 0; i < 2; i++) {
        sprintf(kernelname, "conv_direct_trans_fltbuf_%d%d", 1, 0);
        U32 fc, fn;
        Mem flt_org = ((GCLMem_t)filter[i])->mem;
        Mem flt_tra = ((GCLMem_t)fltmem[i])->mem;
        tensorSelectGet(filterDesc[i], NULL, NULL, &fn, &fc, NULL, NULL);
        CHECK_STATUS(gcl_get_kernel_from_map(handle, kernelname, &kernel));
        CHECK_STATUS(gcl_set_kernelArgs(kernel, fwh, fc, fn, flt_org, flt_tra));
        gs[0] = fwh;
        gs[1] = ALIGN(fc, 4);
        gs[2] = ALIGN(fn, fc_bk[i]);
        CHECK_STATUS(gcl_run_kernel(handle, kernel, dim, gs, ls, kernelname));
    }

    for (U32 i = 2; i < filterDesc.size(); i++) {
        sprintf(kernelname, "conv_direct_trans_fltbuf_%d%d", fc_bc[i], fc_bk[i]);
        U32 fc, fn;
        Mem flt_org = ((GCLMem_t)filter[i])->mem;
        Mem flt_tra = ((GCLMem_t)fltmem[i])->mem;
        tensorSelectGet(filterDesc[i], NULL, NULL, &fn, &fc, NULL, NULL);
        CHECK_STATUS(gcl_get_kernel_from_map(handle, kernelname, &kernel));
        CHECK_STATUS(gcl_set_kernelArgs(kernel, fwh, fc, fn, flt_org, flt_tra));
        gs[0] = fwh;
        gs[1] = (fc + fc_bc[i] - 1) / fc_bk[i];
        gs[2] = ALIGN(fn, fc_bk[i]);
        CHECK_STATUS(gcl_run_kernel(handle, kernel, dim, gs, ls, kernelname));
    }
    return SUCCESS;
}

EE multihead_attention_infer_forward_tmp_bytes_mali_fp16(TensorDesc inputDesc,
    std::vector<TensorDesc> filterDesc,
    std::vector<bool> eltwiseWithLayerNormIn,
    U32 *firstFCSliceNum,
    U32 matmulSliceLen,
    U32 *bytes,
    ForwardRunInfoMali_t forwardRunInfo)
{
    U32 fn[4];
    for (U32 i = 0; i < filterDesc.size(); i++) {
        tensorSelectGet(filterDesc[i], NULL, NULL, &fn[i], NULL, NULL, NULL);
    }
    DataType dt;
    U32 m, k, t;
    get_nlp_mkt_val(inputDesc, &dt, &m, &k, &t);
    U32 fc_bw[4];
    U32 fc_bk[4];
    U32 tn_bw, tn_bk;
    U32 nt_bw, nt_bk, nt_bc;
    set_best_wkc(fc_bw, fc_bk, tn_bw, tn_bk, nt_bw, nt_bk, nt_bc, forwardRunInfo);

    U32 ln_out_flag[2];
    U32 fc_out_flag[3];
    U32 tn_out_flag, nt_out_flag;
    set_mem_flag(ln_out_flag, fc_out_flag, tn_out_flag, nt_out_flag);

    U32 ln_out_w[2];
    U32 ln_out_h[2];
    U32 fc_out_w[3];
    U32 fc_out_h[3];
    U32 tn_out_w, tn_out_h, tn_out_c;
    U32 nt_out_w, nt_out_h, nt_out_c;
    get_ln0_out_wh(t, k, fc_bw, ln_out_w[0], ln_out_h[0], eltwiseWithLayerNormIn);
    get_fc0_out_wh(t, fn[0], fc_bw, fc_bk, tn_bw, tn_bk, nt_bc, fc_out_w[0], fc_out_h[0]);
    U32 Aw = ALIGN(t, tn_bk);
    U32 Bw = ALIGN(t, tn_bw);
    get_tn_sf_out_whc(
        Aw, Bw, t, firstFCSliceNum[0], matmulSliceLen, nt_bw, nt_bc, tn_out_w, tn_out_h, tn_out_c);
    U32 Ah = ALIGN(matmulSliceLen, nt_bk);
    U32 Bh = ALIGN(t, nt_bw);
    get_nt_out_whc(
        Ah, Bh, t, firstFCSliceNum[2], matmulSliceLen, fc_bw, nt_out_w, nt_out_h, nt_out_c);
    Bw = ALIGN(t, fc_bw[1]);
    get_fc1_out_wh(Bw, t, fn[1], fc_bw, fc_bk, fc_out_w[1], fc_out_h[1]);
    ln_out_w[1] = fc_out_w[1];
    ln_out_h[1] = fc_out_h[1];
    Bw = ALIGN(t, fc_bw[2]);
    get_fc2_out_wh(Bw, t, fn[2], fc_bw, fc_bk, fc_out_w[2], fc_out_h[2]);

    U32 sub_size[3];
    get_subbuf_size(dt, ln_out_w, ln_out_h, ln_out_flag, fc_out_w, fc_out_h, fc_out_flag, tn_out_w,
        tn_out_h, tn_out_c, tn_out_flag, nt_out_w, nt_out_h, nt_out_c, nt_out_flag, sub_size);
    *bytes = ALIGN(sub_size[0], 1024) + ALIGN(sub_size[1], 1024) + sub_size[2];
    return SUCCESS;
}

EE multihead_attention_mali_fp16(GCLHandle_t handle,
    TensorDesc inputDesc,
    const GCLMem_t input,
    std::vector<TensorDesc> filterDesc,
    std::vector<void *> filter,
    std::vector<TensorDesc> biasDesc,
    std::vector<void *> bias,
    std::vector<void *> layerNormAlpha,
    std::vector<void *> layerNormBeta,
    void *multiplyAlpha,
    void *multiplyBeta,
    U32 *firstFCSliceNum,
    U32 matmulSliceLen,
    std::vector<bool> eltwiseWithLayerNormIn,
    ActivationMode activation,
    U32 tmpBytes,
    GCLMem_t tmpBuf,
    TensorDesc outputDesc,
    GCLMem_t output,
    ForwardRunInfoMali_t forwardRunInfo)
{
    CHECK_STATUS(multihead_attention_checkpara_mali_fp16(inputDesc, filterDesc, outputDesc));
    CHECK_STATUS(fill_output_zero(handle, output, outputDesc));
    CHECK_STATUS(multihead_attention_core_mali_fp16(handle, inputDesc, input, filterDesc, filter,
        biasDesc, bias, layerNormAlpha, layerNormBeta, multiplyAlpha, multiplyBeta, firstFCSliceNum,
        matmulSliceLen, eltwiseWithLayerNormIn, activation, tmpBytes, tmpBuf, outputDesc, output,
        forwardRunInfo));
    return SUCCESS;
}
