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

EE grucell_fp32(TensorDesc xDesc,
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
    I32 column = hDim;
    if (!(idt == DT_F32 && fdt == DT_F32 && odt == DT_F32)) {
        CHECK_STATUS(NOT_MATCH);
    }
    if (!(3 * column == (I32)fn * 32 && (ix + oh) == fk && in == on)) {
        CHECK_STATUS(NOT_MATCH);
    }
    ActivationMode activationMode = rnnParamSpec.activationMode;
    if (activationMode != ACTIVATION_TANH) {
        CHECK_STATUS(NOT_SUPPORTED);
    }

    const F32 *currentXArray = (const F32 *)currentX;
    F32 *lastHArray = (F32 *)state;
    F32 *tmpArray = (F32 *)tmp;
    F32 *currentHArray = (F32 *)state;
    F32 *outputArray = (F32 *)output;
    F32 *xhArray = tmpArray;
    F32 *intermediateH = xhArray + (xDim + hDim);
    U32 lastHStride = hDim;
    U32 currentHStride = hDim;
    I32 h = 0;
    for (U32 m = 0; m < batch; m++) {
        F32 *lastBatchH = lastHArray + m * lastHStride;
        F32 *currentBatchH = currentHArray + m * currentHStride;
        F32 *currentOutput = outputArray + m * batchStrideH;
        if (xDim > 0) {
            memcpy(xhArray, currentXArray + m * batchStrideX, xDim * sizeof(F32));
            memcpy(xhArray + xDim, lastBatchH, hDim * sizeof(F32));
        } else {
            intermediateH = tmpArray;
            xhArray = lastBatchH;
            memcpy(currentOutput, lastBatchH, hDim * sizeof(F32));
        }

        memcpy(intermediateH, bias[0], column * 2 * sizeof(F32));
        mvm_nkn32(column * 2 / 32, fk, (const F32 *)filter[0], xhArray, intermediateH);
        F32 *out_z = intermediateH;
        F32 *out_r = out_z + column;
        F32 *out_h = out_r + column;

        for (h = 0; h < column - 3; h += 4) {
            float32x4_t out_r_v = vld1q_f32(out_r + h);
            float32x4_t r_v = vsigmoidq_f32(out_r_v);
            vst1q_f32(out_r + h, r_v);
        }
        for (; h < column; h++) {
            out_r[h] = 1.0 / (1.0 + exp(-out_r[h]));
        }

        if (rnnParamSpec.mode == RNN_GRU_LBR) {
            F32 *h_x_b = (F32 *)bias[0] + column * 2;
            F32 *h_h_b = (F32 *)bias[1];
            memcpy(out_h, h_h_b, column * sizeof(F32));
            mvm_nkn32(column / 32, hDim, (const F32 *)filter[0] + column * 2 * fk + column * xDim,
                xhArray + xDim, out_h);
            array_mul_f32(out_r, out_h, out_h, hDim);
            if (xDim > 0) {
                memcpy(out_r, h_x_b, column * sizeof(F32));
                mvm_nkn32(
                    column / 32, xDim, (const F32 *)filter[0] + column * 2 * fk, xhArray, out_r);
                h_x_b = out_r;
            }
            array_add_f32(h_x_b, out_h, out_h, hDim);
        } else {
            array_mul_f32(out_r, xhArray + xDim, xhArray + xDim, hDim);
            memcpy(out_h, (const F32 *)bias[0] + column * 2, column * sizeof(F32));
            mvm_nkn32(column / 32, fk, (const F32 *)filter[0] + column * 2 * fk, xhArray, out_h);
        }
        for (h = 0; h < column - 3; h += 4) {
            float32x4_t out_z_v = vld1q_f32(out_z + h);
            float32x4_t out_h_v = vld1q_f32(out_h + h);
            float32x4_t z_v = vsigmoidq_f32(out_z_v);
            float32x4_t h_v = vtanhq_f32(out_h_v);
            vst1q_f32(out_z + h, z_v);
            vst1q_f32(out_h + h, h_v);
        }
        for (; h < column; h++) {
            out_z[h] = 1.0 / (1.0 + exp(-out_z[h]));
            out_h[h] = tanh(out_h[h]);
        }
        if (xDim > 0) {
            array_mul_f32(out_z, lastBatchH, out_r, column);
        } else {
            array_mul_f32(out_z, currentOutput, out_r, column);
        }
        array_scale_f32(out_z, out_z, column, -1, 1);
        array_mul_f32(out_z, out_h, out_h, column);
        array_add_f32(out_r, out_h, currentOutput, column);
        memcpy(currentBatchH, currentOutput, sizeof(F32) * hDim);
    }
    return SUCCESS;
}
