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

EE grucell_fp16(TensorDesc xDesc,
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
    if (!(idt == DT_F16 && fdt == DT_F16 && odt == DT_F16)) {
        CHECK_STATUS(NOT_MATCH);
    }
    if (!(3 * column == (I32)fn * 32 && (ix + oh) == fk && in == on)) {
        CHECK_STATUS(NOT_MATCH);
    }
    ActivationMode activationMode = rnnParamSpec.activationMode;
    if (activationMode != ACTIVATION_TANH) {
        CHECK_STATUS(NOT_SUPPORTED);
    }

    const F16 *currentXArray = (const F16 *)currentX;
    F16 *lastHArray = (F16 *)state;
    F16 *tmpArray = (F16 *)tmp;
    F16 *currentHArray = (F16 *)state;
    F16 *outputArray = (F16 *)output;
    F16 *xhArray = tmpArray;
    F16 *intermediateH = xhArray + (xDim + hDim);
    U32 lastHStride = hDim;
    U32 currentHStride = hDim;
    I32 h = 0;
    for (U32 m = 0; m < batch; m++) {
        F16 *lastBatchH = lastHArray + m * lastHStride;
        F16 *currentBatchH = currentHArray + m * currentHStride;
        F16 *currentOutput = outputArray + m * batchStrideH;
        if (xDim > 0) {
            memcpy(xhArray, currentXArray + m * batchStrideX, xDim * sizeof(F16));
            memcpy(xhArray + xDim, lastBatchH, hDim * sizeof(F16));
        } else {
            intermediateH = tmpArray;
            xhArray = lastBatchH;
            memcpy(currentOutput, lastBatchH, hDim * sizeof(F16));
        }

        memcpy(intermediateH, bias[0], column * 2 * sizeof(F16));
        mvm_nkn32(column * 2 / 32, fk, (const F16 *)filter[0], xhArray, intermediateH);
        F16 *out_z = intermediateH;
        F16 *out_r = out_z + column;
        F16 *out_h = out_r + column;

        for (h = 0; h < column - 7; h += 8) {
            float16x8_t out_r_v = vld1q_f16(out_r + h);
            float16x8_t r_v = vsigmoidq_f16(out_r_v);
            vst1q_f16(out_r + h, r_v);
        }
        for (; h < column; h++) {
            out_r[h] = 1.0 / (1.0 + exp(-out_r[h]));
        }

        if (rnnParamSpec.mode == RNN_GRU_LBR) {
            F16 *h_x_b = (F16 *)bias[0] + column * 2;
            F16 *h_h_b = (F16 *)bias[1];
            memcpy(out_h, h_h_b, column * sizeof(F16));
            mvm_nkn32(column / 32, hDim, (const F16 *)filter[0] + column * 2 * fk + column * xDim,
                xhArray + xDim, out_h);
            array_mul_f16(out_r, out_h, out_h, hDim);
            if (xDim > 0) {
                memcpy(out_r, h_x_b, column * sizeof(F16));
                mvm_nkn32(
                    column / 32, xDim, (const F16 *)filter[0] + column * 2 * fk, xhArray, out_r);
                h_x_b = out_r;
            }
            array_add_f16(h_x_b, out_h, out_h, hDim);
        } else {
            array_mul_f16(out_r, xhArray + xDim, xhArray + xDim, hDim);
            memcpy(out_h, (const F16 *)bias[0] + column * 2, column * sizeof(F16));
            mvm_nkn32(column / 32, fk, (const F16 *)filter[0] + column * 2 * fk, xhArray, out_h);
        }
        for (h = 0; h < column - 7; h += 8) {
            float16x8_t out_z_v = vld1q_f16(out_z + h);
            float16x8_t out_h_v = vld1q_f16(out_h + h);
            float16x8_t z_v = vsigmoidq_f16(out_z_v);
            float16x8_t h_v = vtanhq_f16(out_h_v);
            vst1q_f16(out_z + h, z_v);
            vst1q_f16(out_h + h, h_v);
        }
        for (; h < column; h++) {
            out_z[h] = 1.0 / (1.0 + exp(-out_z[h]));
            out_h[h] = tanh(out_h[h]);
        }
        if (xDim > 0) {
            array_mul_f16(out_z, lastBatchH, out_r, column);
        } else {
            array_mul_f16(out_z, currentOutput, out_r, column);
        }
        array_scale_f16(out_z, out_z, column, -1, 1);
        array_mul_f16(out_z, out_h, out_h, column);
        array_add_f16(out_r, out_h, currentOutput, column);
        memcpy(currentBatchH, currentOutput, sizeof(F16) * hDim);
    }
    return SUCCESS;
}
