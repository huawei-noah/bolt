// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifdef _USE_INT8
#include <cstring>
#include "cpu/arm/int8/tensor_computing_int8.h"

EE concat_int8(std::vector<TensorDesc> inputDesc,
    std::vector<void *> input,
    F32 *inputScale,
    int concatDim,
    TensorDesc outputDesc,
    void *output,
    F32 *outputScale)
{
    if (inputDesc.size() < 1) {
        CHECK_STATUS(NOT_MATCH);
    }
    if (inputDesc.size() == 1) {
        memcpy(output, input[0], tensorNumBytes(outputDesc));
        return SUCCESS;
    }
    if (concatDim != 0 && concatDim != 1) {
        CHECK_STATUS(NOT_SUPPORTED);
    }

    F32 min_scale = inputScale[0];
    U32 min_idx = 0;

    for (U32 i = 1; i < input.size(); i++) {
        if (min_scale > inputScale[i]) {
            min_scale = inputScale[i];
            min_idx = i;
        }
    }
    *outputScale = min_scale;

    for (U32 i = 0; i < input.size(); i++) {
        if (i == min_idx) {
            continue;
        }

        INT8 *narr = (INT8 *)input[i];
        F32 rescale = min_scale / inputScale[i];
        if (rescale >= 0.9921) {  // Even 127 will not be updated to 126
            continue;
        }
        INT8 factor = rescale * 128;

        if (factor < 2) {
            continue;
        }

        int8x8_t fact = vdup_n_s8(factor);

        U32 num = tensorNumElements(inputDesc[i]);
        U32 i32 = num / 32;

        int8x8_t in[4];
        int16x8_t in16[4];

        for (U32 i = 0; i < i32; i++) {
            for (U32 j = 0; j < 4; j++) {
                in[j] = vld1_s8(narr + j * 8);
            }
            for (U32 j = 0; j < 4; j++) {
                in16[j] = vmull_s8(in[j], fact);
            }
            in[0] = vqshrn_n_s16(in16[0], 7);
            for (U32 j = 1; j < 4; j++) {
                in[j] = vqshrn_n_s16(in16[j], 7);
                vst1_s8(narr + j * 8 - 8, in[j - 1]);
            }
            vst1_s8(narr + 24, in[3]);

            narr += 32;
        }

        U32 remainder = num - i32 * 32;

        for (U32 j = 0; j < remainder; j += 8) {
            int8x8_t in = vld1_s8(narr + j);
            int16x8_t in16 = vmull_s8(in, fact);
            in = vqshrn_n_s16(in16, 7);
            vst1_s8(narr + j, in);
        }
    }

    DataType odt, idt;
    DataFormat odf, idf;
    U32 on = 0, oc = 0, oh = 0, ow = 0, in = 0, ic = 0, ih = 0, iw = 0;
    U32 copySize;

    if (tensorIs4d(outputDesc)) {
        CHECK_STATUS(tensor4dGet(outputDesc, &odt, &odf, &on, &oc, &oh, &ow));
        if (odt != DT_I8) {
            CHECK_STATUS(NOT_MATCH);
        }

        INT8 *out_ptr = (INT8 *)output;

        // batch
        if (concatDim == 0) {
            for (U32 i = 0; i < inputDesc.size(); i++) {
                copySize = tensorNumElements(inputDesc[i]) * sizeof(INT8);

                memcpy(out_ptr, input[i], copySize);
                out_ptr = out_ptr + copySize;
            }
            return SUCCESS;
        }
        // channel
        if (concatDim == 1) {
            for (U32 j = 0; j < on; j++) {
                for (U32 i = 0; i < inputDesc.size(); i++) {
                    CHECK_STATUS(tensor4dGet(inputDesc[i], &idt, &idf, &in, &ic, &ih, &iw));
                    if (odf != idf) {
                        CHECK_STATUS(NOT_MATCH);
                    }

                    copySize = tensorNumElements(inputDesc[i]) / in * sizeof(INT8);

                    memcpy(out_ptr, (INT8 *)input[i] + j * copySize, copySize);
                    out_ptr = out_ptr + copySize;
                }
            }
            return SUCCESS;
        }
    } else {
        return NOT_MATCH;
    }
    return NOT_SUPPORTED;
}
#endif
