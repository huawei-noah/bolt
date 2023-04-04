// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "cpu/tensor_computing_cpu.h"

template <typename T, int cx>
static EE space2depth_kernel(
    TensorDesc inputDesc, T *input, Space2DepthParamSpec p, TensorDesc outputDesc, T *output)
{
    DataType idt, odt;
    DataFormat idf, odf;
    U32 in, ic, ih, iw;
    U32 on, oc, oh, ow;
    int bh = p.block_size;
    int bw = p.block_size;
    if (tensorIs4d(inputDesc)) {
        CHECK_STATUS(tensor4dGet(inputDesc, &idt, &idf, &in, &ic, &ih, &iw));
        CHECK_STATUS(tensor4dGet(outputDesc, &odt, &odf, &on, &oc, &oh, &ow));
    } else if (tensorIs3d(inputDesc)) {
        CHECK_STATUS(tensor3dGet(inputDesc, &idt, &idf, &in, &ic, &ih));
        CHECK_STATUS(tensor3dGet(outputDesc, &odt, &odf, &on, &oc, &oh));
        iw = ow = 1;
        bw = 1;
    } else {
        return NOT_SUPPORTED;
    }
    U32 icx = ic / cx;
    for (U32 n = 0, o_i = 0; n < in; n++) {
        for (U32 c1 = 0; c1 < icx; c1++) {
            for (int c2 = 0; c2 < cx; c2++) {
                for (int i = 0; i < bh; i++) {
                    for (int j = 0; j < bw; j++) {
                        for (U32 h = 0; h < oh; h++) {
                            for (U32 w = 0; w < ow; w++, o_i++) {
                                int i_i =
                                    (((n * icx + c1) * ih + h * bh + i) * iw + w * bw + j) * cx + c2;
                                output[o_i] = input[i_i];
                            }
                        }
                    }
                }
            }
        }
    }
    return SUCCESS;
}

template <typename T>
static EE space2depth_kernel(
    TensorDesc inputDesc, T *input, Space2DepthParamSpec p, TensorDesc outputDesc, T *output)
{
    EE ret;
    DataFormat idf = inputDesc.df;
    if (idf == DF_NCHWC8) {
        ret = space2depth_kernel<T, 8>(inputDesc, input, p, outputDesc, output);
    } else if (idf == DF_NCHWC16) {
        ret = space2depth_kernel<T, 16>(inputDesc, input, p, outputDesc, output);
    } else {
        ret = space2depth_kernel<T, 1>(inputDesc, input, p, outputDesc, output);
    }
    return ret;
}

EE space2depth_cpu(
    TensorDesc inputDesc, void *input, Space2DepthParamSpec p, TensorDesc outputDesc, void *output)
{
    if (nullptr == input || nullptr == output) {
        CHECK_STATUS(NULL_POINTER);
    }
    EE ret = NOT_SUPPORTED;
    switch (inputDesc.dt) {
#ifdef _USE_FP32
        case DT_F32:
            ret = space2depth_kernel<F32>(inputDesc, (F32 *)input, p, outputDesc, (F32 *)output);
            break;
#endif
#ifdef _USE_FP16
        case DT_F16:
            ret = space2depth_kernel<F16>(inputDesc, (F16 *)input, p, outputDesc, (F16 *)output);
            break;
#endif
        default:
            break;
    }
    return ret;
}
