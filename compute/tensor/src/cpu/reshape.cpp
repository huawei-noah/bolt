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

template <typename T>
inline void trans(T *input, T *output, U32 in, U32 ic, U32 ihiw, U32 cx) {
    for (U32 n = 0; n < in; n++) {
        for (U32 c = 0; c < ic; c++) {
            for (U32 hw = 0; hw < ihiw; hw++) {
                for (U32 c8 = 0; c8 < cx; c8++) {
                    U32 iidx = ((n * ic + c) * ihiw + hw) * cx + c8;
                    U32 oidx = ((n * ic + c) * cx + c8) * ihiw + hw;
                    output[oidx] = input[iidx];
                }
            }
        }
    }
}

EE reshape_cpu(TensorDesc inputDesc, void *input, TensorDesc outputDesc, void *output)
{
    if (nullptr == input || nullptr == output) {
        CHECK_STATUS(NULL_POINTER);
    }

    if (tensorNumElements(inputDesc) != tensorNumElements(outputDesc)) {
        // Only allow the removal of padded convolution channels
        CHECK_REQUIREMENT(DF_NCHWC8 == inputDesc.df);
        CHECK_REQUIREMENT(tensorNumElements(inputDesc) >= tensorNumElements(outputDesc));
        inputDesc.df = DF_NCHW;
    }

    bool sameDim = (outputDesc.nDims == inputDesc.nDims);
    for (U32 i = 0; i < inputDesc.nDims; i++) {
        if (outputDesc.dims[i] != inputDesc.dims[i]) {
            sameDim = false;
        }
    }
    if (!sameDim && ((inputDesc.df == DF_NCHWC8) || (inputDesc.df == DF_NCHWC16))) {
        sameDim = isC8HasSameDim(inputDesc, outputDesc);
    }

    if ((DF_NCHWC8 != inputDesc.df && DF_NCHWC16 != inputDesc.df) || sameDim) {
        if (output != input) {
            UNI_MEMCPY(output, input, tensorNumBytes(outputDesc));
        }
    } else {
        CHECK_REQUIREMENT(input != output);
        DataType idt;
        DataFormat idf;
        U32 in, ic, ih, iw;
        if (tensorIs4d(inputDesc)) {
            CHECK_STATUS(tensor4dGet(inputDesc, &idt, &idf, &in, &ic, &ih, &iw));
        } else if (tensorIs3d(inputDesc)) {
            CHECK_STATUS(tensor3dGet(inputDesc, &idt, &idf, &in, &ic, &ih));
            iw = 1;
        } else {
            return NOT_SUPPORTED;
        }

        U32 cx = (DF_NCHWC8 == inputDesc.df) ? 8 : 16;
        ic /= cx;
        U32 ihiw = ih * iw;

        switch (inputDesc.dt) {
            case DT_I32:
            case DT_F32: {
                trans<F32>((F32 *)input, (F32 *)output, in, ic, ihiw, cx);
                break;
            }
#ifdef _USE_FP16
            case DT_F16: {
                trans<F16>((F16 *)input, (F16 *)output, in, ic, ihiw, cx);
                break;
            }
#endif
            case DT_U8:
            case DT_U8_Q:
            case DT_I8: {
                trans<INT8>((INT8 *)input, (INT8 *)output, in, ic, ihiw, cx);
                break;
            }
            default:
                return NOT_SUPPORTED;
        }
    }
    return SUCCESS;
}
