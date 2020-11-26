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

EE embedding_cpu(TensorDesc inputDesc,
    void *input,
    void *weight,
    EmbedParamSpec p,
    TensorDesc outputDesc,
    void *output)
{
    U8 *weightPtr = (U8 *)weight;
    U8 *outputPtr = (U8 *)output;
    U32 len = tensorNumElements(inputDesc);
    U32 elementBytes = bytesOf(outputDesc.dt);
    U32 wordEmbeddingCPUBytes = elementBytes * p.num_output;
    U32 transposeStride = elementBytes * p.input_dim;
    EE ret = SUCCESS;
    for (U32 i = 0; i < len; i++) {
        U32 wordIndex = 0;
        switch (inputDesc.dt) {
            case DT_U32:
                wordIndex = ((U32 *)input)[i];
                break;
            case DT_I32:
                wordIndex = ((I32 *)input)[i];
                break;
            case DT_F32:
                wordIndex = ((F32 *)input)[i];
                break;
#ifdef _USE_FP16
            case DT_F16:
                wordIndex = ((F16 *)input)[i];
                break;
#endif
            default:
                ret = NOT_SUPPORTED;
                break;
        }
        U8 *dest = outputPtr;
        if (p.transpose) {
            U8 *src = weightPtr + wordIndex * elementBytes;
            for (U32 j = 0; j < p.num_output; j++) {
                memcpy(dest, src, elementBytes);
                src += transposeStride;
                dest += elementBytes;
            }
        } else {
            U8 *src = weightPtr + wordIndex * wordEmbeddingCPUBytes;
            memcpy(dest, src, wordEmbeddingCPUBytes);
        }
        outputPtr += wordEmbeddingCPUBytes;
    }
    return ret;
}
