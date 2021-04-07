// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "cpu/general/tensor_computing_general.h"
#include "uni.h"

template <typename T>
static EE attention_mask(TensorDesc inputDesc,
    const T *input,
    I32 attentionLength,
    bool sameLength,
    float maskValue,
    TensorDesc outputDesc,
    T *output)
{
    UNUSED(outputDesc);
    if (nullptr == input || nullptr == output) {
        CHECK_STATUS(NULL_POINTER);
    }
    int qlen = inputDesc.dims[1];
    int klen = inputDesc.dims[0];
    int mlen = klen - qlen;
    std::vector<std::vector<T>> mask;
    if (attentionLength < 0) {
        mask = std::vector<std::vector<T>>(qlen, std::vector<T>(klen, 0));
    } else {
        mask = std::vector<std::vector<T>>(qlen, std::vector<T>(klen, 1));
        for (int i = 0; i < qlen; i++) {
            int start, loops;
            if (attentionLength > 0) {
                int end = mlen + i;
                start = UNI_MAX(end - attentionLength, 0);
                loops = end - start + 1;
            } else {
                if (sameLength) {
                    start = i;
                    loops = qlen + 1;
                } else {
                    start = 0;
                    loops = i + qlen + 1;
                }
            }
            loops = UNI_MAX(loops, 0);
            start = UNI_MIN(start, klen);
            if (start + loops > klen) {
                loops = UNI_MAX(klen - start, 0);
            }
            memset(&mask[i][start], 0, sizeof(T) * loops);
        }
    }
    I32 loops = tensorNumElements(inputDesc) / qlen / klen;
    for (int i = 0, index = 0; i < loops; i++) {
        for (int j = 0; j < qlen; j++) {
            for (int k = 0; k < klen; k++) {
                output[index] = input[index] * (1 - mask[j][k]) - maskValue * mask[j][k];
                index++;
            }
        }
    }
    return SUCCESS;
}

EE attention_mask_general(TensorDesc inputDesc,
    const void *input,
    AttentionMaskParamSpec p,
    TensorDesc outputDesc,
    void *output)
{
    DataType idt = inputDesc.dt;
    EE ret = SUCCESS;
    switch (idt) {
#ifdef _USE_FP32
        case DT_F32: {
            ret = attention_mask<F32>(inputDesc, (const F32 *)input, p.attention_length,
                p.same_length, p.mask, outputDesc, (F32 *)output);
            break;
        }
#endif
#ifdef _USE_FP16
        case DT_F16: {
            ret = attention_mask<F16>(inputDesc, (const F16 *)input, p.attention_length,
                p.same_length, p.mask, outputDesc, (F16 *)output);
            break;
        }
#endif
        default:
            ret = NOT_SUPPORTED;
            break;
    }

    return ret;
}
