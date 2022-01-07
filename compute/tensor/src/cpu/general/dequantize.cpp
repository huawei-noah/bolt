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

template <typename IT, typename OT>
inline static void dequantize_kernel(
    I32 len, IT *q, F32 scale, I32 biasLen, const OT *biasPtr, OT *d)
{
    if (0 != biasLen) {
        CHECK_REQUIREMENT(nullptr != biasPtr);
        CHECK_REQUIREMENT(len % biasLen == 0);
    }
    F32 factor = 1 / scale;
    if (biasLen > 0) {
        for (int i = 0; i < len; i += biasLen) {
            for (int j = 0; j < biasLen; j++) {
                d[i + j] = q[i + j] * factor + biasPtr[j];
            }
        }
    } else {
        for (int i = 0; i < len; i++) {
            d[i] = q[i] * factor;
        }
    }
}

template <typename OT>
inline static EE dequantize_wrapper(TensorDesc qDesc,
    void *qData,
    const F32 *scale,
    TensorDesc bDesc,
    const OT *bData,
    TensorDesc dDesc,
    OT *data)
{
    if (nullptr == data || nullptr == qData || nullptr == scale) {
        CHECK_STATUS(NULL_POINTER);
    }
    int length = tensorNumElements(qDesc);
    int biasLength = tensorNumElements(bDesc);
    EE ret = SUCCESS;
    switch (qDesc.dt) {
        case DT_I8:
            CHECK_REQUIREMENT(biasLength == 0);
            dequantize_kernel<INT8, OT>(length, (INT8 *)qData, scale[0], biasLength, bData, data);
            break;
        case DT_I32:
            dequantize_kernel<I32, OT>(length, (I32 *)qData, scale[0], biasLength, bData, data);
            break;
        default:
            ret = NOT_SUPPORTED;
            break;
    }
    return ret;
}

EE dequantize_general(TensorDesc qDesc,
    void *qData,
    const F32 *scale,
    TensorDesc bDesc,
    void *bData,
    TensorDesc dDesc,
    void *data)
{
    if (nullptr == data || nullptr == qData || nullptr == scale) {
        CHECK_STATUS(NULL_POINTER);
    }
    EE ret = NOT_SUPPORTED;
    switch (dDesc.dt) {
#if defined(_USE_FP32)
        case DT_F32:
            ret = dequantize_wrapper<F32>(
                qDesc, qData, scale, bDesc, (const F32 *)bData, dDesc, (F32 *)data);
            break;
#endif
#if defined(_USE_FP16)
        case DT_F16:
            ret = dequantize_wrapper<F16>(
                qDesc, qData, scale, bDesc, (const F16 *)bData, dDesc, (F16 *)data);
            break;
#endif
        default:
            break;
    }
    return ret;
}
