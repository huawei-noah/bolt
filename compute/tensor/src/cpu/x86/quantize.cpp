// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "cpu/x86/tensor_computing_x86.h"
#ifdef _USE_INT8
#include "cpu/x86/int8/tensor_computing_int8.h"
#endif

EE quantize_x86(TensorDesc dDesc, const void *data, TensorDesc *qDesc, void *qData, F32 *scale, int mode)
{
    EE ret = SUCCESS;
    if (dDesc.dt == DT_F32 || dDesc.dt == DT_F32_8Q) {
        switch (qDesc->dt) {
#ifdef _USE_INT8
            case DT_I8: {
                ret = quantizeF32ToI8(dDesc, (const F32 *)data, qDesc, (INT8 *)qData, scale, mode);
                break;
            }
            case DT_U8_Q: {
                ret = quantizeF32ToU8(dDesc, (const F32 *)data, qDesc, (UINT8 *)qData, scale, mode);
                break;
            }
#endif
            default:
                ret = NOT_SUPPORTED;
                break;
        }
#ifdef _USE_INT8
    } else if (dDesc.dt == DT_U8_Q && qDesc->dt == DT_I8) {
        ret = transformU8ToI8(dDesc, (const UINT8 *)data, qDesc, (INT8 *)qData);
#endif
    } else if (dDesc.dt == DT_I32) {
        switch (qDesc->dt) {
#ifdef _USE_INT8
            case DT_I8: {
                ret = quantizeI32ToI8(dDesc, (const I32 *)data, qDesc, (INT8 *)qData, scale);
                break;
            }
            case DT_U8_Q: {
                ret = quantizeI32ToU8(dDesc, (const I32 *)data, qDesc, (UINT8 *)qData, scale);
                break;
            }
#endif
            default:
                ret = NOT_SUPPORTED;
                break;
        }
    } else {
        ret = NOT_SUPPORTED;
    }
    return ret;
}

EE quantize_bias_offsetC(const void *bias,
    TensorDesc biasDesc,
    DataType qType,
    const void *filter,
    TensorDesc filterDesc,
    const F32 *scale,
    void *offsetCBias)
{
    EE ret = SUCCESS;
    if (bias == nullptr) {
        switch (qType) {
#ifdef _USE_INT8
            case DT_I32: {
                ret = quantizeBiasOffsetCI32(
                    nullptr, biasDesc, (INT8 *)filter, filterDesc, scale, (I32 *)offsetCBias);
                break;
            }
#endif
            default:
                ret = NOT_SUPPORTED;
                break;
        }
    } else if (biasDesc.dt == DT_F32) {
        switch (qType) {
#ifdef _USE_INT8
            case DT_I32: {
                ret = quantizeBiasOffsetCI32((const F32 *)bias, biasDesc, (INT8 *)filter,
                    filterDesc, scale, (I32 *)offsetCBias);
                break;
            }
#endif
            default:
                ret = NOT_SUPPORTED;
                break;
        }
    } else if (biasDesc.dt == DT_I32) {
        switch (qType) {
#ifdef _USE_INT8
            case DT_I32: {
                if (filter == nullptr) {
                    UNI_MEMCPY(offsetCBias, bias, tensorNumBytes(biasDesc));
                } else {
                    for (U32 i = 0; i < tensorNumElements(biasDesc); ++i) {
                        ((I32 *)offsetCBias)[i] = ((I32 *)bias)[i] + ((I32 *)filter)[i];
                    }
                }
                break;
            }
#endif
            default:
                ret = NOT_SUPPORTED;
                break;
        }
    } else {
        ret = NOT_SUPPORTED;
    }
    return ret;
}
