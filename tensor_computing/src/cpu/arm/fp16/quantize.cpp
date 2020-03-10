// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), 
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, 
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE 
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR 
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


#include <math.h>
#include <arm_neon.h>
#include "cpu/arm/fp16/tensor_computing_fp16.h"

inline void apply_scale_f16(U32 numData, F16* array, F16 scale, INT8* qArray)
{
    for (U32 i=0; i<numData; i++) {
        F32 tmp = array[i] * scale;
        qArray[i] = round(tmp);
    }
}

EE quantize_tensor_fp16(TensorDesc dDesc, const void* data, TensorDesc* qDesc, void* qData, F16 *scale)
{
    if (nullptr == data || nullptr == qDesc || nullptr == qData || nullptr == scale) {
        CHECK_STATUS(NULL_POINTER);
    }
    DataType dt;
    DataFormat df;
    U32 n, c, h, w;
    CHECK_STATUS(tensor4dGet(dDesc, &dt, &df, &n, &c, &h, &w));
    switch (dt) {
        case DT_F16: {
            switch (df) {
                case DF_HWNCN8C4:{ // winograd
                    F16 *array = (F16*)data;
                    for (U32 idx=0; idx<36; idx++) {
                        float16x8_t tmp_v = vld1q_f16(array + idx*8*c);
                        float16x8_t max_v = tmp_v;
                        float16x8_t min_v = tmp_v;

                        for (U32 o=0; o<n; o+=8) {
                            F16 *base = array + o*36*c + idx*8*c;
                            for (U32 i=0; i<8*c; i+=8) {
                                tmp_v = vld1q_f16(base+i);
                                max_v = vmaxq_f16(max_v, tmp_v);
                                min_v = vminq_f16(min_v, tmp_v);
                            }
                        }

                        F16 max = vmaxvq_f16(max_v);
                        F16 min = vminvq_f16(min_v);
                        if (max == 0 && min == 0) {
                            return NOT_SUPPORTED;
                        }
                        if (max > 0 && min < 0) {
                            F16 scale_max = 127.0 / max;
                            F16 scale_min = -128.0 / min;
                            scale[idx] = (scale_max < scale_min) ? scale_max : scale_min;
                        } else if (max < 0) {
                            scale[idx] = -128.0 / min;
                        } else { // min > 0
                            scale[idx] = 127.0 / max;
                        }

                        INT8* qArray = (INT8*)qData;
                        for (U32 o=0; o<n; o+=8) {
                            U32 base = o*36*c + idx*8*c;
                            apply_scale_f16(8*c, array+base, scale[idx], qArray+base);
                        }
                    }
                    *qDesc = tensor4df(DT_I8, df, n, c, h, w);
                    break;
                }
                default:{
                    F16 *array = (F16*)data;
                    float16x8_t tmp_v = vld1q_f16(array);
                    float16x8_t max_v = tmp_v;
                    float16x8_t min_v = tmp_v;

                    U32 numData = n * c * h * w;
                    for (U32 i=8; i<numData; i+=8) {
                        tmp_v = vld1q_f16(array+i);
                        max_v = vmaxq_f16(max_v, tmp_v);
                        min_v = vminq_f16(min_v, tmp_v);
                    }

                    F16 max = vmaxvq_f16(max_v);
                    F16 min = vminvq_f16(min_v);
                    if (max == 0 && min == 0) {
                        return NOT_SUPPORTED;
                    }
                    if (max > 0 && min < 0) {
                        F16 scale_max = 127.0 / max;
                        F16 scale_min = -128.0 / min;
                        *scale = (scale_max < scale_min) ? scale_max : scale_min;
                    } else if (max < 0) {
                        *scale = -128.0 / min;
                    } else { // min > 0
                        *scale = 127.0 / max;
                    }

                    INT8* qArray = (INT8*)qData;
                    apply_scale_f16(numData, array, *scale, qArray);

                    *qDesc = tensor4df(DT_I8, df, n, c, h, w);
                    break;
                }
            }
            break;
        }
        default:{
            return NOT_SUPPORTED;
            break;
        }
    }
    return SUCCESS;
}
