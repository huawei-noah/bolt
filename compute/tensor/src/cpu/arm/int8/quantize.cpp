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
#include "cpu/arm/int8/tensor_computing_int8.h"
#include "cpu/arm/int8/convolution_gemm.h"

EE quantize_tensor_int32(
    TensorDesc dDesc, const void *data, TensorDesc *qDesc, void *qData, F32 *scale)
{
    if (nullptr == data || nullptr == qDesc || nullptr == qData || nullptr == scale) {
        CHECK_STATUS(NULL_POINTER);
    }
    DataType dt;
    DataFormat df;
    U32 n, c, h, w;
    if (tensorIs2d(dDesc)) {
        CHECK_STATUS(tensor2dGet(dDesc, &dt, &df, &n, &w));
        c = 1;
        h = 1;
    } else if (tensorIs3d(dDesc)) {
        CHECK_STATUS(tensor3dGet(dDesc, &dt, &df, &n, &h, &w));
        c = 1;
    } else {
        CHECK_STATUS(tensor4dGet(dDesc, &dt, &df, &n, &c, &h, &w));
    }
    switch (dt) {
        case DT_I32: {
            I32 *array = (I32 *)data;
            int32x4_t tmp_v = vld1q_s32(array);
            int32x4_t max_v = tmp_v;
            int32x4_t min_v = tmp_v;

            U32 numData = n * c * h * w;
            CHECK_REQUIREMENT(numData >= 4);
            U32 i = 4;
            for (; i < numData - 3; i += 4) {
                tmp_v = vld1q_s32(array + i);
                max_v = vmaxq_s32(max_v, tmp_v);
                min_v = vminq_s32(min_v, tmp_v);
            }

            I32 max = vmaxvq_s32(max_v);
            I32 min = vminvq_s32(min_v);
            for (; i < numData; i++) {
                I32 tmp = array[i];
                if (tmp > max) {
                    max = tmp;
                }
                if (tmp < min) {
                    min = tmp;
                }
            }
            if (max == 0 && min == 0) {
                CHECK_STATUS(NOT_SUPPORTED);
            }

            I32 factor;
            F32 scaleO;
            if (max > 0 && min < 0) {
                I32 factor_max = 127 * 16777216 / max;
                I32 factor_min = -127 * 16777216 / min;
                factor = (factor_max < factor_min) ? factor_max : factor_min;
                scaleO = (factor_max < factor_min) ? (127.0 / max) : (-127.0 / min);
            } else if (max > 0) {
                factor = 127 * 16777216 / max;
                scaleO = 127.0 / max;
            } else {
                factor = -127 * 16777216 / min;
                scaleO = -127.0 / min;
            }
            UNI_DEBUG_LOG("%d is the max I32 value, and min values is %d, %f is the derived "
                          "scale\n",
                max, min, scaleO);
            *scale *= scaleO;

            U32 main = numData / 16;
            INT8 *qArray = (INT8 *)qData;
            CHECK_STATUS(quantize_I32(main * 4, array, factor, scaleO, qArray));
            for (U32 i = main * 16; i < numData; i++) {
                qArray[i] = array[i] * scaleO;
            }

            if (tensorIs2d(dDesc)) {
                *qDesc = tensor2df(DT_I8, df, n, w);
            } else if (tensorIs3d(dDesc)) {
                *qDesc = tensor3df(DT_I8, df, n, h, w);
            } else {
                *qDesc = tensor4df(DT_I8, df, n, c, h, w);
            }
            break;
        }
        default: {
            CHECK_STATUS(NOT_SUPPORTED);
        }
    }
    return SUCCESS;
}
