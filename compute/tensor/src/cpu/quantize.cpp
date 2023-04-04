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
#include "cpu/cpu_functions.h"
#ifdef _USE_X86
#include "cpu/x86/tensor_computing_x86.h"
#endif

EE quantize_hwncn8c4_cpu(
    TensorDesc dDesc, const void *data, TensorDesc *qDesc, void *qData, F32 *scale, Arch arch)
{
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
    } else if (tensorIs4d(dDesc)) {
        CHECK_STATUS(tensor4dGet(dDesc, &dt, &df, &n, &c, &h, &w));
    } else {
        return NOT_SUPPORTED;
    }

    F32 minmax[2] = {1, -1};
    ArrayMinMaxValueFunction minmax_value = get_array_minmax_value_function(arch);
    ArrayScaleRoundFunction scale_round = get_array_scale_round_function(arch);
    CHECK_REQUIREMENT(h * w == 36);
    for (U32 idx = 0; idx < 36; idx++) {
        F32 min = FLT_MAX;
        F32 max = -FLT_MAX;
        for (U32 o = 0; o < n; o += 8) {
            U32 base = o * 36 * c + idx * 8 * c;
            const U8 *input = (const U8 *)data + base * bytesOf(dt);
            CHECK_STATUS(minmax_value(dt, input, 8 * c, 3, minmax));
            min = UNI_MIN(min, minmax[0]);
            max = UNI_MAX(max, minmax[1]);
        }
        if (max == 0 && min == 0) {
            return NOT_SUPPORTED;
        } else {
            F32 absMax = UNI_MAX(UNI_ABS(max), UNI_ABS(min));
            scale[idx] = 127.0 / absMax;
            INT8 *qArray = (INT8 *)qData;
            for (U32 o = 0; o < n; o += 8) {
                U32 base = o * 36 * c + idx * 8 * c;
                const U8 *input = (const U8 *)data + base * bytesOf(dt);
                CHECK_STATUS(scale_round(dt, input, qArray + base, 8 * c, scale[idx], false));
            }
        }
    }
    return SUCCESS;
}

EE quantize_cpu(
    TensorDesc dDesc, const void *data, TensorDesc *qDesc, void *qData, F32 *scale, Arch arch, int mode)
{
    if (nullptr == data || nullptr == qDesc || nullptr == qData || nullptr == scale) {
        return NULL_POINTER;
    }

#if defined(_USE_X86) && defined(_USE_INT8)
    return quantize_x86(dDesc, data, qDesc, qData, scale, mode);
#endif

    DataType dt = dDesc.dt;
    *qDesc = dDesc;
    qDesc->dt = DT_I8;
    if (dDesc.df == DF_HWNCN8C4) {
        return quantize_hwncn8c4_cpu(dDesc, data, qDesc, qData, scale, arch);
    }
    F32 minmax[2] = {1, -1}; // first is min, second is max
    U32 numData = tensorNumElements(dDesc);
    F32 scaleRaw = scale[0];
    if (!mode || (scaleRaw <= 0)) {
        CHECK_STATUS(get_array_minmax_value_function(arch)(dt, data, numData, 3, minmax));
        F32 absMax = UNI_MAX(UNI_ABS(minmax[1]), UNI_ABS(minmax[0]));
        scaleRaw = 127.0 / absMax;
    }

    EE ret = SUCCESS;
    if (minmax[0] == 0 && minmax[1] == 0) {
        *scale = 1;
        UNI_MEMSET(qData, 0, tensorNumBytes(*qDesc));
    } else {
        bool clamp = false;
        if (dt == DT_I32) {
            F32 factor = scaleRaw;
            if (scale[0] < 0 && scale[1] > 0) {
                scale[0] = scale[1] * scaleRaw;
            } else if (scale[0] > 0 && scale[1] > 0) {
                scaleRaw = scale[0] / scale[1];
            }
            ret = get_array_scale_round_function(arch)(
                dt, data, (INT8 *)qData, numData, factor, (*scale) != scaleRaw);
        } else {
            if (*scale < scaleRaw) {
                *scale = scaleRaw;
            }
            ret = get_array_scale_round_function(arch)(
                dt, data, (INT8 *)qData, numData, *scale, (*scale) != scaleRaw);
        }
    }
    UNI_DETAIL_LOG(
        "tensor min value is %f, max value is %f, scale value is %f.\n", minmax[0], minmax[1], *scale);
    return ret;
}
