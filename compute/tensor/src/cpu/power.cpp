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
#include "affinity_policy.h"
#include "uni.h"

EE power_cpu(TensorDesc inputDesc,
    void *_input,
    PowerParamSpec p,
    TensorDesc outputDesc,
    void *_output,
    Arch arch)
{
    if (nullptr == _input || nullptr == _output) {
        return NULL_POINTER;
    }
    U8 *input = (U8 *)_input;
    U8 *output = (U8 *)_output;
    int ie = bytesOf(inputDesc.dt);
    int oe = bytesOf(outputDesc.dt);
    ArrayScaleFunction scale_func = get_array_scale_function(arch);
    ArrayPowerFunction power_func = get_array_power_function(arch);
    int size = tensorNumElements(inputDesc);
#ifdef _USE_OPENMP
    int tile = UNI_MAX(64, (((size + OMP_NUM_THREADS - 1) / OMP_NUM_THREADS + 7) / 8 * 8));
#pragma omp parallel for num_threads(OMP_NUM_THREADS)
    for (int i = 0; i < size; i += tile)
#else
    int i = 0;
    int tile = size;
#endif
    {
        int num = UNI_MIN(size - i, tile);
        U8* ptr;
        if (inputDesc.dt != outputDesc.dt) {
            ptr = output + i * oe;
            CHECK_REQUIREMENT(transformDataType(inputDesc.dt, input + i * ie, NULL, outputDesc.dt,
                                  ptr, NULL, num) == 0);
        } else {
            ptr = input + i * ie;
        }
        scale_func(outputDesc.dt, ptr, output + i * oe, num, p.scale, p.shift);
        power_func(outputDesc.dt, output + i * oe, output + i * oe, num, p.power);
    }
    return SUCCESS;
}
