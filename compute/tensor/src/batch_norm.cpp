// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "tensor_computing.h"
#include "cpu/cpu_functions.h"

EE batch_norm_infer_output_size(
    Tensor *inputTensor, BatchNormParamSpec bnParamSpec, Tensor *outputTensor, ArchInfo_t archInfo)
{
    UNUSED(bnParamSpec);
    if (inputTensor == nullptr || outputTensor == nullptr) {
        return NULL_POINTER;
    }
    Arch arch = archInfo->arch;
    EE ret = NOT_SUPPORTED;
    if (IS_CPU(arch)) {
        outputTensor->resize(inputTensor->get_desc());
        ret = SUCCESS;
    }
    return ret;
}

EE batch_norm_transform_filter_bytes(Tensor varianceTensor,
    Tensor meanTensor,
    BatchNormParamSpec bnParamSpec,
    U32 *bytes,
    ArchInfo_t archInfo)
{
    UNUSED(bnParamSpec);
    if (nullptr == bytes) {
        return NULL_POINTER;
    }
    Arch arch = archInfo->arch;
    EE ret = NOT_SUPPORTED;
    if (IS_CPU(arch)) {
        bytes[0] = varianceTensor.bytes();
        bytes[1] = meanTensor.bytes();
        ret = SUCCESS;
    }
    return ret;
}

// transform batch norm weight to scale weight
EE batch_norm_transform_filter(Tensor varianceTensor,
    Tensor meanTensor,
    BatchNormParamSpec bnParamSpec,
    Tensor alphaTensor,
    Tensor betaTensor,
    ArchInfo_t archInfo)
{
    Arch arch = archInfo->arch;
    void *var = get_ptr_from_tensor(varianceTensor, arch);
    void *mean = get_ptr_from_tensor(meanTensor, arch);
    void *alpha = get_ptr_from_tensor(alphaTensor, arch);
    void *beta = get_ptr_from_tensor(betaTensor, arch);
    EE ret = NOT_SUPPORTED;
    if (IS_CPU(arch)) {
        TensorDesc desc = varianceTensor.get_desc();
        DataType dt = desc.dt;
        U32 length = varianceTensor.length();
        ArrayPowerFunction power_func = get_array_power_function(arch);
        ArrayScaleFunction scale_func = get_array_scale_function(arch);
        ArrayMulFunction mul_func = get_array_mul_function(arch);
        power_func(dt, var, alpha, length, -0.5);
        scale_func(dt, mean, beta, length, -1, 0);
        mul_func(dt, alpha, beta, beta, length);
        ret = SUCCESS;
    }
    return ret;
}

EE batch_norm(Tensor inputTensor,
    Tensor alphaTensor,
    Tensor betaTensor,
    BatchNormParamSpec p,
    Tensor outputTensor,
    ArchInfo_t archInfo)
{
    auto arch = archInfo->arch;
    void *alpha = get_ptr_from_tensor(alphaTensor, arch);
    void *beta = get_ptr_from_tensor(betaTensor, arch);
    ScaleParamSpec scaleParam;
    scaleParam.axis = p.axis;
    return scale(inputTensor, alpha, beta, scaleParam, outputTensor, archInfo);
}
