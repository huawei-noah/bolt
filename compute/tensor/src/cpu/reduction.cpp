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

template <typename T, typename T1>
static EE reduction_kernel(const TensorDesc &inputDesc,
    const T *input,
    const TensorDesc &maskDesc,
    const T1 *mask,
    I32 axis,
    ReductionMode reductionMode,
    const TensorDesc & outputDesc,
    T *output,
    Arch arch)
{
    if (nullptr == input || nullptr == output) {
        CHECK_STATUS(NULL_POINTER);
    }
    if (axis < 0) {
        axis = inputDesc.nDims + axis;
    }
    axis = inputDesc.nDims - 1 - axis;
    U32 loopInner = 1;
    for (int i = 0; i < axis; i++) {
        loopInner *= inputDesc.dims[i];
    }
    U32 loopOuter = 1;
    for (U32 i = axis + 1; i < inputDesc.nDims; i++) {
        loopOuter *= inputDesc.dims[i];
    }
    U32 len = inputDesc.dims[axis];
    U32 maskLen = tensorNumElements(maskDesc);
    maskLen = (maskLen > 0) ? maskLen : len;
    U32 axisDim = maskLen / len;
    EE ret = SUCCESS;
#ifdef _USE_OPENMP
#pragma omp parallel num_threads(OMP_NUM_THREADS)
#endif
    {
        ArraySumFunction sum_func = get_array_sum_function(arch);
        ArrayMeanFunction mean_func = get_array_mean_function(arch);
        ArrayVarFunction var_func = get_array_var_function(arch);
        ArrayAddFunction add_func = get_array_add_function(arch);
        ArrayMulAndAddFunction mul_and_add_func = get_array_mul_and_add_function(arch);
        ArrayScaleFunction scale_func = get_array_scale_function(arch);
        ArrayMinMaxValueFunction minmax_value_func = get_array_minmax_value_function(arch);
        ArrayMaxFunction max_func = get_array_max_function(arch);
#ifdef _USE_OPENMP
#pragma omp for
#endif
        for (U32 i = 0; i < loopOuter; i++) {
            if (loopInner == 1) {
                const T *array = input + i * len;
                F32 tmpValue = 0;
                switch (reductionMode) {
                    case REDUCTION_SUM:
                        output[i] = sum_func(inputDesc.dt, array, len);
                        break;
                    case REDUCTION_MEAN:
                        output[i] = mean_func(inputDesc.dt, array, len);
                        break;
                    case REDUCTION_STD_DEVIATION: {
                        tmpValue = mean_func(inputDesc.dt, array, len);
                        tmpValue = var_func(inputDesc.dt, array, len, tmpValue);
                        output[i] = sqrt(tmpValue);
                        break;
                    }
                    case REDUCTION_SCALAR_PRODUCT:
                        output[i] = var_func(inputDesc.dt, array, len, 0);
                        break;
                    case REDUCTION_MAX: {
                        F32 maxValue = 0;
                        CHECK_STATUS(minmax_value_func(inputDesc.dt, array, len, 2, &maxValue));
                        output[i] = maxValue;
                        break;
                    }
                    case REDUCTION_L2: {
                        tmpValue = var_func(inputDesc.dt, array, len, 0) * len;
                        output[i] = sqrt(tmpValue);
                        break;
                    }
                    case REDUCTION_MIN: {
                        F32 minValue = 0;
                        CHECK_STATUS(minmax_value_func(inputDesc.dt, array, len, 1, &minValue));
                        output[i] = minValue;
                        break;
                    }
                    default:
                        ret = NOT_SUPPORTED;
                        break;
                }
            } else {
                for (U32 j = 0; j < maskLen; j += len) {
                    U32 axisIndex = j / len;
                    U32 outputIndex = (i * axisDim + axisIndex) * loopInner;
                    auto ptr2 = output + outputIndex;
                    for (U32 k = 0; k < len; k++) {
                        if (mask == nullptr || (mask != nullptr && mask[j + k] == 1)) {
                            auto ptr1 = &input[(i * len + k) * loopInner];
                            if ((k == 0) && (reductionMode != REDUCTION_SCALAR_PRODUCT)) {
                                UNI_MEMCPY(ptr2, ptr1, loopInner * bytesOf(inputDesc.dt));
                                continue;
                            }
                            if (reductionMode == REDUCTION_SUM || reductionMode == REDUCTION_MEAN) {
                                add_func(inputDesc.dt, ptr2, ptr1, ptr2, loopInner);
                            } else if (reductionMode == REDUCTION_SCALAR_PRODUCT) {
                                mul_and_add_func(inputDesc.dt, ptr1, ptr1, ptr2, ptr2, loopInner);
                            } else if (reductionMode == REDUCTION_MAX) {
                                max_func(inputDesc.dt, ptr2, ptr1, ptr2, loopInner);
                            } else {
                                ret = NOT_SUPPORTED;
                            }
                        }
                    }
                    if (reductionMode == REDUCTION_MEAN) {
                        scale_func(inputDesc.dt, ptr2, ptr2, loopInner, 1.0 / len, 0);
                    }
                }
            }
        }
    }
    return ret;
}

template <typename T>
static EE reduction_kernel(const TensorDesc &inputDesc,
    const T *input,
    const TensorDesc &maskDesc,
    const void *mask,
    I32 axis,
    ReductionMode reductionMode,
    const TensorDesc & outputDesc,
    T *output,
    Arch arch)
{
    EE ret = NOT_SUPPORTED;
    DataType mdt = maskDesc.dt;
    if (mask == nullptr || mdt == DT_F32) {
        ret = reduction_kernel<T, F32>(inputDesc, input,
        maskDesc, (const F32 *)mask,
        axis, reductionMode,
        outputDesc, output, arch);
#ifdef _USE_FP16
    } else if (mdt == DT_F16) {
        ret = reduction_kernel<T, F16>(inputDesc, input,
        maskDesc, (const F16 *)mask,
        axis, reductionMode,
        outputDesc, output, arch);
#endif
    }
    return ret;
}
EE reduction_cpu(TensorDesc inputDesc,
    const void *input,
    TensorDesc maskDesc,
    const void *mask,
    ReductionParamSpec p,
    int tmpBytes,
    void *tmp,
    TensorDesc outputDesc,
    void *output,
    Arch arch)
{
    EE ret = SUCCESS;
    ArrayScaleFunction scale_func = get_array_scale_function(arch);
    int start = 0;
    TensorDesc tmpDesc = inputDesc;
    int channel = tmpDesc.nDims - 1;
    if (inputDesc.df == DF_NCHWC8 || inputDesc.df == DF_NCHWC16) {
        U32 cx = (inputDesc.df == DF_NCHWC8) ? 8 : 16;
        for (int i = 0; i < p.num_axes; i++) {
            // channel dimension
            if (p.axes[i] == 1 || p.axes[i] == -channel) {
                start = -1;
                break;
            }
        }
        for (int i = (int)inputDesc.nDims - 1; i >= 0; i--) {
            tmpDesc.dims[i + 1] = tmpDesc.dims[i];
        }
        tmpDesc.dims[channel] /= cx;
        tmpDesc.dims[0] = cx;
        tmpDesc.nDims += 1;
    }
    const void *tmp1 = input;
    void *tmp2 = nullptr;
    int chunk = UNI_MAX(tensorNumBytes(inputDesc), tensorNumBytes(outputDesc));
    for (int i = start; i < p.num_axes; i++) {
        if (p.num_axes - start == 1) {
            tmp2 = output;
        } else {
            tmp2 = (char *)tmp + (i - start) % 2 * chunk;
        }
        int axis;
        if (i == -1) {
            axis = tmpDesc.nDims - 1;
        } else {
            axis = p.axes[i];
        }

        switch (inputDesc.dt) {
#ifdef _USE_FP32
            case DT_F32: {
                ret = reduction_kernel<F32>(tmpDesc, (const F32 *)tmp1, maskDesc,
                    mask, axis, p.mode, outputDesc, (F32 *)tmp2, arch);
                break;
            }
#endif
#ifdef _USE_FP16
            case DT_F16: {
                ret = reduction_kernel<F16>(tmpDesc, (const F16 *)tmp1, maskDesc,
                    mask, axis, p.mode, outputDesc, (F16 *)tmp2, arch);
                break;
            }
#endif
            case DT_I32: {
                ret = reduction_kernel<I32>(tmpDesc, (const I32 *)tmp1, maskDesc,
                    mask, axis, p.mode, outputDesc, (I32 *)tmp2, arch);
                break;
            }
            case DT_U32: {
                ret = reduction_kernel<U32>(tmpDesc, (const U32 *)tmp1, maskDesc,
                    mask, axis, p.mode, outputDesc, (U32 *)tmp2, arch);
                break;
            }
            case DT_I8: {
                ret = reduction_kernel<INT8>(tmpDesc, (const INT8 *)tmp1, maskDesc,
                    mask, axis, p.mode, outputDesc, (INT8 *)tmp2, arch);
                break;
            }
            default:
                ret = NOT_SUPPORTED;
                break;
        }
        tmp1 = tmp2;
        if (axis < 0) {
            axis = tmpDesc.nDims + axis;
        }
        axis = tmpDesc.nDims - 1 - axis;
        tmpDesc.dims[axis] = 1;
    }

    if (tmp2 != output) {
        UNI_MEMCPY(output, tmp2, tensorNumBytes(outputDesc));
    }

    if (p.coeff != 1) {
        scale_func(outputDesc.dt, output, output, tensorNumElements(outputDesc), p.coeff, 0);
    }

    return ret;
}
