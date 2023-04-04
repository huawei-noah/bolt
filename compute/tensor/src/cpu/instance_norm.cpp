// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#include "cpu/cpu_functions.h"
#ifdef _USE_X86
#include "cpu/x86/tensor_computing_x86.h"
#endif

template <typename T, int cx = 8>
inline EE instance_norm_template(
    TensorDesc inputDesc, T *input, T *tmp, T *scale, T *bias, InstanceNormParamSpec p, T *output)
{
    if (nullptr == input || nullptr == output) {
        CHECK_STATUS(NULL_POINTER);
    }
    if (inputDesc.df != DF_NCHWC8) {
        CHECK_STATUS(NOT_MATCH);
    }

    I32 axis = (p.axis + inputDesc.nDims) % inputDesc.nDims;
    axis = inputDesc.nDims - 1 - axis;

    // support axisDim != inputDesc.dims[axis]
    U32 axisDim = (p.axis_dim > 0) ? p.axis_dim : inputDesc.dims[axis];

    I32 loopInner = 1;
    for (I32 i = 0; i < axis; ++i) {
        loopInner *= inputDesc.dims[i];
    }

    I32 loopOuter = 1;
    for (U32 i = axis; i < inputDesc.nDims; ++i) {
        loopOuter *= inputDesc.dims[i];
    }

    // eps will cause some error after sqrt
    const F32 eps = 1e-6;
    if (axisDim == inputDesc.dims[axis]) {
#ifdef _USE_OPENMP
#pragma omp parallel for num_threads(OMP_NUM_THREADS)
#endif
        for (I32 i = 0; i < loopOuter; i += cx) {
            I32 offset = i * loopInner;
            F32 mean[cx] = {0};
            for (I32 j = 0, jj = offset; j < loopInner; ++j, jj += cx) {
                for (U32 ii = 0; ii < cx; ++ii) {
                    mean[ii] += input[jj + ii];
                }
            }
            for (U32 ii = 0; ii < cx; ++ii) {
                mean[ii] = mean[ii] / loopInner;
            }
            F32 var[cx] = {0};
            for (I32 j = 0, jj = offset; j < loopInner; ++j, jj += cx) {
                for (U32 ii = 0; ii < cx; ++ii) {
                    F32 tmpVal = input[jj + ii] - mean[ii];
                    var[ii] += tmpVal * tmpVal;
                }
            }
            F32 *s = var;
            F32 *b = mean;
            for (U32 ii = 0; ii < cx; ++ii) {
                var[ii] = sqrt(var[ii] / loopInner + eps);
                int c = (i + ii) % axisDim;
                s[ii] = scale[c] / var[ii];
                b[ii] = bias[c] - s[ii] * mean[ii];
            }
            for (I32 j = 0, jj = offset; j < loopInner; ++j, jj += cx) {
                for (U32 ii = 0; ii < cx; ++ii) {
                    output[jj + ii] = s[ii] * input[jj + ii] + b[ii];
                }
            }
        }
    } else {
        I32 loopInnerIn = loopInner * inputDesc.dims[axis] * 1.0f / axisDim;
        I32 loopOuterIn = loopOuter * axisDim * 1.0f / inputDesc.dims[axis];

        if (loopOuterIn == 1) {
            F32 mean = 0;
            F32 var = 0;
            for (I32 j = 0; j < loopInnerIn; ++j) {
                mean += input[j];
            }
            mean = mean / loopInnerIn;

            for (I32 j = 0; j < loopInnerIn; ++j) {
                F32 tmpVal = input[j] - mean;
                var += tmpVal * tmpVal;
            }
            var = sqrt(var / loopInnerIn + eps);

            for (I32 j = 0; j < loopInnerIn; ++j) {
                output[j] = scale[0] * (input[j] - mean) / var + bias[0];
            }
            return SUCCESS;
        }

        T *mean = tmp;
        T *var = tmp + loopOuterIn;
        T *tmpI = tmp + loopOuterIn * 2;

        //transform to NCHW
        DataType idt;
        DataFormat idf;
        U32 in, ic, ih, iw;
        if (tensorIs4d(inputDesc)) {
            CHECK_STATUS(tensor4dGet(inputDesc, &idt, &idf, &in, &ic, &ih, &iw));
        } else if (tensorIs3d(inputDesc)) {
            CHECK_STATUS(tensor3dGet(inputDesc, &idt, &idf, &in, &ic, &ih));
            iw = 1;
        } else {
            return NOT_SUPPORTED;
        }
        ic /= cx;
        for (U32 n = 0; n < in; n++) {
            for (U32 c = 0; c < ic; c++) {
                for (U32 hw = 0; hw < ih * iw; hw++) {
                    for (U32 c8 = 0; c8 < cx; c8++) {
                        tmpI[n * ic * cx * ih * iw + (c * cx + c8) * ih * iw + hw] =
                            input[n * ic * ih * iw * cx + c * ih * iw * cx + hw * cx + c8];
                    }
                }
            }
        }

#ifdef _USE_OPENMP
#pragma omp parallel for num_threads(OMP_NUM_THREADS)
#endif
        for (I32 i = 0; i < loopOuterIn; ++i) {
            mean[i] = 0;
            for (I32 j = 0; j < loopInnerIn; ++j) {
                mean[i] += tmpI[i * loopInnerIn + j];
            }
            mean[i] = mean[i] / loopInnerIn;
            var[i] = 0;
            for (I32 j = 0; j < loopInnerIn; ++j) {
                F32 tmpVal = tmpI[i * loopInnerIn + j] - mean[i];
                var[i] += tmpVal * tmpVal;
            }
            var[i] = 1.0f / sqrt(var[i] / loopInnerIn + eps);
        }

#ifdef _USE_OPENMP
#pragma omp parallel for num_threads(OMP_NUM_THREADS)
#endif
        for (I32 i = 0; i < loopOuter; i += cx) {
            for (I32 j = 0; j < loopInner; ++j) {
                for (I32 ii = 0; ii < cx; ++ii) {
                    I32 idx = (((i + ii) * loopInner + j) / loopInnerIn) % axisDim;
                    output[i * loopInner + j * cx + ii] =
                        scale[idx] * (input[i * loopInner + j * cx + ii] - mean[idx]) * var[idx] +
                        bias[idx];
                }
            }
        }
    }

    return SUCCESS;
}

EE instance_norm_infer_forward_tmp_bytes_cpu(
    TensorDesc inputDesc, InstanceNormParamSpec p, U32 *bytes)
{
    EE ret = SUCCESS;

    I32 axis = (p.axis + inputDesc.nDims) % inputDesc.nDims;
    axis = inputDesc.nDims - 1 - axis;

    // support axisDim != inputDesc.dims[axis]
    U32 axisDim = (p.axis_dim > 0) ? p.axis_dim : inputDesc.dims[axis];
    if (axisDim == inputDesc.dims[axis]) {
        *bytes = 0;
        return ret;
    }

    I32 loopInner = 1;
    for (I32 i = 0; i < axis; ++i) {
        loopInner *= inputDesc.dims[i];
    }
    loopInner = loopInner * inputDesc.dims[axis] * 1.0f / axisDim;

    I32 loopOuter = axisDim;
    for (U32 i = axis + 1; i < inputDesc.nDims; ++i) {
        loopOuter *= inputDesc.dims[i];
    }
    *bytes = loopOuter * loopInner * bytesOf(inputDesc.dt);
    *bytes += loopInner * 2 * bytesOf(inputDesc.dt);
    return ret;
}

EE instance_norm_cpu(TensorDesc inputDesc,
    void *input,
    void *scale,
    void *bias,
    InstanceNormParamSpec p,
    void *tmp,
    TensorDesc outputDesc,
    void *output,
    Arch arch)
{
    DataType idt = inputDesc.dt;
    DataFormat idf = inputDesc.df;
    I32 axis = (p.axis + inputDesc.nDims) % inputDesc.nDims;
    axis = inputDesc.nDims - 1 - axis;

    // support axisDim != inputDesc.dims[axis]
    I32 axisDim = (p.axis_dim > 0) ? p.axis_dim : inputDesc.dims[axis];

    EE ret = NOT_SUPPORTED;
    if (idf != DF_NCHWC8) {
        ArrayMeanFunction mean_func = get_array_mean_function(arch);
        ArrayVarFunction var_func = get_array_var_function(arch);
        ArrayNormScalarScaleFunction norm_scale_func = get_norm_scalar_scale_function(arch);

        I32 loopInner = 1;
        for (I32 i = 0; i < axis; ++i) {
            loopInner *= inputDesc.dims[i];
        }
        loopInner = loopInner * inputDesc.dims[axis] * 1.0f / axisDim;

        I32 loopOuter = axisDim;
        for (U32 i = axis + 1; i < inputDesc.nDims; ++i) {
            loopOuter *= inputDesc.dims[i];
        }
#ifdef _USE_OPENMP
#pragma omp parallel for num_threads(OMP_NUM_THREADS)
#endif
        for (I32 i = 0; i < loopOuter; ++i) {
            I32 off = i * loopInner * bytesOf(idt);
            F32 mean = mean_func(idt, (U8 *)input + off, loopInner);
            F32 var = var_func(idt, (U8 *)input + off, loopInner, mean);
            I32 c = i % axisDim;
            norm_scale_func(idt, (U8 *)input + off, (U8 *)output + off, loopInner, mean, var,
                (U8 *)scale + c * bytesOf(idt), (U8 *)bias + c * bytesOf(idt));
        }
        ret = SUCCESS;
    } else {
        if (IS_GENERAL(arch) || IS_ARM(arch)) {
#if defined(_USE_GENERAL) || defined(_USE_NEON)
            switch (idt) {
#ifdef _USE_FP32
                case DT_F32: {
                    ret = instance_norm_template<F32>(inputDesc, (F32 *)input, (F32 *)tmp,
                        (F32 *)scale, (F32 *)bias, p, (F32 *)output);
                    break;
                }
#endif
#ifdef _USE_FP16
                case DT_F16: {
                    ret = instance_norm_template<F16>(inputDesc, (F16 *)input, (F16 *)tmp,
                        (F16 *)scale, (F16 *)bias, p, (F16 *)output);
                    break;
                }
#endif
                default:
                    ret = NOT_SUPPORTED;
                    break;
            }
#endif
#ifdef _USE_X86
        } else if (IS_X86(arch)) {
            ret = instance_norm_x86(inputDesc, input, tmp, scale, bias, p, output);
#endif
        }
    }
    return ret;
}
