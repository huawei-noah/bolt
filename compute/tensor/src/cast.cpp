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
#ifdef _USE_GPU
#include "gpu/mali/tensor_computing_mali.h"
#endif

EE cast_infer_output_size(
    Tensor *inputTensor, Tensor *outputTensor, CastParamSpec p, ArchInfo_t archInfo)
{
    if (inputTensor == nullptr) {
        CHECK_STATUS(NULL_POINTER);
    }
    if (outputTensor == nullptr) {
        CHECK_STATUS(NULL_POINTER);
    }
    TensorDesc inputDesc = inputTensor->get_desc();
    TensorDesc outputDesc = outputTensor->get_desc();
    outputDesc = inputDesc;
    outputDesc.dt = p.targetDt;
    if (IS_GPU(archInfo->arch)) {
#ifdef _USE_GPU
        if (outputDesc.dt != DT_I32 && outputDesc.dt != DT_F16) {
            CHECK_STATUS(NOT_SUPPORTED);
        }
#endif
    }
    outputTensor->resize(outputDesc);
    return SUCCESS;
}

template <typename TI, typename TO>
static EE diffSourceCastKernel(U32 len, TI *inputPtr, TO *outputPtr)
{
    for (U32 i = 0; i < len; ++i) {
        outputPtr[i] = (TO)(inputPtr[i]);
    }
    return SUCCESS;
}

template <typename T>
static EE diffSourceCast(TensorDesc inputDesc, T *inputPtr, void *outputPtr, CastParamSpec p)
{
    EE ret = SUCCESS;
    U32 len = tensorNumElements(inputDesc);
    switch (p.targetDt) {
        case DT_I32: {
            diffSourceCastKernel<T, I32>(len, inputPtr, (I32 *)outputPtr);
            break;
        }
        case DT_U32: {
            diffSourceCastKernel<T, U32>(len, inputPtr, (U32 *)outputPtr);
            break;
        }
#ifdef _USE_FP32
        case DT_F32: {
            diffSourceCastKernel<T, F32>(len, inputPtr, (F32 *)outputPtr);
            break;
        }
#endif
#ifdef _USE_FP16
        case DT_F16: {
            diffSourceCastKernel<T, F16>(len, inputPtr, (F16 *)outputPtr);
            break;
        }
#endif
        case DT_U8: {
            diffSourceCastKernel<T, U8>(len, inputPtr, (U8 *)outputPtr);
            break;
        }
        case DT_I8: {
            diffSourceCastKernel<T, INT8>(len, inputPtr, (INT8 *)outputPtr);
            break;
        }
        default:
            ret = NOT_SUPPORTED;
            break;
    }
    return ret;
}

EE cast(Tensor inputTensor, Tensor outputTensor, CastParamSpec p, ArchInfo_t archInfo)
{
    auto arch = archInfo->arch;
    TensorDesc inputDesc = inputTensor.get_desc();
    void *input = get_ptr_from_tensor(inputTensor, arch);
    void *output = get_ptr_from_tensor(outputTensor, arch);

    EE ret = NOT_SUPPORTED;
    if (IS_CPU(arch)) {
#ifdef _USE_CPU
        switch (inputDesc.dt) {
#ifdef _USE_FP32
            case DT_F32: {
                ret = diffSourceCast<F32>(inputDesc, (F32 *)input, output, p);
                break;
            }
#endif
#ifdef _USE_FP16
            case DT_F16: {
                ret = diffSourceCast<F16>(inputDesc, (F16 *)input, output, p);
                break;
            }
#endif
            case DT_U32: {
                ret = diffSourceCast<U32>(inputDesc, (U32 *)input, output, p);
                break;
            }
            case DT_I32: {
                ret = diffSourceCast<I32>(inputDesc, (I32 *)input, output, p);
                break;
            }
            case DT_U8: {
                ret = diffSourceCast<U8>(inputDesc, (U8 *)input, output, p);
                break;
            }
            case DT_I8: {
                ret = diffSourceCast<INT8>(inputDesc, (INT8 *)input, output, p);
                break;
            }
            default:
                ret = NOT_SUPPORTED;
                break;
        }
#endif
#ifdef _USE_GPU
    } else if (IS_GPU(arch)) {
        TensorDesc outputDesc = outputTensor.get_desc();
        ret = cast_mali(((MaliPara_t)(archInfo->archPara))->handle, inputDesc, (GCLMem_t)input, p,
            outputDesc, (GCLMem_t)output);
#endif
    }
    return ret;
}
