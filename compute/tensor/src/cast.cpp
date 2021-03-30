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
#ifdef _USE_MALI
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
    EE ret = NOT_SUPPORTED;
    if (IS_MALI_GPU(archInfo->arch)) {
#ifdef _USE_MALI
        GCLMemDesc gclmemInputDesc = ocl_get_desc(*inputTensor);
        GCLMemDesc gclmemOutputDesc = ocl_get_desc(*outputTensor);
        ret = cast_infer_output_size_mali(
            inputDesc, p, &outputDesc, &gclmemInputDesc, &gclmemOutputDesc);
        ocl_set_desc(inputTensor, gclmemInputDesc);
        ocl_set_desc(outputTensor, gclmemOutputDesc);
#endif
    } else {
        outputDesc = inputDesc;
        switch (p.targetDt) {
            case DT_I32: {
                outputDesc.dt = DT_I32;
                ret = SUCCESS;
                break;
            }
            default:
                ret = NOT_SUPPORTED;
                break;
        }
    }
    outputTensor->resize(outputDesc);
    return ret;
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
            default:
                ret = NOT_SUPPORTED;
                break;
        }
#endif
#ifdef _USE_MALI
    } else if (IS_MALI_GPU(arch)) {
        TensorDesc outputDesc = outputTensor.get_desc();
        ret = cast_mali(((MaliPara_t)(archInfo->archPara))->handle, inputDesc, (GCLMem_t)input, p,
            outputDesc, (GCLMem_t)output);
#endif
    }
    return ret;
}
