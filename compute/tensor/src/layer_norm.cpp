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
#ifdef _USE_GENERAL
#include "cpu/general/tensor_computing_general.h"
#endif
#ifdef _USE_X86
#include "cpu/x86/tensor_computing_x86.h"
#endif
#ifdef _USE_NEON
#include "cpu/arm/tensor_computing_arm.h"
#endif
#ifdef _USE_GPU
#include "gpu/mali/tensor_computing_mali.h"
#endif
#ifdef _USE_CPU
#include "cpu/tensor_computing_cpu.h"
#endif

EE layer_norm(Tensor inputTensor,
    LayerNormParamSpec p,
    Tensor alphaTensor,
    Tensor betaTensor,
    Tensor tmpTensor,
    Tensor outputTensor,
    ArchInfo_t archInfo)
{
    auto arch = archInfo->arch;
    TensorDesc inputDesc = inputTensor.get_desc();
    void *input = get_ptr_from_tensor(inputTensor, arch);
    void *alpha = get_ptr_from_tensor(alphaTensor, arch);
    void *beta = get_ptr_from_tensor(betaTensor, arch);
    TensorDesc outputDesc = outputTensor.get_desc();
    void *output = get_ptr_from_tensor(outputTensor, arch);

#ifdef _USE_INT8
    TensorDesc qDesc = outputDesc;
    if ((outputDesc.dt == DT_I8 || outputDesc.dt == DT_U8_Q) && outputDesc.dt != inputDesc.dt) {
        outputDesc.dt = inputDesc.dt;
        output = get_ptr_from_tensor(tmpTensor, arch);
    }
#endif

    EE ret = NOT_SUPPORTED;
    if (IS_GENERAL(arch)) {
#ifdef _USE_GENERAL
        ret = layer_norm_general(inputDesc, input, p, alpha, beta, outputDesc, output);
#endif
#ifdef _USE_X86
    } else if (IS_X86(arch)) {
        ret = layer_norm_x86(inputDesc, input, p, alpha, beta, outputDesc, output);
#endif
#ifdef _USE_NEON
    } else if (IS_ARM(arch)) {
        ret = layer_norm_arm(inputDesc, input, p, alpha, beta, outputDesc, output);
#endif
#ifdef _USE_GPU
    } else if (IS_GPU(arch)) {
        void *tmpbuf = get_ptr_from_tensor(tmpTensor, arch);
        if (p.axis == -1) {
            ret = layer_norm_mali(((MaliPara_t)(archInfo->archPara))->handle, inputDesc,
                (GCLMem_t)input, (GCLMem_t)alpha, (GCLMem_t)beta, (GCLMem_t)tmpbuf, outputDesc,
                (GCLMem_t)output);
        } else {
            UNI_WARNING_LOG("please close optimizeTransposeLN in "
                            "model_tools/include/OPOptimizers/LayerNormOptimizer.hpp and "
                            "reconverter model.\n");
        }
#endif
    }

#ifdef _USE_INT8
    if (qDesc.dt != outputDesc.dt) {
        F32 scaleO = -1;
        if (DT_I8 == qDesc.dt || DT_U8_Q == qDesc.dt) {
            CHECK_STATUS(quantize_cpu(outputDesc, output, &qDesc,
                get_ptr_from_tensor(outputTensor, arch), &scaleO, arch));
            outputTensor.set_scale(scaleO);
        } else {
            ret = NOT_SUPPORTED;
        }
    }
#endif

    return ret;
}

EE layer_norm_infer_output_size(Tensor *inputTensor, LayerNormParamSpec p, Tensor *outputTensor, ArchInfo_t archInfo)
{
    if (inputTensor == nullptr) {
        CHECK_STATUS(NULL_POINTER);
    }
    if (outputTensor == nullptr) {
        CHECK_STATUS(NULL_POINTER);
    }
    auto inDesc = inputTensor->get_desc();
    if (inDesc.df == DF_NCHWC8 && p.axis == -1) {
        inDesc.df = DF_NCHW;
    }
    outputTensor->resize(inDesc);
    return SUCCESS;
}

EE layer_norm_infer_forward_tmp_bytes(Tensor inputTensor, U32 *bytes, ArchInfo_t archInfo)
{
    if (bytes == nullptr) {
        CHECK_STATUS(NULL_POINTER);
    }
    EE ret = NOT_SUPPORTED;
    if (IS_GPU(archInfo->arch)) {
#ifdef _USE_GPU
        GCLMemDesc gclmemInputDesc = ocl_get_desc(inputTensor);
        ret = layer_norm_infer_forward_tmp_bytes_mali(gclmemInputDesc, bytes);
#endif
    } else {
        *bytes = 0;
        ret = SUCCESS;
    }
    return ret;
}
