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
#ifdef _USE_CPU
#include "cpu/tensor_computing_cpu.h"
#endif

EE unsqueeze_infer_output_size(
    Tensor *inputTensor, UnsqueezeParamSpec p, Tensor *outputTensor, ArchInfo_t archInfo)
{
    if (inputTensor == nullptr || outputTensor == nullptr) {
        return NULL_POINTER;
    }
    TensorDesc inputDesc = inputTensor->get_desc();
    TensorDesc outputDesc = outputTensor->get_desc();
    EE ret = unsqueeze_infer_output_size_cpu(inputDesc, p, &outputDesc);
#ifdef _USE_CPU
    if (IS_CPU(archInfo->arch) && ret == SUCCESS && tensorIsShape(inputDesc) &&
        tensorIsShape(outputDesc)) {
        for (U32 i = 0; i < tensorNumElements(inputDesc); i++) {
            outputDesc.dims[outputDesc.nDims + i] = inputDesc.dims[inputDesc.nDims + i];
        }
    }
#endif
    outputTensor->resize(outputDesc);
    return ret;
}

EE unsqueeze_infer_forward_tmp_bytes(
    Tensor inputTensor, Tensor outputTensor, U32 *bytes, ArchInfo_t archInfo)
{
    EE ret = SUCCESS;
    *bytes = 0;
    if (IS_GPU(archInfo->arch)) {
#ifdef _USE_GPU
        TensorDesc inputDesc = inputTensor.get_desc();
        TensorDesc outputDesc = outputTensor.get_desc();
        GCLMemDesc gclmemInputDesc = ocl_get_desc(inputTensor);
        GCLMemDesc gclmemOutputDesc = ocl_get_desc(outputTensor);
        ret = unsqueeze_infer_forward_tmp_bytes_mali(
            inputDesc, gclmemInputDesc, outputDesc, gclmemOutputDesc, bytes);
#endif
    }
    return ret;
}

EE unsqueeze(Tensor inputTensor, Tensor tmpTensor, Tensor outputTensor, ArchInfo_t archInfo)
{
    auto arch = archInfo->arch;
    TensorDesc inputDesc = inputTensor.get_desc();
    void *input = get_ptr_from_tensor(inputTensor, arch);
    TensorDesc outputDesc = outputTensor.get_desc();
    void *output = get_ptr_from_tensor(outputTensor, arch);
    if (input == output) {
        return SUCCESS;
    }

    EE ret = NOT_SUPPORTED;
    if (IS_GPU(arch)) {
#ifdef _USE_GPU
        void *tmpbuf = get_ptr_from_tensor(tmpTensor, arch);
        TensorDesc outputDesc = outputTensor.get_desc();
        ret = unsqueeze_mali(((MaliPara_t)(archInfo->archPara))->handle, inputDesc, (GCLMem_t)input,
            (GCLMem_t)tmpbuf, outputDesc, (GCLMem_t)output);
#endif
#ifdef _USE_CPU
    } else {
        if ((inputDesc.df == DF_NCHWC8 || inputDesc.df == DF_NCHWC16) &&
            inputDesc.df != outputDesc.df) {
            TensorDesc nchwDesc = inputDesc;
            nchwDesc.df = DF_NCHW;
            transformToNCHW(inputDesc, input, nchwDesc, output);
        } else {
            UNI_MEMCPY(output, input, tensorNumBytes(inputDesc));
        }
        ret = SUCCESS;
#endif
    }
    return ret;
}
