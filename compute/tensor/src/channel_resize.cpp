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
#ifdef _USE_NEON
#include "cpu/arm/tensor_computing_arm.h"
#endif
#ifdef _USE_MALI
#include "gpu/mali/tensor_computing_mali.h"
#endif

EE channel_resize(
    Tensor inputTensor, ChannelResizeParamSpec p, Tensor outputTensor, ArchInfo_t archInfo)
{
    auto arch = archInfo->arch;
    EE ret = NOT_SUPPORTED;
    if (IS_MALI_GPU(arch)) {
#ifdef _USE_MALI
        TensorDesc inputDesc = inputTensor.get_desc();
        void *input = get_ptr_from_tensor(inputTensor, arch);
        TensorDesc outputDesc = outputTensor.get_desc();
        void *output = get_ptr_from_tensor(outputTensor, arch);
        ret = channel_resize_mali(((MaliPara_t)(archInfo->archPara))->handle, inputDesc,
            (GCLMem_t)input, p, outputDesc, (GCLMem_t)output);
#endif
    }
    return ret;
}

EE channel_resize_infer_output_size(
    Tensor *inputTensor, ChannelResizeParamSpec p, Tensor *outputTensor, ArchInfo_t archInfo)
{
    if (inputTensor == nullptr) {
        CHECK_STATUS(NULL_POINTER);
    }
    if (outputTensor == nullptr) {
        CHECK_STATUS(NULL_POINTER);
    }
    TensorDesc outputDesc = outputTensor->get_desc();
    if (IS_MALI_GPU(archInfo->arch)) {
#ifdef _USE_MALI
        TensorDesc inputDesc = inputTensor->get_desc();
        GCLMemDesc gclmemInputDesc = ocl_get_desc(*inputTensor);
        GCLMemDesc gclmemOutputDesc = ocl_get_desc(*outputTensor);
        CHECK_STATUS(channel_resize_infer_output_size_mali(
            inputDesc, p, &outputDesc, &gclmemInputDesc, &gclmemOutputDesc))
        ocl_set_desc(inputTensor, gclmemInputDesc);
        ocl_set_desc(outputTensor, gclmemOutputDesc);
#endif
    }
    outputTensor->resize(outputDesc);
    return SUCCESS;
}
