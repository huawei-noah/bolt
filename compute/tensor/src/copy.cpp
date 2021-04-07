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

EE copy_infer_output_size(std::vector<Tensor *> inputTensor, ArchInfo_t archInfo)
{
    auto arch = archInfo->arch;
    std::vector<TensorDesc> inputDesc = get_desc_from_tensor_ptrs(inputTensor);
    EE ret = NOT_SUPPORTED;
    if (IS_MALI_GPU(arch)) {
#ifdef _USE_MALI
        std::vector<GCLMemDesc> gclmemInputDescs = ocl_get_descs_ptr(inputTensor);
        ret = copy_infer_output_size_mali(inputDesc, gclmemInputDescs.data());
        ocl_set_descs(inputTensor, gclmemInputDescs);
#endif
    }
    return ret;
}

EE copy(std::vector<Tensor> inputTensor,
    U32 srcOffset,
    U32 dstOffset,
    U32 srcStride,
    U32 dstStride,
    U32 length,
    ArchInfo_t archInfo)
{
    auto arch = archInfo->arch;
    std::vector<TensorDesc> inputDesc = get_desc_from_tensors(inputTensor);
    std::vector<void *> input = get_data_from_tensors<void *>(inputTensor, arch);

    EE ret = NOT_SUPPORTED;
    if (IS_MALI_GPU(arch)) {
#ifdef _USE_MALI
        ret = copy_mali(((MaliPara_t)(archInfo->archPara))->handle, inputDesc, input, srcOffset,
            dstOffset, srcStride, dstStride, length);
#endif
#ifdef _USE_CPU
    } else {
        memcpy((U8 *)input[1] + bytesOf(inputDesc[1].dt) * dstOffset,
            (U8 *)input[0] + bytesOf(inputDesc[0].dt) * srcOffset,
            length * bytesOf(inputDesc[0].dt));
        ret = SUCCESS;
#endif
    }
    return ret;
}
