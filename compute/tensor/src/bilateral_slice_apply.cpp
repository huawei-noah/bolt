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

inline EE bilateral_slice_apply_infer_output_size_cpu()
{
    return NOT_SUPPORTED;
}

EE bilateral_slice_apply_infer_output_size(Tensor *inputTensor,
    Tensor *guideTensor,
    Tensor *gridTensor,
    BilateralSliceApplyParamSpec p,
    Tensor *outputTensor,
    ArchInfo_t archInfo)
{
    if (inputTensor == nullptr) {
        CHECK_STATUS(NULL_POINTER);
    }
    if (guideTensor == nullptr) {
        CHECK_STATUS(NULL_POINTER);
    }
    if (gridTensor == nullptr) {
        CHECK_STATUS(NULL_POINTER);
    }
    if (outputTensor == nullptr) {
        CHECK_STATUS(NULL_POINTER);
    }
    EE ret = NOT_SUPPORTED;
    if (IS_MALI_GPU(archInfo->arch)) {
#ifdef _USE_MALI
        TensorDesc inputDesc = inputTensor->get_desc();
        TensorDesc outputDesc = outputTensor->get_desc();
        TensorDesc guideDesc = guideTensor->get_desc();
        TensorDesc gridDesc = gridTensor->get_desc();

        GCLMemDesc gclmemInputDesc = ocl_get_desc(*inputTensor);
        GCLMemDesc gclmemGuideDesc = ocl_get_desc(*guideTensor);
        GCLMemDesc gclmemGridDesc = ocl_get_desc(*gridTensor);
        GCLMemDesc gclmemOutputDesc = ocl_get_desc(*outputTensor);
        ret = bilateral_slice_apply_infer_output_size_mali(inputDesc, guideDesc, gridDesc, p,
            &outputDesc, &gclmemInputDesc, &gclmemGuideDesc, &gclmemGridDesc, &gclmemOutputDesc);
        ocl_set_desc(inputTensor, gclmemInputDesc);
        ocl_set_desc(guideTensor, gclmemGuideDesc);
        ocl_set_desc(gridTensor, gclmemGridDesc);
        ocl_set_desc(outputTensor, gclmemOutputDesc);
        outputTensor->resize(outputDesc);
#endif
    }
    return ret;
}

EE bilateral_slice_apply_infer_forward_tmp_bytes(Tensor inputTensor,
    Tensor guideTensor,
    Tensor gridTensor,
    BilateralSliceApplyParamSpec p,
    U32 *bytes,
    ArchInfo_t archInfo)
{
    EE ret = NOT_SUPPORTED;
    if (IS_MALI_GPU(archInfo->arch)) {
#ifdef _USE_MALI
        TensorDesc inputDesc = inputTensor.get_desc();
        TensorDesc guideDesc = guideTensor.get_desc();
        TensorDesc gridDesc = gridTensor.get_desc();

        ret = bilateral_slice_apply_infer_forward_tmp_bytes_mali(inputDesc, guideDesc, gridDesc, p,
            ((MaliPara_t)(archInfo->archPara))->forwardRunInfo, bytes);
#endif
    }
    return ret;
}

EE bilateral_slice_apply(Tensor inputTensor,
    Tensor guideTensor,
    Tensor gridTensor,
    BilateralSliceApplyParamSpec p,
    Tensor tmpTensor,
    Tensor outputTensor,
    ArchInfo_t archInfo)
{
    auto arch = archInfo->arch;
    EE ret = NOT_SUPPORTED;
    if (IS_MALI_GPU(arch)) {
#ifdef _USE_MALI
        TensorDesc inputDesc = inputTensor.get_desc();
        void *input = get_ptr_from_tensor(inputTensor, arch);
        U32 tmpBytes = tmpTensor.bytes();
        void *tmp = get_ptr_from_tensor(tmpTensor, arch);
        TensorDesc outputDesc = outputTensor.get_desc();
        void *output = get_ptr_from_tensor(outputTensor, arch);
        TensorDesc guideDesc = guideTensor.get_desc();
        void *guide = get_ptr_from_tensor(guideTensor, arch);
        TensorDesc gridDesc = gridTensor.get_desc();
        void *grid = get_ptr_from_tensor(gridTensor, arch);

        ret = bilateral_slice_apply_mali(((MaliPara_t)(archInfo->archPara))->handle, inputDesc,
            (GCLMem_t)input, guideDesc, (GCLMem_t)guide, gridDesc, (GCLMem_t)grid, p,
            ((MaliPara_t)(archInfo->archPara))->forwardRunInfo, tmpBytes, (GCLMem_t)tmp, outputDesc,
            (GCLMem_t)output);
#endif
    }
    return ret;
}
