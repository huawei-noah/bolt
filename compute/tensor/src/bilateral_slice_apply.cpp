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
#ifdef _USE_CPU
#include "cpu/tensor_computing_cpu.h"
#endif
#ifdef _USE_GPU
#include "gpu/mali/tensor_computing_mali.h"
#endif

static void update(TensorDesc &inputDesc, TensorDesc &guideDesc, TensorDesc &gridDesc)
{
    if (inputDesc.df == DF_NCHW && inputDesc.dims[0] == 3) {
        TensorDesc desc0 = inputDesc;
        TensorDesc desc1 = guideDesc;
        TensorDesc desc2 = gridDesc;
        U32 v = inputDesc.dims[0];
        for (U32 i = 1; i < inputDesc.nDims; i++) {
            inputDesc.dims[i - 1] = inputDesc.dims[i];
        }
        inputDesc.dims[inputDesc.nDims - 2] = v;
        inputDesc.df = DF_NHWC;

        v = guideDesc.dims[0];
        for (U32 i = 1; i < guideDesc.nDims; i++) {
            guideDesc.dims[i - 1] = guideDesc.dims[i];
        }
        guideDesc.dims[guideDesc.nDims - 2] = v;
        guideDesc.df = DF_NHWC;

        if (gridDesc.nDims != 4) {
            UNI_ERROR_LOG("currently only support 4d dimension grid, please transpose to nhwdc and "
                          "reshape to nhw(dc).\n");
        }
        v = gridDesc.dims[0];
        for (U32 i = 1; i < inputDesc.nDims; i++) {
            gridDesc.dims[i - 1] = gridDesc.dims[i];
        }
        gridDesc.dims[gridDesc.nDims - 2] = v;
        gridDesc.df = DF_NHWC;
        UNI_DEBUG_LOG("change input from %s -> %s.\n", tensorDesc2Str(desc0).c_str(),
            tensorDesc2Str(inputDesc).c_str());
        UNI_DEBUG_LOG("change input from %s -> %s.\n", tensorDesc2Str(desc1).c_str(),
            tensorDesc2Str(guideDesc).c_str());
        UNI_DEBUG_LOG("change input from %s -> %s.\n", tensorDesc2Str(desc2).c_str(),
            tensorDesc2Str(gridDesc).c_str());
    }
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
    TensorDesc inputDesc = inputTensor->get_desc();
    TensorDesc outputDesc = outputTensor->get_desc();
    if (IS_GPU(archInfo->arch)) {
#ifdef _USE_GPU
        TensorDesc guideDesc = guideTensor->get_desc();
        TensorDesc gridDesc = gridTensor->get_desc();
        OclMemory *inputMem = (OclMemory *)inputTensor->get_memory();
        OclMemory *guideMem = (OclMemory *)guideTensor->get_memory();
        OclMemory *gridMem = (OclMemory *)gridTensor->get_memory();
        OclMemory *outputMem = (OclMemory *)outputTensor->get_memory();
        update(inputDesc, guideDesc, gridDesc);
        ret = bilateral_slice_padding_input_mali(
            inputDesc, guideDesc, gridDesc, p, &outputDesc, inputMem, guideMem, gridMem, outputMem);
#endif
    } else {
        outputDesc = inputDesc;
        ret = SUCCESS;
    }
    outputTensor->resize(outputDesc);
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
    if (IS_GPU(archInfo->arch)) {
#ifdef _USE_GPU
        TensorDesc inputDesc = inputTensor.get_desc();
        TensorDesc guideDesc = guideTensor.get_desc();
        TensorDesc gridDesc = gridTensor.get_desc();
        update(inputDesc, guideDesc, gridDesc);

        ret = bilateral_slice_apply_infer_forward_tmp_bytes_mali(inputDesc, guideDesc, gridDesc, p,
            ((MaliPara_t)(archInfo->archPara))->forwardRunInfo, bytes);
#endif
    } else {
        *bytes = 0;
        ret = SUCCESS;
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
    update(inputDesc, guideDesc, gridDesc);
    EE ret = NOT_SUPPORTED;
    if (IS_GPU(arch)) {
#ifdef _USE_GPU
        ret = bilateral_slice_apply_mali(((MaliPara_t)(archInfo->archPara))->handle, inputDesc,
            (GCLMem_t)input, guideDesc, (GCLMem_t)guide, gridDesc, (GCLMem_t)grid, p,
            ((MaliPara_t)(archInfo->archPara))->forwardRunInfo, tmpBytes, (GCLMem_t)tmp, outputDesc,
            (GCLMem_t)output);
#endif
    } else {
        ret = bilateral_slice_apply_cpu(
            inputDesc, input, guideDesc, guide, gridDesc, grid, p, outputDesc, output);
    }
    return ret;
}
