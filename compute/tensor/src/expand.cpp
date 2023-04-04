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

EE expand_infer_output_size(
    Tensor *inputTensor, ExpandParamSpec p, Tensor *outputTensor, ArchInfo_t archInfo)
{
    TensorDesc inputDesc = inputTensor->get_desc();
    TensorDesc outputDesc = inputDesc;
    CHECK_REQUIREMENT((I32)inputDesc.nDims <= p.num_shape);
    outputDesc.nDims = (U32)p.num_shape;
    I32 inputDims = inputDesc.nDims;
    for (I32 i = 0; i < p.num_shape; ++i) {
        I32 reverseDim = p.num_shape - 1 - i;
        if ((reverseDim >= inputDims) ||
            (reverseDim < inputDims && inputDesc.dims[reverseDim] == 1)) {
            outputDesc.dims[reverseDim] = p.shape[i];
        } else {
            CHECK_REQUIREMENT(p.shape[i] <= (I32)inputDesc.dims[reverseDim]);
            outputDesc.dims[reverseDim] = inputDesc.dims[reverseDim];
        }
    }
    if (IS_GPU(archInfo->arch)) {
#ifdef _USE_GPU
        if (outputDesc.df == DF_NCHWC4) {
            outputDesc.df = DF_NCHW;
        }
#endif
    }
    if (outputDesc.dt == DT_F32 && outputDesc.nDims == 4 &&
        outputDesc.dims[outputDesc.nDims - 2] % 8 == 0) {
        outputDesc.df = DF_NCHWC8;
    }
    outputTensor->resize(outputDesc);
    return SUCCESS;
}

EE expand_infer_forward_tmp_bytes(
    Tensor inputTensor, Tensor outputTensor, U32 *bytes, ArchInfo_t archInfo)
{
    TensorDesc outputDesc = outputTensor.get_desc();
    TensorDesc inputDesc = inputTensor.get_desc();
    EE ret = NOT_SUPPORTED;
    if (IS_GPU(archInfo->arch)) {
#ifdef _USE_GPU
        GCLMemDesc gclmemInputDesc = ocl_get_desc(inputTensor);
        GCLMemDesc gclmemOutputDesc = ocl_get_desc(outputTensor);
        ret = expand_infer_forward_tmp_bytes_mali(
            inputDesc, outputDesc, gclmemInputDesc, gclmemOutputDesc, bytes);
#endif
    } else {
        *bytes = 0;
        if (!isSameDataFormat(outputDesc.df, inputDesc.df)) {
            *bytes += tensorNumBytes(outputDesc);
        }
        ret = SUCCESS;
    }
    return ret;
}

void expand_copy_kernel(U32 dims,
    U32 inDims,
    U32 outDims,
    U32 *inD,
    U32 *outD,
    U8 *input,
    U8 *output,
    DataType dt,
    U32 lastDims,
    U32 minCopySize)
{
    if (dims >= outDims) {  // out of range
        return;
    }
    if (dims == lastDims) {
        if (dims >= inDims || inD[dims] == 1) {
            for (U32 i = 0; i < outD[dims]; ++i) {
                UNI_MEMCPY(output + i * minCopySize, input, minCopySize);
            }
        } else {
            UNI_MEMCPY(output, input, minCopySize * inD[dims]);
        }
        return;
    }

    U32 oOffSize = 1;
    for (U32 j = 0; j < dims; ++j) {
        oOffSize *= outD[j];
    }
    oOffSize *= bytesOf(dt);
    if (dims >= inDims || inD[dims] == 1) {
        expand_copy_kernel(
            dims - 1, inDims, outDims, inD, outD, input, output, dt, lastDims, minCopySize);
        for (U32 i = 1; i < outD[dims]; ++i) {
            UNI_MEMCPY(output + i * oOffSize, output, oOffSize);
        }
        return;
    }
    if (inD[dims] > 1) {
        U32 iOffSize = 1;
        for (U32 j = 0; j < dims; ++j) {
            iOffSize *= inD[j];
        }
        iOffSize *= bytesOf(dt);
        for (U32 i = 0; i < inD[dims]; ++i) {
            expand_copy_kernel(dims - 1, inDims, outDims, inD, outD, input + i * iOffSize,
                output + i * oOffSize, dt, lastDims, minCopySize);
        }
    }
}

EE expand(
    Tensor inputTensor, ExpandParamSpec p, Tensor tmpTensor, Tensor outputTensor, ArchInfo_t archInfo)
{
    auto arch = archInfo->arch;
    void *input = get_ptr_from_tensor(inputTensor, arch);
    void *output = get_ptr_from_tensor(outputTensor, arch);
    void *tmp = get_ptr_from_tensor(tmpTensor, arch);
    TensorDesc inputDesc = inputTensor.get_desc();
    TensorDesc outputDesc = outputTensor.get_desc();
    EE ret = NOT_SUPPORTED;
    if (IS_GPU(arch)) {
#ifdef _USE_GPU
        ret = expand_mali(((MaliPara_t)(archInfo->archPara))->handle, inputDesc, (GCLMem_t)input, p,
            (GCLMem_t)tmp, outputDesc, (GCLMem_t)output);
#endif
    } else {
        auto outBytes = tensorNumBytes(outputDesc);
        auto inBytes = tensorNumBytes(inputDesc);
        if (!isSameDataFormat(outputDesc.df, inputDesc.df)) {
            output = tmp;
            if (outBytes == inBytes) {
                output = input;
            }
        }
        if (outBytes == inBytes) {
            if (output != input) {
                UNI_MEMCPY(output, input, outBytes);
            }
        } else {
            CHECK_REQUIREMENT(isSameDataFormat(inputDesc.df, DF_NCHW));
            U32 lastDims = 0;
            DataType idt = inputDesc.dt;
            U32 minCopySize = bytesOf(idt);
            for (U32 i = 0; i < outputDesc.nDims; ++i) {
                if (i >= inputDesc.nDims || inputDesc.dims[i] == 1) {
                    break;
                }
                lastDims = i + 1;
                minCopySize *= inputDesc.dims[i];
            }
            expand_copy_kernel((outputDesc.nDims - 1), inputDesc.nDims, outputDesc.nDims,
                inputDesc.dims, outputDesc.dims, (U8 *)input, (U8 *)output, idt, lastDims,
                minCopySize);
        }
        ret = SUCCESS;
        if (!isSameDataFormat(outputDesc.df, inputDesc.df)) {
            TensorDesc oldDesc = outputDesc;
            oldDesc.df = inputDesc.df;
            ret = transformFormat(
                oldDesc, output, outputDesc, get_ptr_from_tensor(outputTensor, arch));
        }
    }
    return ret;
}
