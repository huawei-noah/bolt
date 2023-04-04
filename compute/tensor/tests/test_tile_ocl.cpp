// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,

#include "tensor_computing.h"
#include "ut_util_ocl.h"

int tileTest(int argc, char **argv, DataType dt)
{
    // input dim
    U32 nDims = atoi(argv[1]);
    if (argc != (int)(2 * nDims + 2)) {
        CHECK_STATUS(NOT_MATCH);
    }
    U32 iDim[8];
    TileParamSpec tileParamSpec;
    tileParamSpec.num_repeats = nDims;
    for (U32 i = 2; i < nDims + 2; i++) {
        iDim[i - 2] = atoi(argv[i]);
    }
    for (U32 i = nDims + 2; i < 2 * nDims + 2; i++) {
        tileParamSpec.repeats[i - nDims - 2] = atoi(argv[i]);
    }

    ArchInfo archInfo;
    archInfo.arch = MALI;
    if (gcl_check_device_qualcomm(OCLContext::getInstance().handle.get())) {
        archInfo.arch = QUALCOMM;
    }

    TensorDesc inputDesc;
    inputDesc.nDims = nDims;
    inputDesc.dt = dt;
    if (nDims < 3) {
        inputDesc.df = DF_NORMAL;
    } else if (nDims == 3) {
        inputDesc.df = DF_MTK;
    } else {
        inputDesc.df = DF_NCHW;
    }
    for (U32 i = 0; i < nDims; i++) {
        inputDesc.dims[nDims - i - 1] = iDim[i];
    }
    U32 inputLen = tensorNumElements(inputDesc);
    U8 *input_cpu = ut_input_v(inputLen, dt, UT_INIT_RANDOM);
    F16 *val = (F16 *)input_cpu;
    for (U32 i = 0; i < inputLen; i++) {
        val[i] = i;
    }
    Tensor inputTensorCpu, outputTensorCpu, tmpTensorCpu;
    inputTensorCpu.resize(inputDesc);
    inputTensorCpu.alloc();
    UNI_MEMCPY(
        get_ptr_from_tensor(inputTensorCpu, CPU_GENERAL), input_cpu, tensorNumBytes(inputDesc));

    CHECK_STATUS(tile_infer_output_size(
        &inputTensorCpu, tileParamSpec, &outputTensorCpu, &UT_SERIAL_ARCHINFO));
    outputTensorCpu.alloc();

    CHECK_STATUS(
        tile(inputTensorCpu, tileParamSpec, tmpTensorCpu, outputTensorCpu, &UT_SERIAL_ARCHINFO));
    TensorDesc outputDesc = outputTensorCpu.get_desc();
    U32 outputLen = tensorNumElements(outputDesc);

    std::shared_ptr<GCLHandle> handleSharedPtr = OCLContext::getInstance().handle;
    GCLHandle_t handle = handleSharedPtr.get();
    std::vector<GCLKernelInfo> kernelVec;
    handle->kernelVec = &kernelVec;
    MemoryType memType = OCLMem;
    if (archInfo.arch == QUALCOMM) {
        memType = OCLMemImg;
    }
    Tensor inputTensor = Tensor(memType);
    Tensor outputTensor = Tensor(memType);
    Tensor tmpTensor = Tensor(OCLMem);
    inputTensor.resize(inputDesc);

    MaliPara maliPara;
    maliPara.handle = handle;
    archInfo.archPara = &maliPara;
    CHECK_STATUS(tile_infer_output_size(&inputTensor, tileParamSpec, &outputTensor, &archInfo));
    U8 *output_gpu = ut_input_v(outputLen, dt, UT_INIT_ZERO);

    U32 maxBytes = 0;
    U32 tmpBytes = 0;
    CHECK_STATUS(tile_infer_forward_tmp_bytes(inputTensor, outputTensor, &tmpBytes, &archInfo));
    maxBytes = (tmpBytes > maxBytes) ? tmpBytes : maxBytes;

    GCLMem_t output = alloc(outputTensor);
    GCLMem_t input = alloc(inputTensor);
    CHECK_STATUS(gcl_fill_memory_zero(handle, input));
    tmpBytes = tensorNumBytes(inputDesc);
    maxBytes = (tmpBytes > maxBytes) ? tmpBytes : maxBytes;
    tmpBytes = tensorNumBytes(outputDesc);
    maxBytes = (tmpBytes > maxBytes) ? tmpBytes : maxBytes;
    GCLMem_t tmpbuf = alloc_bytes(tmpTensor, maxBytes);

    CHECK_STATUS(ocl_set_input(handle, input, inputDesc, input_cpu, tmpbuf, true));
    CHECK_STATUS(tile(inputTensor, tileParamSpec, tmpTensor, outputTensor, &archInfo));

    double time = 0;
#ifdef _DEBUG
    for (I32 i = 0; i < UT_LOOPS; i++) {
        CHECK_STATUS(gcl_run_kernelVec_timing(handle, 0, handle->kernelVec->size()));
        time += handle->t_execute * 0.001;
    }
#else
    double start = ut_time_ms();
    for (I32 i = 0; i < UT_LOOPS; i++) {
        CHECK_STATUS(gcl_run_kernelVec(handle));
        CHECK_STATUS(gcl_finish(handle));
    }
    double end = ut_time_ms();
    time = (end - start);
#endif
    time /= UT_LOOPS;

    char buffer[150];
    char params[120];
    UNI_MEMSET(params, 0, 120);
    sprintf(params, "(");
    for (U32 i = 0; i < inputDesc.nDims; i++) {
        if (i != inputDesc.nDims - 1) {
            sprintf(params + i * 2 + 1, "%d ", inputDesc.dims[inputDesc.nDims - 1 - i]);
        } else {
            sprintf(params + i * 2 + 1, "%d) = (", inputDesc.dims[inputDesc.nDims - 1 - i]);
        }
    }
    for (U32 i = 0; i < outputDesc.nDims; i++) {
        I32 index = 0;
        for (; index < 120; index++) {
            if (params[index] == '\0') {
                break;
            }
        }
        if (i != outputDesc.nDims - 1) {
            sprintf(params + index, "%d ", outputDesc.dims[outputDesc.nDims - 1 - i]);
        } else {
            sprintf(params + index, "%d)", outputDesc.dims[outputDesc.nDims - 1 - i]);
        }
    }
    sprintf(buffer, "%20s, %80s", "tile", params);
    double ops = outputLen;
    ut_log(dt, buffer, ops, time);

    CHECK_STATUS(ocl_get_output(handle, output, outputDesc, output_gpu, tmpbuf, true));
    ut_check_v(output_gpu, get_ptr_from_tensor(outputTensorCpu, CPU_GENERAL), outputLen, dt, 0.1);
    CHECK_STATUS(gcl_finish(handle));
    CHECK_STATUS(gcl_clean_kernelVec(handle));
    free(input_cpu);
    free(output_gpu);
    return 0;
}

int main(int argc, char **argv)
{
    tileTest(argc, argv, DT_F16);
    return 0;
}
