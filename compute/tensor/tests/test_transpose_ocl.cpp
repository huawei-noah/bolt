// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include <vector>
#include "tensor_computing.h"
#include "ut_util_ocl.h"

int transposeTest(int argc, char **argv, DataType dt)
{
    TransposeParamSpec p, p_inv;
    DataFormat df = DF_NCHW;
    TensorDesc inputDesc_cpu, inputDesc_gpu;
    U32 nDims = atoi(argv[1]);
    inputDesc_cpu.nDims = nDims;
    inputDesc_cpu.dt = dt;
    if (nDims < 3) {
        inputDesc_cpu.df = DF_NORMAL;
    } else if (nDims == 3) {
        inputDesc_cpu.df = DF_MTK;
    } else {
        inputDesc_cpu.df = DF_NCHW;
    }
    CHECK_REQUIREMENT(argc == (int)(nDims * 2 + 2));
    p.num_axes = nDims;
    p_inv.num_axes = nDims;
    for (U32 i = 0; i < nDims; i++) {
        inputDesc_cpu.dims[nDims - 1 - i] = atoi(argv[2 + i]);
        I32 value = atoi(argv[2 + nDims + i]);
        p.axes[i] = value;
        p_inv.axes[value] = i;
    }
    inputDesc_gpu = inputDesc_cpu;

    ArchInfo archInfo;

    archInfo.arch = MALI;

    TensorDesc outputDesc;
    U32 len = tensorNumElements(inputDesc_cpu);
    U8 *input_cpu = ut_input_v(len, dt, UT_INIT_RANDOM);

    Tensor inputTensorCpu;
    inputTensorCpu.resize(inputDesc_cpu);
    inputTensorCpu.alloc();
    UNI_MEMCPY(
        get_ptr_from_tensor(inputTensorCpu, CPU_GENERAL), input_cpu, tensorNumBytes(inputDesc_cpu));
    Tensor outputTensorCpu;
    Tensor tmpTensorCpu;
    //run on cpu
    CHECK_STATUS(
        transpose_infer_output_size(&inputTensorCpu, p, &outputTensorCpu, &UT_SERIAL_ARCHINFO));
    outputTensorCpu.alloc();
    CHECK_STATUS(transpose(inputTensorCpu, p, tmpTensorCpu, outputTensorCpu, &UT_SERIAL_ARCHINFO));
    //run on gpu
    std::shared_ptr<GCLHandle> handleSharedPtr = OCLContext::getInstance().handle;
    GCLHandle_t handle = handleSharedPtr.get();
    std::vector<GCLKernelInfo> kernelVec;
    handle->kernelVec = &kernelVec;
    Tensor inputTensor = Tensor(OCLMem);
    Tensor outputTensor = Tensor(OCLMem);
    Tensor tmpTensor = Tensor(OCLMem);
    inputTensor.resize(inputDesc_gpu);
    U8 *output_gpu = NULL;
    MaliPara maliPara;
    maliPara.handle = handle;
    archInfo.archPara = &maliPara;

    CHECK_STATUS(transpose_infer_output_size(&inputTensor, p, &outputTensor, &archInfo));
    outputDesc = outputTensor.get_desc();
    output_gpu = ut_input_v(tensorNumElements(outputDesc), dt, UT_INIT_RANDOM);

    U32 maxBytes = 0;
    U32 tmpBytes;
    CHECK_STATUS(transpose_infer_forward_tmp_bytes(inputTensor, outputTensor, &tmpBytes, &archInfo))
    maxBytes = (tmpBytes > maxBytes) ? tmpBytes : maxBytes;

    GCLMem_t output = alloc(outputTensor);
    GCLMem_t input = alloc(inputTensor);
    CHECK_STATUS(gcl_fill_memory_zero(handle, input));

    tmpBytes = tensorNumBytes(inputDesc_gpu);
    maxBytes = (tmpBytes > maxBytes) ? tmpBytes : maxBytes;
    tmpBytes = tensorNumBytes(outputDesc);
    maxBytes = (tmpBytes > maxBytes) ? tmpBytes : maxBytes;
    GCLMem_t tmpbuf = alloc_bytes(tmpTensor, maxBytes);

    CHECK_STATUS(ocl_set_input(handle, input, inputDesc_gpu, input_cpu, tmpbuf, true));
    CHECK_STATUS(transpose(inputTensor, p, tmpTensor, outputTensor, &archInfo));

    for (U32 i = 0; i < UT_WARMUP; i++) {
        CHECK_STATUS(gcl_run_kernelVec(handle));
    }
    CHECK_STATUS(gcl_finish(handle));

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

    CHECK_STATUS(ocl_get_output(handle, output, outputDesc, output_gpu, tmpbuf, true));

    char buffer[150];
    char params[120];
    sprintf(params, "(");
    I32 index = 0;
    for (U32 i = 0; i < nDims; i++) {
        for (; index < 120; index++) {
            if (params[index] == '\0') {
                break;
            }
        }
        if (i == nDims - 1) {
            sprintf(params + index, "%d)=(", inputDesc_cpu.dims[nDims - 1 - i]);
        } else {
            sprintf(params + index, "%d ", inputDesc_cpu.dims[nDims - 1 - i]);
        }
    }
    for (U32 i = 0; i < nDims; i++) {
        for (; index < 120; index++) {
            if (params[index] == '\0') {
                break;
            }
        }
        if (i == nDims - 1) {
            sprintf(params + index, "%d)", outputDesc.dims[nDims - 1 - i]);
        } else {
            sprintf(params + index, "%d ", outputDesc.dims[nDims - 1 - i]);
        }
    }
    sprintf(buffer, "%20s, %80s", "Transpose", params);
    double ops = len;
    ut_log(dt, buffer, ops, time);
    ut_check_v(output_gpu, get_ptr_from_tensor(outputTensorCpu, CPU_GENERAL), len, dt, 0.1);

    CHECK_STATUS(gcl_finish(handle));
    CHECK_STATUS(gcl_clean_kernelVec(handle));
    free(input_cpu);
    free(output_gpu);
    return 0;
}

int main(int argc, char **argv)
{
    transposeTest(argc, argv, DT_F16);
    return 0;
}
