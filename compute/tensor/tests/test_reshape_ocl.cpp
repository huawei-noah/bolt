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
#include "ut_util_ocl.h"

int reshapeTest(int argc, char **argv, DataType dt)
{
    TensorDesc inputDesc;
    inputDesc.df = DF_NCHW;
    inputDesc.dt = dt;
    inputDesc.nDims = atoi(argv[1]);
    ReshapeParamSpec p;
    for (U32 i = 0; i < inputDesc.nDims; i++) {
        inputDesc.dims[inputDesc.nDims - i - 1] = atoi(argv[i + 2]);
    }
    p.num_shape = atoi(argv[inputDesc.nDims + 2]);
    for (I32 i = 0; i < p.num_shape; i++) {
        p.shape[i] = atoi(argv[i + inputDesc.nDims + 3]);
    }

    ArchInfo archInfo;
    archInfo.arch = MALI;

    U32 len = tensorNumElements(inputDesc);
    U8 *input_cpu = ut_input_v(len, dt, UT_INIT_RANDOM);

    Tensor inputTensorCpu;
    inputTensorCpu.resize(inputDesc);
    inputTensorCpu.alloc();
    UNI_MEMCPY(
        get_ptr_from_tensor(inputTensorCpu, CPU_GENERAL), input_cpu, tensorNumBytes(inputDesc));

    Tensor outputTensorCpu;
    Tensor tmpTensorCpu;
    CHECK_STATUS(
        reshape_infer_output_size(&inputTensorCpu, p, &outputTensorCpu, &UT_SERIAL_ARCHINFO));
    outputTensorCpu.alloc();
    CHECK_STATUS(reshape(inputTensorCpu, tmpTensorCpu, outputTensorCpu, &UT_SERIAL_ARCHINFO));

    std::shared_ptr<GCLHandle> handleSharedPtr = OCLContext::getInstance().handle;
    GCLHandle_t handle = handleSharedPtr.get();
    std::vector<GCLKernelInfo> kernelVec;
    handle->kernelVec = &kernelVec;
    Tensor inputTensor = Tensor(OCLMem);
    Tensor outputTensor = Tensor(OCLMem);
    Tensor tmpTensor = Tensor(OCLMem);
    inputTensor.resize(inputDesc);

    MaliPara maliPara;
    maliPara.handle = handle;
    archInfo.archPara = &maliPara;
    CHECK_STATUS(reshape_infer_output_size(&inputTensor, p, &outputTensor, &archInfo));
    TensorDesc outputDesc = outputTensor.get_desc();
    U32 on, oc, oh, ow;
    tensorSelectGet(outputDesc, NULL, NULL, &on, &oc, &oh, &ow);
    U8 *output_gpu = ut_input_v(on * oc * oh * ow, dt, UT_INIT_RANDOM);

    GCLMem_t output = alloc(outputTensor);
    GCLMem_t input = alloc(inputTensor);
    CHECK_STATUS(gcl_fill_memory_zero(handle, input));

    U32 maxBytes = 0;
    U32 tmpBytes = 0;
    tmpBytes = tensorNumBytes(inputDesc);
    maxBytes = (tmpBytes > maxBytes) ? tmpBytes : maxBytes;
    tmpBytes = tensorNumBytes(outputDesc);
    maxBytes = (tmpBytes > maxBytes) ? tmpBytes : maxBytes;
    GCLMem_t tmpbuf = alloc_bytes(tmpTensor, maxBytes);

    CHECK_STATUS(ocl_set_input(handle, input, inputDesc, input_cpu, tmpbuf, true));
    CHECK_STATUS(reshape(inputTensor, tmpTensor, outputTensor, &archInfo));

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
    for (I32 i = 0; i < p.num_shape; i++) {
        I32 index = 0;
        for (; index < 120; index++) {
            if (params[index] == '\0') {
                break;
            }
        }
        if (i != p.num_shape - 1) {
            sprintf(params + index, "%d ", outputDesc.dims[outputDesc.nDims - 1 - i]);
        } else {
            sprintf(params + index, "%d)", outputDesc.dims[outputDesc.nDims - 1 - i]);
        }
    }
    sprintf(buffer, "%20s, %80s", "Reshape", params);
    double ops = len;
    ut_log(dt, buffer, ops, time);

    CHECK_STATUS(ocl_get_output(handle, output, outputDesc, output_gpu, tmpbuf, true));
    output_gpu = output->mapPtrArray.back();
    ut_check_v(output_gpu, get_ptr_from_tensor(outputTensorCpu, CPU_GENERAL), len, dt, 0.1);
    CHECK_STATUS(gcl_finish(handle));
    CHECK_STATUS(gcl_clean_kernelVec(handle));
    free(input_cpu);
    free(output_gpu);
    return 0;
}

int main(int argc, char **argv)
{
    reshapeTest(argc, argv, DT_F16);
    return 0;
}
