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

int powerTest(int argc, char **argv, DataType dt)
{
    U32 in = 1;
    U32 ic = 4;
    U32 ih = 4;
    U32 iw = 4;
    PowerParamSpec p;
    p.scale = 0.5;
    p.shift = 0.5;
    p.power = 2;
    if (argc == 8) {
        in = atoi(argv[1]);
        ic = atoi(argv[2]);
        ih = atoi(argv[3]);
        iw = atoi(argv[4]);
        p.scale = atof(argv[5]);
        p.shift = atof(argv[6]);
        p.power = atof(argv[7]);
    }
    U32 on = in;
    U32 oc = ic;
    U32 oh = ih;
    U32 ow = iw;

    ArchInfo archInfo;
    archInfo.arch = MALI;

    U32 len = in * ic * ih * iw;

    TensorDesc input_desc_cpu = tensor1d(dt, len);
    TensorDesc output_desc_cpu = tensor1d(dt, len);
    TensorDesc input_desc_gpu = tensor4df(dt, DF_NCHW, in, ic, ih, iw);

    U8 *input_cpu = ut_input_v(len, dt, UT_INIT_RANDOM);

    std::shared_ptr<GCLHandle> handleSharedPtr = OCLContext::getInstance().handle;
    GCLHandle_t handle = handleSharedPtr.get();
    std::vector<GCLKernelInfo> kernelVec;
    handle->kernelVec = &kernelVec;
    Tensor inputTensor = Tensor(OCLMem);
    Tensor outputTensor = Tensor(OCLMem);
    Tensor tmpTensor = Tensor(OCLMem);
    inputTensor.resize(input_desc_gpu);

    MaliPara maliPara;
    maliPara.handle = handle;
    archInfo.archPara = &maliPara;
    CHECK_STATUS(power_infer_output_size(&inputTensor, p, dt, &outputTensor, &archInfo));
    TensorDesc output_desc_gpu = outputTensor.get_desc();
    U8 *output_gpu = ut_input_v(on * oc * oh * ow, dt, UT_INIT_RANDOM);

    GCLMem_t output = alloc(outputTensor);
    GCLMem_t input = alloc(inputTensor);
    CHECK_STATUS(gcl_fill_memory_zero(handle, input));

    U32 maxBytes = 0;
    U32 tmpBytes = 0;
    tmpBytes = tensorNumBytes(input_desc_gpu);
    maxBytes = (tmpBytes > maxBytes) ? tmpBytes : maxBytes;
    tmpBytes = tensorNumBytes(output_desc_gpu);
    maxBytes = (tmpBytes > maxBytes) ? tmpBytes : maxBytes;
    GCLMem_t tmpbuf = alloc_bytes(tmpTensor, maxBytes);

    CHECK_STATUS(ocl_set_input(handle, input, input_desc_gpu, input_cpu, tmpbuf, true));
    CHECK_STATUS(power(inputTensor, p, outputTensor, &archInfo));

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

    CHECK_STATUS(ocl_get_output(handle, output, output_desc_gpu, output_gpu, tmpbuf, true));
    output_gpu = output->mapPtrArray.back();
    char buffer[150];
    char params[120];
    sprintf(params, "(%u %u %u %u) = (%u %u %u %u)", in, ic, ih, iw, on, oc, oh, ow);
    sprintf(buffer, "%20s, %80s", "Power", params);
    double ops = (2.0 * on * oc * oh * ow);
    ut_log(dt, buffer, ops, time);
    Tensor inputTensorCpu;
    inputTensorCpu.resize(input_desc_cpu);
    inputTensorCpu.alloc();
    UNI_MEMCPY(get_ptr_from_tensor(inputTensorCpu, CPU_GENERAL), input_cpu,
        tensorNumBytes(input_desc_cpu));

    Tensor outputTensorCpu;
    outputTensorCpu.resize(output_desc_cpu);
    outputTensorCpu.alloc();

    CHECK_STATUS(power(inputTensorCpu, p, outputTensorCpu, &UT_SERIAL_ARCHINFO));
    ut_check_v(output_gpu, get_ptr_from_tensor(outputTensorCpu, CPU_GENERAL), on * oc * ow * oh, dt, 0.3);

    CHECK_STATUS(gcl_finish(handle));
    CHECK_STATUS(gcl_clean_kernelVec(handle));
    free(input_cpu);
    free(output_gpu);
    return 0;
}

int main(int argc, char **argv)
{
    powerTest(argc, argv, DT_F16);
    return 0;
}
