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

int softmaxTest(int argc, char **argv, DataType dt)
{
    U32 in, ic, ih, iw, axis;
    in = 1;
    ic = 4;
    ih = 1;
    iw = 1;
    axis = 1;

    if (argc == 6) {
        in = atoi(argv[1]);
        ic = atoi(argv[2]);
        ih = atoi(argv[3]);
        iw = atoi(argv[4]);
        axis = atoi(argv[5]);
    }

    SoftmaxParamSpec p;
    p.axis = axis;

    ArchInfo archInfo;
    archInfo.arch = MALI;
    if (gcl_check_device_qualcomm(OCLContext::getInstance().handle.get())) {
        archInfo.arch = QUALCOMM;
    }

    TensorDesc in_desc, in_desc_gpu, out_desc;
    in_desc = tensor4df(dt, DF_NCHW, in, ic, ih, iw);

    U8 *input_cpu = ut_input_v(in * ic * ih * iw, dt, UT_INIT_RANDOM);
    in_desc_gpu = tensor4df(dt, DF_NCHWC4, in, ic, ih, iw);

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
    inputTensor.resize(in_desc_gpu);

    MaliPara maliPara;
    maliPara.handle = handle;
    archInfo.archPara = &maliPara;
    CHECK_STATUS(softmax_infer_output_size(&inputTensor, p, &outputTensor, &archInfo));
    out_desc = outputTensor.get_desc();
    U8 *output_gpu = ut_input_v(tensorNumElements(out_desc), dt, UT_INIT_RANDOM);

    U32 maxBytes = 0;
    U32 tmpBytes;
    CHECK_STATUS(softmax_infer_forward_tmp_bytes(inputTensor, p, &tmpBytes, &archInfo));
    maxBytes = (tmpBytes > maxBytes) ? tmpBytes : maxBytes;  // 18

    GCLMem_t output = alloc(outputTensor);
    GCLMem_t input = alloc(inputTensor);
    CHECK_STATUS(gcl_fill_memory_zero(handle, input));
    tmpBytes = tensorNumBytes(in_desc_gpu);
    maxBytes = (tmpBytes > maxBytes) ? tmpBytes : maxBytes;
    tmpBytes = tensorNumBytes(out_desc);
    maxBytes = (tmpBytes > maxBytes) ? tmpBytes : maxBytes;
    GCLMem_t tmpbuf = alloc_bytes(tmpTensor, maxBytes);

    CHECK_STATUS(ocl_set_input(handle, input, in_desc_gpu, input_cpu, tmpbuf, true));
    CHECK_STATUS(softmax(inputTensor, p, tmpTensor, outputTensor, &archInfo));

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
    sprintf(params, "(%u %u %u %u)", in, ic, ih, iw);
    sprintf(buffer, "%20s, %80s", "softmax", params);
    double ops = 1;
    ut_log(dt, buffer, ops, time);

    CHECK_STATUS(ocl_get_output(handle, output, out_desc, output_gpu, tmpbuf, true));
    Tensor inputTensorCpu;
    inputTensorCpu.resize(in_desc);
    inputTensorCpu.alloc();
    UNI_MEMCPY(get_ptr_from_tensor(inputTensorCpu, CPU_GENERAL), input_cpu, tensorNumBytes(in_desc));

    Tensor outputTensorCpu;
    outputTensorCpu.resize(out_desc);
    outputTensorCpu.alloc();

    Tensor tmpTensorCpu;
    CHECK_STATUS(softmax(inputTensorCpu, p, tmpTensorCpu, outputTensorCpu, &UT_SERIAL_ARCHINFO));

    ut_check_v(output_gpu, get_ptr_from_tensor(outputTensorCpu, CPU_GENERAL), in * ih * iw * ic, dt, 0.3);
    CHECK_STATUS(gcl_finish(handle));
    CHECK_STATUS(gcl_clean_kernelVec(handle));

    free(output_gpu);
    free(input_cpu);
    return 0;
}

int main(int argc, char **argv)
{
    softmaxTest(argc, argv, DT_F16);
    return 0;
}
