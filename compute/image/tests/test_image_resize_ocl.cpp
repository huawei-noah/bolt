// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "image.h"
#include "tensor_computing.h"
#include "ut_util_ocl.h"

int resizeTest(int argc, char *argv[], DataType dt)
{
    CHECK_REQUIREMENT(argc == 9);
    // in data
    U32 in = atoi(argv[1]);
    U32 ic = atoi(argv[2]);
    U32 ih = atoi(argv[3]);
    U32 iw = atoi(argv[4]);
    // output
    U32 on = atoi(argv[5]);
    U32 oc = atoi(argv[6]);
    U32 oh = atoi(argv[7]);
    U32 ow = atoi(argv[8]);

    CHECK_REQUIREMENT(in == 1 && on == 1);

    ResizeParamSpec p;
    //p.mode = RESIZE_LINEAR;
    p.mode = RESIZE_NEAREST;
    p.trans_mode = COORDINATE_TRANS_ASYMMETRIC;
    p.num_scales = 0;
    p.num_sizes = 2;
    p.sizes[0] = oh;
    p.sizes[1] = ow;

    // setup input
    TensorDesc inputDesc = tensor4df(dt, DF_NCHW, in, ic, ih, iw);
    U8 *input_cpu = ut_input_v(in * ic * ih * iw, dt, UT_INIT_RANDOM);
    Tensor inputTensorCpu = Tensor::alloc_sized<CPUMem>(inputDesc);
    UNI_MEMCPY(get_ptr_from_tensor(inputTensorCpu, UT_ARCH), input_cpu, tensorNumBytes(inputDesc));
    Tensor outputTensorCpu;
    CHECK_STATUS(
        resize_infer_output_size(&inputTensorCpu, p, &outputTensorCpu, &UT_SERIAL_ARCHINFO));
    outputTensorCpu.alloc();
    U32 cpuTmpBytes = 0;
    CHECK_STATUS(resize_infer_forward_tmp_bytes(
        inputTensorCpu, p, outputTensorCpu, &cpuTmpBytes, &UT_SERIAL_ARCHINFO));
    Tensor tmpTensorCpu = Tensor::alloc_sized<CPUMem>(tensor1d(DT_I8, cpuTmpBytes));

    // CPU output
    CHECK_STATUS(resize(inputTensorCpu, p, tmpTensorCpu, outputTensorCpu, &UT_SERIAL_ARCHINFO));

    ArchInfo archInfo;
    archInfo.arch = MALI;
    std::shared_ptr<GCLHandle> handleSharedPtr = OCLContext::getInstance().handle;
    GCLHandle_t handle = handleSharedPtr.get();
    std::vector<GCLKernelInfo> kernelVec;
    handle->kernelVec = &kernelVec;
    MaliPara maliPara;
    maliPara.handle = handle;
    archInfo.archPara = &maliPara;

    Tensor inputTensor = Tensor(OCLMem);
    Tensor outputTensor = Tensor(OCLMem);
    Tensor tmpTensor = Tensor(OCLMem);
    inputTensor.resize(inputDesc);

    CHECK_STATUS(resize_infer_output_size(&inputTensor, p, &outputTensor, &archInfo));
    U32 maxBytes = 0;
    U32 tmpBytes = 0;

    GCLMem_t output = alloc(outputTensor);
    GCLMem_t input = alloc(inputTensor);
    CHECK_STATUS(gcl_fill_memory_zero(handle, input));
    TensorDesc outputDesc_gpu = outputTensor.get_desc();
    U8 *output_gpu = ut_input_v(on * oc * oh * ow, dt, UT_INIT_RANDOM);
    tmpBytes = tensorNumBytes(inputDesc);
    maxBytes = (tmpBytes > maxBytes) ? tmpBytes : maxBytes;
    tmpBytes = tensorNumBytes(outputDesc_gpu);
    maxBytes = (tmpBytes > maxBytes) ? tmpBytes : maxBytes;
    GCLMem_t tmpbuf = alloc_bytes(tmpTensor, maxBytes);
    CHECK_STATUS(ocl_set_input(handle, input, inputDesc, input_cpu, tmpbuf, true));

    CHECK_STATUS(resize(inputTensor, p, tmpTensor, outputTensor, &archInfo));

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
    sprintf(params, "(%u %u %u %u)->(%u %u %u %u)", in, ic, ih, iw, on, oc, oh, ow);
    sprintf(buffer, "%20s, %80s", "Resize", params);
    double ops = on * oc * oh * ow * 4;
    ut_log(dt, buffer, ops, time);

    CHECK_STATUS(ocl_get_output(handle, output, outputDesc_gpu, output_gpu, tmpbuf, true));
    ut_check_v(output_gpu, get_ptr_from_tensor(outputTensorCpu, UT_ARCH), on * oc * ow * oh, dt, 0.3);
    CHECK_STATUS(gcl_finish(handle));
    CHECK_STATUS(gcl_clean_kernelVec(handle));
    free(input_cpu);
    free(output_gpu);
    return 0;
}

int main(int argc, char **argv)
{
    resizeTest(argc, argv, DT_F16);
    return 0;
}
