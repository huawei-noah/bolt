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

int channelresizeTest(int argc, char *argv[], DataType dt)
{
    CHECK_REQUIREMENT(argc == 8);
    // in data
    U32 in = atoi(argv[1]);
    U32 ic = atoi(argv[2]);
    U32 ih = atoi(argv[3]);
    U32 iw = atoi(argv[4]);

    ChannelResizeParamSpec p;
    p.channel_before = atoi(argv[5]);
    p.channel_after = atoi(argv[6]);
    p.group = atoi(argv[7]);
    // output
    U32 on = in;
    U32 oc = p.channel_after;
    U32 oh = ih;
    U32 ow = iw;

    CHECK_REQUIREMENT(in == 1 && on == 1);
    CHECK_REQUIREMENT(p.channel_before == (int)ic);

    TensorDesc inputDesc_cpu, inputDesc_gpu, outputDesc;
    inputDesc_cpu = tensor4df(dt, DF_NCHW, in, ic, ih, iw);
    inputDesc_gpu = tensor4df(dt, DF_NCHW, in, ic, ih, iw);
    outputDesc = tensor4df(dt, DF_NCHW, in, oc, oh, ow);

    // setup input
    U8 *input_cpu = ut_input_v(in * ic * ih * iw, dt, UT_INIT_RANDOM);
    U8 *output_cpu = ut_input_v(on * oc * oh * ow, dt, UT_INIT_RANDOM);
    U8 *output_gpu = ut_input_v(on * oc * oh * ow, dt, UT_INIT_RANDOM);
    F16 *in_val = (F16 *)input_cpu;
    U32 len_in = tensorNumElements(inputDesc_cpu);
    for (U32 i = 0; i < len_in; i++) {
        in_val[i] = i;
    }

    U32 len = tensorNumElements(outputDesc);
    F16 *out_val = (F16 *)output_cpu;
    for (U32 i = 0; i < len; i++) {
        out_val[i] = in_val[i];
    }

    std::shared_ptr<GCLHandle> handleSharedPtr = OCLContext::getInstance().handle;
    GCLHandle_t handle = handleSharedPtr.get();
    std::vector<GCLKernelInfo> kernelVec;
    handle->kernelVec = &kernelVec;
    Tensor inputTensor = Tensor(OCLMem);
    Tensor outputTensor = Tensor(OCLMem);
    Tensor tmpTensor = Tensor(OCLMem);
    inputTensor.resize(inputDesc_gpu);

    MaliPara maliPara;
    maliPara.handle = handle;
    ArchInfo archInfo;
    archInfo.arch = MALI;
    archInfo.archPara = &maliPara;

    CHECK_STATUS(channel_resize_infer_output_size(&inputTensor, p, &outputTensor, &archInfo));
    outputDesc = outputTensor.get_desc();
    CHECK_REQUIREMENT(tensorNumElements(outputDesc) == on * oc * oh * ow);

    GCLMem_t output = alloc(outputTensor);
    GCLMem_t input = alloc(inputTensor);
    CHECK_STATUS(gcl_fill_memory_zero(handle, input));

    U32 maxBytes = 0;
    U32 tmpBytes = 0;
    tmpBytes = tensorNumBytes(inputDesc_gpu);
    maxBytes = (tmpBytes > maxBytes) ? tmpBytes : maxBytes;
    tmpBytes = tensorNumBytes(outputDesc);
    maxBytes = (tmpBytes > maxBytes) ? tmpBytes : maxBytes;
    GCLMem_t tmpbuf = alloc_bytes(tmpTensor, maxBytes);

    CHECK_STATUS(ocl_set_input(handle, input, inputDesc_gpu, input_cpu, tmpbuf, true));
    CHECK_STATUS(channel_resize(inputTensor, p, outputTensor, &archInfo));

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
    sprintf(params, "(%u %u %u %u)->(%u %u %u %u)", in, ic, ih, iw, on, oc, oh, ow);
    sprintf(buffer, "%20s, %80s", "ChannelResize", params);
    double ops = 1.0 * on * oc * oh * ow;
    ut_log(dt, buffer, ops, time);

    ut_check_v(output_gpu, output_cpu, on * oc * ow * oh, dt, 0.3);
    CHECK_STATUS(gcl_finish(handle));
    CHECK_STATUS(gcl_clean_kernelVec(handle));
    free(output_gpu);
    free(input_cpu);
    free(output_cpu);
    return 0;
}

int main(int argc, char **argv)
{
    channelresizeTest(argc, argv, DT_F16);
    return 0;
}
