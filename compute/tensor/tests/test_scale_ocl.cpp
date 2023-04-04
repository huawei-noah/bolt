
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

int scaleTest(int argc, char **argv, DataType dt)
{
    U32 in = atoi(argv[1]);
    U32 ic = atoi(argv[2]);
    U32 ih = atoi(argv[3]);
    U32 iw = atoi(argv[4]);
    U32 axis = atoi(argv[5]);
    U32 axisLen = atoi(argv[6]);
    U32 useAlpha = 1;
    U32 useBeta = 1;
    if (argc > 7) {
        useAlpha = atoi(argv[7]);
    }
    if (argc > 8) {
        useBeta = atoi(argv[8]);
    }

    ArchInfo archInfo;
    archInfo.arch = MALI;

    ScaleParamSpec p;
    p.axis = axis;
    TensorDesc inputDesc, outputDesc, alphaDesc, betaDesc;
    inputDesc = tensor4df(dt, DF_NCHW, in, ic, ih, iw);
    alphaDesc = tensor1d(dt, axisLen);
    betaDesc = tensor1d(dt, axisLen);
    U32 inputLen = tensorNumElements(inputDesc);
    U8 *inputCpu = ut_input_v(inputLen, dt, UT_INIT_RANDOM);
    F16 *val = (F16 *)inputCpu;
    for (U32 i = 0; i < inputLen; i++)
        val[i] = i * 0.1;

    U8 *alphaCpu = nullptr;
    U8 *betaCpu = nullptr;
    if (useAlpha) {
        alphaCpu = ut_input_v(axisLen, dt, UT_INIT_RANDOM);
        val = (F16 *)alphaCpu;
        for (U32 i = 0; i < axisLen; i++)
            val[i] = i;
    }
    if (useBeta) {
        betaCpu = ut_input_v(axisLen, dt, UT_INIT_RANDOM);
        val = (F16 *)betaCpu;
        for (U32 i = 0; i < axisLen; i++)
            val[i] = i;
    }

    Tensor inputTensorCpu, outputTensorCpu;
    inputTensorCpu.resize(inputDesc);
    inputTensorCpu.alloc();
    UNI_MEMCPY(
        get_ptr_from_tensor(inputTensorCpu, CPU_GENERAL), inputCpu, tensorNumBytes(inputDesc));
    CHECK_STATUS(
        scale_infer_output_size(&inputTensorCpu, p, axisLen, &outputTensorCpu, &UT_SERIAL_ARCHINFO));
    outputTensorCpu.alloc();
    CHECK_STATUS(scale(inputTensorCpu, alphaCpu, betaCpu, p, outputTensorCpu, &UT_SERIAL_ARCHINFO));

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
    Tensor alphaTensor = Tensor(OCLMem);
    Tensor betaTensor = Tensor(OCLMem);
    inputDesc.df = DF_NCHWC4;
    inputTensor.resize(inputDesc);
    GCLMem_t alpha = nullptr;
    GCLMem_t beta = nullptr;
    if (useAlpha) {
        alphaTensor.resize(alphaDesc);
        U32 pr = (axisLen + 3) / 4 * 4 - axisLen;
        alpha = alloc_padding(alphaTensor, 0, pr, 0, 0, alphaCpu);
    }
    if (useBeta) {
        betaTensor.resize(betaDesc);
        U32 pr = (axisLen + 3) / 4 * 4 - axisLen;
        beta = alloc_padding(betaTensor, 0, pr, 0, 0, betaCpu);
    }

    MaliPara maliPara;
    maliPara.handle = handle;
    archInfo.archPara = &maliPara;

    CHECK_STATUS(scale_infer_output_size(&inputTensor, p, axisLen, &outputTensor, &archInfo));
    outputDesc = outputTensor.get_desc();
    U8 *outputGpu = ut_input_v(tensorNumBytes(outputDesc), dt, UT_INIT_RANDOM);

    GCLMem_t output = alloc(outputTensor);
    GCLMem_t input = alloc(inputTensor);
    CHECK_STATUS(gcl_fill_memory_zero(handle, input));

    Tensor tmpTensor = Tensor(OCLMem);
    U32 tmpBytes;
    U32 maxBytes = 0;
    tmpBytes = tensorNumBytes(inputDesc);
    maxBytes = (tmpBytes > maxBytes) ? tmpBytes : maxBytes;
    tmpBytes = tensorNumBytes(outputDesc);
    maxBytes = (tmpBytes > maxBytes) ? tmpBytes : maxBytes;
    GCLMem_t tmpbuf = alloc_bytes(tmpTensor, maxBytes);

    CHECK_STATUS(ocl_set_input(handle, input, inputDesc, inputCpu, tmpbuf, true));
    CHECK_STATUS(scale(inputTensor, alpha, beta, p, outputTensor, &archInfo));

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
    U32 on, oc, oh, ow;
    tensorSelectGet(outputDesc, NULL, NULL, &on, &oc, &oh, &ow);
    sprintf(params, "(%u %u %u %u)->(%u %u %u %u)", in, ic, ih, iw, on, oc, oh, ow);
    sprintf(buffer, "%20s, %80s", "Scale", params);
    double ops = 2.0 * on * oc * oh * ow;
    ut_log(dt, buffer, ops, time);

    CHECK_STATUS(ocl_get_output(handle, output, outputDesc, outputGpu, tmpbuf, true));
    ut_check_v(outputGpu, get_ptr_from_tensor(outputTensorCpu, CPU_GENERAL), on * oc * ow * oh, dt, 0.3);

    CHECK_STATUS(gcl_finish(handle));
    CHECK_STATUS(gcl_clean_kernelVec(handle));
    free(outputGpu);
    free(inputCpu);
    if (useAlpha) {
        free(alphaCpu);
    }
    if (useBeta) {
        free(betaCpu);
    }
    return 0;
}

int main(int argc, char **argv)
{
    scaleTest(argc, argv, DT_F16);
    return 0;
}
