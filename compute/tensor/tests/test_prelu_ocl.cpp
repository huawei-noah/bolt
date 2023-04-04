
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

int preluTest(int argc, char **argv, DataType dt)
{
    U32 in = atoi(argv[1]);
    U32 ic = atoi(argv[2]);
    U32 ih = atoi(argv[3]);
    U32 iw = atoi(argv[4]);
    U32 prop = atoi(argv[5]);
    U32 weightNum;

    ArchInfo archInfo;
    archInfo.arch = MALI;

    TensorDesc inputDescGPU, outputDescGPU, weightDescGPU;
    inputDescGPU = tensor4df(dt, DF_NCHWC4, in, ic, ih, iw);

    U32 input_len = tensorNumElements(inputDescGPU);
    U8 *inputCPU = ut_input_v(input_len, dt, UT_INIT_RANDOM);
    U8 *weightCPU = NULL;
    PReLUParamSpec preluDesc;
    if (prop) {
        preluDesc.propagate_down = true;
        weightCPU = ut_input_v(1, dt, UT_INIT_RANDOM);
        weightDescGPU = tensor1d(dt, 1);
    } else {
        preluDesc.propagate_down = false;
        weightCPU = ut_input_v(ic, dt, UT_INIT_RANDOM);
        weightDescGPU = tensor1d(dt, ic);
    }
    std::shared_ptr<GCLHandle> handleSharedPtr = OCLContext::getInstance().handle;
    GCLHandle_t handle = handleSharedPtr.get();
    std::vector<GCLKernelInfo> kernelVec;
    handle->kernelVec = &kernelVec;
    Tensor inputTensor = Tensor(OCLMem);
    Tensor outputTensor = Tensor(OCLMem);
    Tensor tmpTensor = Tensor(OCLMem);
    Tensor weightTensor = Tensor(OCLMem);
    inputTensor.resize(inputDescGPU);
    weightTensor.resize(weightDescGPU);

    MaliPara maliPara;
    maliPara.handle = handle;
    archInfo.archPara = &maliPara;

    CHECK_STATUS(prelu_infer_output_size(&inputTensor, &outputTensor, &archInfo));
    outputDescGPU = outputTensor.get_desc();
    U8 *outputGPU = ut_input_v(input_len, dt, UT_INIT_RANDOM);

    GCLMem_t output = alloc(outputTensor);
    GCLMem_t input = alloc(inputTensor);
    CHECK_STATUS(gcl_fill_memory_zero(handle, input));

    U32 icAlign = (ic + 3) / 4 * 4;
    if (!preluDesc.propagate_down) {
        U8 *weightAlign = ut_input_v(icAlign, dt, UT_INIT_ZERO);
        UNI_MEMCPY(weightAlign, weightCPU, ic * bytesOf(dt));
        free(weightCPU);
        weightCPU = weightAlign;
        alloc_padding(weightTensor, 0, icAlign - ic, 0, 0, weightCPU);
    } else {
        alloc_padding(weightTensor, 0, 0, 0, 0, weightCPU);
    }

    U32 tmpBytes;
    U32 maxBytes = 0;
    tmpBytes = tensorNumBytes(inputDescGPU);
    maxBytes = (tmpBytes > maxBytes) ? tmpBytes : maxBytes;
    tmpBytes = tensorNumBytes(outputDescGPU);
    maxBytes = (tmpBytes > maxBytes) ? tmpBytes : maxBytes;
    GCLMem_t tmpbuf = alloc_bytes(tmpTensor, maxBytes);

    CHECK_STATUS(ocl_set_input(handle, input, inputDescGPU, inputCPU, tmpbuf, true));
    CHECK_STATUS(prelu(inputTensor, weightTensor, preluDesc, outputTensor, &archInfo));

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

    CHECK_STATUS(ocl_get_output(handle, output, outputDescGPU, outputGPU, tmpbuf, true));

    char buffer[150];
    char params[120];
    sprintf(params, "(%u %u %u %u)->(%u %u %u %u)", in, ic, ih, iw, in, ic, ih, iw);
    sprintf(buffer, "%20s, %80s", "PRelu", params);
    double ops = tensorNumElements(inputDescGPU);
    ut_log(dt, buffer, ops, time);

    CHECK_STATUS(gcl_finish(handle));
    CHECK_STATUS(gcl_clean_kernelVec(handle));
    free(outputGPU);
    free(inputCPU);
    free(weightCPU);
    return 0;
}

int main(int argc, char **argv)
{
    preluTest(argc, argv, DT_F16);
    return 0;
}
