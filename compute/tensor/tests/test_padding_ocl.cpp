
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

int paddingTest(int argc, char **argv, DataType dt)
{
    // input dim
    U32 in = atoi(argv[1]);
    U32 ic = atoi(argv[2]);
    U32 ih = atoi(argv[3]);
    U32 iw = atoi(argv[4]);

    // padding info
    U32 h_top = atoi(argv[5]);
    U32 h_bot = atoi(argv[7]);
    U32 w_lef = atoi(argv[6]);
    U32 w_rig = atoi(argv[8]);
    U32 mode = atoi(argv[9]);

    ArchInfo archInfo;
    archInfo.arch = MALI;

    PadParamSpec padParamSpec;

    padParamSpec.top = h_top;
    padParamSpec.bottom = h_bot;
    padParamSpec.left = w_lef;
    padParamSpec.right = w_rig;
    padParamSpec.front = 0;
    padParamSpec.back = 0;
    padParamSpec.constant_value = 0.0;
    switch (mode) {
        case 0: {
            padParamSpec.pad_mode = PAD_CONSTANT;
            break;
        }
        case 1: {
            padParamSpec.pad_mode = PAD_EDGE;
            break;
        }
        case 2: {
            // limitation: the h_fir and the h_sec should lower than 0
            padParamSpec.pad_mode = PAD_REFLECT;
            break;
        }
        case 3: {
            padParamSpec.pad_mode = PAD_SYMMETRIC;
            break;
        }
        default: {
            UNI_ERROR_LOG("unknown paddding mode %d\n", mode);
            break;
        }
    }

    TensorDesc inputDescCPU, inputDescGPU, outputDescCPU, outputDescGPU;
    inputDescCPU = tensor4df(dt, DF_NCHW, in, ic, ih, iw);
    inputDescGPU = tensor4df(dt, DF_NCHW, in, ic, ih, iw);

    U32 input_len = tensorNumElements(inputDescCPU);
    U8 *inputCPU = ut_input_v(input_len, dt, UT_INIT_RANDOM);
    F16 *val = (F16 *)inputCPU;
    for (U32 i = 0; i < input_len; i++)
        val[i] = i;

    std::shared_ptr<GCLHandle> handleSharedPtr = OCLContext::getInstance().handle;
    GCLHandle_t handle = handleSharedPtr.get();
    std::vector<GCLKernelInfo> kernelVec;
    handle->kernelVec = &kernelVec;
    Tensor inputTensor = Tensor(OCLMem);
    Tensor outputTensor = Tensor(OCLMem);
    Tensor tmpTensor = Tensor(OCLMem);
    inputTensor.resize(inputDescGPU);
    MaliPara maliPara;
    maliPara.handle = handle;
    archInfo.archPara = &maliPara;

    CHECK_STATUS(padding_infer_output_size(&inputTensor, padParamSpec, &outputTensor, &archInfo));
    outputDescGPU = outputTensor.get_desc();
    U32 on, oc, oh, ow;
    on = outputDescGPU.dims[3];
    oc = outputDescGPU.dims[2];
    oh = outputDescGPU.dims[1];
    ow = outputDescGPU.dims[0];
    U8 *outputGPU = ut_input_v(on * oc * oh * ow, dt, UT_INIT_RANDOM);

    GCLMem_t output = alloc(outputTensor);
    GCLMem_t input = alloc(inputTensor);
    CHECK_STATUS(gcl_fill_memory_zero(handle, input));

    U32 maxBytes = 0;
    U32 tmpBytes = 0;
    tmpBytes = tensorNumBytes(inputDescGPU);
    maxBytes = (tmpBytes > maxBytes) ? tmpBytes : maxBytes;
    tmpBytes = tensorNumBytes(outputDescGPU);
    maxBytes = (tmpBytes > maxBytes) ? tmpBytes : maxBytes;
    GCLMem_t tmpbuf = alloc_bytes(tmpTensor, maxBytes);

    CHECK_STATUS(ocl_set_input(handle, input, inputDescGPU, inputCPU, tmpbuf, true));
    CHECK_STATUS(padding(inputTensor, padParamSpec, outputTensor, &archInfo));

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
    sprintf(params, "(%u %u %u %u)->(%u %u %u %u)", in, ic, ih, iw, on, oc, oh, ow);
    sprintf(buffer, "%20s, %80s", "padding", params);
    double ops = on * oc * oh * ow * 4;
    ut_log(dt, buffer, ops, time);
    Tensor inputTensorCpu;
    inputTensorCpu.resize(inputDescCPU);
    inputTensorCpu.alloc();
    UNI_MEMCPY(
        get_ptr_from_tensor(inputTensorCpu, CPU_GENERAL), inputCPU, tensorNumBytes(inputDescCPU));

    Tensor outputTensorCpu;
    CHECK_STATUS(padding_infer_output_size(
        &inputTensorCpu, padParamSpec, &outputTensorCpu, &UT_SERIAL_ARCHINFO));
    outputTensorCpu.alloc();

    if (UT_CHECK) {
        CHECK_STATUS(padding(inputTensorCpu, padParamSpec, outputTensorCpu, &UT_SERIAL_ARCHINFO));
    }
    TensorDesc desc = outputTensorCpu.get_desc();
    ut_check_v(
        outputGPU, get_ptr_from_tensor(outputTensorCpu, CPU_GENERAL), tensorNumElements(desc), dt, 0.3);

    CHECK_STATUS(gcl_finish(handle));
    CHECK_STATUS(gcl_clean_kernelVec(handle));
    free(inputCPU);
    free(outputGPU);
    return 0;
}

int main(int argc, char **argv)
{
    paddingTest(argc, argv, DT_F16);
    return 0;
}
