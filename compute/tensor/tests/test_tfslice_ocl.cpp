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

int tfsliceTest(int argc, char **argv, DataType dt)
{
    CHECK_REQUIREMENT(argc == 25);
    U32 in = atoi(argv[1]);
    U32 ic = atoi(argv[2]);
    U32 ih = atoi(argv[3]);
    U32 iw = atoi(argv[4]);
    TfSliceParamSpec p;
    for (U32 i = 0; i < 4; i++) {
        p.begin[i] = atoi(argv[5 + i]);
    }
    for (U32 i = 0; i < 4; i++) {
        p.end[i] = atoi(argv[9 + i]);
    }
    for (U32 i = 0; i < 4; i++) {
        p.strides[i] = atoi(argv[13 + i]);
    }
    for (U32 i = 0; i < 4; i++) {
        p.begin_mask[i] = atoi(argv[17 + i]);
    }
    for (U32 i = 0; i < 4; i++) {
        p.end_mask[i] = atoi(argv[21 + i]);
    }

    ArchInfo archInfo;
    archInfo.arch = MALI;

    DataFormat df = DF_NCHW;
    TensorDesc inputDesc = tensor4df(dt, df, in, ic, ih, iw);
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
        tfslice_infer_output_size(&inputTensorCpu, p, &outputTensorCpu, &UT_SERIAL_ARCHINFO));
    outputTensorCpu.alloc();
    CHECK_STATUS(tfslice(inputTensorCpu, p, tmpTensorCpu, outputTensorCpu, &UT_SERIAL_ARCHINFO));

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
    CHECK_STATUS(tfslice_infer_output_size(&inputTensor, p, &outputTensor, &archInfo));
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
    CHECK_STATUS(tfslice_infer_forward_tmp_bytes(inputTensor, outputTensor, &tmpBytes, &archInfo));
    maxBytes = (tmpBytes > maxBytes) ? tmpBytes : maxBytes;
    GCLMem_t tmpbuf = alloc_bytes(tmpTensor, maxBytes);

    CHECK_STATUS(ocl_set_input(handle, input, inputDesc, input_cpu, tmpbuf, true));
    CHECK_STATUS(tfslice(inputTensor, p, tmpTensor, outputTensor, &archInfo));

    /*warp up*/
    UNI_INFO_LOG("warm up gpu:\n")
    for (U32 i = 0; i < 2; i++) {
        CHECK_STATUS(gcl_run_kernelVec(handle));
    }

    UNI_INFO_LOG("Run:\n")
#ifdef _DEBUG
    CHECK_STATUS(gcl_run_kernelVec_timing(handle, 0, handle->kernelVec->size()));
    double time = handle->t_execute * 0.001;
#else
    CHECK_STATUS(gcl_run_kernelVec(handle));
#endif
    CHECK_STATUS(ocl_get_output(handle, output, outputDesc, output_gpu, tmpbuf, true));

    char buffer[150];
    char params[120];
    UNI_MEMSET(params, 0, 120);
    sprintf(params, "(%u %u %u %u)=(%u %u %u %u)", in, ic, ih, iw, on, oc, oh, ow);
    sprintf(buffer, "%20s, %80s", "tfslice", params);
#ifdef _DEBUG
    double ops = on * oc * oh * ow;
    ut_log(dt, buffer, ops, time);
#endif
    ut_check_a(output_gpu, get_ptr_from_tensor(outputTensorCpu, CPU_GENERAL), on * oc * oh * ow, dt);
    CHECK_STATUS(gcl_finish(handle));
    CHECK_STATUS(gcl_clean_kernelVec(handle));
    free(input_cpu);
    free(output_gpu);
    return 0;
}

int main(int argc, char **argv)
{
    tfsliceTest(argc, argv, DT_F16);
    return 0;
}
