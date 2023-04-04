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

int sliceTest(int argc, char **argv, DataType dt)
{
    CHECK_REQUIREMENT(argc > 2);
    I32 num = atoi(argv[1]);
    CHECK_REQUIREMENT(argc == 2 + 4 + 1 + num - 1);
    U32 in = atoi(argv[2]);
    U32 ic = atoi(argv[3]);
    U32 ih = atoi(argv[4]);
    U32 iw = atoi(argv[5]);
    SliceParamSpec p;
    p.axis = atoi(argv[6]);
    p.num_slice = num - 1;
    for (U32 i = 0; i < p.num_slice; i++) {
        p.slice_points[i] = atoi(argv[7 + i]);
    }
    ArchInfo archInfo;
    archInfo.arch = MALI;

    DataFormat df = DF_NCHW;
    TensorDesc inDesc = tensor4df(dt, df, in, ic, ih, iw);
    U32 len = tensorNumElements(inDesc);
    U8 *inputCpu = ut_input_v(len, dt, UT_INIT_RANDOM);
    F16 *val = (F16 *)inputCpu;
    for (U32 i = 0; i < len; i++) {
        val[i] = i;
    }
    Tensor inputTensorCpu;
    inputTensorCpu.resize(inDesc);
    inputTensorCpu.alloc();
    UNI_MEMCPY(get_ptr_from_tensor(inputTensorCpu, CPU_GENERAL), inputCpu, tensorNumBytes(inDesc));
    std::vector<Tensor> outputTensorsCpu(num);
    std::vector<Tensor *> outputTensorsPtrCpu(num);
    for (I32 i = 0; i < num; i++) {
        outputTensorsPtrCpu[i] = &outputTensorsCpu[i];
    }
    CHECK_STATUS(
        slice_infer_output_size(&inputTensorCpu, p, outputTensorsPtrCpu, &UT_SERIAL_ARCHINFO));
    for (I32 i = 0; i < num; i++) {
        outputTensorsCpu[i].alloc();
    }
    Tensor tmpTensorCpu;
    CHECK_STATUS(slice(inputTensorCpu, p, tmpTensorCpu, outputTensorsCpu, &UT_SERIAL_ARCHINFO));

    std::shared_ptr<GCLHandle> handleSharedPtr = OCLContext::getInstance().handle;
    GCLHandle_t handle = handleSharedPtr.get();
    std::vector<GCLKernelInfo> kernelVec;
    handle->kernelVec = &kernelVec;

    Tensor inputTensor = Tensor(OCLMem);
    inputTensor.resize(inDesc);
    std::vector<Tensor> outputTensors(num);
    std::vector<Tensor *> outputTensorsPtr;
    Tensor tmpTensor = Tensor(OCLMem);
    for (I32 i = 0; i < num; i++) {
        outputTensors[i] = Tensor(OCLMem);
        outputTensorsPtr.push_back(&outputTensors[i]);
    }

    MaliPara maliPara;
    maliPara.handle = handle;
    archInfo.archPara = &maliPara;
    U32 maxBytes = 0;
    U32 tmpBytes;
    CHECK_STATUS(slice_infer_output_size(&inputTensor, p, outputTensorsPtr, &archInfo));
    CHECK_STATUS(slice_infer_forward_tmp_bytes(inputTensor, p, outputTensors, &tmpBytes, &archInfo));
    maxBytes = (tmpBytes > maxBytes) ? tmpBytes : maxBytes;
    tmpBytes = tensorNumBytes(inDesc);
    maxBytes = (tmpBytes > maxBytes) ? tmpBytes : maxBytes;

    GCLMem_t input = alloc(inputTensor);
    std::vector<GCLMem_t> outputVec;
    for (I32 i = 0; i < num; i++) {
        TensorDesc outputDesc = outputTensors[i].get_desc();
        tmpBytes = tensorNumBytes(outputDesc);
        maxBytes = (tmpBytes > maxBytes) ? tmpBytes : maxBytes;
        GCLMem_t output = alloc(outputTensors[i]);
        outputVec.push_back(output);
    }
    GCLMem_t tmpbuf = alloc_bytes(tmpTensor, maxBytes);

    CHECK_STATUS(ocl_set_input(handle, input, inDesc, (U8 *)inputCpu, tmpbuf, true));
    CHECK_STATUS(slice(inputTensor, p, tmpTensor, outputTensors, &archInfo));

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
    sprintf(params, "(%u %u %u %u)=(%u %u %u %u)/%u", in, ic, ih, iw, in, ic, ih, iw, num);
    sprintf(buffer, "%20s, %80s", "Slice", params);
    double ops = num * len;
    ut_log(dt, buffer, ops, time);

    for (I32 i = 0; i < num; i++) {
        TensorDesc outputDesc = outputTensors[i].get_desc();
        U8 *output_gpu = ut_input_v(tensorNumElements(outputDesc), dt, UT_INIT_RANDOM);
        CHECK_STATUS(ocl_get_output(handle, outputVec[i], outputDesc, output_gpu, tmpbuf, true));
        U8 *output_cpu = (U8 *)get_ptr_from_tensor(outputTensorsCpu[i], CPU_GENERAL);
        ut_check_v(output_gpu, output_cpu, tensorNumElements(outputDesc), outputDesc.dt, 0.1);
        free(output_gpu);
    }

    CHECK_STATUS(gcl_finish(handle));
    CHECK_STATUS(gcl_clean_kernelVec(handle));
    free(inputCpu);
    return 0;
}

int main(int argc, char **argv)
{
    sliceTest(argc, argv, DT_F16);
    return 0;
}
