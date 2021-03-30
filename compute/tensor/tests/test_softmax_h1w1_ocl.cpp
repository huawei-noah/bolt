// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include <string.h>
#include "tensor_computing.h"
#include "ut_util.h"
#include "gcl.h"
#include "libkernelsource.h"
inline GCLMem_t alloc(Tensor tensor)
{
    auto mem = (OclMemory *)tensor.get_memory();
    mem->alloc();
    return (GCLMem_t)mem->get_ptr();
}

inline GCLMem_t alloc_map(Tensor tensor)
{
    auto mem = (OclMemory *)tensor.get_memory();
    mem->mapped_alloc();
    return (GCLMem_t)mem->get_ptr();
}

inline GCLMem_t alloc_bytes(Tensor tensor, U32 size)
{
    auto mem = (OclMemory *)tensor.get_memory();
    GCLMem_t ptr = NULL;
    if (size > 0) {
        mem->resize(tensor1d(DT_U8, size));
        mem->alloc();
        ptr = (GCLMem_t)mem->get_ptr();
    }
    return ptr;
}

int softmaxTest(int argc, char **argv, DataType dt)
{
    U32 in, ic, ih, iw;
    in = 1;
    ic = 4;
    ih = 1;
    iw = 1;

    if (argc == 2) {
        ic = atoi(argv[1]);
    }

    SoftmaxParamSpec p;
    p.axis = 1;

    ArchInfo archInfo;
    archInfo.arch = MALI;
    ArchInfo archInfo_org;
    archInfo_org.arch = CPU_GENERAL;

    TensorDesc in_desc, in_desc_gpu, out_desc;
    in_desc = tensor4df(dt, DF_NCHW, in, ic, ih, iw);

    U8 *input_cpu = ut_input_v(in * ic * ih * iw, dt, UT_INIT_RANDOM);
    U8 *output_gpu = NULL;
    in_desc_gpu = tensor4df(dt, DF_NCHW, in, ic, ih, iw);

    std::shared_ptr<GCLHandle> handleSharedPtr = OCLContext::getInstance().handle;
    GCLHandle_t handle = handleSharedPtr.get();
    std::vector<GCLKernelInfo> kernelVec;
    handle->kernelVec = &kernelVec;
    Tensor inputTensor = Tensor(OCLMem);
    Tensor outputTensor = Tensor(OCLMem);
    Tensor tmpTensor = Tensor(OCLMem);
    inputTensor.resize(in_desc_gpu);

    MaliPara maliPara;
    maliPara.handle = handle;
    archInfo.archPara = &maliPara;
    CHECK_STATUS(softmax_infer_output_size(&inputTensor, &outputTensor, &archInfo));
    U32 maxBytes = 0;
    U32 tmpBytes;
    CHECK_STATUS(softmax_infer_forward_tmp_bytes(inputTensor, &tmpBytes, &archInfo));
    maxBytes = (tmpBytes > maxBytes) ? tmpBytes : maxBytes;  // 18

    GCLMem_t output = alloc_map(outputTensor);
    GCLMem_t input = alloc(inputTensor);
    CHECK_STATUS(gcl_fill_memory_zero(handle, input));
    tmpBytes = tensorNumBytes(in_desc_gpu);
    maxBytes = (tmpBytes > maxBytes) ? tmpBytes : maxBytes;
    GCLMem_t tmpbuf = alloc_bytes(tmpTensor, maxBytes);

    CHECK_STATUS(ocl_set_input(handle, input, in_desc_gpu, input_cpu, tmpbuf, true));
    CHECK_STATUS(softmax(inputTensor, p, tmpTensor, outputTensor, &archInfo));
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
    out_desc = outputTensor.get_desc();
    CHECK_STATUS(ocl_get_output(handle, output, out_desc, true));
    output_gpu = output->mapPtrArray.back();

    char buffer[150];
    char params[120];
    sprintf(params, "(%u %u %u %u)", in, ic, ih, iw);
    sprintf(buffer, "%20s, %80s", "softmax_h1w1", params);
#ifdef _DEBUG
    double ops = 1;
    ut_log(dt, buffer, ops, time);
#endif
    Tensor inputTensorCpu;
    inputTensorCpu.resize(in_desc);
    inputTensorCpu.alloc();
    memcpy(get_ptr_from_tensor(inputTensorCpu, UT_ARCH), input_cpu, tensorNumBytes(in_desc));

    Tensor outputTensorCpu;
    outputTensorCpu.resize(out_desc);
    outputTensorCpu.alloc();

    Tensor tmpTensorCpu;
    CHECK_STATUS(softmax(inputTensorCpu, p, tmpTensorCpu, outputTensorCpu, &archInfo_org));

    ut_check_a(output_gpu, get_ptr_from_tensor(outputTensorCpu, UT_ARCH), in * ih * iw * ic, dt);
    CHECK_STATUS(gcl_finish(handle));
    CHECK_STATUS(gcl_clean_kernelVec(handle));

    free(input_cpu);
    return 0;
}

int main(int argc, char **argv)
{
#ifdef _USE_FP16
    softmaxTest(argc, argv, DT_F16);
#endif
    return 0;
}
