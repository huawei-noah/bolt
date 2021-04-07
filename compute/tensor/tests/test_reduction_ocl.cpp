// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifdef _USE_FP16
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

int reductionTest(int argc, char **argv, DataType dt)
{
    CHECK_REQUIREMENT(argc >= 9);
    U32 in = atoi(argv[1]);
    U32 ic = atoi(argv[2]);
    U32 ih = atoi(argv[3]);
    U32 iw = atoi(argv[4]);
    ReductionParamSpec p;
    p.keep_dim = atoi(argv[5]);
    bool use_c4 = atoi(argv[6]);
    p.axes_num = atoi(argv[7]);
    for (int i = 0; i < p.axes_num; i++) {
        p.axes[i] = atoi(argv[8 + i]);
    }
    p.reduction_mode = REDUCTION_MEAN;
    p.coeff = 1.0;
    TensorDesc maskDesc;
    maskDesc.nDims = 0;
    ArchInfo archInfo;
    archInfo.arch = MALI;
    ArchInfo archInfo_org;
    archInfo_org.arch = CPU_GENERAL;

    DataFormat df = DF_MTK;
    TensorDesc inputDesc = tensor3df(dt, df, ic, ih, iw);
    //DataFormat df = DF_NCHW;
    //TensorDesc inputDesc = tensor4df(dt, df, in, ic, ih, iw);
    U32 len = tensorNumElements(inputDesc);
    U8 *input_cpu = ut_input_v(len, dt, UT_INIT_RANDOM);

    Tensor inputTensorCpu;
    inputTensorCpu.resize(inputDesc);
    inputTensorCpu.alloc();
    memcpy(get_ptr_from_tensor(inputTensorCpu, UT_ARCH), input_cpu, tensorNumBytes(inputDesc));

    Tensor maskTensorCpu;
    maskTensorCpu.resize(maskDesc);
    Tensor outputTensorCpu;
    Tensor tmpTensorCpu;
    CHECK_STATUS(reduction_infer_output_size(
        &inputTensorCpu, maskTensorCpu, p, &outputTensorCpu, &archInfo_org));
    outputTensorCpu.alloc();
    CHECK_STATUS(
        reduction(inputTensorCpu, maskTensorCpu, p, tmpTensorCpu, outputTensorCpu, &archInfo_org));

    U8 *output_gpu = NULL;
    std::shared_ptr<GCLHandle> handleSharedPtr = OCLContext::getInstance().handle;
    GCLHandle_t handle = handleSharedPtr.get();
    std::vector<GCLKernelInfo> kernelVec;
    handle->kernelVec = &kernelVec;
    Tensor inputTensor = Tensor(OCLMem);
    Tensor maskTensor = Tensor(OCLMem);
    Tensor outputTensor = Tensor(OCLMem);
    Tensor tmpTensor = Tensor(OCLMem);
    inputTensor.resize(inputDesc);
    maskTensor.resize(maskDesc);
    if (use_c4) {
        U32 str[3] = {1, 1, 1};
        U32 off[3] = {0, 0, 0};
        GCLMemDesc inputMemDesc = gcl_mem_desc(str, off, DT_U8, DF_NCWHC4);
        ocl_set_desc(&inputTensor, inputMemDesc);
    }

    MaliPara maliPara;
    maliPara.handle = handle;
    archInfo.archPara = &maliPara;
    CHECK_STATUS(reduction_infer_output_size(&inputTensor, maskTensor, p, &outputTensor, &archInfo));
    TensorDesc outputDesc = outputTensor.get_desc();
    U32 on, oc, oh, ow;
    tensorSelectGet(outputDesc, NULL, NULL, &on, &oc, &oh, &ow);

    GCLMem_t output = alloc_map(outputTensor);
    GCLMem_t input = alloc(inputTensor);
    CHECK_STATUS(gcl_fill_memory_zero(handle, input));

    U32 maxBytes = 0;
    U32 tmpBytes = 0;
    CHECK_STATUS(
        reduction_infer_forward_tmp_bytes(inputTensor, p, outputTensor, &tmpBytes, &archInfo));
    maxBytes = tensorNumBytes(inputDesc);
    maxBytes = (tmpBytes > maxBytes) ? tmpBytes : maxBytes;
    GCLMem_t tmpbuf = alloc_bytes(tmpTensor, maxBytes);

    CHECK_STATUS(ocl_set_input(handle, input, inputDesc, input_cpu, tmpbuf, true));
    CHECK_STATUS(reduction(inputTensor, maskTensor, p, tmpTensor, outputTensor, &archInfo));

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
    outputDesc = outputTensor.get_desc();
    CHECK_STATUS(ocl_get_output(handle, output, outputDesc, true));
    output_gpu = output->mapPtrArray.back();

    char buffer[150];
    char params[120];
    sprintf(params, "(%u %u %u %u) %d =(%u %u %u %u)", in, ic, ih, iw, p.axes_num, on, oc, oh, ow);
    sprintf(buffer, "%20s, %80s", "Reduction", params);
#ifdef _DEBUG
    double ops = len;
    ut_log(dt, buffer, ops, time);
#endif
    ut_check_a(output_gpu, get_ptr_from_tensor(outputTensorCpu, UT_ARCH), on * oc * oh * ow, dt);
    CHECK_STATUS(gcl_finish(handle));
    CHECK_STATUS(gcl_clean_kernelVec(handle));
    free(input_cpu);
    return 0;
}

int main(int argc, char **argv)
{
    reductionTest(argc, argv, DT_F16);
    return 0;
}
#endif
