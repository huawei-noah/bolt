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
#include <iostream>
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

inline GCLMem_t alloc_desc(Tensor tensor, GCLMemDesc desc)
{
    auto mem = (OclMemory *)tensor.get_memory();
    mem->padding(desc);
    mem->alloc();
    return (GCLMem_t)mem->get_ptr();
}

int deconvolutionTest(int argc, char *argv[], DataType dt)
{
    U32 biasNum;
    ArchInfo archInfo;
    archInfo.arch = MALI;
    ArchInfo archInfo_org;
    archInfo_org.arch = CPU_GENERAL;
    U32 in = 1;
    U32 ic = 4;
    U32 ih = 2;
    U32 iw = 2;
    U32 fn = 4;
    U32 fh = 2;
    U32 fw = 2;
    U32 fc = 4;
    U32 stride = 2;
    U32 padding = 0;
    U32 group = 1;
    if (argc == 9) {
        ic = atoi(argv[1]);
        ih = atoi(argv[2]);
        iw = atoi(argv[3]);
        fc = atoi(argv[4]);
        fh = atoi(argv[5]);
        fw = atoi(argv[6]);
        stride = atoi(argv[7]);
        padding = atoi(argv[8]);
        fn = ic;
    }
    U32 on = 1;
    U32 oc = fc;
    U32 oh = fh + stride * (ih - 1) - padding - padding;
    U32 ow = fw + stride * (iw - 1) - padding - padding;

    ActivationParamSpec activationDesc;
    activationDesc.mode = ACTIVATION_NULL;
    ConvolutionParamSpec convParamSpec = createConvolutionParamSpec(group, 1, fh, fw, 1, stride,
        stride, 0, 0, padding, padding, padding, padding, 1, 1, 1, fn, Convolution_Deconvolution);

    TensorDesc inputDesc = tensor4df(dt, DF_NCHW, in, ic, ih, iw);
    TensorDesc filterDesc = tensor4df(dt, DF_NCHW, fn, fc, fh, fw);
    TensorDesc biasDesc = tensor1d(dt, oc);
    U8 *input_cpu = ut_input_v(in * ic * ih * iw, dt, UT_INIT_RANDOM);
    U8 *filter_cpu = ut_input_v(fn * fc * fh * fw, dt, UT_INIT_RANDOM);
    U8 *bias_cpu = ut_input_v(oc, dt, UT_INIT_RANDOM);
    U8 *output_gpu = NULL;
    TensorDesc inputDesc_gpu = tensor4df(dt, DF_NCHW, in, ic, ih, iw);

    std::shared_ptr<GCLHandle> handleSharedPtr = OCLContext::getInstance().handle;
    GCLHandle_t handle = handleSharedPtr.get();
    std::vector<GCLKernelInfo> kernelVec;
    handle->kernelVec = &kernelVec;
    Tensor inputTensor = Tensor(OCLMem);
    Tensor outputTensor = Tensor(OCLMem);
    Tensor filterTensorOrg = Tensor(OCLMem);
    Tensor filterTensor = Tensor(OCLMem);
    Tensor biasTensor = Tensor(OCLMem);
    Tensor tmpTensor = Tensor(OCLMem);
    inputTensor.resize(inputDesc_gpu);
    filterTensor.resize(filterDesc);
    filterTensorOrg.resize(filterDesc);
    biasTensor.resize(biasDesc);

    MaliPara maliPara;
    ForwardRunInfoMali runInfo;
    runInfo.algorithm = (I32)(CONVOLUTION_ALGORITHM_NULL);
    maliPara.handle = handle;
    maliPara.forwardRunInfo = &runInfo;
    archInfo.archPara = &maliPara;

    CHECK_STATUS(deconvolution_infer_output_size(
        &inputTensor, filterTensor, convParamSpec, &outputTensor, dt, &archInfo));

    ConvolutionPolicy policy = CONVOLUTION_TUNNING;
    ConvolutionForwardAlgorithm alg = CONVOLUTION_ALGORITHM_NULL;
    CHECK_STATUS(deconvolution_infer_forward_algorithm(inputTensor, filterTensor, outputTensor,
        convParamSpec, policy, &alg, dt, activationDesc, &archInfo));

    U32 maxBytes = 0;
    U32 tmpBytes;
    CHECK_STATUS(deconvolution_infer_forward_tmp_bytes(
        inputTensor, filterTensor, outputTensor, convParamSpec, alg, &tmpBytes, &archInfo));
    maxBytes = (tmpBytes > maxBytes) ? tmpBytes : maxBytes;

    U32 ftmBytes;
    U32 str[3] = {0, 0, 0};
    U32 off[3] = {0, 0, 0};
    GCLMemDesc filterMemDesc = gcl_mem_desc(str, off, DT_U8, DF_NCWHC4);
    maliPara.gclmemFilterDesc = &filterMemDesc;
    CHECK_STATUS(deconvolution_transform_filter_bytes(
        filterTensor, convParamSpec, alg, &ftmBytes, &archInfo));

    GCLMem_t output = alloc_map(outputTensor);
    GCLMem_t input = alloc(inputTensor);
    CHECK_STATUS(gcl_fill_memory_zero(handle, input));

    GCLMemDesc desc = gclmem_build_desc();
    biasNum = (oc + 3) / 4;
    desc.memType = GCL_MEM_IMG_1D;
    desc.byteSize = biasNum * 4 * bytesOf(dt);
    desc.stride[0] = biasNum;
    desc.stride[1] = 1;
    desc.stride[2] = 1;
    desc.offset[0] = 0;
    desc.offset[1] = 0;
    desc.offset[2] = 0;
    desc.num = biasNum;
    desc.memFormat = DF_NHWC;
    desc.flags = CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR;
    U8 *bias_cpu_align = NULL;
    if ((oc & 3) != 0) {
        U8 *bias_cpu_align = ut_input_v((oc + 3) / 4 * 4, dt, UT_INIT_ZERO);
        memcpy(bias_cpu_align, bias_cpu, (oc + 3) / 4 * 4 * bytesOf(dt));
        desc.host_ptr = bias_cpu_align;
    } else {
        desc.host_ptr = bias_cpu;
    }

    alloc_desc(biasTensor, desc);

    desc = filterMemDesc;
    alloc_desc(filterTensor, desc);

    desc.stride[0] = fw * fh;
    desc.stride[1] = fc;
    desc.stride[2] = fn;
    desc.offset[0] = 0;
    desc.offset[1] = 0;
    desc.offset[2] = 0;
    desc.byteSize = fw * fh * fc * fn * bytesOf(dt);
    desc.num = fw * fh * fc * fn;
    desc.memType = GCL_MEM_BUF;
    desc.memFormat = DF_NCHW;
    desc.flags = CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR;
    desc.host_ptr = filter_cpu;
    alloc_desc(filterTensorOrg, desc);

    tmpBytes = tensorNumBytes(inputDesc_gpu);
    maxBytes = (tmpBytes > maxBytes) ? tmpBytes : maxBytes;
    GCLMem_t tmpbuf = alloc_bytes(tmpTensor, maxBytes);

    CHECK_STATUS(deconvolution_transform_filter(
        filterTensorOrg, convParamSpec, alg, tmpTensor, &filterTensor, &archInfo));
    CHECK_STATUS(ocl_set_input(handle, input, inputDesc_gpu, input_cpu, tmpbuf, true));

    CHECK_STATUS(deconvolution(inputTensor, filterTensor, convParamSpec, alg, nullptr, biasTensor,
        tmpTensor, outputTensor, activationDesc, &archInfo));
    /*warp up*/
    UNI_INFO_LOG("warm up gpu:\n")
    for (U32 i = 0; i < 2; i++) {
        CHECK_STATUS(gcl_run_kernelVec(handle));
    }

#ifdef _DEBUG
    CHECK_STATUS(gcl_run_kernelVec_timing(handle, 0, handle->kernelVec->size()));
    double time = handle->t_execute * 0.001;
#else
    CHECK_STATUS(gcl_run_kernelVec(handle));
#endif
    TensorDesc outputDesc = outputTensor.get_desc();
    CHECK_STATUS(ocl_get_output(handle, output, outputDesc, true));
    output_gpu = output->mapPtrArray.back();
    char buffer[150];
    char params[120];
    sprintf(params, "(%u %u %u %u)+(%u %u %u %u)/(%u %u)=(%u %u %u %u)", in, ic, ih, iw, fn, fc, fh,
        fw, stride, padding, on, oc, oh, ow);
    sprintf(buffer, "%20s, %80s", "Deonvolution", params);
#ifdef _DEBUG
    double ops = (1.0 * on * oc * ih * iw) * (2.0 * ic * fh * fw + fh * fw);
    ut_log(dt, buffer, ops, time);
#endif
    Tensor inputTensorCpu;
    inputTensorCpu.resize(inputDesc);
    inputTensorCpu.alloc();
    memcpy(get_ptr_from_tensor(inputTensorCpu, UT_ARCH), input_cpu, tensorNumBytes(inputDesc));

    Tensor filterTensorCpu;
    filterTensorCpu.resize(filterDesc);
    filterTensorCpu.alloc();
    memcpy(get_ptr_from_tensor(filterTensorCpu, UT_ARCH), filter_cpu, tensorNumBytes(filterDesc));

    Tensor biasTensorCpu;
    biasTensorCpu.resize(biasDesc);
    biasTensorCpu.alloc();
    memcpy(get_ptr_from_tensor(biasTensorCpu, UT_ARCH), bias_cpu, tensorNumBytes(biasDesc));

    Tensor outputTensorCpu;
    outputTensorCpu.resize(outputDesc);
    outputTensorCpu.alloc();

    Tensor tmpTensorCpu;
    CHECK_STATUS(
        deconvolution(inputTensorCpu, filterTensorCpu, convParamSpec, CONVOLUTION_ALGORITHM_GEMM,
            nullptr, biasTensorCpu, tmpTensorCpu, outputTensorCpu, activationDesc, &archInfo_org));
    ut_check_a(output_gpu, get_ptr_from_tensor(outputTensorCpu, UT_ARCH), on * oc * ow * oh, dt);

    CHECK_STATUS(gcl_finish(handle));
    CHECK_STATUS(gcl_clean_kernelVec(handle));
    free(input_cpu);
    free(filter_cpu);
    free(bias_cpu);
    if (bias_cpu_align) {
        free(bias_cpu_align);
    }
    return 0;
}

int main(int argc, char **argv)
{
#ifdef _USE_FP16
    deconvolutionTest(argc, argv, DT_F16);
#endif
    return 0;
}
