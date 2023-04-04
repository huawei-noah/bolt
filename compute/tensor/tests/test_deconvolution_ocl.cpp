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

int deconvolutionTest(int argc, char *argv[], DataType dt)
{
    U32 biasNum;
    ArchInfo archInfo;
    archInfo.arch = MALI;

    if (gcl_check_device_qualcomm(OCLContext::getInstance().handle.get())) {
        archInfo.arch = QUALCOMM;
    }
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
        stride, 0, 0, padding, padding, padding, padding, 1, 1, 1, fn, CONVOLUTION_DECONVOLUTION);

    TensorDesc inputDesc = tensor4df(dt, DF_NCHW, in, ic, ih, iw);
    TensorDesc filterDesc = tensor4df(dt, DF_NCHW, fn, fc, fh, fw);
    TensorDesc biasDesc = tensor1d(dt, oc);
    TensorDesc inputDescGpu = tensor4df(dt, DF_NCHWC4, in, ic, ih, iw);
    U8 *input_cpu = ut_input_v(in * ic * ih * iw, dt, UT_INIT_RANDOM);
    U8 *filter_cpu = ut_input_v(fn * fc * fh * fw, dt, UT_INIT_RANDOM);
    U8 *bias_cpu = ut_input_v(oc, dt, UT_INIT_RANDOM);
    U8 *output_gpu = ut_input_v(on * oc * oh * ow, dt, UT_INIT_RANDOM);

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
    Tensor filterTensorOrg = Tensor(OCLMem);
    Tensor filterTensor = Tensor(OCLMem);
    Tensor biasTensor = Tensor(OCLMemImg1D);
    Tensor tmpTensor = Tensor(OCLMem);
    inputTensor.resize(inputDescGpu);
    filterTensorOrg.resize(filterDesc);
    biasTensor.resize(biasDesc);

    MaliPara maliPara;
    ForwardRunInfoMali runInfo;
    runInfo.algorithm = (I32)(CONVOLUTION_ALGORITHM_NULL);
    maliPara.handle = handle;
    maliPara.forwardRunInfo = &runInfo;
    archInfo.archPara = &maliPara;

    CHECK_STATUS(deconvolution_infer_output_size(
        &inputTensor, filterTensorOrg, convParamSpec, &outputTensor, dt, &archInfo));

    ConvolutionPolicy policy = CONVOLUTION_TUNNING;
    ConvolutionForwardAlgorithm alg = CONVOLUTION_ALGORITHM_NULL;
    CHECK_STATUS(deconvolution_infer_forward_algorithm(inputTensor, filterTensorOrg, outputTensor,
        convParamSpec, policy, &alg, dt, activationDesc, &archInfo));

    U32 maxBytes = 0;
    U32 tmpBytes;
    CHECK_STATUS(deconvolution_infer_forward_tmp_bytes(
        inputTensor, filterTensorOrg, outputTensor, convParamSpec, alg, &tmpBytes, &archInfo));
    maxBytes = (tmpBytes > maxBytes) ? tmpBytes : maxBytes;

    TensorDesc ftmDesc;
    CHECK_STATUS(deconvolution_transform_filter_bytes(
        filterTensorOrg, convParamSpec, alg, &ftmDesc, &archInfo));

    GCLMem_t output = alloc(outputTensor);
    GCLMem_t input = alloc(inputTensor);
    alloc_host_ptr(filterTensorOrg, filter_cpu);
    filterTensor.resize(ftmDesc);
    alloc(filterTensor);
    CHECK_STATUS(gcl_fill_memory_zero(handle, input));
    if ((oc & 3) != 0) {
        U32 ocAlign = (oc + 3) / 4 * 4;
        U8 *bias_cpu_align = ut_input_v(ocAlign, dt, UT_INIT_ZERO);
        UNI_MEMCPY(bias_cpu_align, bias_cpu, oc * bytesOf(dt));
        free(bias_cpu);
        bias_cpu = bias_cpu_align;
    }
    alloc_host_ptr(biasTensor, bias_cpu);
    TensorDesc outputDesc = outputTensor.get_desc();
    tmpBytes = tensorNumBytes(inputDescGpu);
    maxBytes = (tmpBytes > maxBytes) ? tmpBytes : maxBytes;
    tmpBytes = tensorNumBytes(outputDesc);
    maxBytes = (tmpBytes > maxBytes) ? tmpBytes : maxBytes;
    GCLMem_t tmpbuf = alloc_bytes(tmpTensor, maxBytes);

    CHECK_STATUS(deconvolution_transform_filter(
        filterTensorOrg, convParamSpec, alg, tmpTensor, &filterTensor, &archInfo));
    CHECK_STATUS(ocl_set_input(handle, input, inputDesc, input_cpu, tmpbuf, true));

    CHECK_STATUS(deconvolution(inputTensor, filterTensor, convParamSpec, alg, nullptr, biasTensor,
        tmpTensor, outputTensor, activationDesc, &archInfo));

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
    sprintf(params, "(%u %u %u %u)+(%u %u %u %u)/(%u %u)=(%u %u %u %u)", in, ic, ih, iw, fn, fc, fh,
        fw, stride, padding, on, oc, oh, ow);
    sprintf(buffer, "%20s, %80s", "Deonvolution", params);
    double ops = (1.0 * on * oc * ih * iw) * (2.0 * ic * fh * fw + fh * fw);
    ut_log(dt, buffer, ops, time);

    inputDesc.df = DF_NCHW;
    outputDesc.df = DF_NCHW;
    Tensor inputTensorCpu;
    inputTensorCpu.resize(inputDesc);
    inputTensorCpu.alloc();
    UNI_MEMCPY(
        get_ptr_from_tensor(inputTensorCpu, CPU_GENERAL), input_cpu, tensorNumBytes(inputDesc));

    Tensor filterTensorCpu;
    filterTensorCpu.resize(filterDesc);
    filterTensorCpu.alloc();
    UNI_MEMCPY(
        get_ptr_from_tensor(filterTensorCpu, CPU_GENERAL), filter_cpu, tensorNumBytes(filterDesc));

    Tensor biasTensorCpu;
    biasTensorCpu.resize(biasDesc);
    biasTensorCpu.alloc();
    UNI_MEMCPY(get_ptr_from_tensor(biasTensorCpu, CPU_GENERAL), bias_cpu, tensorNumBytes(biasDesc));

    Tensor outputTensorCpu;
    outputTensorCpu.resize(outputDesc);
    outputTensorCpu.alloc();

    Tensor tmpTensorCpu;
    CHECK_STATUS(deconvolution(inputTensorCpu, filterTensorCpu, convParamSpec,
        CONVOLUTION_ALGORITHM_GEMM, nullptr, biasTensorCpu, tmpTensorCpu, outputTensorCpu,
        activationDesc, &UT_SERIAL_ARCHINFO));
    ut_check_v(output_gpu, get_ptr_from_tensor(outputTensorCpu, CPU_GENERAL), on * oc * ow * oh, dt, 0.3);

    CHECK_STATUS(gcl_finish(handle));
    CHECK_STATUS(gcl_clean_kernelVec(handle));
    free(output_gpu);
    free(input_cpu);
    free(filter_cpu);
    free(bias_cpu);
    return 0;
}

int main(int argc, char **argv)
{
    deconvolutionTest(argc, argv, DT_F16);
    return 0;
}
