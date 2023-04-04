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

int depthwiseConvolutionTest(int argc, char *argv[], DataFormat filterDataFormat, DataType dt)
{
    U32 in, ic, ih, iw;
    U32 fn, fc, fh, fw, fhd, fwd;
    U32 group, stride, padding, dila;
    U32 on, oc, oh, ow;
    U32 biasNum;
    ArchInfo archInfo;
    archInfo.arch = MALI;

    if (gcl_check_device_qualcomm(OCLContext::getInstance().handle.get())) {
        archInfo.arch = QUALCOMM;
    }

    in = 1;
    ic = 8;
    ih = 4;
    iw = 4;
    fn = 1;
    fc = 8;
    fh = 3;
    fw = 3;
    group = 1;
    stride = 1;
    padding = 1;
    dila = 1;

    bool useNchw = false;
    if (argc == 9 || argc == 10) {
        ic = atoi(argv[1]);
        ih = atoi(argv[2]);
        iw = atoi(argv[3]);
        fc = atoi(argv[4]);
        fh = atoi(argv[5]);
        fw = atoi(argv[6]);
        stride = atoi(argv[7]);
        padding = atoi(argv[8]);
        if (argc == 10) {
            dila = atoi(argv[9]);
        }
    } else if (argc == 11 || argc == 12) {
        in = atoi(argv[1]);
        ic = atoi(argv[2]);
        ih = atoi(argv[3]);
        iw = atoi(argv[4]);
        fc = atoi(argv[5]);
        fh = atoi(argv[6]);
        fw = atoi(argv[7]);
        stride = atoi(argv[8]);
        padding = atoi(argv[9]);
        dila = atoi(argv[10]);
        if (argc == 12) {
            useNchw = atoi(argv[11]);
        }
    } else {
        CHECK_STATUS(NOT_MATCH);
    }

    fhd = (fh - 1) * dila + 1;
    fwd = (fw - 1) * dila + 1;
    on = in;
    oc = fc;
    oh = (ih + padding * 2 - fhd) / stride + 1;
    ow = (iw + padding * 2 - fwd) / stride + 1;
    ActivationParamSpec dwActivationParamSpec;
    dwActivationParamSpec.mode = ACTIVATION_NULL;
    ConvolutionParamSpec convParamSpec = createConvolutionParamSpec(group, 1, fh, fw, 1, stride,
        stride, 0, 0, padding, padding, padding, padding, dila, dila, dila, fn,
        CONVOLUTION_DEPTHWISE);

    U32 filterLen = fn * fc * fh * fw;
    U32 biasLen = oc;
    TensorDesc inputDesc;
    if (useNchw) {
        inputDesc = tensor4df(dt, DF_NCHW, in, ic, ih, iw);
    } else {
        inputDesc = tensor4df(dt, DF_NCHWC4, in, ic, ih, iw);
    }
    TensorDesc filterDesc = tensor4df(dt, filterDataFormat, fn, fc, fh, fw);
    TensorDesc biasDesc = tensor1d(dt, biasLen);
    U8 *input_cpu = ut_input_v(in * ic * ih * iw, dt, UT_INIT_RANDOM);
    U8 *filter_cpu = ut_input_v(filterLen, dt, UT_INIT_RANDOM);
    U8 *bias_cpu = ut_input_v(biasLen, dt, UT_INIT_RANDOM);
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
    Tensor tmpTensorImg = Tensor(OCLMemImg);
    inputTensor.resize(inputDesc);
    filterTensorOrg.resize(filterDesc);
    biasTensor.resize(biasDesc);

    MaliPara maliPara;
    ForwardRunInfoMali runInfo;
    runInfo.algorithm = (I32)(DEPTHWISE_CONVOLUTION_ALGORITHM_NULL);
    maliPara.handle = handle;
    maliPara.forwardRunInfo = &runInfo;
    archInfo.archPara = &maliPara;

    CHECK_STATUS(depthwise_convolution_infer_output_size(
        &inputTensor, filterTensorOrg, convParamSpec, &outputTensor, dt, &archInfo));
    ConvolutionPolicy policy = CONVOLUTION_TUNNING;
    DepthwiseConvolutionForwardAlgorithm alg = DEPTHWISE_CONVOLUTION_ALGORITHM_NULL;
    CHECK_STATUS(depthwise_convolution_infer_forward_algorithm(inputTensor, filterTensorOrg,
        outputTensor, convParamSpec, policy, &alg, dt, dwActivationParamSpec, &archInfo));

    U32 maxBytes[4] = {0};
    CHECK_STATUS(depthwise_convolution_infer_forward_tmp_bytes(
        inputTensor, filterTensorOrg, outputTensor, convParamSpec, alg, maxBytes, &archInfo));

    TensorDesc ftmDesc;
    CHECK_STATUS(depthwise_convolution_transform_filter_bytes(
        filterTensorOrg, convParamSpec, alg, &ftmDesc, &archInfo));
    filterTensor.resize(ftmDesc);

    GCLMem_t output = alloc(outputTensor);
    GCLMem_t input = alloc(inputTensor);
    alloc_host_ptr(filterTensorOrg, filter_cpu);
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
    U32 tmpBytes = tensorNumBytes(inputDesc);
    maxBytes[0] = (tmpBytes > maxBytes[0]) ? tmpBytes : maxBytes[0];
    tmpBytes = tensorNumBytes(outputDesc);
    maxBytes[0] = (tmpBytes > maxBytes[0]) ? tmpBytes : maxBytes[0];
    GCLMem_t tmpbuf = alloc_bytes(tmpTensor, maxBytes[0]);
    alloc_img(tmpTensorImg, maxBytes + 1);
    Tensor tmp = tmpTensor;
    if (maxBytes[1] > 0 && maxBytes[2] > 0 && maxBytes[3] > 0) {
        tmp = tmpTensorImg;
    }

    CHECK_STATUS(depthwise_convolution_transform_filter(
        filterTensorOrg, convParamSpec, alg, &filterTensor, &archInfo));

    CHECK_STATUS(ocl_set_input(handle, input, inputDesc, input_cpu, tmpbuf, true));

    CHECK_STATUS(depthwise_convolution(inputTensor, filterTensor, convParamSpec, alg, nullptr,
        biasTensor, tmp, outputTensor, dwActivationParamSpec, &archInfo));

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
    sprintf(buffer, "%20s, %80s", "DepthwiseConvolution", params);
    double ops = 2.0 * in * ic * ih * iw * fh * fw + in * ic * oh * ow;
    ut_log(dt, buffer, ops, time);
    Tensor inputTensorCpu;
    inputDesc.df = DF_NCHW;
    outputDesc.df = DF_NCHW;
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
    // setup tmp
    CHECK_STATUS(depthwise_convolution_infer_forward_tmp_bytes(inputTensorCpu, filterTensorCpu,
        outputTensorCpu, convParamSpec, alg, &tmpBytes, &archInfo));
    tmpTensorCpu.resize(tensor1d(dt, tmpBytes / bytesOf(dt)));
    tmpTensorCpu.alloc();

    CHECK_STATUS(depthwise_convolution(inputTensorCpu, filterTensorCpu, convParamSpec,
        DEPTHWISE_CONVOLUTION_ALGORITHM_DIRECT, nullptr, biasTensorCpu, tmpTensorCpu,
        outputTensorCpu, dwActivationParamSpec, &UT_SERIAL_ARCHINFO));
    ut_check_v(
        output_gpu, get_ptr_from_tensor(outputTensorCpu, CPU_GENERAL), on * oc * ow * oh, dt, 0.3);

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
#ifdef _USE_FP16
    depthwiseConvolutionTest(argc, argv, DF_NCHW, DT_F16);
#else
    depthwiseConvolutionTest(argc, argv, DF_NCHW, DT_F32);
#endif
    return 0;
}
