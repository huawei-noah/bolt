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

int depthwisePointwiseConvolutionTest(
    int argc, char *argv[], DataFormat filterDataFormat, DataType dt)
{
    U32 in, ic, ih, iw;
    U32 fn, fc, fh, fw;
    U32 group, stride, padding, dilation;
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
    fn = 8;
    fh = 3;
    fw = 3;
    group = 1;
    stride = 1;
    padding = 1;
    dilation = 1;
    bool useNchw = false;

    if (argc == 9 || argc == 10) {
        ic = atoi(argv[1]);
        ih = atoi(argv[2]);
        iw = atoi(argv[3]);
        fn = atoi(argv[4]);
        fh = atoi(argv[5]);
        fw = atoi(argv[6]);
        stride = atoi(argv[7]);
        padding = atoi(argv[8]);
        if (argc == 10) {
            dilation = atoi(argv[9]);
        }
    } else if (argc == 11 || argc == 12) {
        in = atoi(argv[1]);
        ic = atoi(argv[2]);
        ih = atoi(argv[3]);
        iw = atoi(argv[4]);
        fn = atoi(argv[5]);
        fh = atoi(argv[6]);
        fw = atoi(argv[7]);
        stride = atoi(argv[8]);
        padding = atoi(argv[9]);
        dilation = atoi(argv[10]);
        if (argc == 12) {
            useNchw = atoi(argv[11]);
        }
    } else {
        CHECK_STATUS(NOT_MATCH);
    }
    U32 pr = padding;
    U32 pl = padding;
    U32 pt = padding;
    U32 pb = padding;
    fc = ic;
    U32 fhd = (fh - 1) * dilation + 1;
    U32 fwd = (fw - 1) * dilation + 1;
    on = in;
    oc = fn;
    oh = (ih + pt + pb - fhd) / stride + 1;
    ow = (iw + pr + pl - fwd) / stride + 1;
    ActivationParamSpec dwActivationParamSpec;
    ActivationParamSpec pwActivationParamSpec;
    dwActivationParamSpec.mode = ACTIVATION_NULL;
    pwActivationParamSpec.mode = ACTIVATION_NULL;
    ConvolutionParamSpec convParamSpec = createConvolutionParamSpec(group, 1, fh, fw, 1, stride,
        stride, 0, 0, pt, pb, pl, pr, dilation, dilation, dilation, fn,
        CONVOLUTION_DEPTHWISE_POINTWISE);

    U32 dwFilterLen = 1 * fc * fh * fw;
    U32 pwFilterLen = fn * fc * 1 * 1;
    U32 dwBiasLen = fc;
    U32 pwBiasLen = fn;

    TensorDesc inputDesc;
    if (useNchw) {
        inputDesc = tensor4df(dt, DF_NCHW, in, ic, ih, iw);
    } else {
        inputDesc = tensor4df(dt, DF_NCHWC4, in, ic, ih, iw);
    }
    TensorDesc dwFilterDesc = tensor4df(dt, filterDataFormat, 1, fc, fh, fw);
    TensorDesc pwFilterDesc = tensor4df(dt, filterDataFormat, fn, fc, 1, 1);
    TensorDesc dwBiasDesc = tensor1d(dt, dwBiasLen);
    TensorDesc pwBiasDesc = tensor1d(dt, pwBiasLen);

    U8 *input_cpu = ut_input_v(in * ic * ih * iw, dt, UT_INIT_RANDOM);
    U8 *dw_filter_cpu = ut_input_v(dwFilterLen, dt, UT_INIT_RANDOM);
    U8 *pw_filter_cpu = ut_input_v(pwFilterLen, dt, UT_INIT_RANDOM);
    U8 *dw_bias_cpu = ut_input_v(dwBiasLen, dt, UT_INIT_RANDOM);
    U8 *pw_bias_cpu = ut_input_v(pwBiasLen, dt, UT_INIT_RANDOM);
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
    Tensor dwFilterTensorOrg = Tensor(OCLMem);
    Tensor dwFilterTensor = Tensor(OCLMem);
    Tensor pwFilterTensorOrg = Tensor(OCLMem);
    Tensor pwFilterTensor = Tensor(OCLMem);
    Tensor dwBiasTensor = Tensor(OCLMemImg1D);
    Tensor pwBiasTensor = Tensor(OCLMem);
    Tensor pwBiasTensorBuf = Tensor(OCLMem);
    Tensor pwBiasTensorImg = Tensor(OCLMemImg1D);
    Tensor tmpTensor = Tensor(OCLMem);
    Tensor tmpTensorImgA = Tensor(OCLMemImg);
    Tensor tmpTensorImgB = Tensor(OCLMemImg);
    inputTensor.resize(inputDesc);
    dwFilterTensorOrg.resize(dwFilterDesc);
    pwFilterTensorOrg.resize(pwFilterDesc);
    dwBiasTensor.resize(dwBiasDesc);
    pwBiasTensor.resize(pwBiasDesc);
    pwBiasTensorBuf.resize(pwBiasDesc);
    pwBiasTensorImg.resize(pwBiasDesc);

    MaliPara maliPara;
    ForwardRunInfoMali runInfo;
    runInfo.algorithm = (I32)(DEPTHWISE_CONVOLUTION_ALGORITHM_NULL);
    maliPara.handle = handle;
    maliPara.forwardRunInfo = &runInfo;
    archInfo.archPara = &maliPara;

    CHECK_STATUS(depthwise_pointwise_convolution_infer_output_size(&inputTensor, dwFilterTensorOrg,
        pwFilterTensorOrg, convParamSpec, &outputTensor, dt, &archInfo));
    ConvolutionPolicy policy = CONVOLUTION_TUNNING;
    DepthwiseConvolutionForwardAlgorithm alg = DEPTHWISE_CONVOLUTION_ALGORITHM_NULL;
    CHECK_STATUS(depthwise_pointwise_convolution_infer_forward_algorithm(inputTensor,
        dwFilterTensorOrg, pwFilterTensorOrg, outputTensor, convParamSpec, policy, &alg, DT_F16,
        dwActivationParamSpec, pwActivationParamSpec, &archInfo));

    U32 maxBytes[4] = {0};
    U32 tmpBytes;
    CHECK_STATUS(
        depthwise_pointwise_convolution_infer_forward_tmp_bytes(inputTensor, dwFilterTensorOrg,
            pwFilterTensorOrg, outputTensor, convParamSpec, alg, maxBytes, &archInfo));

    TensorDesc dwFtmDesc;
    TensorDesc pwFtmDesc;
    CHECK_STATUS(depthwise_pointwise_convolution_transform_filter_bytes(dwFilterTensorOrg,
        pwFilterTensorOrg, convParamSpec, alg, &dwFtmDesc, &pwFtmDesc, &archInfo));
    dwFilterTensor.resize(dwFtmDesc);
    pwFilterTensor.resize(pwFtmDesc);

    GCLMem_t output = alloc(outputTensor);
    GCLMem_t input = alloc(inputTensor);
    alloc_host_ptr(dwFilterTensorOrg, dw_filter_cpu);
    alloc_host_ptr(pwFilterTensorOrg, pw_filter_cpu);
    alloc(dwFilterTensor);
    alloc(pwFilterTensor);
    CHECK_STATUS(gcl_fill_memory_zero(handle, input));

    if ((ic & 3) != 0) {
        U32 icAlign = (ic + 3) / 4 * 4;
        U8 *tmp = ut_input_v(icAlign, dt, UT_INIT_ZERO);
        UNI_MEMCPY(tmp, dw_bias_cpu, ic * bytesOf(dt));
        free(dw_bias_cpu);
        dw_bias_cpu = tmp;
    }
    alloc_host_ptr(dwBiasTensor, dw_bias_cpu);

    U8 *pw_bias_val = ut_input_v(oc + 8, dt, UT_INIT_ZERO);
    UNI_MEMCPY(pw_bias_val, pw_bias_cpu, oc * bytesOf(dt));
    free(pw_bias_cpu);
    pw_bias_cpu = pw_bias_val;
    alloc_host_ptr(pwBiasTensorImg, pw_bias_cpu);
    alloc_padding(pwBiasTensorBuf, 0, 8, 0, 0, pw_bias_cpu);

    TensorDesc outputDesc = outputTensor.get_desc();
    tmpBytes = tensorNumBytes(inputDesc);
    maxBytes[0] = (tmpBytes > maxBytes[0]) ? tmpBytes : maxBytes[0];
    tmpBytes = tensorNumBytes(outputDesc);
    maxBytes[0] = (tmpBytes > maxBytes[0]) ? tmpBytes : maxBytes[0];
    GCLMem_t tmpbuf = alloc_bytes(tmpTensor, maxBytes[0]);
    alloc_img(tmpTensorImgA, maxBytes + 1);
    alloc_img(tmpTensorImgB, maxBytes + 4);
    std::vector<Tensor> tmpTensors(3);
    tmpTensors[0] = tmpTensor;
    tmpTensors[1] = tmpTensorImgA;
    tmpTensors[2] = tmpTensorImgB;

    CHECK_STATUS(depthwise_pointwise_convolution_transform_filter(dwFilterTensorOrg,
        pwFilterTensorOrg, convParamSpec, alg, &dwFilterTensor, &pwFilterTensor, &archInfo));
    pwBiasTensor = (runInfo.algorithm == (I32)(DEPTHWISE_POINTWISE_CONVOLUTION_ALGORITHM_GEMM))
        ? pwBiasTensorBuf
        : pwBiasTensorImg;
    CHECK_STATUS(ocl_set_input(handle, input, inputDesc, input_cpu, tmpbuf, true));

    std::vector<Tensor> inputTensors(1, inputTensor);
    CHECK_STATUS(depthwise_pointwise_convolution(inputTensors, dwFilterTensor, pwFilterTensor,
        convParamSpec, alg, nullptr, dwBiasTensor, pwBiasTensor, tmpTensors, outputTensor,
        dwActivationParamSpec, pwActivationParamSpec, &archInfo));

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
    sprintf(buffer, "%20s, %80s", "DepthwisePointwise", params);
    double ops = 2.0 * in * ic * ih * iw * fh * fw + in * ic * oh * ow +
        2.0 * on * oc * oh * ow * ic + on * oc * oh * ow;
    ut_log(dt, buffer, ops, time);

    inputDesc.df = DF_NCHW;
    outputDesc.df = DF_NCHW;
    Tensor inputTensorCpu;
    inputTensorCpu.resize(inputDesc);
    inputTensorCpu.alloc();
    UNI_MEMCPY(
        get_ptr_from_tensor(inputTensorCpu, CPU_GENERAL), input_cpu, tensorNumBytes(inputDesc));

    Tensor dwFilterTensorCpu;
    dwFilterTensorCpu.resize(dwFilterDesc);
    dwFilterTensorCpu.alloc();
    UNI_MEMCPY(get_ptr_from_tensor(dwFilterTensorCpu, CPU_GENERAL), dw_filter_cpu,
        tensorNumBytes(dwFilterDesc));

    Tensor pwFilterTensorCpu;
    pwFilterTensorCpu.resize(pwFilterDesc);
    pwFilterTensorCpu.alloc();
    UNI_MEMCPY(get_ptr_from_tensor(pwFilterTensorCpu, CPU_GENERAL), pw_filter_cpu,
        tensorNumBytes(pwFilterDesc));

    Tensor dwBiasTensorCpu;
    dwBiasTensorCpu.resize(dwBiasDesc);
    dwBiasTensorCpu.alloc();
    UNI_MEMCPY(
        get_ptr_from_tensor(dwBiasTensorCpu, CPU_GENERAL), dw_bias_cpu, tensorNumBytes(dwBiasDesc));

    Tensor pwBiasTensorCpu;
    pwBiasTensorCpu.resize(pwBiasDesc);
    pwBiasTensorCpu.alloc();
    UNI_MEMCPY(
        get_ptr_from_tensor(pwBiasTensorCpu, CPU_GENERAL), pw_bias_cpu, tensorNumBytes(pwBiasDesc));

    Tensor outputTensorCpu;
    outputTensorCpu.resize(outputDesc);
    outputTensorCpu.alloc();

    Tensor tmpTensorCpu;
    // setup tmp
    CHECK_STATUS(
        depthwise_pointwise_convolution_infer_forward_tmp_bytes(inputTensorCpu, dwFilterTensorCpu,
            pwFilterTensorCpu, outputTensorCpu, convParamSpec, alg, &tmpBytes, &archInfo));
    tmpTensorCpu.resize(tensor1d(DT_F16, tmpBytes / bytesOf(DT_F16)));
    tmpTensorCpu.alloc();

    std::vector<Tensor> inputTensorsCpu(1, inputTensorCpu);
    std::vector<Tensor> tmpTensorsCpu(1, tmpTensorCpu);
    CHECK_STATUS(depthwise_pointwise_convolution(inputTensorsCpu, dwFilterTensorCpu,
        pwFilterTensorCpu, convParamSpec, DEPTHWISE_CONVOLUTION_ALGORITHM_DIRECT, nullptr,
        dwBiasTensorCpu, pwBiasTensorCpu, tmpTensorsCpu, outputTensorCpu, dwActivationParamSpec,
        pwActivationParamSpec, &UT_SERIAL_ARCHINFO));
    ut_check_v(output_gpu, get_ptr_from_tensor(outputTensorCpu, CPU_GENERAL), on * oc * ow * oh, dt, 0.3);

    CHECK_STATUS(gcl_finish(handle));
    CHECK_STATUS(gcl_clean_kernelVec(handle));
    free(output_gpu);
    free(input_cpu);
    free(dw_filter_cpu);
    free(pw_filter_cpu);
    free(dw_bias_cpu);
    free(pw_bias_cpu);
    return 0;
}

int main(int argc, char **argv)
{
#ifdef _USE_FP16
    depthwisePointwiseConvolutionTest(argc, argv, DF_NCHW, DT_F16);
#else
    depthwisePointwiseConvolutionTest(argc, argv, DF_NCHW, DT_F32);
#endif
    return 0;
}
