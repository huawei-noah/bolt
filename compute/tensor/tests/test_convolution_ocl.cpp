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

int convolutionTest(int argc, char *argv[], DataType dt)
{
    U32 biasNum;
    ArchInfo archInfo;
    archInfo.arch = MALI;
    if (gcl_check_device_qualcomm(OCLContext::getInstance().handle.get())) {
        archInfo.arch = QUALCOMM;
    }

    U32 in = 1;
    U32 ic = 4;
    U32 ih = 4;
    U32 iw = 4;
    U32 fn = 4;
    U32 fh = 3;
    U32 fw = 3;
    U32 group = 1;
    U32 strideW = 1;
    U32 strideH = 1;
    U32 paddingT = 1;
    U32 paddingB = 1;
    U32 paddingL = 1;
    U32 paddingR = 1;
    U32 it = 1;
    U32 fc = ic;
    U32 ft = 1;
    U32 strideT = 1;
    U32 paddingTF = 0;
    U32 paddingTB = 0;
    U32 use_nchw = 0;
    U32 dilationH = 1;
    U32 dilationW = 1;

    if (argc == 9 || argc == 10) {
        ic = atoi(argv[1]);
        ih = atoi(argv[2]);
        iw = atoi(argv[3]);
        fn = atoi(argv[4]);
        fh = atoi(argv[5]);
        fw = atoi(argv[6]);
        strideH = atoi(argv[7]);
        strideW = atoi(argv[7]);
        paddingT = atoi(argv[8]);
        paddingB = atoi(argv[8]);
        paddingL = atoi(argv[8]);
        paddingR = atoi(argv[8]);
        if (argc == 10) {
            use_nchw = atoi(argv[9]);
        }
    }
    if (argc == 13) {
        ic = atoi(argv[1]);
        ih = atoi(argv[2]);
        iw = atoi(argv[3]);
        fn = atoi(argv[4]);
        fh = atoi(argv[5]);
        fw = atoi(argv[6]);
        strideH = atoi(argv[7]);
        strideW = atoi(argv[8]);
        paddingT = atoi(argv[9]);
        paddingB = atoi(argv[10]);
        paddingL = atoi(argv[11]);
        paddingR = atoi(argv[12]);
    }

    if (argc == 16 || argc == 17) {
        in = atoi(argv[1]);
        ic = atoi(argv[2]);
        ih = atoi(argv[3]);
        iw = atoi(argv[4]);
        fn = atoi(argv[5]);
        fh = atoi(argv[6]);
        fw = atoi(argv[7]);
        strideH = atoi(argv[8]);
        strideW = atoi(argv[9]);
        paddingT = atoi(argv[10]);
        paddingB = atoi(argv[11]);
        paddingL = atoi(argv[12]);
        paddingR = atoi(argv[13]);
        dilationH = atoi(argv[14]);
        dilationW = atoi(argv[15]);
        if (argc == 17) {
            use_nchw = atoi(argv[16]);
        }
    }

    if (argc == 20 || argc == 21) {
        in = atoi(argv[1]);
        ic = atoi(argv[2]);
        it = atoi(argv[3]);
        ih = atoi(argv[4]);
        iw = atoi(argv[5]);
        fn = atoi(argv[6]);
        fc = atoi(argv[7]);
        ft = atoi(argv[8]);
        fh = atoi(argv[9]);
        fw = atoi(argv[10]);
        strideT = atoi(argv[11]);
        strideH = atoi(argv[12]);
        strideW = atoi(argv[13]);
        paddingTF = atoi(argv[14]);
        paddingTB = atoi(argv[15]);
        paddingT = atoi(argv[16]);
        paddingB = atoi(argv[17]);
        paddingL = atoi(argv[18]);
        paddingR = atoi(argv[19]);
        if (argc == 21) {
            use_nchw = atoi(argv[6]);
        }
    }
    fc = ic;
    U32 fhd = (fh - 1) * dilationH + 1;
    U32 fwd = (fw - 1) * dilationW + 1;
    U32 on = in;
    U32 oc = fn;
    U32 oh = (ih + paddingT + paddingB - fhd) / strideH + 1;
    U32 ow = (iw + paddingL + paddingR - fwd) / strideW + 1;
    U32 ot = (it + paddingTB + paddingTF - ft) / strideT + 1;
    ActivationParamSpec activationDesc;
    activationDesc.mode = ACTIVATION_NULL;
    ConvolutionParamSpec convParamSpec = createConvolutionParamSpec(group, ft, fh, fw, strideT,
        strideH, strideW, paddingTF, paddingTB, paddingT, paddingB, paddingL, paddingR, 1,
        dilationH, dilationW, fn, CONVOLUTION_DEPTHWISE_POINTWISE);

    TensorDesc inputDesc, filterDesc, inputDesc_gpu;
    if (it > 1) {
        inputDesc = tensor5df(dt, DF_NCHW, in, ic, it, ih, iw);
        filterDesc = tensor5df(dt, DF_NCHW, fn, fc, ft, fh, fw);
        if (use_nchw) {
            inputDesc_gpu = tensor5df(dt, DF_NCHW, in, ic, it, ih, iw);
        } else {
            inputDesc_gpu = tensor5df(dt, DF_NCHWC4, in, ic, it, ih, iw);
        }
    } else {
        inputDesc = tensor4df(dt, DF_NCHW, in, ic, ih, iw);
        filterDesc = tensor4df(dt, DF_NCHW, fn, fc, fh, fw);
        if (use_nchw) {
            inputDesc_gpu = tensor4df(dt, DF_NCHW, in, ic, ih, iw);
        } else {
            inputDesc_gpu = tensor4df(dt, DF_NCHWC4, in, ic, ih, iw);
        }
    }

    TensorDesc biasDesc = tensor1d(dt, oc);
    U8 *input_cpu = ut_input_v(in * ic * it * ih * iw, dt, UT_INIT_RANDOM);
    U8 *filter_cpu = ut_input_v(fn * fc * ft * fh * fw, dt, UT_INIT_RANDOM);
    U8 *bias_cpu = ut_input_v(oc, dt, UT_INIT_RANDOM);

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
    Tensor filterTensorImg = Tensor(OCLMemImg);
    Tensor biasTensor = Tensor(OCLMem);
    Tensor tmpTensor = Tensor(OCLMem);
    Tensor tmpTensorImgA = Tensor(OCLMemImg);
    Tensor tmpTensorImgB = Tensor(OCLMemImg);
    if (use_nchw) {
        inputTensor = Tensor(OCLMem);
    }
    inputTensor.resize(inputDesc_gpu);
    filterTensorOrg.resize(filterDesc);
    biasTensor.resize(biasDesc);

    MaliPara maliPara;
    ForwardRunInfoMali runInfo;
    runInfo.algorithm = (I32)(CONVOLUTION_ALGORITHM_NULL);
    maliPara.handle = handle;
    maliPara.forwardRunInfo = &runInfo;
    archInfo.archPara = &maliPara;

    CHECK_STATUS(convolution_infer_output_size(
        &inputTensor, filterTensorOrg, convParamSpec, &outputTensor, dt, &archInfo));

    ConvolutionPolicy policy = CONVOLUTION_TUNNING;
    ConvolutionForwardAlgorithm alg = CONVOLUTION_ALGORITHM_NULL;
    CHECK_STATUS(convolution_infer_forward_algorithm(inputTensor, filterTensorOrg, outputTensor,
        convParamSpec, policy, &alg, dt, activationDesc, &archInfo));
    alg = (ConvolutionForwardAlgorithm)runInfo.algorithm;

    U32 maxBytes[7] = {0};
    CHECK_STATUS(convolution_infer_forward_tmp_bytes(
        inputTensor, filterTensorOrg, outputTensor, convParamSpec, alg, maxBytes, &archInfo));

    TensorDesc ftmDesc;
    CHECK_STATUS(convolution_transform_filter_bytes(
        filterTensorOrg, convParamSpec, alg, &ftmDesc, &archInfo));
    filterTensor.resize(ftmDesc);
    GCLMem_t input = alloc(inputTensor);
    GCLMem_t output = alloc(outputTensor);

    alloc_host_ptr(filterTensorOrg, filter_cpu);
    alloc(filterTensor);
    CHECK_STATUS(gcl_fill_memory_zero(handle, input));

    U32 ocAlign = UNI_ALIGN(oc, 4);
    if (ocAlign != oc) {
        U8 *bias_cpu_align = ut_input_v(ocAlign, dt, UT_INIT_ZERO);
        UNI_MEMCPY(bias_cpu_align, bias_cpu, oc * bytesOf(dt));
        free(bias_cpu);
        bias_cpu = bias_cpu_align;
    }
    if (runInfo.best_k[0] > 1 || alg == CONVOLUTION_ALGORITHM_WINOGRAD) {
        biasTensor = Tensor(OCLMemImg1D);
        biasTensor.resize(biasDesc);
        alloc_host_ptr(biasTensor, bias_cpu);
    } else {
        alloc_padding(biasTensor, 0, ocAlign - oc, 0, 0, bias_cpu);
    }
    TensorDesc outputDesc = outputTensor.get_desc();
    GCLMem_t tmpbuf;
    std::vector<Tensor> tmp(3, Tensor(OCLMem));
    U32 tmpBytes;
    tmpBytes = tensorNumBytes(inputDesc_gpu);
    maxBytes[0] = (tmpBytes > maxBytes[0]) ? tmpBytes : maxBytes[0];
    tmpBytes = tensorNumBytes(outputDesc);
    maxBytes[0] = (tmpBytes > maxBytes[0]) ? tmpBytes : maxBytes[0];
    tmpbuf = alloc_bytes(tmpTensor, maxBytes[0]);
    tmp[0] = tmpTensor;
    if (alloc_img(tmpTensorImgA, maxBytes + 1)) {
        tmp[0] = tmpTensorImgA;
    }
    alloc_img(tmpTensorImgB, maxBytes + 4);
    Tensor filterTensorTran = filterTensor;

    if (alg == CONVOLUTION_ALGORITHM_WINOGRAD && archInfo.arch == QUALCOMM) {
        tmp[0] = tmpTensor;
        tmp[1] = tmpTensorImgA;
        tmp[2] = tmpTensorImgB;
        filterTensorImg.resize(ftmDesc);
        alloc(filterTensorImg);
        filterTensorTran = filterTensorImg;
    }
    CHECK_STATUS(convolution_transform_filter(
        filterTensorOrg, convParamSpec, alg, tmpTensor, &filterTensorTran, &archInfo));

    CHECK_STATUS(ocl_set_input(handle, input, inputDesc_gpu, input_cpu, tmpbuf, true));
    std::vector<Tensor> inputTensors(1, inputTensor);
    CHECK_STATUS(convolution(inputTensors, filterTensorTran, convParamSpec, alg, nullptr,
        biasTensor, tmp, outputTensor, activationDesc, &archInfo));

    for (U32 i = 0; i < 20; i++) {
        CHECK_STATUS(gcl_run_kernelVec(handle));
    }
    CHECK_STATUS(gcl_finish(handle));

    std::vector<U32> kernelIndex;
    for (U32 i = 0; i < handle->kernelVec->size(); i++) {
        kernelIndex.push_back(i);
    }
    CHECK_STATUS(gcl_run_kernelVec_select_ls(handle, kernelIndex));
    CHECK_STATUS(gcl_finish(handle));
    double time = 0;
    for (I32 i = 0; i < UT_LOOPS; i++) {
        CHECK_STATUS(gcl_run_kernelVec_timing(handle, 0, handle->kernelVec->size()));
        time += handle->t_execute * 0.001;
    }
    time /= UT_LOOPS;

    U8 *output_gpu = ut_input_v(on * oc * ot * oh * ow, dt, UT_INIT_RANDOM);
    CHECK_STATUS(ocl_get_output(handle, output, outputDesc, output_gpu, tmpbuf, true));

    char buffer[150];
    char params[120];
    sprintf(params,
        "(%u %u %u %u %u)+(%u %u %u %u %u)/(%u %u %u %u %u %u %u %u %u %u)=(%u %u %u %u %u)", in,
        ic, it, ih, iw, fn, fc, ft, fh, fw, group, strideT, strideH, strideW, paddingTF, paddingTB,
        paddingT, paddingB, paddingL, paddingR, on, oc, ot, oh, ow);
    sprintf(buffer, "%20s, %80s", "Convolution", params);
    double ops = (1.0 * on * oc * oh * ow * ot) * (2.0 * ic * ft * fh * fw / group + 1);
    ut_log(dt, buffer, ops, time);
    Tensor inputTensorCpu = Tensor::alloc_sized<CPUMem>(inputDesc);
    Tensor filterTensorCpu = Tensor::alloc_sized<CPUMem>(filterDesc);
    Tensor biasTensorCpu = Tensor::alloc_sized<CPUMem>(biasDesc);
    outputDesc.df = DF_NCHW;
    Tensor outputTensorCpu = Tensor::alloc_sized<CPUMem>(outputDesc);
    UNI_MEMCPY(
        get_ptr_from_tensor(inputTensorCpu, CPU_GENERAL), input_cpu, tensorNumBytes(inputDesc));
    UNI_MEMCPY(
        get_ptr_from_tensor(filterTensorCpu, CPU_GENERAL), filter_cpu, tensorNumBytes(filterDesc));
    UNI_MEMCPY(get_ptr_from_tensor(biasTensorCpu, CPU_GENERAL), bias_cpu, tensorNumBytes(biasDesc));

    Tensor tmpTensorCpu;
    std::vector<Tensor> inputTensorsCpu(1, inputTensorCpu);
    std::vector<Tensor> tmpTensorsCpu(1, tmpTensorCpu);
    CHECK_STATUS(convolution(inputTensorsCpu, filterTensorCpu, convParamSpec,
        CONVOLUTION_ALGORITHM_GEMM, nullptr, biasTensorCpu, tmpTensorsCpu, outputTensorCpu,
        activationDesc, &UT_SERIAL_ARCHINFO));
    float threshold = 0.3;
    if (fh == fw && fw == 3) {
        threshold = 20;
    }
    ut_check_v(output_gpu, get_ptr_from_tensor(outputTensorCpu, CPU_GENERAL),
        on * oc * ow * oh * ot, dt, threshold);

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
    convolutionTest(argc, argv, DT_F16);
#else
    convolutionTest(argc, argv, DT_F32);
#endif
    return 0;
}
