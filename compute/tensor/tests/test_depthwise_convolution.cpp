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
#include "ut_util.h"

int depthwiseConvolutionTest(int argc, char *argv[], bool isFusedWithPw, DataType dt)
{
    CHECK_REQUIREMENT(argc == 16);
    // in data
    U32 in = atoi(argv[1]);
    U32 ic = atoi(argv[2]);
    U32 ih = atoi(argv[3]);
    U32 iw = atoi(argv[4]);
    // weight
    U32 fn = atoi(argv[5]);
    U32 fc = atoi(argv[6]);
    U32 fh = atoi(argv[7]);
    U32 fw = atoi(argv[8]);
    U32 group = atoi(argv[9]);
    // stride & padding
    U32 stride = atoi(argv[10]);
    U32 padding = atoi(argv[11]);
    // output
    U32 on = atoi(argv[12]);
    U32 oc = atoi(argv[13]);
    U32 oh = atoi(argv[14]);
    U32 ow = atoi(argv[15]);
    CHECK_REQUIREMENT(in == 1 && on == 1);

    ActivationParamSpec dwActivationParamSpec;
    ActivationParamSpec pwActivationParamSpec;
    dwActivationParamSpec.mode = ACTIVATION_NULL;
    pwActivationParamSpec.mode = ACTIVATION_NULL;

    TensorDesc inputDesc, dwFilterDesc, pwFilterDesc, outputDesc, dwBiasDesc, pwBiasDesc;
    inputDesc = tensor4df(dt, DF_NCHWC8, in, ic, ih, iw);
    dwFilterDesc = tensor4df(dt, DF_NCHW, 1, ic, fh, fw);
    dwBiasDesc = tensor1d(dt, ic);
    if (isFusedWithPw) {
        pwFilterDesc = tensor4df(dt, DF_NCHW, oc, ic, 1, 1);
        pwBiasDesc = tensor1d(dt, oc);
    }
    ConvolutionParamSpec p = createConvolutionParamSpec(group, 1, fh, fw, 1, stride, stride, 0, 0,
        padding, padding, padding, padding, 1, 1, 1, fn, CONVOLUTION_DEPTHWISE);

    // setup input, filter, bias
    U8 *dwFilter = nullptr;
    U8 *dwBias = nullptr;
    U8 *pwFilter = nullptr;
    U8 *pwBias = nullptr;

    U8 *input = ut_input_v(in * ic * ih * iw, dt, UT_INIT_RANDOM);
    dwFilter = ut_input_v(tensorNumElements(dwFilterDesc), dt, UT_INIT_RANDOM);
    dwBias = ut_input_v(tensorNumElements(dwBiasDesc), dt, UT_INIT_RANDOM);
    Tensor inputTensor;
    Tensor inputTensorRef;
    Tensor dwFilterTensor;
    Tensor dwFilterTensorRef;
    Tensor outputTensor;
    Tensor outputTensorRef;
    Tensor dwBiasTensor;

    inputTensor.resize(inputDesc);
    inputTensorRef.resize(inputDesc);
    dwFilterTensor.resize(dwFilterDesc);
    dwFilterTensorRef.resize(dwFilterDesc);
    dwBiasTensor.resize(dwBiasDesc);

    inputTensor.alloc();
    inputTensorRef.alloc();
    dwFilterTensor.alloc();
    dwFilterTensorRef.alloc();
    dwBiasTensor.alloc();
    UNI_MEMCPY(
        get_ptr_from_tensor(inputTensor, CPU_GENERAL), input, bytesOf(dt) * in * ic * ih * iw);
    UNI_MEMCPY(
        get_ptr_from_tensor(inputTensorRef, CPU_GENERAL), input, bytesOf(dt) * in * ic * ih * iw);
    UNI_MEMCPY(
        get_ptr_from_tensor(dwFilterTensor, CPU_GENERAL), dwFilter, bytesOf(dt) * 1 * ic * fh * fw);
    UNI_MEMCPY(get_ptr_from_tensor(dwFilterTensorRef, CPU_GENERAL), dwFilter,
        bytesOf(dt) * 1 * ic * fh * fw);
    UNI_MEMCPY(get_ptr_from_tensor(dwBiasTensor, CPU_GENERAL), dwBias, bytesOf(dt) * ic);
    Tensor pwFilterTensor;
    Tensor pwFilterTensorRef;
    Tensor pwBiasTensor;
    if (isFusedWithPw) {
        pwFilter = ut_input_v(tensorNumElements(pwFilterDesc), dt, UT_INIT_RANDOM);
        pwBias = ut_input_v(tensorNumElements(pwBiasDesc), dt, UT_INIT_RANDOM);
        pwFilterTensor.resize(pwFilterDesc);
        pwFilterTensorRef.resize(pwFilterDesc);
        pwBiasTensor.resize(pwBiasDesc);
        pwFilterTensor.alloc();
        pwFilterTensorRef.alloc();
        pwBiasTensor.alloc();
        UNI_MEMCPY(get_ptr_from_tensor(pwFilterTensor, CPU_GENERAL), pwFilter,
            bytesOf(dt) * oc * ic * 1 * 1);
        UNI_MEMCPY(get_ptr_from_tensor(pwFilterTensorRef, CPU_GENERAL), pwFilter,
            bytesOf(dt) * oc * ic * 1 * 1);
        UNI_MEMCPY(get_ptr_from_tensor(pwBiasTensor, CPU_GENERAL), pwBias, bytesOf(dt) * oc);
    }

    // setup output, bias
    if (isFusedWithPw) {
        CHECK_STATUS(depthwise_pointwise_convolution_infer_output_size(
            &inputTensor, dwFilterTensor, pwFilterTensor, p, &outputTensor, dt, &UT_CPU_ARCHINFO));
    } else {
        CHECK_STATUS(depthwise_convolution_infer_output_size(
            &inputTensor, dwFilterTensor, p, &outputTensor, dt, &UT_CPU_ARCHINFO));
    }
    DataType odt;
    DataFormat odf;
    CHECK_STATUS(tensor4dGet(outputTensor.get_desc(), &odt, &odf, &on, &oc, &oh, &ow));

    outputTensor.alloc();
    outputTensorRef.resize(outputTensor.get_desc());
    outputTensorRef.alloc();

    // setup alg
    ConvolutionPolicy policy = CONVOLUTION_FASTEST;
    DepthwiseConvolutionForwardAlgorithm alg = DEPTHWISE_CONVOLUTION_ALGORITHM_NULL;
    if (isFusedWithPw) {
        CHECK_STATUS(depthwise_pointwise_convolution_infer_forward_algorithm(inputTensor,
            dwFilterTensor, pwFilterTensor, outputTensor, p, policy, &alg, dt,
            dwActivationParamSpec, pwActivationParamSpec, &UT_CPU_ARCHINFO));
    } else {
        CHECK_STATUS(depthwise_convolution_infer_forward_algorithm(inputTensor, dwFilterTensor,
            outputTensor, p, policy, &alg, dt, dwActivationParamSpec, &UT_CPU_ARCHINFO));
    }

    // setup tmp
    U32 tmpBytes;
    if (isFusedWithPw) {
        CHECK_STATUS(depthwise_pointwise_convolution_infer_forward_tmp_bytes(inputTensor,
            dwFilterTensor, pwFilterTensor, outputTensor, p, alg, &tmpBytes, &UT_CPU_ARCHINFO));
    } else {
        CHECK_STATUS(depthwise_convolution_infer_forward_tmp_bytes(
            inputTensor, dwFilterTensor, outputTensor, p, alg, &tmpBytes, &UT_CPU_ARCHINFO));
    }
    Tensor tmpTensor;
    tmpTensor.resize(tensor1d(DT_U8, tmpBytes));
    tmpTensor.alloc();

    // setup filter trans
    U32 dwBytes, pwBytes;
    if (isFusedWithPw) {
        CHECK_STATUS(depthwise_pointwise_convolution_transform_filter_bytes(
            dwFilterTensor, pwFilterTensor, p, alg, &dwBytes, &pwBytes, &UT_CPU_ARCHINFO));
    } else {
        CHECK_STATUS(depthwise_convolution_transform_filter_bytes(
            dwFilterTensor, p, alg, &dwBytes, &UT_CPU_ARCHINFO));
    }
    Tensor dwFtmTensor;
    dwFtmTensor.resize(tensor1d(DT_U8, dwBytes));
    dwFtmTensor.alloc();
    Tensor pwFtmTensor;
    if (isFusedWithPw) {
        pwFtmTensor.resize(tensor1d(DT_U8, pwBytes));
        pwFtmTensor.alloc();
    }

    // trans filter
    if (isFusedWithPw) {
        CHECK_STATUS(depthwise_pointwise_convolution_transform_filter(
            dwFilterTensor, pwFilterTensor, p, alg, &dwFtmTensor, &pwFtmTensor, &UT_CPU_ARCHINFO));
    } else {
        CHECK_STATUS(depthwise_convolution_transform_filter(
            dwFilterTensor, p, alg, &dwFtmTensor, &UT_CPU_ARCHINFO));
    }

    std::vector<Tensor> inputTensors(1, inputTensor);
    std::vector<Tensor> inputTensorsRef(1, inputTensorRef);
    std::vector<Tensor> tmpTensors(1, tmpTensor);

    if (UT_CHECK) {
        if (isFusedWithPw) {
            CHECK_STATUS(depthwise_pointwise_convolution(inputTensors, dwFtmTensor, pwFtmTensor, p,
                alg, nullptr, dwBiasTensor, pwBiasTensor, tmpTensors, outputTensor,
                dwActivationParamSpec, pwActivationParamSpec, &UT_CPU_ARCHINFO));

            // naive implement
            CHECK_STATUS(depthwise_pointwise_convolution(inputTensorsRef, dwFilterTensorRef,
                pwFilterTensorRef, p, alg, nullptr, dwBiasTensor, pwBiasTensor, tmpTensors,
                outputTensorRef, dwActivationParamSpec, pwActivationParamSpec, &UT_SERIAL_ARCHINFO));
        } else {
            CHECK_STATUS(depthwise_convolution(inputTensor, dwFtmTensor, p, alg, nullptr,
                dwBiasTensor, tmpTensor, outputTensor, dwActivationParamSpec, &UT_CPU_ARCHINFO));

            // naive implement
            CHECK_STATUS(depthwise_convolution(inputTensorRef, dwFilterTensorRef, p, alg, nullptr,
                dwBiasTensor, tmpTensor, outputTensorRef, dwActivationParamSpec,
                &UT_SERIAL_ARCHINFO));
        }

        // check
        ut_check_v(get_ptr_from_tensor(outputTensor, CPU_GENERAL),
            get_ptr_from_tensor(outputTensorRef, CPU_GENERAL), outputTensor.length(), dt, 0.1);
    }

    // benchmark
    double time_start = ut_time_ms();
    for (int iter = 0; iter < UT_LOOPS; iter++) {
        if (isFusedWithPw) {
            CHECK_STATUS(depthwise_pointwise_convolution(inputTensors, dwFtmTensor, pwFtmTensor, p,
                alg, nullptr, dwBiasTensor, pwBiasTensor, tmpTensors, outputTensor,
                dwActivationParamSpec, pwActivationParamSpec, &UT_CPU_ARCHINFO));
        } else {
            CHECK_STATUS(depthwise_convolution(inputTensor, dwFtmTensor, p, alg, nullptr,
                dwBiasTensor, tmpTensor, outputTensor, dwActivationParamSpec, &UT_CPU_ARCHINFO));
        }
    }
    double time_end = ut_time_ms();
    double time = (time_end - time_start) / UT_LOOPS;

    // log performance data
    char buffer[150];
    char params[120];
    sprintf(params, "(%u %u %u %u)+(%u %u %u %u)/(%u %u)=(%u %u %u %u)", in, ic, ih, iw, fn, fc, fh,
        fw, stride, padding, on, oc, oh, ow);
    double ops = 0;
    if (isFusedWithPw) {
        sprintf(buffer, "%20s, %80s", "DepthwisePointwise", params);
        ops = 2.0 * in * ic * oh * ow * fh * fw + in * ic * oh * ow + 2.0 * on * oc * oh * ow * ic +
            on * oc * oh * ow;
    } else {
        sprintf(buffer, "%20s, %80s", "DepthwiseConvolution", params);
        ops = 1.0 * in * ic * oh * ow * (2.0 * fh * fw + 1);
    }
    ut_log(dt, buffer, ops, time);

    free(input);
    free(dwFilter);
    free(dwBias);
    if (isFusedWithPw) {
        free(pwFilter);
        free(pwBias);
    }
    return 0;
}

int main(int argc, char *argv[])
{
#ifdef _USE_FP16
    depthwiseConvolutionTest(argc, argv, true, DT_F16);
    depthwiseConvolutionTest(argc, argv, false, DT_F16);
#endif
#ifdef _USE_FP32
    depthwiseConvolutionTest(argc, argv, true, DT_F32);
    depthwiseConvolutionTest(argc, argv, false, DT_F32);
#endif
    return 0;
}
