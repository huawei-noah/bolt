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

int main(int argc, char *argv[])
{
#ifdef _USE_INT8
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
    ArchInfo archInfo;
    archInfo.arch = UT_ARCH;
    ArchInfo archInfo_org;
    archInfo_org.arch = CPU_GENERAL;

    DataType dt = DT_I8;
    DataType odt = DT_I32;
    ActivationParamSpec dwActivationParamSpec;
    ActivationParamSpec pwActivationParamSpec;
    dwActivationParamSpec.mode = ACTIVATION_RELU6;
    pwActivationParamSpec.mode = ACTIVATION_RELU6;

    TensorDesc inputDesc, dwFilterDesc, pwFilterDesc, outputDesc, dwBiasDesc, pwBiasDesc;
    inputDesc = tensor4df(dt, DF_NCHWC8, in, ic, ih, iw);
    dwFilterDesc = tensor4df(dt, DF_NCHW, 1, ic, fh, fw);
    pwFilterDesc = tensor4df(dt, DF_NCHW, oc, ic, 1, 1);
    dwBiasDesc = tensor1d(odt, ic);
    pwBiasDesc = tensor1d(odt, oc);
    ConvolutionParamSpec convParamSpec = createConvolutionParamSpec(group, 1, fh, fw, 1, stride,
        stride, 0, 0, padding, padding, padding, padding, 1, 1, 1, fn, Convolution_Depthwise);

    // setup input, filter, bias
    INT8 *input = (INT8 *)ut_input_v(in * ic * ih * iw, DT_I8, UT_INIT_RANDOM);
    INT8 *dwFilter = (INT8 *)ut_input_v(tensorNumElements(dwFilterDesc), DT_I8, UT_INIT_RANDOM);
    INT8 *pwFilter = (INT8 *)ut_input_v(tensorNumElements(pwFilterDesc), DT_I8, UT_INIT_RANDOM);
    I32 *dwBias = (I32 *)ut_input_v(ic, DT_I32, UT_INIT_RANDOM);
    I32 *pwBias = (I32 *)ut_input_v(oc, DT_I32, UT_INIT_RANDOM);

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
    memcpy(get_ptr_from_tensor(inputTensor, UT_ARCH), input, bytesOf(dt) * in * ic * ih * iw);
    memcpy(get_ptr_from_tensor(inputTensorRef, UT_ARCH), input, bytesOf(dt) * in * ic * ih * iw);
    memcpy(get_ptr_from_tensor(dwFilterTensor, UT_ARCH), dwFilter, bytesOf(dt) * 1 * ic * fh * fw);
    memcpy(
        get_ptr_from_tensor(dwFilterTensorRef, UT_ARCH), dwFilter, bytesOf(dt) * 1 * ic * fh * fw);
    memcpy(get_ptr_from_tensor(dwBiasTensor, UT_ARCH), dwBias, bytesOf(dt) * ic);

    Tensor pwFilterTensor;
    Tensor pwFilterTensorRef;
    Tensor pwBiasTensor;
    pwFilterTensor.resize(pwFilterDesc);
    pwFilterTensorRef.resize(pwFilterDesc);
    pwBiasTensor.resize(pwBiasDesc);
    pwFilterTensor.alloc();
    pwFilterTensorRef.alloc();
    pwBiasTensor.alloc();
    memcpy(get_ptr_from_tensor(pwFilterTensor, UT_ARCH), pwFilter, bytesOf(dt) * oc * ic * 1 * 1);
    memcpy(get_ptr_from_tensor(pwFilterTensorRef, UT_ARCH), pwFilter, bytesOf(dt) * oc * ic * 1 * 1);
    memcpy(get_ptr_from_tensor(pwBiasTensor, UT_ARCH), pwBias, bytesOf(dt) * oc);

    // setup output, bias
    CHECK_STATUS(depthwise_pointwise_convolution_infer_output_size(&inputTensor, dwFilterTensor,
        pwFilterTensor, convParamSpec, &outputTensor, odt, &archInfo));
    outputTensor.alloc();
    outputTensorRef.resize(outputTensor.get_desc());
    outputTensorRef.alloc();

    // setup alg
    ConvolutionPolicy policy = CONVOLUTION_FASTEST;
    DepthwiseConvolutionForwardAlgorithm alg = DEPTHWISE_CONVOLUTION_ALGORITHM_NULL;
    CHECK_STATUS(depthwise_pointwise_convolution_infer_forward_algorithm(inputTensor,
        dwFilterTensor, pwFilterTensor, outputTensor, convParamSpec, policy, &alg, dt,
        dwActivationParamSpec, pwActivationParamSpec, &archInfo));

    // setup tmp
    U32 tmpBytes;
    CHECK_STATUS(depthwise_pointwise_convolution_infer_forward_tmp_bytes(inputTensor,
        dwFilterTensor, pwFilterTensor, outputTensor, convParamSpec, alg, &tmpBytes, &archInfo));
    Tensor tmpTensor;
    tmpTensor.resize(tensor1d(DT_U8, tmpBytes));
    tmpTensor.alloc();

    // setup filter trans
    U32 dwBytes, pwBytes;
    CHECK_STATUS(depthwise_pointwise_convolution_transform_filter_bytes(
        dwFilterTensor, pwFilterTensor, convParamSpec, alg, &dwBytes, &pwBytes, &archInfo));
    Tensor dwFtmTensor;
    dwFtmTensor.resize(tensor1d(DT_U8, dwBytes));
    dwFtmTensor.alloc();
    Tensor pwFtmTensor;
    pwFtmTensor.resize(tensor1d(DT_U8, pwBytes));
    pwFtmTensor.alloc();
    // trans filter
    CHECK_STATUS(depthwise_pointwise_convolution_transform_filter(
        dwFilterTensor, pwFilterTensor, convParamSpec, alg, &dwFtmTensor, &pwFtmTensor, &archInfo));

    if (UT_CHECK) {
        CHECK_STATUS(depthwise_pointwise_convolution(inputTensor, dwFtmTensor, pwFtmTensor,
            convParamSpec, alg, dwBiasTensor, pwBiasTensor, tmpTensor, outputTensor,
            dwActivationParamSpec, pwActivationParamSpec, &archInfo));

        // naive implement
        CHECK_STATUS(depthwise_pointwise_convolution(inputTensorRef, dwFilterTensorRef,
            pwFilterTensorRef, convParamSpec, alg, dwBiasTensor, pwBiasTensor, tmpTensor,
            outputTensorRef, dwActivationParamSpec, pwActivationParamSpec, &archInfo_org));

        // check
        ut_check_v(get_ptr_from_tensor(outputTensor, UT_ARCH),
            get_ptr_from_tensor(outputTensorRef, UT_ARCH), outputTensor.length(), DT_I32, 1,
            __FILE__, __LINE__);
    }

    // benchmark
    double time_start = ut_time_ms();
    for (int iter = 0; iter < UT_LOOPS; iter++) {
        CHECK_STATUS(depthwise_pointwise_convolution(inputTensor, dwFtmTensor, pwFtmTensor,
            convParamSpec, alg, dwBiasTensor, pwBiasTensor, tmpTensor, outputTensor,
            dwActivationParamSpec, pwActivationParamSpec, &archInfo));
    }
    double time_end = ut_time_ms();
    double time = (time_end - time_start) / UT_LOOPS;

    // log performance data
    char buffer[150];
    char params[120];
    sprintf(params, "(%u %u %u %u)+(%u %u %u %u)/(%u %u)=(%u %u %u %u)", in, ic, ih, iw, fn, fc, fh,
        fw, stride, padding, on, oc, oh, ow);
    sprintf(buffer, "%20s, %80s", "DepthwiseConvolution", params);
    double ops = 2.0 * in * ic * ih * iw * fh * fw + in * ic * oh * ow +
        2.0 * on * oc * oh * ow * ic + on * oc * oh * ow;
    ut_log(DT_I8, buffer, ops, time);

    free(input);
    free(dwFilter);
    free(pwFilter);
    free(dwBias);
    free(pwBias);
#endif

    return 0;
}
