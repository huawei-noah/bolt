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

int convolutionTest(int argc, char *argv[], DataType dt)
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

    ActivationParamSpec activationDesc;
    activationDesc.mode = ACTIVATION_RELU;
    activationDesc.value[0] = 0;

    TensorDesc inputDesc, outputDesc;
    if (ic % 8 != 0) {
        inputDesc = tensor4df(dt, DF_NCHW, in, ic, ih, iw);
    } else {
        inputDesc = tensor4df(dt, DF_NCHWC8, in, ic, ih, iw);
    }
    TensorDesc filterDesc = tensor4df(dt, DF_NCHW, fn, fc, fh, fw);
    TensorDesc biasDesc = tensor1d(dt, oc);
    ConvolutionParamSpec p = createConvolutionParamSpec(group, 1, fh, fw, 1, stride, stride, 0, 0,
        padding, padding, padding, padding, 1, 1, 1, fn, CONVOLUTION_DEPTHWISE_POINTWISE);

    // setup input, filter, bias
    U8 *input = ut_input_v(in * ic * ih * iw, dt, UT_INIT_RANDOM);
    U8 *filter = ut_input_v(fn * fc * fh * fw, dt, UT_INIT_RANDOM);
    U8 *bias = ut_input_v(oc, dt, UT_INIT_RANDOM);
    Tensor inputTensor;
    Tensor inputTensorRef;
    Tensor filterTensor;
    Tensor filterTensorRef;
    Tensor outputTensor;
    Tensor outputTensorRef;
    Tensor biasTensor;

    inputTensor.resize(inputDesc);
    inputTensorRef.resize(inputDesc);
    filterTensor.resize(filterDesc);
    filterTensorRef.resize(filterDesc);
    biasTensor.resize(biasDesc);

    inputTensor.alloc();
    inputTensorRef.alloc();
    filterTensor.alloc();
    filterTensorRef.alloc();
    biasTensor.alloc();
    UNI_MEMCPY(
        get_ptr_from_tensor(inputTensor, CPU_GENERAL), input, bytesOf(dt) * in * ic * ih * iw);
    UNI_MEMCPY(
        get_ptr_from_tensor(inputTensorRef, CPU_GENERAL), input, bytesOf(dt) * in * ic * ih * iw);
    UNI_MEMCPY(
        get_ptr_from_tensor(filterTensor, CPU_GENERAL), filter, bytesOf(dt) * fn * fc * fh * fw);
    UNI_MEMCPY(
        get_ptr_from_tensor(filterTensorRef, CPU_GENERAL), filter, bytesOf(dt) * fn * fc * fh * fw);
    UNI_MEMCPY(get_ptr_from_tensor(biasTensor, CPU_GENERAL), bias, bytesOf(dt) * oc);

    // setup output, bias
    CHECK_STATUS(convolution_infer_output_size(
        &inputTensor, filterTensor, p, &outputTensor, dt, &UT_CPU_ARCHINFO));
    outputDesc = outputTensor.get_desc();

    outputTensor.alloc();
    outputTensorRef.resize(outputTensor.get_desc());
    outputTensorRef.alloc();

    // setup alg
    ConvolutionPolicy policy = CONVOLUTION_FASTEST;
    ConvolutionForwardAlgorithm alg = CONVOLUTION_ALGORITHM_NULL;
    CHECK_STATUS(convolution_infer_forward_algorithm(inputTensor, filterTensor, outputTensor, p,
        policy, &alg, dt, activationDesc, &UT_CPU_ARCHINFO));

    // setup tmp
    U32 tmpBytes;
    CHECK_STATUS(convolution_infer_forward_tmp_bytes(
        inputTensor, filterTensor, outputTensor, p, alg, &tmpBytes, &UT_CPU_ARCHINFO));
    Tensor tmpTensor;
    tmpTensor.resize(tensor1d(DT_U8, tmpBytes));
    tmpTensor.alloc();

    // setup filter trans
    U32 ftmBytes;
    CHECK_STATUS(
        convolution_transform_filter_bytes(filterTensor, p, alg, &ftmBytes, &UT_CPU_ARCHINFO));
    // trans filter
    Tensor ftmTensor;
    ftmTensor.resize(tensor1d(DT_U8, ftmBytes));
    ftmTensor.alloc();
    CHECK_STATUS(
        convolution_transform_filter(filterTensor, p, alg, tmpTensor, &ftmTensor, &UT_CPU_ARCHINFO));
    std::vector<Tensor> inputTensors(1, inputTensor);
    std::vector<Tensor> inputTensorsRef(1, inputTensorRef);
    std::vector<Tensor> tmpTensors(1, tmpTensor);
    if (UT_CHECK) {
        CHECK_STATUS(convolution(inputTensors, ftmTensor, p, alg, nullptr, biasTensor, tmpTensors,
            outputTensor, activationDesc, &UT_CPU_ARCHINFO));

        // naive implement
        CHECK_STATUS(convolution(inputTensorsRef, filterTensorRef, p, alg, nullptr, biasTensor,
            tmpTensors, outputTensorRef, activationDesc, &UT_SERIAL_ARCHINFO));
        // check
        ut_check_v(get_ptr_from_tensor(outputTensor, CPU_GENERAL),
            get_ptr_from_tensor(outputTensorRef, CPU_GENERAL), outputTensor.length(), dt, 5,
            __FILE__, __LINE__);
    }

    // benchmark
    double time_start = ut_time_ms();
    for (int iter = 0; iter < UT_LOOPS; iter++) {
        CHECK_STATUS(convolution(inputTensors, ftmTensor, p, alg, nullptr, biasTensor, tmpTensors,
            outputTensor, activationDesc, &UT_CPU_ARCHINFO));
    }
    double time_end = ut_time_ms();
    double time = (time_end - time_start) / UT_LOOPS;

    // log performance data
    char buffer[150];
    char params[120];
    DataFormat df;
    CHECK_STATUS(tensor4dGet(outputDesc, &dt, &df, &on, &oc, &oh, &ow));
    sprintf(params, "(%u %u %u %u)+(%u %u %u %u)/(%u %u %u)=(%u %u %u %u)", in, ic, ih, iw, fn, fc,
        fh, fw, group, stride, padding, on, oc, oh, ow);
    sprintf(buffer, "%20s, %80s", "Convolution", params);
    double ops = (1.0 * on * oc * oh * ow) * (2.0 * ic / group * fh * fw + 1);
    ut_log(dt, buffer, ops, time);

    free(input);
    free(filter);
    free(bias);
    return 0;
}

int main(int argc, char **argv)
{
#ifdef _USE_FP16
    convolutionTest(argc, argv, DT_F16);
#endif
#ifdef _USE_FP32
    convolutionTest(argc, argv, DT_F32);
#endif
    return 0;
}
