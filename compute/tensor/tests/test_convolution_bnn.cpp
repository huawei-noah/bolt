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

int bnnConvolutionTest(int argc, char *argv[], DataType dt)
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
    ArchInfo archInfo;
    archInfo.arch = UT_ARCH;
    ArchInfo archInfo_org;
    archInfo_org.arch = CPU_GENERAL;

    CHECK_REQUIREMENT(in == 1 && on == 1);

    DataType fdt = DT_BIN11;  // Use dt to distinguish DoReFa and XNOR
    ActivationParamSpec activationDesc;
    activationDesc.mode = ACTIVATION_NULL;

    TensorDesc inputDesc = tensor4df(dt, DF_NCHWC8, in, ic, ih, iw);
    TensorDesc filterDesc = tensor4df(fdt, DF_NCHW, oc, ic, fh, fw);
    TensorDesc biasDesc = tensor1d(dt, oc * 2);  // including scale and bias
    ConvolutionParamSpec p = createConvolutionParamSpec(group, 1, fh, fw, 1, stride, stride, 0, 0,
        padding, padding, padding, padding, 1, 1, 1, oc, Convolution_Depthwise_Pointwise);

    // setup input, filter, bias
    U8 *input = ut_input_v(in * ic * ih * iw, dt, UT_INIT_RANDOM);
    if (fdt == DT_BIN01) {
        for (U32 i = 0; i < in * ic * ih * iw; i++) {
            switch (dt) {
#ifdef _USE_FP16
                case DT_F16:
                    ((F16 *)input)[i] += 0.5;
                    break;
#endif
#ifdef _USE_FP32
                case DT_F32:
                    ((F32 *)input)[i] += 0.5;
                    break;
#endif
                default:
                    break;
            }
        }
    }

    BIN8 *filter = (BIN8 *)ut_input_v(fn * fc * fh * fw / 8, fdt, UT_INIT_POS);
    U8 *bias = ut_input_v(oc * 2, dt, UT_INIT_RANDOM);
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
    memcpy(get_ptr_from_tensor(inputTensor, UT_ARCH), input, bytesOf(dt) * in * ic * ih * iw);
    memcpy(get_ptr_from_tensor(inputTensorRef, UT_ARCH), input, bytesOf(dt) * in * ic * ih * iw);
    memcpy(get_ptr_from_tensor(filterTensor, UT_ARCH), filter, tensorNumBytes(filterDesc));
    memcpy(get_ptr_from_tensor(filterTensorRef, UT_ARCH), filter, tensorNumBytes(filterDesc));
    memcpy(get_ptr_from_tensor(biasTensor, UT_ARCH), bias, tensorNumBytes(biasDesc));

    // setup output, bias
    CHECK_STATUS(
        convolution_infer_output_size(&inputTensor, filterTensor, p, &outputTensor, dt, &archInfo));

    outputTensor.alloc();
    outputTensorRef.resize(outputTensor.get_desc());
    outputTensorRef.alloc();

    // setup alg
    ConvolutionPolicy policy = CONVOLUTION_FASTEST;
    ConvolutionForwardAlgorithm alg = CONVOLUTION_ALGORITHM_NULL;
    CHECK_STATUS(convolution_infer_forward_algorithm(
        inputTensor, filterTensor, outputTensor, p, policy, &alg, fdt, activationDesc, &archInfo));

    // setup tmp
    U32 tmpBytes;
    CHECK_STATUS(convolution_infer_forward_tmp_bytes(
        inputTensor, filterTensor, outputTensor, p, alg, &tmpBytes, &archInfo));
    Tensor tmpTensor;
    tmpTensor.resize(tensor1d(DT_U8, tmpBytes));
    tmpTensor.alloc();
    // setup filter trans
    U32 ftmBytes;
    CHECK_STATUS(convolution_transform_filter_bytes(filterTensor, p, alg, &ftmBytes, &archInfo));
    // trans filter
    Tensor ftmTensor;
    ftmTensor.resize(tensor1d(DT_U8, ftmBytes));
    ftmTensor.alloc();

    CHECK_STATUS(
        convolution_transform_filter(filterTensor, p, alg, tmpTensor, &ftmTensor, &archInfo));

    std::vector<Tensor> inputTensors(1, inputTensor);
    std::vector<Tensor> inputTensorsRef(1, inputTensorRef);
    if (UT_CHECK) {
        CHECK_STATUS(convolution(inputTensors, ftmTensor, p, alg, nullptr, biasTensor, tmpTensor,
            outputTensor, activationDesc, &archInfo));
        // naive implement
        CHECK_STATUS(convolution(inputTensorsRef, filterTensorRef, p, alg, nullptr, biasTensor,
            tmpTensor, outputTensorRef, activationDesc, &archInfo_org));
        // check
        ut_check_v(get_ptr_from_tensor(outputTensor, UT_ARCH),
            get_ptr_from_tensor(outputTensorRef, UT_ARCH), outputTensor.length(), dt, 1, __FILE__,
            __LINE__);
    }

    // benchmark
    double time_start = ut_time_ms();
    for (int iter = 0; iter < UT_LOOPS; iter++) {
        CHECK_STATUS(convolution(inputTensors, ftmTensor, p, alg, nullptr, biasTensor, tmpTensor,
            outputTensor, activationDesc, &archInfo));
    }
    double time_end = ut_time_ms();
    double time = (time_end - time_start) / UT_LOOPS;

    // log performance data
    char buffer[150];
    char params[120];
    sprintf(params, "(%u %u %u %u)+(%u %u %u %u)/(%u %u)=(%u %u %u %u)", in, ic, ih, iw, fn, fc, fh,
        fw, stride, padding, on, oc, oh, ow);
    sprintf(buffer, "%20s, %80s", "BNN Convolution", params);
    double ops = (1.0 * on * oc * oh * ow) * (2.0 * ic * fh * fw + 1);
    ut_log(DT_I8, buffer, ops, time);

    free(input);
    free(filter);
    free(bias);
    return 0;
}

int main(int argc, char **argv)
{
#ifdef _USE_FP16
    bnnConvolutionTest(argc, argv, DT_F16);
#endif
    return 0;
}
